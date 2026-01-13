import json
import uuid


from src.messages.tool_message import ToolCall


def _extract_tool_calls_anthropic(response):
    """
    Extract tool calls from Anthropic Claude response.

    Anthropic returns content as a list of blocks. Tool calls are blocks with:
    - type="tool_use"
    - id: unique identifier
    - name: tool name
    - input: arguments (dict)

    Args:
        response: Anthropic Message response object with .content attribute

    Returns:
        List of ToolCall objects extracted from tool_use blocks
    """
    tool_calls = []

    if not hasattr(response, "content"):
        return []

    for block in response.content:
        # Handles both dict and object notation
        block_type = (
            block.get("type") if isinstance(block, dict)
            else getattr(block, "type", None)
        )

        if block_type == "tool_use":
            tool_id = block.get("id") if isinstance(block, dict) else block.id
            tool_name = block.get("name") if isinstance(block, dict) else block.name
            tool_input = block.get("input") if isinstance(block, dict) else block.input

            tool_calls.append(
                ToolCall(
                    id=tool_id,
                    name=tool_name,
                    arguments=tool_input
                )
            )

    return tool_calls



def _has_structured_tool_calls(resp):
    """Check if response has structured tool calls."""
    return (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            resp.data.chat_response.choices and
            resp.data.chat_response.choices[0].message.tool_calls)


def _parse_double_encoded_json(arguments):
    """Parse potentially double-encoded JSON arguments."""
    if not isinstance(arguments, str):
        return arguments

    try:
        parsed = json.loads(arguments)

        # Handle double encoding - keep parsing until we get a non-string
        while isinstance(parsed, str):
            try:
                # Try to parse again
                next_parsed = json.loads(parsed)
                parsed = next_parsed
            except json.JSONDecodeError:
                # Try fixing invalid JSON by replacing single quotes
                try:
                    fixed = parsed.replace("'", '"')
                    next_parsed = json.loads(fixed)
                    parsed = next_parsed
                except json.JSONDecodeError:
                    # If all parsing attempts fail, break and return what we have
                    break

        return parsed
    except json.JSONDecodeError:
        # If initial parsing fails, return original arguments
        return arguments


def _extract_structured_tool_calls(resp):
    """Extract tool calls from structured response format."""
    if not _has_structured_tool_calls(resp):
        return []

    structured_calls = []
    for call in resp.data.chat_response.choices[0].message.tool_calls:
        if call.type != "FUNCTION":
            continue

        arguments = _parse_double_encoded_json(call.arguments)
        structured_calls.append(ToolCall(
            id=call.id,
            name=call.name,
            arguments=arguments
        ))

    return structured_calls


def _has_content_in_response(resp):
    """Check if response has content field."""
    return (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            resp.data.chat_response.choices and
            resp.data.chat_response.choices[0].message.content)


def _extract_content_from_response(resp):
    """Extract text content from response."""
    if not _has_content_in_response(resp):
        return ""

    return "".join([
        c.text for c in resp.data.chat_response.choices[0].message.content
        if hasattr(c, 'text') and c.type == "TEXT"
    ])


def _find_json_start_position(content):
    """Find the starting position of JSON in content."""
    return content.find('{')


def _preprocess_json_content(content, start_pos):
    """Preprocess JSON content by removing invalid escapes."""
    return content[start_pos:].replace("\\'", "'")


def _extract_arguments_from_object(obj):
    """Extract arguments from parsed JSON object."""
    arguments = obj.get("parameters") or obj.get("arguments", {})

    # If no explicit arguments, use all keys except "name" and "type"
    if not arguments or not isinstance(arguments, dict):
        return {key: value for key, value in obj.items()
                if key not in ["name", "type"]}

    return arguments


def _parse_tool_call_from_content(content):
    """Parse tool call from text content using JSON decoder."""
    start_pos = _find_json_start_position(content)
    if start_pos == -1:
        return []

    json_content = _preprocess_json_content(content, start_pos)

    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(json_content, 0)

        # Verify it's a tool call structure
        if not isinstance(obj, dict) or "name" not in obj:
            return []

        tool_id = str(uuid.uuid4())
        arguments = _extract_arguments_from_object(obj)

        return [ToolCall(
            id=tool_id,
            name=obj["name"],
            arguments=arguments
        )]
    except json.JSONDecodeError:
        return []


#####---------OCI GENERIC (Llama, Gemini, Grok, OpenAI)
def _extract_tool_calls_generic(resp):
    """
    Extract tool calls for GenericChatRequest (Llama, Gemini, Grok, OpenAI).
    Path: resp.data.chat_response.choices[0].message.tool_calls
    """
    # Try structured tool calls first
    structured_calls = _extract_structured_tool_calls(resp)
    if structured_calls:
        return structured_calls

    # Fallback: parse from content
    content = _extract_content_from_response(resp)
    if not content:
        return []

    return _parse_tool_call_from_content(content)


#####---------OCI COHERE (Cohere Command models)
def _extract_tool_calls_cohere(resp):
    """
    Extract tool calls for CohereChatRequest (Cohere Command models).
    Path: resp.data.chat_response.tool_calls
    Structure: {name: str, parameters: dict}
    """

    # 1. First try: structured tool calls (standard Cohere mode)
    structured_calls = []

    # Cohere: tool_calls are directly in chat_response, not inside choices[0].message
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            hasattr(resp.data.chat_response, 'tool_calls') and
            resp.data.chat_response.tool_calls):

        for call in resp.data.chat_response.tool_calls:
            # Cohere uses 'name' and 'parameters' (not 'arguments')
            tool_id = str(uuid.uuid4())  # Cohere doesn't provide ID, we generate it

            structured_calls.append(ToolCall(
                id=tool_id,
                name=call.name if hasattr(call, 'name') else "",
                arguments=call.parameters if hasattr(call, 'parameters') else {}
            ))

    # If we have structured tool calls, use them
    if structured_calls:
        return structured_calls

    # 2. Fallback: robust JSON parsing from content
    content = ""
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            hasattr(resp.data.chat_response, 'text')):
        content = resp.data.chat_response.text

    if not content:
        return []

    parsed_calls = []

    # Find the first brace
    start_pos = content.find('{')
    if start_pos == -1:
        return []

    # Pre-processing: remove invalid escapes (e.g. \' which is not valid in JSON)
    json_content = content[start_pos:].replace("\\'", "'")

    # Use JSONDecoder for automatic and robust parsing
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(json_content, 0)

        # Verify it has the structure of a tool call
        if isinstance(obj, dict) and "name" in obj:
            tool_id = str(uuid.uuid4())

            # Extract parameters - Cohere uses "parameters"
            arguments = obj.get("parameters") or obj.get("arguments", {})

            # If arguments is empty, use all keys except "name" and "type"
            if not arguments or not isinstance(arguments, dict):
                arguments = {}
                for key, value in obj.items():
                    if key not in ["name", "type"]:
                        arguments[key] = value

            parsed_calls.append(ToolCall(
                id=tool_id,
                name=obj["name"],
                arguments=arguments
            ))
    except json.JSONDecodeError:
        pass

    return parsed_calls


# Keep for backward compatibility
def _extract_tool_calls_hybrid(resp):
    """
    [DEPRECATED] Use _extract_tool_calls_generic or _extract_tool_calls_cohere.
    Kept for backward compatibility with existing code.
    """
    return _extract_tool_calls_generic(resp)


def _extract_tool_calls_ibm_watson_hybrid(resp):
    """Extract tool calls for IBM Watson: first structured, then from content"""

    # 1. First try: structured tool calls (standard IBM Watson mode)
    structured_calls = []
    if (resp.get("choices") and
            resp["choices"][0].get("message", {}).get("tool_calls")):
        structured_calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=json.loads(call["function"]["arguments"])
                if isinstance(call["function"]["arguments"], str)
                else call["function"]["arguments"]
            )
            for call in resp["choices"][0]["message"]["tool_calls"]
            if call.get("type") == "function"
        ]

    # If we have structured tool calls, use them
    if structured_calls:
        return structured_calls

    # 2. Fallback: robust JSON parsing from content
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        return []

    parsed_calls = []

    # Find the first brace
    start_pos = content.find('{')
    if start_pos == -1:
        return []

    # Pre-processing: remove invalid escapes (e.g. \' which is not valid in JSON)
    json_content = content[start_pos:].replace("\\'", "'")

    # Use JSONDecoder for automatic and robust parsing
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(json_content, 0)

        # Verify it has the structure of a tool call
        if isinstance(obj, dict) and "name" in obj:
            tool_id = str(uuid.uuid4())

            # Extract parameters - supports both "parameters" and "arguments"
            arguments = obj.get("parameters") or obj.get("arguments", {})

            # If arguments is empty, use all keys except "name" and "type"
            if not arguments or not isinstance(arguments, dict):
                arguments = {}
                for key, value in obj.items():
                    if key not in ["name", "type"]:
                        arguments[key] = value

            parsed_calls.append(ToolCall(
                id=tool_id,
                name=obj["name"],
                arguments=arguments
            ))
    except json.JSONDecodeError:
        pass

    return parsed_calls

def _extract_tool_calls_ollama(resp):
    """
    Extract tool calls for Ollama ChatResponse.
    Ollama follows OpenAI-compatible format: response.message.tool_calls
    Structure: ChatResponse with message.tool_calls = [{function: {name, arguments}}]
    """
    # 1. First try: structured tool calls from message
    structured_calls = []

    # Ollama: response is a ChatResponse object with message.tool_calls
    if (hasattr(resp, 'message') and
        hasattr(resp.message, 'tool_calls') and
        resp.message.tool_calls):

        for call in resp.message.tool_calls:
            # Ollama tool_call has: function: {name, arguments}
            if hasattr(call, 'function'):
                tool_id = getattr(call, 'id', str(uuid.uuid4()))

                # Arguments can be JSON string or dict
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                structured_calls.append(ToolCall(
                    id=tool_id,
                    name=call.function.name,
                    arguments=arguments
                ))

    # If we have structured tool calls, use them
    if structured_calls:
        return structured_calls

    # 2. Fallback: parsing from content
    content = ""
    if hasattr(resp, 'message') and hasattr(resp.message, 'content'):
        content = resp.message.content

    if not content:
        return []

    return _parse_tool_call_from_content(content)


def _extract_tool_calls_vllm(output, tools):
    """
    Estrae tool calls per vLLM offline inference.
    vLLM con llm.chat() ritorna RequestOutput con outputs[0].text contenente
    il JSON dei tool calls quando tools sono definiti.

    Args:
        output: RequestOutput di vLLM
        tools: Lista di tools passati alla richiesta

    Returns:
        Lista di ToolCall
    """
    # Se non ci sono tools definiti, non ci aspettiamo tool calls
    if not tools:
        return []

    # Estrae il testo dalla risposta
    content = ""
    if hasattr(output, 'outputs') and output.outputs:
        first_output = output.outputs[0]
        if hasattr(first_output, 'text'):
            content = first_output.text.strip()

    if not content:
        return []

    # vLLM può ritornare tool calls come JSON array nel content
    # Formato tipico: [{"name": "function_name", "arguments": {...}}]
    try:
        # Prova a parsare l'intero content come JSON array
        parsed = json.loads(content)

        # Se è un array di tool calls
        if isinstance(parsed, list):
            tool_calls = []
            for call in parsed:
                if isinstance(call, dict) and "name" in call:
                    tool_id = call.get("id", str(uuid.uuid4()))
                    arguments = call.get("arguments", call.get("parameters", {}))

                    tool_calls.append(ToolCall(
                        id=tool_id,
                        name=call["name"],
                        arguments=arguments
                    ))

            if tool_calls:
                return tool_calls

        # Se è un singolo tool call (dict)
        if isinstance(parsed, dict) and "name" in parsed:
            tool_id = parsed.get("id", str(uuid.uuid4()))
            arguments = parsed.get("arguments", parsed.get("parameters", {}))

            return [ToolCall(
                id=tool_id,
                name=parsed["name"],
                arguments=arguments
            )]

    except json.JSONDecodeError:
        pass

    # Fallback: usa il parser generico per trovare JSON nel content
    return _parse_tool_call_from_content(content)


def _extract_tool_calls_openai(response):
    """
    Extract tool calls from OpenAI ChatCompletion response.

    OpenAI returns response.choices[0].message.tool_calls as a list of objects:
    - id: unique identifier
    - type: "function"
    - function.name: tool name
    - function.arguments: JSON string of arguments

    Args:
        response: OpenAI ChatCompletion response object

    Returns:
        List of ToolCall objects extracted from tool_calls
    """
    # 1. First try: structured tool calls
    structured_calls = []

    if (hasattr(response, 'choices') and
            response.choices and
            hasattr(response.choices[0], 'message') and
            hasattr(response.choices[0].message, 'tool_calls') and
            response.choices[0].message.tool_calls):

        for call in response.choices[0].message.tool_calls:
            if call.type != "function":
                continue

            # Arguments is a JSON string in OpenAI API
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            structured_calls.append(ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=arguments
            ))

    if structured_calls:
        return structured_calls

    # 2. Fallback: parse from content
    content = ""
    if (hasattr(response, 'choices') and
            response.choices and
            hasattr(response.choices[0], 'message') and
            response.choices[0].message.content):
        content = response.choices[0].message.content

    if not content:
        return []

    return _parse_tool_call_from_content(content)
