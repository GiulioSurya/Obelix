import json
import uuid


from src.messages.tool_message import ToolCall


def _extract_tool_calls_anthropic(response):
    """
    Estrae tool calls da response Anthropic Claude.

    Anthropic ritorna content come lista di blocks. I tool calls sono blocks con:
    - type="tool_use"
    - id: identificatore univoco
    - name: nome del tool
    - input: argomenti (dict)

    Args:
        response: Anthropic Message response object con .content attribute

    Returns:
        Lista di ToolCall objects estratti dai tool_use blocks
    """
    tool_calls = []

    if not hasattr(response, "content"):
        return []

    for block in response.content:
        # Gestisce sia dict che object notation
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
    Estrae tool calls per GenericChatRequest (Llama, Gemini, Grok, OpenAI).
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
    Estrae tool calls per CohereChatRequest (Cohere Command models).
    Path: resp.data.chat_response.tool_calls
    Struttura: {name: str, parameters: dict}
    """

    # 1. Prima prova: tool calls strutturati (modo standard Cohere)
    structured_calls = []

    # Cohere: tool_calls sono direttamente in chat_response, non dentro choices[0].message
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            hasattr(resp.data.chat_response, 'tool_calls') and
            resp.data.chat_response.tool_calls):

        for call in resp.data.chat_response.tool_calls:
            # Cohere usa 'name' e 'parameters' (non 'arguments')
            tool_id = str(uuid.uuid4())  # Cohere non fornisce ID, lo generiamo

            structured_calls.append(ToolCall(
                id=tool_id,
                name=call.name if hasattr(call, 'name') else "",
                arguments=call.parameters if hasattr(call, 'parameters') else {}
            ))

    # Se abbiamo tool calls strutturati, usali
    if structured_calls:
        return structured_calls

    # 2. Fallback: parsing robusto del JSON dal content
    content = ""
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            hasattr(resp.data.chat_response, 'text')):
        content = resp.data.chat_response.text

    if not content:
        return []

    parsed_calls = []

    # Trova la prima graffa
    start_pos = content.find('{')
    if start_pos == -1:
        return []

    # Pre-processing: rimuovi escape invalidi (es. \' che non è valido in JSON)
    json_content = content[start_pos:].replace("\\'", "'")

    # Usa JSONDecoder per parsing automatico e robusto
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(json_content, 0)

        # Verifica se ha la struttura di un tool call
        if isinstance(obj, dict) and "name" in obj:
            tool_id = str(uuid.uuid4())

            # Estrai parametri - Cohere usa "parameters"
            arguments = obj.get("parameters") or obj.get("arguments", {})

            # Se arguments è vuoto, usa tutte le chiavi tranne "name" e "type"
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


# Manteniamo per backward compatibility
def _extract_tool_calls_hybrid(resp):
    """
    [DEPRECATED] Usa _extract_tool_calls_generic o _extract_tool_calls_cohere.
    Manteniamo per backward compatibility con codice esistente.
    """
    return _extract_tool_calls_generic(resp)


def _extract_tool_calls_ibm_watson_hybrid(resp):
    """Estrae tool calls per IBM Watson: prima strutturati, poi dal content"""

    # 1. Prima prova: tool calls strutturati (modo standard IBM Watson)
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

    # Se abbiamo tool calls strutturati, usali
    if structured_calls:
        return structured_calls

    # 2. Fallback: parsing robusto del JSON dal content
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        return []

    parsed_calls = []

    # Trova la prima graffa
    start_pos = content.find('{')
    if start_pos == -1:
        return []

    # Pre-processing: rimuovi escape invalidi (es. \' che non è valido in JSON)
    json_content = content[start_pos:].replace("\\'", "'")

    # Usa JSONDecoder per parsing automatico e robusto
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(json_content, 0)

        # Verifica se ha la struttura di un tool call
        if isinstance(obj, dict) and "name" in obj:
            tool_id = str(uuid.uuid4())

            # Estrai parametri - supporta sia "parameters" che "arguments"
            arguments = obj.get("parameters") or obj.get("arguments", {})

            # Se arguments è vuoto, usa tutte le chiavi tranne "name" e "type"
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
    Estrae tool calls per Ollama ChatResponse.
    Ollama segue formato OpenAI-compatibile: response.message.tool_calls
    Struttura: ChatResponse con message.tool_calls = [{function: {name, arguments}}]
    """
    # 1. Prima prova: tool calls strutturati dal message
    structured_calls = []

    # Ollama: response è un oggetto ChatResponse con message.tool_calls
    if (hasattr(resp, 'message') and
        hasattr(resp.message, 'tool_calls') and
        resp.message.tool_calls):

        for call in resp.message.tool_calls:
            # Ollama tool_call ha: function: {name, arguments}
            if hasattr(call, 'function'):
                tool_id = getattr(call, 'id', str(uuid.uuid4()))

                # Arguments possono essere string JSON o dict
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

    # Se abbiamo tool calls strutturati, usali
    if structured_calls:
        return structured_calls

    # 2. Fallback: parsing dal content
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
