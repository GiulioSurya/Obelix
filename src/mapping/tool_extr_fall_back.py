import json
import re
import uuid


from src.messages.tool_message import ToolCall



#####---------OCI GENERIC (Llama, Gemini, Grok, OpenAI)
def _extract_tool_calls_generic(resp):
    """
    Estrae tool calls per GenericChatRequest (Llama, Gemini, Grok, OpenAI).
    Path: resp.data.chat_response.choices[0].message.tool_calls
    """

    # 1. Prima prova: tool calls strutturati (modo standard)
    structured_calls = []
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            resp.data.chat_response.choices and
            resp.data.chat_response.choices[0].message.tool_calls):

        for call in resp.data.chat_response.choices[0].message.tool_calls:
            if call.type == "FUNCTION":
                # Gestisci il doppio encoding JSON
                arguments = call.arguments
                if isinstance(arguments, str):
                    parsed = json.loads(arguments)
                    # Se dopo il primo parsing è ancora una stringa, fai un secondo parsing
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    arguments = parsed

                structured_calls.append(ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=arguments
                ))

    # Se abbiamo tool calls strutturati, usali
    if structured_calls:
        return structured_calls

    # 2. Fallback: parsa JSON dal content
    content = ""
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            resp.data.chat_response.choices and
            resp.data.chat_response.choices[0].message.content):
        content = "".join([
            c.text for c in resp.data.chat_response.choices[0].message.content
            if hasattr(c, 'text') and c.type == "TEXT"
        ])

    # Cerca pattern JSON nel content
    json_pattern = r'\{"type"\s*:\s*"function".*?\}'
    matches = re.findall(json_pattern, content, re.DOTALL)

    parsed_calls = []
    for match in matches:
        try:
            tool_data = json.loads(match)
            if tool_data.get("type") == "function":
                tool_id = str(uuid.uuid4())
                parsed_calls.append(ToolCall(
                    id=tool_id,
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("parameters", {})
                ))
        except json.JSONDecodeError:
            continue

    return parsed_calls


#####---------OCI COHERE (Cohere Command models)
def _extract_tool_calls_cohere(resp):
    """
    Estrae tool calls per CohereChatRequest (Cohere Command models).
    Path: resp.data.chat_response.tool_calls
    Struttura: {name: str, parameters: dict}
    """

    tool_calls = []

    # Cohere: tool_calls sono direttamente in chat_response, non dentro choices[0].message
    if (hasattr(resp, 'data') and
            hasattr(resp.data, 'chat_response') and
            hasattr(resp.data.chat_response, 'tool_calls') and
            resp.data.chat_response.tool_calls):

        for call in resp.data.chat_response.tool_calls:
            # Cohere usa 'name' e 'parameters' (non 'arguments')
            tool_id = str(uuid.uuid4())  # Cohere non fornisce ID, lo generiamo

            tool_calls.append(ToolCall(
                id=tool_id,
                name=call.name if hasattr(call, 'name') else "",
                arguments=call.parameters if hasattr(call, 'parameters') else {}
            ))

    return tool_calls


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

    # Usa JSONDecoder per parsing automatico e robusto
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(content, start_pos)

        # Verifica se ha la struttura di un tool call
        if isinstance(obj, dict) and "name" in obj:
            tool_id = str(uuid.uuid4())

            # Estrai parametri
            arguments = {}
            for key, value in obj.items():
                if key != "name":
                    arguments[key] = value

            parsed_calls.append(ToolCall(
                id=tool_id,
                name=obj["name"],
                arguments=arguments
            ))
    except json.JSONDecodeError:
        pass

    return parsed_calls