import json
from src.providers import Providers, ProviderRegistry


from oci.generative_ai_inference.models import (
    UserMessage, SystemMessage, AssistantMessage, ToolMessage as OCIToolMessage,
    TextContent, FunctionCall, Message,
    CohereUserMessage, CohereSystemMessage, CohereChatBotMessage, CohereToolMessage,
    CohereTool, CohereParameterDefinition, CohereToolResult, CohereToolCall
)
from src.mapping.tool_extr_fall_back import (
    _extract_tool_calls_anthropic,
    _extract_tool_calls_generic,
    _extract_tool_calls_cohere,
    _extract_tool_calls_hybrid,
    _extract_tool_calls_ibm_watson_hybrid,
    _extract_tool_calls_ollama,
    _extract_tool_calls_vllm,
    _extract_tool_calls_openai
)

# ===== ANTHROPIC MAPPING =====
ANTHROPIC = {
    "tool_input": {
        "tool_schema": lambda tool_schema: {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "input_schema": tool_schema.inputSchema  # Anthropic usa JSON Schema nativo
        }
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_anthropic
    },
    "message_input": {
        "human_message": lambda msg: {
            "role": "user",
            "content": msg.content
        },

        "system_message": lambda msg: msg.content,  # Ritorna solo content (usato come parametro system)

        "assistant_message": lambda msg: {
            "role": "assistant",
            "content": [
                *(
                    [{"type": "text", "text": msg.content}]
                    if msg.content
                    else []
                ),
                *(
                    [
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        }
                        for tc in msg.tool_calls
                    ]
                    if msg.tool_calls
                    else []
                ),
            ],
        },

        "tool_message": lambda msg: [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": str(result.result) if result.result else (result.error or "No result")
                }
                for result in msg.tool_results
            ]
        }]
    },
}

ProviderRegistry.register(Providers.ANTHROPIC, ANTHROPIC)

# ===== IBM WATSON MAPPING =====
IBM_WATSON = {
    "tool_input": {
        "tool_schema": lambda tool_schema: {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": tool_schema.inputSchema
            }
        }
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_ibm_watson_hybrid
    },
    "message_input": {
        "human_message": lambda msg: {
            "role": "user",
            "content": msg.content
        },

        "system_message": lambda msg: {
            "role": "system",
            "content": msg.content
        },

        "assistant_message": lambda msg: {
            "role": "assistant",
            "content": msg.content if msg.content else None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments)
                    }
                }
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        } if msg.tool_calls or msg.content else {
            "role": "assistant",
            "content": ""
        },

        "tool_message": lambda msg: [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": str(result.result) if result.result is not None else (result.error or "No result")
            }
            for result in msg.tool_results
        ]
    },
}

ProviderRegistry.register(Providers.IBM_WATSON, IBM_WATSON)

from oci.generative_ai_inference.models import FunctionDefinition

OCI_GENERATIVE_AI = {
    "tool_input": {
        "tool_schema": lambda tool_schema: FunctionDefinition(
            type="FUNCTION",
            name=tool_schema.name,
            description=tool_schema.description,
            parameters=tool_schema.inputSchema
        )
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_hybrid
    },
     "message_input": {
        "human_message": lambda msg: UserMessage(
            content=[TextContent(text=msg.content)]
        ),
        "system_message": lambda msg: SystemMessage(
            content=[TextContent(text=msg.content)]
        ),
        "assistant_message": lambda msg: AssistantMessage(
            content=[TextContent(text=msg.content)] if msg.content else None,
            tool_calls=[
                FunctionCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(tool_call.arguments)
                )
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        ),
        "tool_message": lambda msg: [
            OCIToolMessage(
                content=[TextContent(text=str(result.result) if result.result else result.error)],
                tool_call_id=result.tool_call_id
            )
            for result in msg.tool_results
        ]
    }
}

ProviderRegistry.register(Providers.OCI_GENERATIVE_AI, OCI_GENERATIVE_AI)

# ===== MAPPINGS SPECIFICI PER STRATEGY =====
# Questi non vengono registrati in ProviderRegistry, sono usati direttamente dalle strategy

# TODO: BUG with Gemini and parallel tool calls
# ==============================================
# PROBLEM:
#   When the LLM (e.g. google.gemini-2.5-flash) generates N tool_calls in a single turn
#   (e.g. 2 subagents called in parallel), Gemini expects responses in a specific format.
#
# OCI ERROR:
#   "Please ensure that the number of function response parts is equal to the number
#   of function call parts of the function call turn."
#
# ROOT CAUSE:
#   The "tool_message" mapping below converts 1 ToolMessage (with N tool_results) into
#   N separate OCIToolMessage objects. This works for GPT but NOT for Gemini.
#
#   Example with 2 parallel tool_calls:
#   - Input:  ToolMessage(tool_results=[result_1, result_2])
#   - Output: [OCIToolMessage(result_1), OCIToolMessage(result_2)]  <-- 2 separate messages
#
#   Gemini expects responses aggregated in the same "turn" as the tool_calls.
#
# POSSIBLE SOLUTIONS:
#   1. Create a Gemini-specific mapping that aggregates responses differently
#   2. Modify the generic strategy to handle the Gemini case
#   3. Check OCI/Gemini documentation for correct function response format
#
# CURRENT WORKAROUND:
#   Use GPT models (e.g. openai.gpt-oss-120b) instead of Gemini for orchestrators
#   that make parallel subagent calls.
# ==============================================

OCI_GENERATIVE_AI_GENERIC = {
    "tool_input": {
        "tool_schema": lambda tool_schema: FunctionDefinition(
            type="FUNCTION",
            name=tool_schema.name,
            description=tool_schema.description,
            parameters=tool_schema.inputSchema
        )
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_generic
    },
    "message_input": {
        "human_message": lambda msg: UserMessage(
            content=[TextContent(text=msg.content)]
        ),
        "system_message": lambda msg: SystemMessage(
            content=[TextContent(text=msg.content)]
        ),
        "assistant_message": lambda msg: AssistantMessage(
            content=[TextContent(text=msg.content)] if msg.content else None,
            tool_calls=[
                FunctionCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=json.dumps(tool_call.arguments)
                )
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        ),
        "tool_message": lambda msg: [
            OCIToolMessage(
                content=[TextContent(text=str(result.result) if result.result else result.error)],
                tool_call_id=result.tool_call_id
            )
            for result in msg.tool_results
        ]
    }
}

OCI_GENERATIVE_AI_COHERE = {
    "tool_input": {
        "tool_schema": lambda tool_schema: CohereTool(
            name=tool_schema.name,
            description=tool_schema.description,
            parameter_definitions={
                param_name: CohereParameterDefinition(
                    description=param_props.get("description", ""),
                    type=param_props.get("type", "string").upper(),
                    is_required=param_name in tool_schema.inputSchema.get("required", [])
                )
                for param_name, param_props in tool_schema.inputSchema.get("properties", {}).items()
            }
        )
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_cohere
    },
    "message_input": {
        "human_message": lambda msg: CohereUserMessage(
            message=msg.content
        ),
        "system_message": lambda msg: CohereSystemMessage(
            message=msg.content
        ),
        "assistant_message": lambda msg: CohereChatBotMessage(
            message=msg.content if msg.content else ""
        ),
        "tool_message": lambda msg: [
            CohereToolMessage(
                role="TOOL",
                tool_results=[{
                    "call": {"name": result.tool_name, "parameters": {}},
                    "outputs": [{
                        "text": str(result.result) if result.result else result.error
                    }]
                }]
            )
            for result in msg.tool_results
        ]
    }
}
# ===== OLLAMA MAPPING =====
OLLAMA = {
    "tool_input": {
        "tool_schema": lambda tool_schema: {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": tool_schema.inputSchema
            }
        }
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_ollama
    },
    "message_input": {
        "human_message": lambda msg: {
            "role": "user",
            "content": msg.content
        },

        "system_message": lambda msg: {
            "role": "system",
            "content": msg.content
        },

        "assistant_message": lambda msg: {
            "role": "assistant",
            "content": msg.content if msg.content else None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments)
                    }
                }
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        } if msg.tool_calls or msg.content else {
            "role": "assistant",
            "content": ""
        },

        "tool_message": lambda msg: [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": str(result.result) if result.result is not None else (result.error or "No result")
            }
            for result in msg.tool_results
        ]
    },
}

ProviderRegistry.register(Providers.OLLAMA, OLLAMA)

# ===== VLLM MAPPING =====
VLLM = {
    "tool_input": {
        "tool_schema": lambda tool_schema: {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": tool_schema.inputSchema
            }
        }
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_vllm
    },
    "message_input": {
        "human_message": lambda msg: {
            "role": "user",
            "content": msg.content
        },

        "system_message": lambda msg: {
            "role": "system",
            "content": msg.content
        },

        "assistant_message": lambda msg: {
            "role": "assistant",
            "content": msg.content if msg.content else None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments)
                    }
                }
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        } if msg.tool_calls or msg.content else {
            "role": "assistant",
            "content": ""
        },

        "tool_message": lambda msg: [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": str(result.result) if result.result is not None else (result.error or "No result")
            }
            for result in msg.tool_results
        ]
    },
}

ProviderRegistry.register(Providers.VLLM, VLLM)

# ===== OPENAI MAPPING =====
OPENAI = {
    "tool_input": {
        "tool_schema": lambda tool_schema: {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": tool_schema.inputSchema
            }
        }
    },
    "tool_output": {
        "tool_calls": _extract_tool_calls_openai
    },
    "message_input": {
        "human_message": lambda msg: {
            "role": "user",
            "content": msg.content
        },

        "system_message": lambda msg: {
            "role": "system",
            "content": msg.content
        },

        "assistant_message": lambda msg: {
            "role": "assistant",
            "content": msg.content if msg.content else None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments)
                    }
                }
                for tool_call in msg.tool_calls
            ] if msg.tool_calls else None
        } if msg.tool_calls or msg.content else {
            "role": "assistant",
            "content": ""
        },

        "tool_message": lambda msg: [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": str(result.result) if result.result is not None else (result.error or "No result")
            }
            for result in msg.tool_results
        ]
    },
}

ProviderRegistry.register(Providers.OPENAI, OPENAI)
