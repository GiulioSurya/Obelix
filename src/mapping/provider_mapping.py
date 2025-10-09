import json
from src.providers import Providers, ProviderRegistry
from src.messages.tool_message import ToolCall

from oci.generative_ai_inference.models import (
    UserMessage, SystemMessage, AssistantMessage, ToolMessage as OCIToolMessage,
    TextContent, FunctionCall, Message,
    CohereUserMessage, CohereSystemMessage, CohereChatBotMessage, CohereToolMessage,
    CohereTool, CohereParameterDefinition
)
from src.mapping.tool_extr_fall_back import (
    _extract_tool_calls_generic,
    _extract_tool_calls_cohere,
    _extract_tool_calls_hybrid,
    _extract_tool_calls_ibm_watson_hybrid
)

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