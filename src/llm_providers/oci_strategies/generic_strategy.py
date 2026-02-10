# src/llm_providers/oci_strategies/generic_strategy.py
"""
Generic Request Strategy for OCI Generative AI.

Supports: Meta Llama, Google Gemini, xAI Grok, OpenAI GPT-OSS

This strategy is self-contained - all conversion logic is inline,
no external mapping dependencies.
"""
import json
import uuid
from typing import List, Any, Optional, Dict

from pydantic import ValidationError

from oci.generative_ai_inference.models import (
    GenericChatRequest,
    BaseChatRequest,
    UserMessage,
    SystemMessage as OCISystemMessage,
    AssistantMessage as OCIAssistantMessage,
    ToolMessage as OCIToolMessage,
    TextContent,
    FunctionCall,
    FunctionDefinition,
    ToolChoiceAuto,
    ToolChoiceRequired,
    ToolChoiceNone
)

from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.messages.standard_message import StandardMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage, ToolCall
from src.tools.tool_base import ToolBase
from src.logging_config import get_logger, format_message_for_trace

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction fails due to malformed JSON or invalid structure."""
    pass


def _inline_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inline JSON Schema $ref using local $defs to avoid OCI Generic rejection.

    OCI Generic (Gemini) rejects $defs/$ref in tool parameters, so we resolve
    local references and drop $defs in the output.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    key = ref.split("/")[-1]
                    resolved = defs.get(key, {})
                    return resolve(resolved)
                return node
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


class GenericRequestStrategy(OCIRequestStrategy):
    """
    Strategy for models using the GENERIC API format.
    Supports: Meta Llama, Google Gemini, xAI Grok, OpenAI GPT-OSS

    Self-contained: all conversion logic is inline.
    """

    # ========== MESSAGE CONVERSION ==========

    def convert_messages(self, messages: List[StandardMessage]) -> List[Any]:
        """
        Convert StandardMessage objects to OCI Generic format.

        Mapping:
        - HumanMessage -> UserMessage(content=[TextContent(text=...)])
        - SystemMessage -> SystemMessage(content=[TextContent(text=...)])
        - AssistantMessage -> AssistantMessage(content=[TextContent], tool_calls=[FunctionCall])
        - ToolMessage -> [ToolMessage(content=[TextContent], tool_call_id=...)]
        """
        logger.debug(f"Converting {len(messages)} messages to OCI GENERIC format")

        converted_messages = []

        for i, message in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(message)}")

            if isinstance(message, HumanMessage):
                converted_messages.append(
                    UserMessage(content=[TextContent(text=message.content)])
                )

            elif isinstance(message, SystemMessage):
                converted_messages.append(
                    OCISystemMessage(content=[TextContent(text=message.content)])
                )

            elif isinstance(message, AssistantMessage):
                converted_messages.append(
                    self._convert_assistant_message(message)
                )

            elif isinstance(message, ToolMessage):
                # ToolMessage produces a list of OCIToolMessage (one per result)
                converted_messages.extend(
                    self._convert_tool_message(message)
                )

        logger.debug(f"Converted {len(converted_messages)} messages to OCI GENERIC format")
        return converted_messages

    def _convert_assistant_message(self, msg: AssistantMessage) -> OCIAssistantMessage:
        """Convert internal AssistantMessage to OCI AssistantMessage."""
        content = [TextContent(text=msg.content)] if msg.content else []

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                FunctionCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=json.dumps(tc.arguments)
                )
                for tc in msg.tool_calls
            ]

        return OCIAssistantMessage(content=content, tool_calls=tool_calls)

    def _convert_tool_message(self, msg: ToolMessage) -> List[OCIToolMessage]:
        """Convert internal ToolMessage to list of OCI ToolMessage."""
        return [
            OCIToolMessage(
                content=[TextContent(
                    text=str(result.result) if result.result else result.error
                )],
                tool_call_id=result.tool_call_id
            )
            for result in msg.tool_results
        ]

    # ========== TOOL CONVERSION ==========

    def convert_tools(self, tools: List[ToolBase]) -> List[FunctionDefinition]:
        """
        Convert ToolBase objects to OCI FunctionDefinition format.
        """
        if not tools:
            logger.debug("No tools to convert for OCI GENERIC format")
            return []

        logger.debug(f"Converting {len(tools)} tools to OCI GENERIC format")

        converted_tools = []
        for tool in tools:
            schema = tool.create_schema()

            # Inline $refs for OCI compatibility
            input_schema = schema.inputSchema
            if isinstance(input_schema, dict) and input_schema:
                input_schema = _inline_schema_refs(input_schema)

            converted_tools.append(
                FunctionDefinition(
                    type="FUNCTION",
                    name=schema.name,
                    description=schema.description,
                    parameters=input_schema
                )
            )

        logger.debug(f"Converted {len(converted_tools)} tools to OCI GENERIC format")
        return converted_tools

    # ========== TOOL CALL EXTRACTION ==========

    def extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract tool calls from OCI Generic response.

        Path: response.data.chat_response.choices[0].message.tool_calls

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails
        """
        chat_response = response.data.chat_response if response.data else None
        if not chat_response:
            return []

        choices = chat_response.choices
        if not choices or len(choices) == 0:
            return []

        message = choices[0].message
        if not message or not message.tool_calls:
            return []

        tool_calls = []
        errors = []

        for call in message.tool_calls:
            if call.type != "FUNCTION":
                continue

            try:
                # Parse arguments (strict=False tolera control char come \n \t nelle stringhe)
                arguments = call.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                # Handle double-encoding (some models do this)
                while isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                # Create ToolCall - Pydantic validates arguments is Dict[str, Any]
                tool_call = ToolCall(
                    id=call.id or str(uuid.uuid4()),
                    name=call.name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                errors.append(
                    f"Tool '{call.name}': Invalid JSON in arguments - {e.msg}"
                )
            except ValidationError as e:
                errors.append(
                    f"Tool '{call.name}': {e.errors()[0]['msg']}"
                )

        if errors:
            raise ToolCallExtractionError(
                f"Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== REQUEST BUILDING ==========

    def build_request(
        self,
        converted_messages: List[Any],
        converted_tools: List[Any],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        is_stream: bool = False,
        **kwargs
    ) -> BaseChatRequest:
        """
        Build a GenericChatRequest for Llama, Gemini, Grok, and OpenAI models.
        """
        request_params = {
            "api_format": BaseChatRequest.API_FORMAT_GENERIC,
            "messages": converted_messages,
            "tools": converted_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add standard optional parameters
        if top_p is not None:
            request_params["top_p"] = top_p
        if top_k is not None:
            request_params["top_k"] = top_k
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if stop_sequences is not None:
            request_params["stop"] = stop_sequences
        if is_stream:
            request_params["is_stream"] = is_stream

        # Add GENERIC-specific parameters from kwargs
        generic_specific_params = [
            "reasoning_effort",
            "verbosity",
            "num_generations",
            "log_probs",
            "logit_bias",
            "is_parallel_tool_calls",
            "seed",
            "metadata"
        ]

        for param in generic_specific_params:
            if param in kwargs and kwargs[param] is not None:
                request_params[param] = kwargs[param]

        # Handle tool_choice conversion from string to OCI ToolChoice object
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            tool_choice_value = kwargs["tool_choice"]
            if isinstance(tool_choice_value, str):
                tool_choice_map = {
                    "AUTO": ToolChoiceAuto(),
                    "REQUIRED": ToolChoiceRequired(),
                    "NONE": ToolChoiceNone()
                }
                if tool_choice_value.upper() in tool_choice_map:
                    request_params["tool_choice"] = tool_choice_map[tool_choice_value.upper()]
                else:
                    raise ValueError(
                        f"Invalid tool_choice value: '{tool_choice_value}'. "
                        f"Expected one of: AUTO, REQUIRED, NONE"
                    )
            else:
                request_params["tool_choice"] = tool_choice_value

        return GenericChatRequest(**request_params)

    # ========== METADATA ==========

    def get_api_format(self) -> str:
        """Returns API_FORMAT_GENERIC"""
        return BaseChatRequest.API_FORMAT_GENERIC

    def get_supported_model_prefixes(self) -> List[str]:
        """
        Returns model ID prefixes supported by GENERIC format.
        """
        return ["meta.", "google.", "xai.", "openai."]
