# src/client_adapters/oci_strategies/cohere_strategy.py
"""
Cohere Request Strategy for OCI Generative AI.

Supports: Cohere Command A, Command R, Command R+

This strategy is self-contained - all conversion logic is inline,
no external mapping dependencies.

Key difference: Cohere separates the last user message from chat_history:
- message: str (the LAST user message)
- chat_history: List[CohereMessage] (all PREVIOUS messages)
"""
import json
import uuid
from typing import List, Any, Optional, Dict

from pydantic import ValidationError

from oci.generative_ai_inference.models import (
    CohereChatRequest,
    BaseChatRequest,
    CohereUserMessage,
    CohereSystemMessage,
    CohereChatBotMessage,
    CohereToolMessage as OCICohereToolMessage,
    CohereTool,
    CohereParameterDefinition,
    CohereToolResult,
    CohereToolCall
)

from src.adapters.outbound.oci.strategies.base_strategy import OCIRequestStrategy
from src.adapters.outbound.oci.strategies.generic_strategy import ToolCallExtractionError
from src.core.model.standard_message import StandardMessage
from src.core.model.human_message import HumanMessage
from src.core.model.system_message import SystemMessage
from src.core.model.assistant_message import AssistantMessage
from src.core.model.tool_message import ToolMessage, ToolCall
from src.core.tool.tool_base import ToolBase
from src.infrastructure.logging import get_logger, format_message_for_trace

logger = get_logger(__name__)


class CohereRequestStrategy(OCIRequestStrategy):
    """
    Strategy for models using the COHERE API format.
    Supports: Cohere Command A, Command R, Command R+

    Self-contained: all conversion logic is inline.
    """

    # ========== MESSAGE CONVERSION ==========

    def convert_messages(self, messages: List[StandardMessage]) -> Dict[str, Any]:
        """
        Convert StandardMessage objects to Cohere format.

        Cohere Architecture:
        - message: str (the LAST user message - REQUIRED)
        - chat_history: List[CohereMessage] (all PREVIOUS messages, NO ToolMessages)
        - tool_results: List[CohereToolResult] (SEPARATE field for tool results)

        Returns:
            Dict with:
                - message: str (last user message content)
                - chat_history: List[CohereMessage] (previous messages without tools)
                - tool_results: List[CohereToolResult] (tool results in separate field)
        """
        logger.debug(f"Converting {len(messages)} messages to OCI COHERE format")

        chat_history = []
        tool_results = []
        last_user_message = None
        last_assistant_tool_calls = {}  # Map tool_call_id -> ToolCall

        for i, message in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(message)}")

            if isinstance(message, HumanMessage):
                if last_user_message is not None:
                    chat_history.append(
                        CohereUserMessage(message=last_user_message)
                    )
                last_user_message = message.content

            elif isinstance(message, SystemMessage):
                chat_history.append(
                    CohereSystemMessage(message=message.content)
                )

            elif isinstance(message, AssistantMessage):
                chat_history.append(
                    CohereChatBotMessage(message=message.content if message.content else "")
                )
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        last_assistant_tool_calls[tool_call.id] = tool_call

            elif isinstance(message, ToolMessage):
                for result in message.tool_results:
                    original_call = last_assistant_tool_calls.get(result.tool_call_id)

                    cohere_call = CohereToolCall(
                        name=result.tool_name,
                        parameters=original_call.arguments if original_call else {}
                    )

                    cohere_result = CohereToolResult(
                        call=cohere_call,
                        outputs=[{
                            "text": str(result.result) if result.result else result.error
                        }]
                    )
                    tool_results.append(cohere_result)

        # Determine final message value
        # IMPORTANT: In multistep mode (when tool_results present), CANNOT specify message
        if tool_results:
            final_message = None
        elif last_user_message is not None and last_user_message.strip():
            final_message = last_user_message
        else:
            final_message = "continue"

        logger.debug(
            f"Converted to COHERE format: message={'<set>' if final_message else 'None'}, "
            f"chat_history={len(chat_history)} msgs, tool_results={len(tool_results)} results"
        )

        return {
            "message": final_message,
            "chat_history": chat_history if chat_history else None,
            "tool_results": tool_results if tool_results else None
        }

    # ========== TOOL CONVERSION ==========

    def convert_tools(self, tools: List[ToolBase]) -> List[CohereTool]:
        """
        Convert ToolBase objects to CohereTool format.
        """
        if not tools:
            logger.debug("No tools to convert for OCI COHERE format")
            return []

        logger.debug(f"Converting {len(tools)} tools to OCI COHERE format")

        converted_tools = []
        for tool in tools:
            schema = tool.create_schema()

            parameter_definitions = {}
            input_schema = schema.inputSchema or {}
            properties = input_schema.get("properties", {})
            required_params = input_schema.get("required", [])

            for param_name, param_props in properties.items():
                parameter_definitions[param_name] = CohereParameterDefinition(
                    description=param_props.get("description", ""),
                    type=param_props.get("type", "string").upper(),
                    is_required=param_name in required_params
                )

            converted_tools.append(
                CohereTool(
                    name=schema.name,
                    description=schema.description,
                    parameter_definitions=parameter_definitions
                )
            )

        logger.debug(f"Converted {len(converted_tools)} tools to OCI COHERE format")
        return converted_tools

    # ========== TOOL CALL EXTRACTION ==========

    def extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract tool calls from OCI Cohere response.

        Path: response.data.chat_response.tool_calls

        Raises:
            ToolCallExtractionError: If validation fails
        """
        chat_response = response.data.chat_response if response.data else None
        if not chat_response:
            return []

        raw_tool_calls = getattr(chat_response, 'tool_calls', None)
        if not raw_tool_calls:
            return []

        tool_calls = []
        errors = []

        for call in raw_tool_calls:
            try:
                # Cohere uses 'name' and 'parameters' (not 'arguments')
                tool_call = ToolCall(
                    id=str(uuid.uuid4()),  # Cohere doesn't provide ID
                    name=call.name if hasattr(call, 'name') else "",
                    arguments=call.parameters if hasattr(call, 'parameters') else {}
                )
                tool_calls.append(tool_call)

            except ValidationError as e:
                errors.append(
                    f"Tool '{getattr(call, 'name', 'unknown')}': {e.errors()[0]['msg']}"
                )

        if errors:
            raise ToolCallExtractionError(
                f"Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== REQUEST BUILDING ==========

    def build_request(
        self,
        converted_messages: Dict[str, Any],
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
        Build a CohereChatRequest for Cohere Command models.
        """
        request_params = {
            "api_format": BaseChatRequest.API_FORMAT_COHERE,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add message field
        message_value = converted_messages.get("message")
        if message_value is None:
            request_params["message"] = ""
        else:
            request_params["message"] = message_value

        if converted_messages.get("chat_history"):
            request_params["chat_history"] = converted_messages["chat_history"]

        if converted_messages.get("tool_results"):
            request_params["tool_results"] = converted_messages["tool_results"]

        if converted_tools:
            request_params["tools"] = converted_tools

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
            request_params["stop_sequences"] = stop_sequences
        if is_stream:
            request_params["is_stream"] = is_stream

        # Add COHERE-specific parameters from kwargs
        cohere_specific_params = [
            "preamble_override",
            "documents",
            "safety_mode",
            "citation_quality",
            "prompt_truncation",
            "raw_prompting",
            "search_queries_only",
            "is_force_single_step",
            "response_format"
        ]

        for param in cohere_specific_params:
            if param in kwargs and kwargs[param] is not None:
                request_params[param] = kwargs[param]

        return CohereChatRequest(**request_params)

    # ========== METADATA ==========

    def get_api_format(self) -> str:
        """Returns API_FORMAT_COHERE"""
        return BaseChatRequest.API_FORMAT_COHERE

    def get_supported_model_prefixes(self) -> List[str]:
        """
        Returns model ID prefixes supported by COHERE format.
        """
        return ["cohere."]
