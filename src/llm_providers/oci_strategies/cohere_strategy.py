# src/llm_providers/oci_strategies/cohere_strategy.py
from typing import List, Any, Optional, Dict
from oci.generative_ai_inference.models import CohereChatRequest, BaseChatRequest

from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.messages.standard_message import StandardMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage
from src.tools.tool_base import ToolBase
from src.mapping.provider_mapping import OCI_GENERATIVE_AI_COHERE
from src.logging_config import get_logger

# Logger per CohereRequestStrategy
logger = get_logger(__name__)


class CohereRequestStrategy(OCIRequestStrategy):
    """
    Strategy for models using the COHERE API format.
    Supports: Cohere Command A, Command R, Command R+

    Key difference: Cohere separates the last user message from chat_history:
    - message: str (the LAST user message)
    - chat_history: List[CohereMessage] (all PREVIOUS messages)
    """

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

        from oci.generative_ai_inference.models import CohereToolResult, CohereToolCall

        mapping = self.get_mapping()
        message_converters = mapping["message_input"]

        chat_history = []
        tool_results = []
        last_user_message = None
        last_assistant_tool_calls = {}  # Map tool_call_id -> ToolCall

        # Process messages
        for i, message in enumerate(messages):
            msg_type = type(message).__name__

            # TRACE: preview del contenuto messaggio
            content_preview = ""
            if hasattr(message, 'content') and message.content:
                content_preview = str(message.content)[:100]
            logger.trace(f"msg[{i}] {msg_type}: {content_preview}")
            if isinstance(message, HumanMessage):
                # Store user messages; last one goes to 'message' field
                if last_user_message is not None:
                    # Add previous user message to history
                    chat_history.append(message_converters["human_message"](
                        HumanMessage(content=last_user_message)
                    ))
                last_user_message = message.content

            elif isinstance(message, SystemMessage):
                chat_history.append(message_converters["system_message"](message))

            elif isinstance(message, AssistantMessage):
                # Add assistant messages to history
                chat_history.append(message_converters["assistant_message"](message))

                # Track tool calls for matching with tool results later
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        last_assistant_tool_calls[tool_call.id] = tool_call

            elif isinstance(message, ToolMessage):
                # Convert ToolMessage to CohereToolResult format
                # Match tool results with original tool calls to get parameters
                for result in message.tool_results:
                    original_call = last_assistant_tool_calls.get(result.tool_call_id)

                    # Create CohereToolCall with name and parameters
                    cohere_call = CohereToolCall(
                        name=result.tool_name,
                        parameters=original_call.arguments if original_call else {}
                    )

                    # Create CohereToolResult with call and outputs
                    cohere_result = CohereToolResult(
                        call=cohere_call,
                        outputs=[{
                            "text": str(result.result) if result.result else result.error
                        }]
                    )

                    tool_results.append(cohere_result)

        # Determine final message value
        # IMPORTANT Cohere Rule: In multistep mode (when tool_results present), CANNOT specify message
        if tool_results:
            # When tool_results are present, message must be None (multistep mode rule)
            final_message = None
        elif last_user_message is not None and last_user_message.strip():
            # Use last user message if not empty
            final_message = last_user_message
        else:
            # No valid user message found - use placeholder
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

    def convert_tools(self, tools: List[ToolBase]) -> List[Any]:
        """
        Convert ToolBase objects to CohereTool format
        """
        if not tools:
            logger.debug("No tools to convert for OCI COHERE format")
            return []

        logger.debug(f"Converting {len(tools)} tools to OCI COHERE format")

        mapping = self.get_mapping()
        tool_mapper = mapping["tool_input"]["tool_schema"]

        converted_tools = [tool_mapper(tool.create_schema()) for tool in tools]
        logger.debug(f"Converted {len(converted_tools)} tools to OCI COHERE format")

        return converted_tools

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

        Args:
            converted_messages: Dict from convert_messages() with {message: str, chat_history: List}
            converted_tools: List[CohereTool] from convert_tools()

        Additional kwargs supported:
            - preamble_override: str (custom system preamble)
            - documents: list (contextual reference documents)
            - safety_mode: str (content safety level)
            - citation_quality: str (FAST, ACCURATE)
            - prompt_truncation: str (OFF, AUTO)
            - raw_prompting: bool
            - search_queries_only: bool
            - is_force_single_step: bool
            - response_format: dict
        """
        request_params = {
            "api_format": BaseChatRequest.API_FORMAT_COHERE,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add message field (required by Cohere)
        message_value = converted_messages.get("message")
        # Python SDK requires message to be present (not None), but when tool_results are present,
        # use empty string to satisfy SDK while letting server handle the multistep mode correctly
        if message_value is None:
            request_params["message"] = ""
        else:
            request_params["message"] = message_value

        # Add chat_history if present
        if converted_messages.get("chat_history"):
            request_params["chat_history"] = converted_messages["chat_history"]

        # Add tool_results if present (SEPARATE field from chat_history)
        if converted_messages.get("tool_results"):
            request_params["tool_results"] = converted_messages["tool_results"]

        # Add tools
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

    def get_mapping(self) -> Dict[str, Any]:
        """Returns the OCI_GENERATIVE_AI_COHERE mapping"""
        return OCI_GENERATIVE_AI_COHERE

    def get_api_format(self) -> str:
        """Returns API_FORMAT_COHERE"""
        return BaseChatRequest.API_FORMAT_COHERE

    def get_supported_model_prefixes(self) -> List[str]:
        """
        Returns model ID prefixes supported by COHERE format.

        Supported models:
        - cohere.* (Command A, Command R, Command R+)
        """
        return ["cohere."]