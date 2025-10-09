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

        Cohere Rule: "cannot specify message if the last entry in chat history contains tool results"
        - If last message is ToolMessage → all messages go to chat_history, message=""
        - Otherwise → last HumanMessage goes to message, rest to chat_history

        Returns:
            Dict with:
                - message: str (last user message content OR empty if last is ToolMessage)
                - chat_history: List[CohereMessage] (all previous/all messages)
        """
        mapping = self.get_mapping()
        message_converters = mapping["message_input"]

        # Check if last message is a ToolMessage
        last_is_tool_message = len(messages) > 0 and isinstance(messages[-1], ToolMessage)

        chat_history = []
        last_user_message = None

        # Process messages
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                if last_is_tool_message:
                    # If last is ToolMessage, put ALL HumanMessages in chat_history
                    chat_history.append(message_converters["human_message"](message))
                else:
                    # Otherwise, separate last HumanMessage for 'message' field
                    if last_user_message is not None:
                        # Add previous user message to history
                        chat_history.append(message_converters["human_message"](
                            HumanMessage(content=last_user_message)
                        ))
                    last_user_message = message.content

            elif isinstance(message, SystemMessage):
                chat_history.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                # Cohere requires all messages to have non-empty content
                # Skip AssistantMessages without content (e.g., only tool_calls)
                if message.content:
                    chat_history.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                # ToolMessage converter returns a list
                chat_history.extend(message_converters["tool_message"](message))

        # Determine 'message' value based on whether last is ToolMessage
        if last_is_tool_message:
            # Last is ToolMessage → message must be empty, all in chat_history
            final_message = ""
        elif last_user_message is not None:
            # Normal case: extract last user message
            final_message = last_user_message
        else:
            # No user message found
            final_message = ""

        return {
            "message": final_message,
            "chat_history": chat_history if chat_history else None
        }

    def convert_tools(self, tools: List[ToolBase]) -> List[Any]:
        """
        Convert ToolBase objects to CohereTool format
        """
        mapping = self.get_mapping()
        tool_mapper = mapping["tool_input"]["tool_schema"]

        return [tool_mapper(tool.create_schema()) for tool in tools]

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
            "message": converted_messages["message"],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add chat_history if present
        if converted_messages.get("chat_history"):
            request_params["chat_history"] = converted_messages["chat_history"]

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