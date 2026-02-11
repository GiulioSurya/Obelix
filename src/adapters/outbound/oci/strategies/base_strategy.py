# src/client_adapters/oci_strategies/base_strategy.py
"""
Base Strategy for OCI Request Strategies.

Each strategy handles a specific API format (GENERIC or COHERE) and is self-contained:
- Convert StandardMessage to provider-specific message format
- Convert ToolBase to provider-specific tool format
- Extract tool calls from response (with validation)
- Build provider-specific chat request
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union

from oci.generative_ai_inference.models import BaseChatRequest, Message

from src.domain.model.standard_message import StandardMessage
from src.domain.model.tool_message import ToolCall
from src.domain.tool.tool_base import ToolBase


class OCIRequestStrategy(ABC):
    """
    Abstract base class for OCI request strategies.

    Each strategy handles a specific API format (GENERIC or COHERE).
    Strategies are self-contained - no external mapping dependencies.
    """

    @abstractmethod
    def convert_messages(self, messages: List[StandardMessage]) -> Union[List[Message], Dict[str, Any]]:
        """
        Convert StandardMessage objects to provider-specific message format.

        Args:
            messages: List of StandardMessage objects

        Returns:
            GenericStrategy: List[Message]
            CohereStrategy: Dict with {message, chat_history, tool_results}
        """
        pass

    @abstractmethod
    def convert_tools(self, tools: List[ToolBase]) -> List[Any]:
        """
        Convert ToolBase objects to provider-specific tool format.

        Args:
            tools: List of ToolBase objects

        Returns:
            GenericStrategy: List[FunctionDefinition]
            CohereStrategy: List[CohereTool]
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract and validate tool calls from OCI response.

        Args:
            response: OCI chat response

        Returns:
            List[ToolCall]: Validated tool calls

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails
        """
        pass

    @abstractmethod
    def build_request(
        self,
        converted_messages: Union[List[Message], Dict[str, Any]],
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
        Build the appropriate chat request for this strategy.

        Args:
            converted_messages: Output from convert_messages()
            converted_tools: Output from convert_tools()
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Stop sequences
            is_stream: Enable streaming
            **kwargs: Additional strategy-specific parameters

        Returns:
            BaseChatRequest: The constructed request object
        """
        pass

    @abstractmethod
    def get_api_format(self) -> str:
        """
        Get the API format for this strategy.

        Returns:
            str: Either API_FORMAT_GENERIC or API_FORMAT_COHERE
        """
        pass

    @abstractmethod
    def get_supported_model_prefixes(self) -> List[str]:
        """
        Get the list of model ID prefixes supported by this strategy.

        Returns:
            List[str]: List of model ID prefixes (e.g., ["meta.", "google."])
        """
        pass
