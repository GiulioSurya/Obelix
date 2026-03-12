# src/client_adapters/llm_abstraction.py
"""
Abstract LLM Provider interface.

Pure ABC contract: each provider implements invoke() and is fully
self-contained for message/tool conversion and response parsing.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel

from obelix.core.model.assistant_message import AssistantMessage, StreamEvent
from obelix.core.model.standard_message import StandardMessage
from obelix.core.tool.tool_base import Tool

if TYPE_CHECKING:
    from obelix.infrastructure.providers import Providers


class AbstractLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Contract:
    - provider_type: identifies the provider (Providers enum)
    - invoke(): messages + tools in, AssistantMessage out

    Optional:
    - invoke_stream(): streaming variant, yields StreamEvent chunks.
      Default raises NotImplementedError. Providers override to enable streaming.

    Each provider is self-contained: message conversion, tool conversion,
    and response parsing are implementation details handled internally
    (inline, via strategy pattern, or any other approach).
    """

    @property
    @abstractmethod
    def provider_type(self) -> "Providers":
        """Returns the Providers enum for this provider."""
        pass

    @abstractmethod
    async def invoke(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """
        Call the LLM with standardized messages and tools.

        Args:
            messages: List of messages in StandardMessage format
            tools: List of available tools
            response_schema: Optional Pydantic BaseModel class for structured JSON output.
                If None, the LLM responds with free-form text.
                If specified, the LLM response follows the JSON schema derived from the model.

        Returns:
            AssistantMessage with model response
        """
        pass

    async def invoke_stream(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming variant of invoke(). Yields StreamEvent chunks.

        Text chunks are yielded as they arrive. The final StreamEvent
        (is_final=True) contains the complete AssistantMessage with
        accumulated content, tool_calls, and usage.

        Providers that support streaming override this method.
        Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            "Override invoke_stream() to enable it."
        )
        # yield is needed to make this a valid AsyncIterator signature
        yield  # pragma: no cover
