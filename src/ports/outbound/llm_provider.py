# src/client_adapters/llm_abstraction.py
"""
Abstract LLM Provider interface.

Pure ABC contract: each provider implements invoke() and is fully
self-contained for message/tool conversion and response parsing.
"""
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel

from src.core.model.standard_message import StandardMessage
from src.core.model.assistant_message import AssistantMessage
from src.core.tool.tool_base import ToolBase
from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from src.infrastructure.providers import Providers

logger = get_logger(__name__)


class AbstractLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Contract:
    - provider_type: identifies the provider (Providers enum)
    - invoke(): messages + tools in, AssistantMessage out

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
        messages: List[StandardMessage],
        tools: List[ToolBase],
        response_schema: Optional[Type[BaseModel]] = None,
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

    # ========== SHARED UTILITY ==========

    @staticmethod
    def _get_connection_from_global_config(
        expected_provider: "Providers",
        provider_class_name: str
    ) -> Any:
        """
        Get connection from GlobalConfig with provider validation.

        Raises:
            ValueError: If GlobalConfig has no provider or a different provider set
        """
        from src.infrastructure.config import GlobalConfig

        logger.debug(f"Getting connection from GlobalConfig for {provider_class_name} (expected: {expected_provider.name})")

        config = GlobalConfig()

        try:
            current_provider = config.get_current_provider()
        except ValueError:
            raise ValueError(
                f"Cannot create {provider_class_name} without connection. "
                "Options: \n"
                f"1. Pass connection explicitly\n"
                f"2. Set GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        if current_provider != expected_provider:
            raise ValueError(
                f"GlobalConfig has provider '{current_provider.value}' set, "
                f"but you are creating {provider_class_name}.\n"
                f"Options:\n"
                f"1. Pass connection explicitly\n"
                f"2. Change GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        if current_provider not in config._connections:
            config._connections[current_provider] = config._create_connection(current_provider)

        return config._connections[current_provider]
