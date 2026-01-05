# src/llm_providers/llm_abstraction.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.tool_message import ToolMessage
from src.tools.tool_base import ToolBase
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.providers import Providers

# Logger for tracing common provider operations
logger = get_logger(__name__)


class AbstractLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides:
    - Common methods for message and tool conversion (using ProviderRegistry)
    - Helper for tool_calls extraction
    - Centralized method to get connection from GlobalConfig
    - Mandatory public interface: invoke()

    Each provider must:
    - Implement `provider_type` property
    - Implement `invoke()` method
    - (Optional) Override conversion methods if using different patterns (e.g. strategy)
    """

    @property
    @abstractmethod
    def provider_type(self) -> "Providers":
        """
        Returns the Providers enum for this provider.
        Used to get the correct mapping from ProviderRegistry.
        """
        pass

    @abstractmethod
    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Calls the LLM model with standardized messages and tools

        Args:
            messages: List of messages in StandardMessage format
            tools: List of available tools

        Returns:
            AssistantMessage with model response (includes populated usage field)
        """
        pass

    # ========== CENTRALIZED METHOD FOR CONNECTION ==========

    @staticmethod
    def _get_connection_from_global_config(
        expected_provider: "Providers",
        provider_class_name: str
    ) -> Any:
        """
        Common logic to get connection from GlobalConfig.

        Verifies that GlobalConfig has the correct provider set,
        otherwise raises ValueError with clear instructions.

        Args:
            expected_provider: Expected Providers enum (e.g. Providers.OCI_GENERATIVE_AI)
            provider_class_name: Class name for error messages (e.g. "OCILLm")

        Returns:
            Connection from GlobalConfig

        Raises:
            ValueError: If GlobalConfig has no provider set or has a different provider
        """
        from src.config import GlobalConfig

        logger.debug(f"Getting connection from GlobalConfig for {provider_class_name} (expected: {expected_provider.name})")

        config = GlobalConfig()

        # Verify that a provider is set
        try:
            current_provider = config.get_current_provider()
        except ValueError:
            logger.error(f"GlobalConfig has no provider set, cannot create {provider_class_name}")
            raise ValueError(
                f"Cannot create {provider_class_name} without connection. "
                "Options: \n"
                f"1. Pass connection explicitly\n"
                f"2. Set GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        # Verify that it is the expected provider
        if current_provider != expected_provider:
            logger.error(
                f"Provider mismatch: GlobalConfig has {current_provider.value}, "
                f"but creating {provider_class_name} (expected {expected_provider.value})"
            )
            raise ValueError(
                f"GlobalConfig has provider '{current_provider.value}' set, "
                f"but you are creating {provider_class_name}.\n"
                f"Options:\n"
                f"1. Pass connection explicitly\n"
                f"2. Change GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        # Reuse connection from GlobalConfig (lazy init if necessary)
        if current_provider not in config._connections:
            logger.debug(f"Creating new connection for {current_provider.value}")
            config._connections[current_provider] = config._create_connection(current_provider)
        else:
            logger.debug(f"Reusing existing connection for {current_provider.value}")

        return config._connections[current_provider]

    # ========== COMMON CONVERSION METHODS ==========

    def _convert_messages_to_provider_format(self, messages: List[StandardMessage]) -> List[Any]:
        """
        Converts StandardMessage to provider-specific format using ProviderRegistry.

        This method implements the common pattern used by IBM, Ollama, vLLM, Anthropic.
        Providers with different logic (e.g. OCI with strategy) can override.

        Args:
            messages: List of StandardMessage

        Returns:
            List of messages in provider format
        """
        from src.providers import ProviderRegistry

        logger.debug(f"Converting {len(messages)} messages to {self.provider_type.value} format")

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        message_converters = mapping["message_input"]

        converted_messages = []

        for i, message in enumerate(messages):
            msg_type = type(message).__name__

            # TRACE: preview of message content
            content_preview = ""
            if hasattr(message, 'content') and message.content:
                content_preview = str(message.content)[:100]
            logger.trace(f"msg[{i}] {msg_type}: {content_preview}")

            if isinstance(message, HumanMessage):
                converted_messages.append(message_converters["human_message"](message))
            elif isinstance(message, SystemMessage):
                converted_messages.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                converted_messages.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                # ToolMessage can generate multiple messages
                converted_messages.extend(message_converters["tool_message"](message))

        logger.debug(f"Converted {len(converted_messages)} messages for {self.provider_type.value}")
        return converted_messages

    def _convert_tools_to_provider_format(self, tools: List[ToolBase]) -> List[Any]:
        """
        Converts ToolBase to provider-specific format using ProviderRegistry.

        This method implements the common pattern used by IBM, Ollama, vLLM, Anthropic.
        Providers with different logic (e.g. OCI with strategy) can override.

        Args:
            tools: List of ToolBase

        Returns:
            List of tools in provider format (empty if tools is empty)
        """
        if not tools:
            logger.debug("No tools to convert")
            return []

        logger.debug(f"Converting {len(tools)} tools to {self.provider_type.value} format")

        from src.providers import ProviderRegistry

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        tool_mapper = mapping["tool_input"]["tool_schema"]

        converted_tools = [tool_mapper(tool.create_schema()) for tool in tools]
        logger.debug(f"Converted {len(converted_tools)} tools for {self.provider_type.value}")

        return converted_tools

    def _extract_tool_calls(self, response: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Extracts tool_calls from response using provider mapping.

        Args:
            response: Response object from provider API
            **kwargs: Additional parameters (e.g. tools for vLLM)

        Returns:
            List of extracted tool calls
        """
        logger.debug(f"Extracting tool_calls from {self.provider_type.value} response")

        from src.providers import ProviderRegistry

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        extractor = mapping["tool_output"]["tool_calls"]

        # Some extractors require additional parameters (e.g. vLLM requires tools)
        if kwargs:
            tool_calls = extractor(response, **kwargs)
        else:
            tool_calls = extractor(response)

        logger.debug(f"Extracted {len(tool_calls)} tool_calls")
        return tool_calls
