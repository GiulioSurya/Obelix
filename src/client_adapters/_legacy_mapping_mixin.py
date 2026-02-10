# src/client_adapters/_legacy_mapping_mixin.py
"""
TEMPORARY mixin for providers not yet migrated to self-contained architecture.

Provides shared conversion methods that delegate to ProviderRegistry mappings.
Used by: IBM Watson, Ollama, vLLM, OpenAI (until they are migrated).

TODO: Remove this file after all providers are self-contained.
"""
from typing import List, Dict, Any

from src.obelix_types.standard_message import StandardMessage
from src.obelix_types.assistant_message import AssistantMessage
from src.obelix_types.human_message import HumanMessage
from src.obelix_types.system_message import SystemMessage
from src.obelix_types.tool_message import ToolMessage
from src.tools.tool_base import ToolBase
from src.logging_config import get_logger, format_message_for_trace

logger = get_logger(__name__)


class LegacyMappingMixin:
    """
    Mixin that provides ProviderRegistry-based conversion methods.

    Requires the class to have a `provider_type` property returning a Providers enum.
    """

    def _convert_messages_to_provider_format(self, messages: List[StandardMessage]) -> List[Any]:
        from src.providers import ProviderRegistry

        logger.debug(f"Converting {len(messages)} messages to {self.provider_type.value} format")

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        message_converters = mapping["message_input"]

        converted_messages = []

        for i, message in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(message)}")

            if isinstance(message, HumanMessage):
                converted_messages.append(message_converters["human_message"](message))
            elif isinstance(message, SystemMessage):
                converted_messages.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                converted_messages.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                converted_messages.extend(message_converters["tool_message"](message))

        logger.debug(f"Converted {len(converted_messages)} messages for {self.provider_type.value}")
        return converted_messages

    def _convert_tools_to_provider_format(self, tools: List[ToolBase]) -> List[Any]:
        if not tools:
            return []

        from src.providers import ProviderRegistry

        logger.debug(f"Converting {len(tools)} tools to {self.provider_type.value} format")

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        tool_mapper = mapping["tool_input"]["tool_schema"]

        converted_tools = [tool_mapper(tool.create_schema()) for tool in tools]
        logger.debug(f"Converted {len(converted_tools)} tools for {self.provider_type.value}")
        return converted_tools

    def _extract_tool_calls(self, response: Any, **kwargs) -> List[Dict[str, Any]]:
        from src.providers import ProviderRegistry

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        extractor = mapping["tool_output"]["tool_calls"]

        if kwargs:
            tool_calls = extractor(response, **kwargs)
        else:
            tool_calls = extractor(response)

        logger.debug(f"Extracted {len(tool_calls)} tool_calls")
        return tool_calls