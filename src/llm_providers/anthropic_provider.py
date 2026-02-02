# src/llm_providers/anthropic_provider.py
import logging
from typing import List, Dict, Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.messages.system_message import SystemMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import AnthropicConnection
from src.logging_config import get_logger

# Logger for Anthropic provider
logger = get_logger(__name__)


class AnthropicProvider(AbstractLLMProvider):
    """
    Provider for Anthropic Claude with configurable parameters.

    Supports:
    - System messages (as separate parameter, not in messages array)
    - Tool calling with content blocks
    - Multi-turn conversations
    - Usage tracking
    """

    @property
    def provider_type(self) -> Providers:
        return Providers.ANTHROPIC

    def __init__(self,
                 connection: Optional[AnthropicConnection] = None,
                 model_id: str = "claude-haiku-4-5-20251001",
                 max_tokens: int = 3000,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 thinking_mode: bool = False,
                 thinking_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Anthropic provider with dependency injection of connection

        Args:
            connection: AnthropicConnection singleton (default: None, reuse from GlobalConfig if provider matches)
            model_id: Claude model ID (default: "claude-haiku-4-5-20251001")
            max_tokens: Maximum number of tokens (default: 3000)
            temperature: Sampling temperature (default: 0.1)
            top_p: Top-p sampling (default: None)
            thinking_mode: Enable extended thinking (default: False)
            thinking_params: Parameters for thinking mode (default: None)

        Raises:
            ValueError: If connection=None and GlobalConfig does not have ANTHROPIC set
        """
        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.ANTHROPIC,
                "AnthropicProvider"
            )

        self.connection = connection

        # Save parameters
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = 1 if thinking_mode else temperature
        self.top_p = top_p
        self.thinking_mode = thinking_mode
        if thinking_mode:
            if thinking_params is None:
                print("Default parameters: {'type': 'enabled', 'budget_tokens': 2000}")
                self.thinking_params = {"type": "enabled", "budget_tokens": 2000}
            else:
                self.thinking_params = thinking_params
        else:
            self.thinking_params = {"type": "disabled"}

    @retry(
        retry=retry_if_exception_type((
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Calls the Anthropic model with standardized messages and tools (async).

        Uses AsyncAnthropic client for native async support.
        Retries automatically on transient errors (connection, timeout, rate limit, server errors).
        """
        logger.debug(f"Anthropic invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}, thinking_mode={self.thinking_mode}")

        # 1. Separate SystemMessage from others (Anthropic wants it as parameter, not in messages array)
        system_content = None
        other_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # SystemMessage goes as system parameter, convert individually
                system_content = self._convert_messages_to_provider_format([msg])[0]
            else:
                other_messages.append(msg)

        # 2. Convert other messages (use base class method)
        conversation_messages = self._convert_messages_to_provider_format(other_messages)

        # 3. Convert tools to Anthropic format (use base class method)
        anthropic_tools = self._convert_tools_to_provider_format(tools) if tools else None

        # 4. Build API call parameters
        api_params: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation_messages,
            "thinking": self.thinking_params
        }

        if system_content:
            api_params["system"] = system_content

        if anthropic_tools:
            api_params["tools"] = anthropic_tools

        # 5. Call Anthropic API using async client from connection
        client = self.connection.get_client()
        response = await client.messages.create(**api_params)

        logger.info(f"Anthropic chat completed: {self.model_id}")

        # Log usage
        if hasattr(response, 'usage'):
            logger.debug(f"Anthropic tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}, total={response.usage.input_tokens + response.usage.output_tokens}")

        # 6. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert Anthropic response to standardized AssistantMessage
        """
        # Extract tool_calls using centralized method
        tool_calls = self._extract_tool_calls(response)
        text_content = self._extract_text_from_content_blocks(response)
        usage = self._extract_usage_info(response)

        logger.debug(f"Anthropic response: content_length={len(text_content)}, tool_calls={len(tool_calls)}, content_blocks={len(response.content) if hasattr(response, 'content') else 0}")

        return AssistantMessage(
            content=text_content,
            tool_calls=tool_calls if tool_calls else [],
            usage=usage
        )

    def _extract_text_from_content_blocks(self, response) -> str:
        """
        Extract text content from response content blocks

        Handles both dict and object notation for compatibility.

        Args:
            response: Response object from Anthropic API

        Returns:
            Text content concatenated from "text" type blocks
        """
        if not hasattr(response, "content"):
            return ""

        text_content = ""
        for block in response.content:
            block_type = self._get_block_attribute(block, "type")
            if block_type == "text":
                text = self._get_block_attribute(block, "text")
                text_content += text

        return text_content

    def _get_block_attribute(self, block, attribute: str) -> Any:
        """
        Helper to get attribute from block handling dict and object notation

        Args:
            block: Content block (can be dict or object)
            attribute: Attribute name to extract

        Returns:
            Attribute value or None if not found
        """
        if isinstance(block, dict):
            return block.get(attribute)
        return getattr(block, attribute, None)

    def _extract_usage_info(self, response) -> Optional[Usage]:
        """
        Extract usage information from response

        Args:
            response: Response object from Anthropic API

        Returns:
            Usage object if available, None otherwise
        """
        if not hasattr(response, "usage"):
            return None

        return Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )
