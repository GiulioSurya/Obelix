# src/client_adapters/anthropic_provider.py
"""
Anthropic Claude Provider.

Self-contained provider with inline message/tool conversion.
No external mapping dependencies.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

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

from obelix.ports.outbound.llm_provider import AbstractLLMProvider
from obelix.domain.model import SystemMessage, StandardMessage, AssistantMessage, HumanMessage, ToolMessage
from obelix.domain.model.tool_message import ToolCall
from obelix.domain.model.usage import Usage
from obelix.domain.tool.tool_base import ToolBase
from obelix.infrastructure.providers import Providers
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.infrastructure.logging import get_logger, format_message_for_trace

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""
    pass


class AnthropicProvider(AbstractLLMProvider):
    """
    Provider for Anthropic Claude with configurable parameters.

    Supports:
    - System messages (as separate parameter, not in messages array)
    - Tool calling with content blocks
    - Multi-turn conversations
    - Usage tracking
    - Extended thinking mode
    """

    # Max retries for tool call extraction errors
    MAX_EXTRACTION_RETRIES = 3

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
        Initialize the Anthropic provider with dependency injection of connection.

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
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.ANTHROPIC,
                "AnthropicProvider"
            )

        self.connection = connection

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = 1 if thinking_mode else temperature
        self.top_p = top_p
        self.thinking_mode = thinking_mode

        if thinking_mode:
            self.thinking_params = thinking_params or {"type": "enabled", "budget_tokens": 2000}
            logger.debug(f"Thinking mode enabled: {self.thinking_params}")
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
        Call the Anthropic model with standardized messages and tools.

        Uses AsyncAnthropic client for native async support.
        Retries automatically on transient errors (connection, timeout, rate limit, server errors).
        If tool call extraction fails, retries with error feedback to LLM.
        """
        client = self.connection.get_client()

        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            logger.debug(f"Anthropic invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}, thinking_mode={self.thinking_mode}")

            system_content, conversation_messages = self._convert_messages(working_messages)

            api_params: Dict[str, Any] = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": conversation_messages,
                "thinking": self.thinking_params
            }

            if system_content:
                api_params["system"] = system_content

            if self.top_p is not None:
                api_params["top_p"] = self.top_p

            if converted_tools:
                api_params["tools"] = converted_tools

            response = await client.messages.create(**api_params)

            logger.info(f"Anthropic chat completed: {self.model_id}")

            if hasattr(response, 'usage'):
                logger.debug(f"Anthropic tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}, total={response.usage.input_tokens + response.usage.output_tokens}")

            try:
                return self._convert_response_to_assistant_message(response)
            except ToolCallExtractionError as e:
                if attempt >= self.MAX_EXTRACTION_RETRIES:
                    logger.error(f"Tool call extraction failed after {attempt} attempts: {e}")
                    raise

                logger.warning(f"Tool call extraction failed (attempt {attempt}): {e}")

                error_feedback = HumanMessage(
                    content=f"ERROR: Your tool call was malformed.\n{e}\nPlease retry with valid JSON arguments."
                )
                working_messages.append(error_feedback)

        raise RuntimeError("Unexpected end of invoke loop")

    def _convert_messages(self, messages: List[StandardMessage]) -> Tuple[Optional[str], List[dict]]:
        """
        Convert standardized messages to Anthropic format.

        Anthropic requires system message as separate parameter, not in messages array.

        Returns:
            Tuple of (system_content, conversation_messages)
        """
        system_content = None
        conversation_messages = []

        for i, msg in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(msg)}")

            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                conversation_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AssistantMessage):
                content_blocks = []

                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })

                conversation_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            elif isinstance(msg, ToolMessage):
                tool_results_content = []
                for result in msg.tool_results:
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": str(result.result) if result.result else (result.error or "No result")
                    })

                conversation_messages.append({
                    "role": "user",
                    "content": tool_results_content
                })

        logger.debug(f"Converted {len(messages)} messages: system={'present' if system_content else 'absent'}, conversation={len(conversation_messages)}")

        return system_content, conversation_messages

    def _convert_tools(self, tools: List[ToolBase]) -> List[dict]:
        """
        Convert tool list to Anthropic format.

        Args:
            tools: List of ToolBase instances

        Returns:
            List of Anthropic tool schemas
        """
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            schema = tool.create_schema()
            anthropic_tools.append({
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.inputSchema
            })

        logger.debug(f"Converted {len(tools)} tools to Anthropic format")

        return anthropic_tools

    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract and validate tool calls from Anthropic response.

        Raises:
            ToolCallExtractionError: If tool calls are malformed or validation fails
        """
        if not hasattr(response, "content"):
            return []

        tool_calls = []
        for block in response.content:
            block_type = self._get_block_attribute(block, "type")
            if block_type == "tool_use":
                try:
                    tool_id = self._get_block_attribute(block, "id")
                    tool_name = self._get_block_attribute(block, "name")
                    tool_input = self._get_block_attribute(block, "input")

                    if not tool_id or not tool_name:
                        raise ToolCallExtractionError(
                            f"Missing required fields: id={tool_id}, name={tool_name}"
                        )

                    if not isinstance(tool_input, dict):
                        raise ToolCallExtractionError(
                            f"Tool input must be a dict, got {type(tool_input).__name__}"
                        )

                    tool_call = ToolCall(
                        id=tool_id,
                        name=tool_name,
                        arguments=tool_input
                    )
                    tool_calls.append(tool_call)

                except Exception as e:
                    if isinstance(e, ToolCallExtractionError):
                        raise
                    raise ToolCallExtractionError(
                        f"Failed to parse tool call block: {e}"
                    ) from e

        logger.debug(f"Extracted {len(tool_calls)} tool calls from response")

        return tool_calls

    def _extract_content(self, response) -> str:
        """
        Extract text content from response content blocks.

        Handles both dict and object notation for compatibility.

        Args:
            response: Response object from Anthropic API

        Returns:
            Text content concatenated from "text" type blocks
        """
        if not hasattr(response, "content"):
            return ""

        text_parts = []
        for block in response.content:
            block_type = self._get_block_attribute(block, "type")
            if block_type == "text":
                text = self._get_block_attribute(block, "text")
                if text:
                    text_parts.append(text)

        return "".join(text_parts)

    def _extract_usage(self, response) -> Optional[Usage]:
        """
        Extract usage information from response.

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

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert Anthropic response to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction or validation fails
        """
        tool_calls = self._extract_tool_calls(response)
        text_content = self._extract_content(response)
        usage = self._extract_usage(response)

        logger.debug(
            f"Anthropic response: content_length={len(text_content)}, "
            f"tool_calls={len(tool_calls)}, "
            f"content_blocks={len(response.content) if hasattr(response, 'content') else 0}"
        )

        return AssistantMessage(
            content=text_content,
            tool_calls=tool_calls,
            usage=usage
        )

    def _get_block_attribute(self, block, attribute: str) -> Any:
        """
        Helper to get attribute from block handling dict and object notation.

        Args:
            block: Content block (can be dict or object)
            attribute: Attribute name to extract

        Returns:
            Attribute value or None if not found
        """
        if isinstance(block, dict):
            return block.get(attribute)
        return getattr(block, attribute, None)