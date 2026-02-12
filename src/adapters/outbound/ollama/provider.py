# src/adapters/outbound/ollama/provider.py
"""
Ollama Provider.

Self-contained provider with inline message/tool conversion.
No external mapping dependencies.

Uses OpenAI-compatible dict format for messages.
Ollama SDK has NO built-in retry - tenacity handles transient errors.
AsyncClient is natively async - no asyncio.to_thread() needed.
"""
import json
import logging
from typing import List, Dict, Any, Optional

from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

try:
    from ollama import AsyncClient, ResponseError, RequestError
except ImportError:
    raise ImportError(
        "ollama is not installed. Install with: pip install ollama"
    )

from src.ports.outbound.llm_provider import AbstractLLMProvider
from src.core.model import SystemMessage, HumanMessage, AssistantMessage, ToolMessage, StandardMessage
from src.core.model.tool_message import ToolCall
from src.core.model.usage import Usage
from src.core.tool.tool_base import ToolBase
from src.infrastructure.providers import Providers
from src.infrastructure.logging import get_logger, format_message_for_trace

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""
    pass


class OllamaProvider(AbstractLLMProvider):
    """
    Provider for Ollama with configurable parameters.

    Self-contained: all conversion logic is inline.
    Uses OpenAI-compatible dict format for messages.
    """

    MAX_EXTRACTION_RETRIES = 3

    @property
    def provider_type(self) -> Providers:
        return Providers.OLLAMA

    def __init__(self,
                 model_id: str = "a-kore/Arctic-Text2SQL-R1-7B",
                 base_url: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = 2000,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 keep_alive: Optional[str] = None):
        """
        Initialize the Ollama provider.

        Args:
            model_id: Ollama model ID (default: "a-kore/Arctic-Text2SQL-R1-7B")
            base_url: Base URL of Ollama server (default: None = http://localhost:11434)
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum number of tokens (default: 2000)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            keep_alive: Keep model in memory (default: None)
        """
        self.model_id = model_id

        if base_url:
            self.client = AsyncClient(host=base_url)
        else:
            self.client = AsyncClient()

        # Build options dict with only non-None parameters
        self.options: Dict[str, Any] = {}
        if temperature is not None:
            self.options["temperature"] = temperature
        if max_tokens is not None:
            self.options["num_predict"] = max_tokens
        if top_p is not None:
            self.options["top_p"] = top_p
        if top_k is not None:
            self.options["top_k"] = top_k
        if seed is not None:
            self.options["seed"] = seed
        if stop is not None:
            self.options["stop"] = stop

        self.keep_alive = keep_alive

    # ========== INVOKE ==========

    @retry(
        retry=retry_if_exception_type((
            ResponseError,
            ConnectionError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the Ollama model with standardized messages and tools.

        Uses AsyncClient for native async support.
        If tool call extraction fails, retries with error feedback to LLM.
        """
        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            logger.debug(f"Ollama invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}")

            converted_messages = self._convert_messages(working_messages)

            chat_params: Dict[str, Any] = {
                "model": self.model_id,
                "messages": converted_messages,
            }

            if converted_tools:
                chat_params["tools"] = converted_tools

            if self.options:
                chat_params["options"] = self.options

            if self.keep_alive is not None:
                chat_params["keep_alive"] = self.keep_alive

            response = await self.client.chat(**chat_params)

            logger.info(f"Ollama chat completed: {self.model_id}")

            if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
                logger.debug(f"Ollama tokens: prompt={response.prompt_eval_count}, completion={response.eval_count}")

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

    # ========== MESSAGE CONVERSION ==========

    def _convert_messages(self, messages: List[StandardMessage]) -> List[dict]:
        """
        Convert standardized messages to Ollama format (OpenAI-compatible dicts).
        """
        converted = []

        for i, msg in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(msg)}")

            if isinstance(msg, SystemMessage):
                converted.append({
                    "role": "system",
                    "content": msg.content
                })

            elif isinstance(msg, HumanMessage):
                converted.append({
                    "role": "user",
                    "content": msg.content
                })

            elif isinstance(msg, AssistantMessage):
                assistant_msg: Dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content

                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "type": "function",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]

                if not msg.content and not msg.tool_calls:
                    assistant_msg["content"] = ""

                converted.append(assistant_msg)

            elif isinstance(msg, ToolMessage):
                for result in msg.tool_results:
                    converted.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": str(result.result) if result.result is not None else (result.error or "No result")
                    })

        logger.debug(f"Converted {len(messages)} messages to {len(converted)} Ollama messages")
        return converted

    # ========== TOOL CONVERSION ==========

    def _convert_tools(self, tools: List[ToolBase]) -> List[dict]:
        """Convert tool list to OpenAI-compatible function format."""
        if not tools:
            return []

        converted = []
        for tool in tools:
            schema = tool.create_schema()
            converted.append({
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.inputSchema
                }
            })

        logger.debug(f"Converted {len(tools)} tools to Ollama format")
        return converted

    # ========== TOOL CALL EXTRACTION ==========

    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract and validate tool calls from Ollama response.

        Path: response.message.tool_calls[i].function.{name, arguments}
        Ollama returns arguments as Mapping[str, Any] (already parsed, not JSON string).

        Raises:
            ToolCallExtractionError: If validation fails
        """
        if not hasattr(response, 'message') or not response.message:
            return []

        raw_tool_calls = getattr(response.message, 'tool_calls', None)
        if not raw_tool_calls:
            return []

        tool_calls = []
        errors = []

        for call in raw_tool_calls:
            if not hasattr(call, 'function'):
                continue

            try:
                # Ollama arguments is already a Mapping[str, Any]
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                tool_call = ToolCall(
                    id=getattr(call, 'id', None) or f"ollama_{len(tool_calls)}",
                    name=call.function.name,
                    arguments=dict(arguments)
                )
                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                errors.append(
                    f"Tool '{call.function.name}': Invalid JSON in arguments - {e.msg}"
                )
            except (ValidationError, AttributeError) as e:
                errors.append(
                    f"Tool '{getattr(call.function, 'name', 'unknown')}': {e}"
                )

        if errors:
            raise ToolCallExtractionError(
                "Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== RESPONSE EXTRACTION ==========

    def _extract_content(self, response) -> str:
        """Extract text content from Ollama response."""
        if not hasattr(response, 'message') or not response.message:
            return ""

        return response.message.content or ""

    def _extract_usage(self, response) -> Optional[Usage]:
        """Extract usage information from Ollama response."""
        prompt_tokens = getattr(response, 'prompt_eval_count', None)
        completion_tokens = getattr(response, 'eval_count', None)

        if prompt_tokens is None and completion_tokens is None:
            return None

        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0

        return Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert Ollama response to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction fails
        """
        tool_calls = self._extract_tool_calls(response)
        content = self._extract_content(response)
        usage = self._extract_usage(response)

        logger.debug(
            f"Ollama response: content_length={len(content)}, tool_calls={len(tool_calls)}"
        )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )