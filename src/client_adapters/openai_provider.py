# src/client_adapters/openai_provider.py
"""
OpenAI Provider.

Self-contained provider with inline message/tool conversion.
No external mapping dependencies.

Supports OpenAI GPT models and any OpenAI-compatible API
(via custom base_url in OpenAIConnection).
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
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    ConflictError,
    APIResponseValidationError,
)

from src.client_adapters.llm_abstraction import AbstractLLMProvider
from src.obelix_types import SystemMessage, HumanMessage, AssistantMessage, ToolMessage, StandardMessage
from src.obelix_types.tool_message import ToolCall
from src.obelix_types.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection.openai_connection import OpenAIConnection
from src.logging_config import get_logger, format_message_for_trace

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""
    pass


class OpenAIProvider(AbstractLLMProvider):
    """
    Provider for OpenAI GPT models with configurable parameters.

    Self-contained: all conversion logic is inline.

    Supports:
    - System obelix_types (in obelix_types array with role="system")
    - Tool calling with function format
    - Tool choice control (auto, required, none, specific function)
    - Parallel tool calls
    - Response format (text, json_object, json_schema)
    - Reasoning effort (for o-series models)
    - Multi-turn conversations
    - Usage tracking
    """

    MAX_EXTRACTION_RETRIES = 3

    @property
    def provider_type(self) -> Providers:
        return Providers.OPENAI

    def __init__(self,
                 connection: Optional[OpenAIConnection] = None,
                 model_id: str = "gpt-4o",
                 max_tokens: int = 4096,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 tool_choice: Optional[str] = None,
                 parallel_tool_calls: Optional[bool] = None,
                 response_format: Optional[Dict[str, Any]] = None,
                 reasoning_effort: Optional[str] = None):
        """
        Initialize the OpenAI provider with dependency injection of connection.

        Args:
            connection: OpenAIConnection singleton
            model_id: OpenAI model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            seed: Seed for deterministic sampling
            stop: Stop sequences
            tool_choice: Tool choice mode ("auto", "required", "none")
            parallel_tool_calls: Enable parallel function calls
            response_format: Response format (e.g. {"type": "json_object"})
            reasoning_effort: Reasoning depth for o-series ("low", "medium", "high")
        """
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OPENAI,
                "OpenAIProvider"
            )

        self.connection = connection

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort

    # ========== INVOKE ==========

    @retry(
        retry=retry_if_exception_type((
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            ConflictError,
            APIResponseValidationError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the OpenAI model with standardized obelix_types and tools.

        Uses AsyncOpenAI client for native async support.
        If tool call extraction fails, retries with error feedback to LLM.
        """
        client = self.connection.get_client()

        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            logger.debug(
                f"OpenAI invoke: model={self.model_id}, obelix_types={len(working_messages)}, "
                f"tools={len(converted_tools)}, attempt={attempt}"
            )

            converted_messages = self._convert_messages(working_messages)

            api_params: Dict[str, Any] = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "obelix_types": converted_messages,
            }

            if self.top_p is not None:
                api_params["top_p"] = self.top_p
            if self.frequency_penalty is not None:
                api_params["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                api_params["presence_penalty"] = self.presence_penalty
            if self.seed is not None:
                api_params["seed"] = self.seed
            if self.stop is not None:
                api_params["stop"] = self.stop
            if self.reasoning_effort is not None:
                api_params["reasoning_effort"] = self.reasoning_effort
            if self.response_format is not None:
                api_params["response_format"] = self.response_format

            if converted_tools:
                api_params["tools"] = converted_tools
                if self.tool_choice is not None:
                    api_params["tool_choice"] = self.tool_choice
                if self.parallel_tool_calls is not None:
                    api_params["parallel_tool_calls"] = self.parallel_tool_calls

            response = await client.chat.completions.create(**api_params)

            logger.info(f"OpenAI chat completed: {self.model_id}")

            if hasattr(response, 'usage') and response.usage:
                logger.debug(
                    f"OpenAI tokens: input={response.usage.prompt_tokens}, "
                    f"output={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

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
        Convert standardized obelix_types to OpenAI format.

        OpenAI keeps system obelix_types in the obelix_types array (unlike Anthropic).
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

        logger.debug(f"Converted {len(messages)} obelix_types to {len(converted)} OpenAI obelix_types")
        return converted

    # ========== TOOL CONVERSION ==========

    def _convert_tools(self, tools: List[ToolBase]) -> List[dict]:
        """Convert tool list to OpenAI function format."""
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

        logger.debug(f"Converted {len(tools)} tools to OpenAI format")
        return converted

    # ========== TOOL CALL EXTRACTION ==========

    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """
        Extract and validate tool calls from OpenAI response.

        Path: response.choices[0].message.tool_calls

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails
        """
        if not hasattr(response, 'choices') or not response.choices:
            return []

        message = response.choices[0].message
        if not message.tool_calls:
            return []

        tool_calls = []
        errors = []

        for call in message.tool_calls:
            if getattr(call, 'type', None) != "function":
                continue

            try:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                # Handle double-encoding
                while isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                tool_call = ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                errors.append(
                    f"Tool '{call.function.name}': Invalid JSON in arguments - {e.msg}"
                )
            except ValidationError as e:
                errors.append(
                    f"Tool '{call.function.name}': {e.errors()[0]['msg']}"
                )

        if errors:
            raise ToolCallExtractionError(
                "Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== RESPONSE EXTRACTION ==========

    def _extract_content(self, response) -> str:
        """Extract text content from OpenAI response."""
        if not hasattr(response, 'choices') or not response.choices:
            return ""

        message = response.choices[0].message
        return message.content if message.content else ""

    def _extract_usage(self, response) -> Optional[Usage]:
        """Extract usage information from OpenAI response."""
        if not hasattr(response, 'usage') or not response.usage:
            return None

        return Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert OpenAI response to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction fails
        """
        tool_calls = self._extract_tool_calls(response)
        content = self._extract_content(response)
        usage = self._extract_usage(response)

        logger.debug(
            f"OpenAI response: content_length={len(content)}, tool_calls={len(tool_calls)}"
        )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )