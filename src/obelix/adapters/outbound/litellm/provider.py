"""
LiteLLM Provider.

Self-contained provider that wraps LiteLLM as a universal adapter for 100+ LLM
providers (OpenAI, Anthropic, Ollama, Azure, Bedrock, OCI, vLLM, etc.)
through a single OpenAI-compatible interface.

The model string controls routing: "anthropic/claude-3-5-sonnet",
"ollama/llama3", "openai/gpt-4o", "azure/my-deployment", etc.

No connection object needed: LiteLLM is stateless, api_key and base_url
are passed per-call.

Supports both synchronous (invoke) and streaming (invoke_stream) modes.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from obelix.core.model import (
    AssistantMessage,
    HumanMessage,
    StandardMessage,
    StreamEvent,
    SystemMessage,
    ToolMessage,
)
from obelix.core.model.tool_message import ToolCall
from obelix.core.model.usage import Usage
from obelix.core.tool.tool_base import Tool
from obelix.infrastructure.logging import format_message_for_trace, get_logger
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

logger = get_logger(__name__)


def _get_litellm():
    """Lazy import to avoid heavy load when litellm extra is not installed."""
    try:
        import litellm

        return litellm
    except ImportError as e:
        raise ImportError(
            "litellm is not installed. Install it with: uv sync --extra litellm"
        ) from e


def _get_litellm_exceptions():
    """Lazy import of litellm exception types for tenacity retry."""
    import litellm.exceptions

    return (
        litellm.exceptions.RateLimitError,
        litellm.exceptions.Timeout,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.APIConnectionError,
    )


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""


class LiteLLMProvider(AbstractLLMProvider):
    """
    Universal LLM provider via LiteLLM.

    Supports 100+ providers through a single interface.
    The model string controls routing (e.g. "anthropic/claude-3-5-sonnet",
    "ollama/llama3", "openai/gpt-4o").

    Constructor exposes the most common parameters explicitly.
    Any additional litellm parameter can be passed via **kwargs.

    Example:
        provider = LiteLLMProvider(
            model_id="anthropic/claude-3-5-sonnet-20241022",
            api_key="sk-ant-...",
            max_tokens=4096,
        )
        agent = BaseAgent(system_message="...", provider=provider)
    """

    MAX_EXTRACTION_RETRIES = 3

    @property
    def provider_type(self) -> Providers:
        return Providers.LITELLM

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        tool_choice: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        drop_params: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the LiteLLM provider.

        Args:
            model_id: LiteLLM model string with provider prefix
                (e.g. "anthropic/claude-3-5-sonnet", "ollama/llama3",
                "openai/gpt-4o", "azure/my-deployment").
            api_key: API key for the target provider. If None, litellm
                reads from provider-specific env vars (ANTHROPIC_API_KEY,
                OPENAI_API_KEY, etc.).
            base_url: Override API base URL (useful for proxies, self-hosted
                endpoints, or OpenAI-compatible APIs).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            frequency_penalty: Frequency penalty (-2.0 to 2.0).
            presence_penalty: Presence penalty (-2.0 to 2.0).
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            tool_choice: Tool choice mode ("auto", "required", "none").
            reasoning_effort: Reasoning depth for supported models
                ("low", "medium", "high").
            timeout: Request timeout in seconds.
            extra_headers: Additional HTTP headers for the request.
            drop_params: If True (default), silently drop parameters not
                supported by the target provider instead of raising an error.
                Highly recommended for a universal adapter.
            **kwargs: Any additional parameter supported by litellm.completion()
                (e.g. custom_llm_provider, api_version, num_retries, metadata).
        """
        # Validate litellm is available at construction time
        _get_litellm()

        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.tool_choice = tool_choice
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.extra_headers = extra_headers
        self.drop_params = drop_params
        self._extra_kwargs = kwargs

    # ========== INVOKE ==========

    async def invoke(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type["BaseModel"] | None = None,
    ) -> AssistantMessage:
        """
        Call the LLM via LiteLLM with standardized messages and tools.

        Uses litellm.acompletion() for async execution.
        If tool call extraction fails, retries with error feedback to LLM.
        Transport errors (rate limit, timeout, server errors) are retried
        via tenacity.
        """
        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            response = await self._call_litellm(
                working_messages, converted_tools, response_schema, attempt
            )

            _usage = getattr(response, "usage", None)
            _input_tokens = getattr(_usage, "prompt_tokens", None) if _usage else None
            _output_tokens = (
                getattr(_usage, "completion_tokens", None) if _usage else None
            )
            _total_tokens = getattr(_usage, "total_tokens", None) if _usage else None
            logger.info(
                f"[LiteLLMProvider] LLM call completed | model={self.model_id} "
                f"input_tokens={_input_tokens} output_tokens={_output_tokens} "
                f"total_tokens={_total_tokens}"
            )

            try:
                return self._convert_response_to_assistant_message(response)
            except ToolCallExtractionError as e:
                if attempt >= self.MAX_EXTRACTION_RETRIES:
                    logger.error(
                        f"[LiteLLMProvider] Tool call extraction failed — max retries "
                        f"exhausted | model={self.model_id} attempt={attempt} error={e}",
                        exc_info=True,
                    )
                    raise

                logger.warning(
                    f"[LiteLLMProvider] Tool call extraction failed — retrying | "
                    f"model={self.model_id} attempt={attempt}/{self.MAX_EXTRACTION_RETRIES} "
                    f"error={e}"
                )

                error_feedback = HumanMessage(
                    content=(
                        f"ERROR: Your tool call was malformed.\n{e}\n"
                        "Please retry with valid JSON arguments."
                    )
                )
                working_messages.append(error_feedback)

        raise RuntimeError("Unexpected end of invoke loop")

    def _build_api_params(
        self,
        messages: list[StandardMessage],
        converted_tools: list[dict],
        response_schema: type["BaseModel"] | None,
    ) -> dict[str, Any]:
        """Build the API params dict shared by invoke and invoke_stream."""
        converted_messages = self._convert_messages(messages)

        api_params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": converted_messages,
            "drop_params": self.drop_params,
        }

        if self.api_key is not None:
            api_params["api_key"] = self.api_key
        if self.base_url is not None:
            api_params["base_url"] = self.base_url
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
        if self.timeout is not None:
            api_params["timeout"] = self.timeout
        if self.extra_headers is not None:
            api_params["extra_headers"] = self.extra_headers

        if response_schema is not None:
            api_params["response_format"] = response_schema

        if converted_tools:
            api_params["tools"] = converted_tools
            if self.tool_choice is not None:
                api_params["tool_choice"] = self.tool_choice

        # Merge any extra kwargs from constructor
        api_params.update(self._extra_kwargs)

        return api_params

    async def _call_litellm(
        self,
        messages: list[StandardMessage],
        converted_tools: list[dict],
        response_schema: type["BaseModel"] | None,
        attempt: int,
    ):
        """
        Build API params and call litellm.acompletion() with tenacity retry.

        Separated from invoke() to allow tenacity decorator on the actual
        network call without interfering with the tool-extraction retry loop.
        """
        _get_litellm()  # Validate availability before building params

        logger.debug(
            f"[LiteLLMProvider] Invoking model | model={self.model_id} "
            f"messages={len(messages)} tools={len(converted_tools)} "
            f"attempt={attempt}/{self.MAX_EXTRACTION_RETRIES}"
        )

        api_params = self._build_api_params(messages, converted_tools, response_schema)

        return await self._acompletion_with_retry(**api_params)

    async def _acompletion_with_retry(self, **api_params):
        """
        Call litellm.acompletion with tenacity retry on transient errors
        and structured logging for non-retryable HTTP errors.

        Tenacity decorator is applied dynamically because litellm exceptions
        are lazily imported.
        """
        litellm = _get_litellm()
        retryable_exceptions = _get_litellm_exceptions()

        @retry(
            retry=retry_if_exception_type(retryable_exceptions),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _do_call():
            return await litellm.acompletion(**api_params)

        try:
            return await _do_call()
        except retryable_exceptions:
            # Already logged by tenacity before_sleep; re-raise as-is
            raise
        except litellm.exceptions.AuthenticationError as e:
            logger.error(
                f"[LiteLLMProvider] Authentication failed | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} "
                f"status={getattr(e, 'status_code', 401)} — "
                f"check api_key or provider credentials"
            )
            raise
        except litellm.exceptions.BadRequestError as e:
            logger.error(
                f"[LiteLLMProvider] Bad request | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} "
                f"status={getattr(e, 'status_code', 400)} — {e.message}"
            )
            raise
        except litellm.exceptions.NotFoundError as e:
            logger.error(
                f"[LiteLLMProvider] Model or endpoint not found | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} "
                f"status={getattr(e, 'status_code', 404)} — {e.message}"
            )
            raise
        except litellm.exceptions.PermissionDeniedError as e:
            logger.error(
                f"[LiteLLMProvider] Permission denied | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} "
                f"status={getattr(e, 'status_code', 403)} — {e.message}"
            )
            raise
        except litellm.exceptions.ContextWindowExceededError as e:
            logger.error(
                f"[LiteLLMProvider] Context window exceeded | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} — {e.message}"
            )
            raise
        except litellm.exceptions.ContentPolicyViolationError as e:
            logger.warning(
                f"[LiteLLMProvider] Content policy violation | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} — {e.message}"
            )
            raise
        except litellm.exceptions.APIError as e:
            logger.error(
                f"[LiteLLMProvider] API error | model={self.model_id} "
                f"provider={getattr(e, 'llm_provider', 'unknown')} "
                f"status={getattr(e, 'status_code', 'unknown')} — {e.message}"
            )
            raise

    # ========== INVOKE STREAM ==========

    async def invoke_stream(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type["BaseModel"] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming variant of invoke(). Yields StreamEvent chunks.

        Text tokens are yielded as StreamEvent(token=...) as they arrive.
        Tool calls are accumulated silently (no token yield).
        The final StreamEvent (is_final=True) carries the complete
        AssistantMessage with accumulated content, tool_calls, and usage.

        Note: tool call extraction retry (the feedback loop in invoke())
        is NOT supported in streaming mode. If tool call JSON is malformed,
        the error propagates to the caller.
        """
        _get_litellm()
        converted_tools = self._convert_tools(tools)

        api_params = self._build_api_params(messages, converted_tools, response_schema)
        api_params["stream"] = True
        api_params["stream_options"] = {"include_usage": True}

        logger.debug(
            f"[LiteLLMProvider] Invoking model (stream) | model={self.model_id} "
            f"messages={len(messages)} tools={len(converted_tools)}"
        )

        response = await self._acompletion_with_retry(**api_params)

        # Accumulators
        content_parts: list[str] = []
        # tool_calls_acc: {index: {"id": str, "name": str, "arguments": str}}
        tool_calls_acc: dict[int, dict[str, str]] = {}
        usage: Usage | None = None

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None

            # Extract usage from the final chunk (stream_options.include_usage)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage and getattr(chunk_usage, "prompt_tokens", None) is not None:
                usage = Usage(
                    input_tokens=chunk_usage.prompt_tokens,
                    output_tokens=chunk_usage.completion_tokens,
                    total_tokens=chunk_usage.total_tokens,
                )

            if not choice:
                continue

            delta = choice.delta

            # Text content
            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)
                yield StreamEvent(token=delta_content)

            # Tool calls (accumulate fragments)
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc_chunk in delta_tool_calls:
                    idx = getattr(tc_chunk, "index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}

                    if getattr(tc_chunk, "id", None):
                        tool_calls_acc[idx]["id"] = tc_chunk.id
                    fn = getattr(tc_chunk, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            tool_calls_acc[idx]["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            tool_calls_acc[idx]["arguments"] += fn.arguments

        # Build complete AssistantMessage
        full_content = "".join(content_parts)

        tool_calls: list[ToolCall] = []
        if tool_calls_acc:
            errors = []
            for _idx, tc_data in sorted(tool_calls_acc.items()):
                try:
                    arguments = tc_data["arguments"]
                    if isinstance(arguments, str) and arguments:
                        arguments = json.loads(arguments, strict=False)
                        while isinstance(arguments, str):
                            arguments = json.loads(arguments, strict=False)
                    elif not arguments:
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=arguments,
                        )
                    )
                except json.JSONDecodeError as e:
                    errors.append(
                        f"Tool '{tc_data['name']}': Invalid JSON in arguments - {e.msg}"
                    )
                except ValidationError as e:
                    errors.append(f"Tool '{tc_data['name']}': {e.errors()[0]['msg']}")

            if errors:
                raise ToolCallExtractionError(
                    "Failed to extract tool calls from stream:\n"
                    + "\n".join(f"  - {err}" for err in errors)
                )

        assistant_message = AssistantMessage(
            content=full_content,
            tool_calls=tool_calls,
            usage=usage,
        )

        logger.info(
            f"[LiteLLMProvider] Stream completed | model={self.model_id} "
            f"content_chars={len(full_content)} tool_calls={len(tool_calls)} "
            f"input_tokens={usage.input_tokens if usage else None} "
            f"output_tokens={usage.output_tokens if usage else None}"
        )

        yield StreamEvent(assistant_message=assistant_message, is_final=True)

    # ========== MESSAGE CONVERSION ==========

    def _convert_messages(self, messages: list[StandardMessage]) -> list[dict]:
        """
        Convert standardized messages to OpenAI format.

        LiteLLM uses OpenAI message format for all providers.
        System messages stay in the messages array — LiteLLM handles
        provider-specific extraction (e.g. Anthropic's system param)
        internally.
        """
        converted = []

        for i, msg in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(msg)}")

            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})

            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})

            elif isinstance(msg, AssistantMessage):
                assistant_msg: dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content

                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "type": "function",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                if not msg.content and not msg.tool_calls:
                    assistant_msg["content"] = ""

                converted.append(assistant_msg)

            elif isinstance(msg, ToolMessage):
                for result in msg.tool_results:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": str(result.result)
                            if result.result is not None
                            else (result.error or "No result"),
                        }
                    )

        logger.debug(
            f"[LiteLLMProvider] Messages converted | "
            f"input={len(messages)} output={len(converted)}"
        )
        return converted

    # ========== TOOL CONVERSION ==========

    def _convert_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert tool list to OpenAI function format (used by all LiteLLM providers)."""
        if not tools:
            return []

        converted = []
        for tool in tools:
            schema = tool.create_schema()
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.inputSchema,
                    },
                }
            )

        logger.debug(
            f"[LiteLLMProvider] Tools converted | count={len(tools)} "
            f"names={[t.create_schema().name for t in tools]}"
        )
        return converted

    # ========== TOOL CALL EXTRACTION ==========

    def _extract_tool_calls(self, response) -> list[ToolCall]:
        """
        Extract and validate tool calls from LiteLLM response.

        LiteLLM returns OpenAI-compatible ModelResponse:
        response.choices[0].message.tool_calls -> List[ChatCompletionMessageToolCall]

        Each tool call has:
        - id: str
        - type: "function"
        - function.name: str
        - function.arguments: str (JSON)

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails.
        """
        if not hasattr(response, "choices") or not response.choices:
            return []

        message = response.choices[0].message
        if not getattr(message, "tool_calls", None):
            return []

        tool_calls = []
        errors = []

        for call in message.tool_calls:
            if getattr(call, "type", None) != "function":
                continue

            try:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                # Handle double-encoding
                while isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                tool_call = ToolCall(
                    id=call.id, name=call.function.name, arguments=arguments
                )
                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                errors.append(
                    f"Tool '{call.function.name}': Invalid JSON in arguments - {e.msg}"
                )
            except ValidationError as e:
                errors.append(f"Tool '{call.function.name}': {e.errors()[0]['msg']}")

        if errors:
            raise ToolCallExtractionError(
                "Failed to extract tool calls:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== RESPONSE EXTRACTION ==========

    def _extract_content(self, response) -> str:
        """Extract text content from LiteLLM ModelResponse."""
        if not hasattr(response, "choices") or not response.choices:
            return ""

        message = response.choices[0].message
        return message.content if message.content else ""

    def _extract_usage(self, response) -> Usage | None:
        """Extract usage information from LiteLLM ModelResponse."""
        if not hasattr(response, "usage") or not response.usage:
            return None

        return Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert LiteLLM ModelResponse to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction fails.
        """
        tool_calls = self._extract_tool_calls(response)
        content = self._extract_content(response)
        usage = self._extract_usage(response)

        logger.debug(
            f"[LiteLLMProvider] Response parsed | content_chars={len(content)} "
            f"tool_calls={len(tool_calls)}"
        )

        return AssistantMessage(content=content, tool_calls=tool_calls, usage=usage)
