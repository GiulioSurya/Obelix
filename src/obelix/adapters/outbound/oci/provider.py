# src/client_adapters/oci_provider.py
"""
OCI Generative AI Provider.

Self-contained provider with strategy-based request handling.
No external mapping dependencies - all conversion logic is in strategies.
"""

import json
import logging
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import oci
    from oci.generative_ai_inference.models import (
        ChatDetails,
        JsonSchemaResponseFormat,
        OnDemandServingMode,
        ResponseJsonSchema,
    )
except ImportError as err:
    raise ImportError("oci is not installed. Install with: pip install oci") from err

from obelix.adapters.outbound.oci.connection import (
    OCIConnection,
    OCIIncorrectStateError,
    OCIRateLimitError,
    OCIServerError,
)

# Import strategies
from obelix.adapters.outbound.oci.strategies.base_strategy import OCIRequestStrategy
from obelix.adapters.outbound.oci.strategies.cohere_strategy import (
    CohereRequestStrategy,
)
from obelix.adapters.outbound.oci.strategies.generic_strategy import (
    GenericRequestStrategy,
    ToolCallExtractionError,
)
from obelix.core.model import (
    AssistantMessage,
    StandardMessage,
    StreamEvent,
    SystemMessage,
)
from obelix.core.model.tool_message import ToolCall
from obelix.core.model.usage import Usage
from obelix.core.tool.tool_base import Tool
from obelix.infrastructure.logging import get_logger
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

logger = get_logger(__name__)


class OCILLm(AbstractLLMProvider):
    """
    Provider for OCI Generative AI with configurable parameters.

    Self-contained: uses strategies for all format conversion.
    No external mapping dependencies.
    """

    # Available strategies
    _STRATEGIES = [GenericRequestStrategy(), CohereRequestStrategy()]

    @property
    def provider_type(self) -> Providers:
        return Providers.OCI_GENERATIVE_AI

    def __init__(
        self,
        connection: OCIConnection,
        compartment_id: str,
        model_id: str = "openai.gpt-oss-120b",
        max_tokens: int = 3500,
        temperature: float = 0.1,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
        is_stream: bool = False,
        strategy: OCIRequestStrategy | None = None,
        logger: bool = False,
        **strategy_kwargs,
    ):
        """
        Initialize the OCI provider with dependency injection of connection.

        Args:
            connection: OCIConnection singleton
            compartment_id: OCI compartment OCID
            model_id: OCI model ID
            max_tokens: Maximum number of tokens
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Stop sequences
            is_stream: Enable streaming
            strategy: Specific strategy to use (default: auto-detect from model_id)
            **strategy_kwargs: Strategy-specific parameters
        """
        if logger is True:
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger("oci").setLevel(logging.DEBUG)
            oci.base_client.is_http_log_enabled(True)
        else:
            logging.getLogger("oci").setLevel(logging.WARNING)
            oci.base_client.is_http_log_enabled(False)

        self.connection = connection
        self.compartment_id = compartment_id

        # Save parameters
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences
        self.is_stream = is_stream
        self.strategy_kwargs = strategy_kwargs

        # Auto-detect or use provided strategy
        self.strategy = strategy if strategy else self._detect_strategy(model_id)

    @classmethod
    def _detect_strategy(cls, model_id: str) -> OCIRequestStrategy:
        """Auto-detect the appropriate strategy based on model_id prefix."""
        for strategy in cls._STRATEGIES:
            for prefix in strategy.get_supported_model_prefixes():
                if model_id.startswith(prefix):
                    return strategy

        raise ValueError(
            f"No strategy found for model_id '{model_id}'. "
            f"Supported prefixes: {[p for s in cls._STRATEGIES for p in s.get_supported_model_prefixes()]}"
        )

    # Max retries for tool call extraction errors
    MAX_EXTRACTION_RETRIES = 3

    @retry(
        retry=retry_if_exception_type(
            (
                OCIRateLimitError,
                OCIServerError,
                OCIIncorrectStateError,
                httpx.TimeoutException,
                httpx.ConnectError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def invoke(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """
        Call the OCI model with standardized messages and tools.

        Uses the strategy for all format conversion (messages, tools, response).
        If tool call extraction fails, retries with error feedback to LLM.

        Args:
            messages: List of messages in StandardMessage format
            tools: List of available tools
            response_schema: Optional Pydantic BaseModel for structured JSON output
        """
        client = self.connection.get_client()

        # Work with a copy of messages for retry loop
        working_messages = list(messages)
        converted_tools = self.strategy.convert_tools(tools)

        # Convert Pydantic BaseModel -> OCI JsonSchemaResponseFormat
        oci_response_format = None
        if response_schema is not None:
            oci_response_format = JsonSchemaResponseFormat(
                json_schema=ResponseJsonSchema(
                    name=response_schema.__name__,
                    schema=response_schema.model_json_schema(),
                    is_strict=True,
                )
            )
            logger.debug(
                f"OCI structured output enabled: schema={response_schema.__name__}"
            )

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            # 1. Convert messages using strategy
            converted_messages = self.strategy.convert_messages(working_messages)

            logger.debug(
                f"OCI invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}"
            )

            # 2. Build strategy kwargs (merge user kwargs + response_format)
            strategy_kwargs = dict(self.strategy_kwargs)
            if oci_response_format is not None:
                strategy_kwargs["response_format"] = oci_response_format

            # 3. Build request using strategy
            chat_request = self.strategy.build_request(
                converted_messages=converted_messages,
                converted_tools=converted_tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop_sequences=self.stop_sequences,
                is_stream=self.is_stream,
                **strategy_kwargs,
            )

            # 3. Call OCI
            chat_details = ChatDetails(
                compartment_id=self.compartment_id,
                serving_mode=OnDemandServingMode(model_id=self.model_id),
                chat_request=chat_request,
            )

            response = await client.chat(chat_details)

            logger.info(f"OCI chat completed: {response.data.model_id}")
            logger.debug(
                f"OCI response total tokens: {response.data.chat_response.usage.total_tokens}"
            )

            # 4. Try to convert response - may raise ToolCallExtractionError
            try:
                return self._convert_response_to_assistant_message(response)
            except ToolCallExtractionError as e:
                if attempt >= self.MAX_EXTRACTION_RETRIES:
                    logger.error(
                        f"Tool call extraction failed after {attempt} attempts: {e}"
                    )
                    raise

                logger.warning(f"Tool call extraction failed (attempt {attempt}): {e}")

                # Add error feedback to messages for retry
                error_feedback = SystemMessage(
                    content=f"ERROR: Your tool call was malformed.\n{e}\nPlease retry with valid JSON arguments."
                )
                working_messages.append(error_feedback)

        # Should not reach here, but just in case
        raise RuntimeError("Unexpected end of invoke loop")

    # ========== INVOKE STREAM ==========

    async def invoke_stream(
        self,
        messages: list[StandardMessage],
        tools: list[Tool],
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming variant of invoke(). Yields StreamEvent chunks via OCI SSE.

        Text tokens are yielded as StreamEvent(token=...) as they arrive.
        Tool calls are accumulated silently (no token yield).
        The final StreamEvent (is_final=True) carries the complete
        AssistantMessage with accumulated content, tool_calls, and usage.

        Note: tool call extraction retry is NOT supported in streaming mode.
        """
        client = self.connection.get_client()

        converted_tools = self.strategy.convert_tools(tools)
        converted_messages = self.strategy.convert_messages(messages)

        # Build request with streaming enabled
        strategy_kwargs = dict(self.strategy_kwargs)

        if response_schema is not None:
            oci_response_format = JsonSchemaResponseFormat(
                json_schema=ResponseJsonSchema(
                    name=response_schema.__name__,
                    schema=response_schema.model_json_schema(),
                    is_strict=True,
                )
            )
            strategy_kwargs["response_format"] = oci_response_format

        chat_request = self.strategy.build_request(
            converted_messages=converted_messages,
            converted_tools=converted_tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop_sequences=self.stop_sequences,
            is_stream=True,
            **strategy_kwargs,
        )

        chat_details = ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=OnDemandServingMode(model_id=self.model_id),
            chat_request=chat_request,
        )

        logger.debug(
            f"[OCILLm] Invoking model (stream) | model={self.model_id} "
            f"messages={len(messages)} tools={len(converted_tools)}"
        )

        # Accumulators
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None

        async for chunk in client.chat_stream(chat_details):
            # OCI SSE format — each chunk is a choice directly:
            #   {"index": 0, "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "..."}]}, "pad": "..."}
            #   {"index": 0, "message": {"role": "ASSISTANT", "reasoningContent": "..."}, "pad": "..."}
            #   {"finishReason": "stop", "pad": "..."}
            #   {"usage": {"promptTokens": ..., "completionTokens": ..., "totalTokens": ...}}
            #
            # Some models may also use the wrapped format:
            #   {"chatResponse": {"choices": [{"delta": {"content": [...]}}], "usage": {...}}}

            # --- Extract message from chunk (flat or wrapped) ---
            message = chunk.get("message") or chunk.get("delta")
            if message is None:
                # Try wrapped format: chatResponse.choices[0].{delta|message}
                chat_response = chunk.get("chatResponse", {})
                choices = chat_response.get("choices", [])
                if choices:
                    choice = choices[0]
                    message = choice.get("delta") or choice.get("message")

                # Extract usage from wrapped format
                chunk_usage = chat_response.get("usage")
                if chunk_usage:
                    input_tokens = chunk_usage.get("promptTokens", input_tokens)
                    output_tokens = chunk_usage.get("completionTokens", output_tokens)
                    total_tokens = chunk_usage.get("totalTokens", total_tokens)

            # --- Extract usage from flat format ---
            flat_usage = chunk.get("usage")
            if flat_usage:
                input_tokens = flat_usage.get("promptTokens", input_tokens)
                output_tokens = flat_usage.get("completionTokens", output_tokens)
                total_tokens = flat_usage.get("totalTokens", total_tokens)

            # --- Reasoning content (accumulate silently, no token yield) ---
            if message and "reasoningContent" in message:
                reasoning_text = message["reasoningContent"]
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)

            if message is None:
                continue

            # --- Text content ---
            msg_content = message.get("content", [])
            if isinstance(msg_content, str):
                if msg_content:
                    content_parts.append(msg_content)
                    yield StreamEvent(token=msg_content)
            elif isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict) and block.get("type") == "TEXT":
                        text = block.get("text", "")
                        if text:
                            content_parts.append(text)
                            yield StreamEvent(token=text)
                    elif isinstance(block, str):
                        content_parts.append(block)
                        yield StreamEvent(token=block)

            # --- Tool calls ---
            msg_tool_calls = message.get("toolCalls", [])
            for tc_chunk in msg_tool_calls:
                # OCI SSE: first chunk has id+name, subsequent chunks only
                # have arguments fragments — no "index" field.
                # A new id or name signals a new tool call; otherwise
                # accumulate into the most recent one.
                if tc_chunk.get("id") or tc_chunk.get("name"):
                    # New tool call starting
                    idx = tc_chunk.get("index", len(tool_calls_acc))
                    tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                else:
                    # Continuation of the most recent tool call
                    idx = max(tool_calls_acc.keys()) if tool_calls_acc else 0
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}

                if tc_chunk.get("id"):
                    tool_calls_acc[idx]["id"] = tc_chunk["id"]
                if tc_chunk.get("name"):
                    tool_calls_acc[idx]["name"] = tc_chunk["name"]
                tc_args = tc_chunk.get("arguments")
                if tc_args is not None:
                    if isinstance(tc_args, dict):
                        tool_calls_acc[idx]["arguments"] = json.dumps(tc_args)
                    elif isinstance(tc_args, str):
                        tool_calls_acc[idx]["arguments"] += tc_args
                    else:
                        tool_calls_acc[idx]["arguments"] += json.dumps(tc_args)

        # Build complete AssistantMessage
        full_content = "".join(content_parts)

        tool_calls: list[ToolCall] = []
        if tool_calls_acc:
            import uuid

            errors = []
            for _idx, tc_data in sorted(tool_calls_acc.items()):
                try:
                    arguments = tc_data["arguments"]
                    if isinstance(arguments, str) and arguments:
                        arguments = json.loads(arguments, strict=False)
                        while isinstance(arguments, str):
                            arguments = json.loads(arguments, strict=False)
                    if not isinstance(arguments, dict):
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"] or str(uuid.uuid4()),
                            name=tc_data["name"],
                            arguments=arguments,
                        )
                    )
                except json.JSONDecodeError as e:
                    errors.append(
                        f"Tool '{tc_data['name']}': Invalid JSON in arguments - {e.msg}"
                    )

            if errors:
                raise ToolCallExtractionError(
                    "Failed to extract tool calls from stream:\n"
                    + "\n".join(f"  - {err}" for err in errors)
                )

        usage = None
        if input_tokens is not None or output_tokens is not None:
            _in = input_tokens or 0
            _out = output_tokens or 0
            _tot = total_tokens or (_in + _out)
            usage = Usage(input_tokens=_in, output_tokens=_out, total_tokens=_tot)

        metadata = {}
        if reasoning_parts:
            metadata["reasoning"] = "".join(reasoning_parts)

        assistant_message = AssistantMessage(
            content=full_content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
        )

        logger.info(
            f"[OCILLm] Stream completed | model={self.model_id} "
            f"content_chars={len(full_content)} tool_calls={len(tool_calls)} "
            f"reasoning_chars={len(metadata.get('reasoning', ''))} "
            f"input_tokens={input_tokens} output_tokens={output_tokens}"
        )

        yield StreamEvent(assistant_message=assistant_message, is_final=True)

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert OCI response to standardized AssistantMessage.

        Uses strategy.extract_tool_calls() for validation.
        Raises ToolCallExtractionError if tool calls are malformed.
        """
        # Extract tool calls using strategy (with validation)
        tool_calls = self.strategy.extract_tool_calls(response)

        # Extract text content
        content = self._extract_content(response)

        # Extract usage
        usage = self._extract_usage(response)

        reasoning = self._extract_reasoning(response)
        metadata = {}
        if reasoning:
            metadata["reasoning"] = reasoning

        return AssistantMessage(
            content=content, tool_calls=tool_calls, usage=usage, metadata=metadata
        )

    def _extract_content(self, response) -> str:
        """Extract text content from OCI response."""
        content = ""

        chat_response = response.data.chat_response if response.data else None
        if not chat_response:
            return content

        # Try Generic format first (choices[0].message.content)
        choices = chat_response.choices if hasattr(chat_response, "choices") else None
        if choices and len(choices) > 0:
            message = choices[0].message
            msg_content = message.content if message else None
            if msg_content:
                parts = []
                for c in msg_content:
                    c_type = (
                        c.type
                        if hasattr(c, "type")
                        else (c.get("type") if isinstance(c, dict) else None)
                    )
                    if c_type == "TEXT":
                        text = (
                            c.text
                            if hasattr(c, "text")
                            else (c.get("text") if isinstance(c, dict) else None)
                        )
                        if text:
                            parts.append(str(text))
                content = "".join(parts)

        # Try Cohere format (chat_response.text)
        if not content:
            cohere_text = getattr(chat_response, "text", None)
            if cohere_text:
                content = str(cohere_text)

        return content or ""

    def _extract_reasoning(self, response) -> str | None:
        """Extract reasoning content from OCI response, if available."""
        try:
            message = response.data.data["chatResponse"]["choices"][0]["message"]
            reasoning = message.get("reasoningContent")
            if reasoning:
                return str(reasoning)
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def _extract_usage(self, response) -> Usage | None:
        """Extract usage information from OCI response."""
        try:
            usage_data = response.data.chat_response.usage
            return Usage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        except AttributeError:
            return None
