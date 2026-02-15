# src/client_adapters/oci_provider.py
"""
OCI Generative AI Provider.

Self-contained provider with strategy-based request handling.
No external mapping dependencies - all conversion logic is in strategies.
"""

import logging

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
except ImportError:
    raise ImportError("oci is not installed. Install with: pip install oci")

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
from obelix.core.model import AssistantMessage, StandardMessage, SystemMessage
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
        connection: OCIConnection | None = None,
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
            connection: OCIConnection singleton (default: None, reuse from GlobalConfig)
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

        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OCI_GENERATIVE_AI, "OCILLm"
            )

        self.connection = connection

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
        import os

        from obelix.infrastructure.k8s import YamlConfig

        # todo, questo andra eliminato
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_config = infra_config.get("llm_providers.oci")
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
                compartment_id=oci_config["compartment_id"],
                serving_mode=OnDemandServingMode(model_id=self.model_id),
                chat_request=chat_request,
            )

            response = await client.chat(chat_details)

            logger.info(f"OCI chat completed: {response.data.model_id}")
            logger.debug(
                f"OCI response total tokens: {response.data.chat_response.usage.total_tokens}"
            )

            # DEBUG TEMP: print tool calls for column_filter and sql_query_executor
            try:
                raw_calls = response.data.data["chatResponse"]["choices"][0][
                    "message"
                ].get("toolCalls", [])
                for tc in raw_calls:
                    if tc.get("name") in (
                        "column_filter",
                        "sql_query_executor",
                        "sql_agent",
                    ):
                        import json

                        print(f"TOOL CALL: {tc['name']}")
                        args = tc.get("arguments", "")
                        try:
                            parsed = json.loads(args)
                            for k, v in parsed.items():
                                print(f"  {k}: {v}")
                        except (json.JSONDecodeError, TypeError):
                            print(args)
                        print(f"{'=' * 50}")
            except (KeyError, IndexError, TypeError):
                pass

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

        return AssistantMessage(content=content, tool_calls=tool_calls, usage=usage)

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
