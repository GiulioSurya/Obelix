# src/llm_providers/oci_provider.py
"""
OCI Generative AI Provider.

Self-contained provider with strategy-based request handling.
No external mapping dependencies - all conversion logic is in strategies.
"""
import logging
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

try:
    import oci
    from oci.generative_ai_inference.models import (
        ChatDetails,
        OnDemandServingMode,
    )
except ImportError:
    raise ImportError(
        "oci is not installed. Install with: pip install oci"
    )

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages import SystemMessage, StandardMessage, AssistantMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import OCIConnection
from src.connections.llm_connection.oci_connection import (
    OCIServiceError,
    OCIRateLimitError,
    OCIServerError,
    OCIIncorrectStateError,
)
from src.logging_config import get_logger

# Import strategies
from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.llm_providers.oci_strategies.generic_strategy import (
    GenericRequestStrategy,
    ToolCallExtractionError,
)
from src.llm_providers.oci_strategies.gemini_strategy import GeminiRequestStrategy
from src.llm_providers.oci_strategies.cohere_strategy import CohereRequestStrategy

logger = get_logger(__name__)


class OCILLm(AbstractLLMProvider):
    """
    Provider for OCI Generative AI with configurable parameters.

    Self-contained: uses strategies for all format conversion.
    No external mapping dependencies.
    """

    # Available strategies (order matters: first match on model prefix wins)
    _STRATEGIES = [
        GeminiRequestStrategy(),
        GenericRequestStrategy(),
        CohereRequestStrategy(),
    ]

    @property
    def provider_type(self) -> Providers:
        return Providers.OCI_GENERATIVE_AI

    def __init__(self,
                 connection: Optional[OCIConnection] = None,
                 model_id: str = "openai.gpt-oss-120b",
                 max_tokens: int = 3500,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 stop_sequences: Optional[List[str]] = None,
                 is_stream: bool = False,
                 strategy: Optional[OCIRequestStrategy] = None,
                 logger: bool = False,
                 **strategy_kwargs):
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
            logging.getLogger('oci').setLevel(logging.DEBUG)
            oci.base_client.is_http_log_enabled(True)
        else:
            logging.getLogger('oci').setLevel(logging.WARNING)
            oci.base_client.is_http_log_enabled(False)

        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OCI_GENERATIVE_AI,
                "OCILLm"
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
        retry=retry_if_exception_type((
            OCIRateLimitError,
            OCIServerError,
            OCIIncorrectStateError,
            httpx.TimeoutException,
            httpx.ConnectError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the OCI model with standardized messages and tools.

        Uses the strategy for all format conversion (messages, tools, response).
        If tool call extraction fails, retries with error feedback to LLM.
        """
        from src.k8s_config import YamlConfig
        import os
        #todo, questo andra eliminato
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_config = infra_config.get("llm_providers.oci")
        client = self.connection.get_client()

        # Work with a copy of messages for retry loop
        working_messages = list(messages)
        converted_tools = self.strategy.convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            # 1. Convert messages using strategy
            converted_messages = self.strategy.convert_messages(working_messages)

            logger.debug(f"OCI invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}")

            # 2. Build request using strategy
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
                **self.strategy_kwargs
            )

            # 3. Call OCI
            chat_details = ChatDetails(
                compartment_id=oci_config["compartment_id"],
                serving_mode=OnDemandServingMode(model_id=self.model_id),
                chat_request=chat_request
            )

            # TODO: remove after debugging Gemini parallel tool calls
            from src.connections.llm_connection.oci_connection import _serialize_oci_model
            serialized = _serialize_oci_model(chat_details)
            msgs = serialized.get("chatRequest", {}).get("messages", [])
            for idx, m in enumerate(msgs):
                logger.trace(f"OCI serialized msg[{idx}]: role={m.get('role')} "
                             f"content={m.get('content', 'ABSENT')} "
                             f"toolCalls={len(m.get('toolCalls') or [])} "
                             f"toolCallId={m.get('toolCallId', '-')}")

            response = await client.chat(chat_details)
            logger.info(f"OCI chat completed: {response.data.model_id}")
            logger.debug(f"OCI response total tokens: {response.data.chat_response.usage.total_tokens}")

            # 4. Try to convert response - may raise ToolCallExtractionError
            try:
                return self._convert_response_to_assistant_message(response)
            except ToolCallExtractionError as e:
                if attempt >= self.MAX_EXTRACTION_RETRIES:
                    logger.error(f"Tool call extraction failed after {attempt} attempts: {e}")
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

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )

    def _extract_content(self, response) -> str:
        """Extract text content from OCI response."""
        content = ""

        chat_response = response.data.chat_response if response.data else None
        if not chat_response:
            return content

        # Try Generic format first (choices[0].message.content)
        choices = chat_response.choices if hasattr(chat_response, 'choices') else None
        if choices and len(choices) > 0:
            message = choices[0].message
            msg_content = message.content if message else None
            if msg_content:
                parts = []
                for c in msg_content:
                    c_type = c.type if hasattr(c, 'type') else (c.get("type") if isinstance(c, dict) else None)
                    if c_type == "TEXT":
                        text = c.text if hasattr(c, 'text') else (c.get("text") if isinstance(c, dict) else None)
                        if text:
                            parts.append(str(text))
                content = "".join(parts)

        # Try Cohere format (chat_response.text)
        if not content:
            cohere_text = getattr(chat_response, 'text', None)
            if cohere_text:
                content = str(cohere_text)

        return content or ""

    def _extract_usage(self, response) -> Optional[Usage]:
        """Extract usage information from OCI response."""
        try:
            usage_data = response.data.chat_response.usage
            return Usage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens
            )
        except AttributeError:
            return None
