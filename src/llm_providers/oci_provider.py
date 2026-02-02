# src/llm_providers/oci_provider.py
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
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import OCIConnection
from src.connections.llm_connection.oci_connection import (
    OCIRateLimitError,
    OCIServerError,
    OCIIncorrectStateError,
)
from src.logging_config import get_logger

# Import strategies
from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.llm_providers.oci_strategies.generic_strategy import GenericRequestStrategy
from src.llm_providers.oci_strategies.cohere_strategy import CohereRequestStrategy

# Logger per trace delle chiamate OCI
logger = get_logger(__name__)

class OCILLm(AbstractLLMProvider):
    """Provider for OCI Generative AI with configurable parameters"""

    # Available strategies
    _STRATEGIES = [
        GenericRequestStrategy(),
        CohereRequestStrategy()
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
        Initialize the OCI provider with dependency injection of connection

        Args:
            connection: OCIConnection singleton (default: None, reuse from GlobalConfig if provider matches)
            model_id: OCI model ID (default: "meta.llama-3.3-70b-instruct")
            max_tokens: Maximum number of tokens (default: 2000)
            temperature: Sampling temperature (default: 0.1)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            frequency_penalty: Frequency penalty (default: None)
            presence_penalty: Presence penalty (default: None)
            stop_sequences: Stop sequences (default: None)
            is_stream: Enable streaming (default: False)
            strategy: Specific strategy to use (default: auto-detect from model_id)
            **strategy_kwargs: Strategy-specific parameters
                Generic: reasoning_effort, verbosity, num_generations, log_probs, etc.
                Cohere: preamble_override, safety_mode, documents, citation_quality, etc.

        Raises:
            ValueError: If connection=None and GlobalConfig does not have OCI_GENERATIVE_AI set
        """
        # Configure OCI SDK logging if enabled in infrastructure.yaml


        if logger is True:
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger('oci').setLevel(logging.DEBUG)
            oci.base_client.is_http_log_enabled(True)
        else:
            # Disable OCI SDK logging if not requested
            logging.getLogger('oci').setLevel(logging.WARNING)
            oci.base_client.is_http_log_enabled(False)

        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OCI_GENERATIVE_AI,
                "OCILLm"
            )

        self.connection = connection

        # Save parameters for use in calls
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
        """
        Auto-detect the appropriate strategy based on model_id prefix.

        Args:
            model_id: The OCI model identifier (e.g., "meta.llama-3.3-70b-instruct")

        Returns:
            OCIRequestStrategy: The appropriate strategy for the model

        Raises:
            ValueError: If no strategy supports the model_id prefix
        """
        for strategy in cls._STRATEGIES:
            for prefix in strategy.get_supported_model_prefixes():
                if model_id.startswith(prefix):
                    return strategy

        raise ValueError(
            f"No strategy found for model_id '{model_id}'. "
            f"Supported prefixes: {[p for s in cls._STRATEGIES for p in s.get_supported_model_prefixes()]}"
        )

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
        Call the OCI model with standardized messages and tools (async).

        Uses native async HTTP client (httpx) for non-blocking I/O.
        Uses the appropriate strategy (auto-detected or specified) for the entire flow.
        Retries automatically on transient errors (rate limit, server errors, timeout, connection errors).
        """
        # 1. The strategy converts messages and tools to provider-specific format
        converted_messages = self.strategy.convert_messages(messages)
        converted_tools = self.strategy.convert_tools(tools)

        logger.debug(f"OCI invoke: model={self.model_id}, messages={len(converted_messages)}, tools={len(converted_tools)}")

        # 2. The strategy builds the specific request (Generic or Cohere)
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

        # 3. Call OCI using async client from connection (native async, no thread blocking)
        from src.k8s_config import YamlConfig
        import os
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_config = infra_config.get("llm_providers.oci")

        client = self.connection.get_client()
        chat_details = ChatDetails(
            compartment_id=oci_config["compartment_id"],
            serving_mode=OnDemandServingMode(model_id=self.model_id),
            chat_request=chat_request
        )

        # Native async call - no thread blocking
        response = await client.chat(chat_details)
        logger.info(f"OCI chat completed: {response.data.model_id}")
        logger.debug(f"OCI response total tokens: {response.data.chat_response.usage.total_tokens}")

        # 4. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert OCI response to standardized AssistantMessage.
        Uses the current strategy to extract tool_calls in the correct format.
        """
        # Use the strategy's mapping to extract tool_calls (Generic or Cohere)
        mapping = self.strategy.get_mapping()

        tool_calls = mapping["tool_output"]["tool_calls"](response)

        # Extract text content
        # GENERIC: content in choices[0].message.content
        # COHERE: content in chat_response.text
        content = ""

        # Try Generic format first
        chat_response = response.data.chat_response if response.data else None
        choices = chat_response.choices if chat_response else None

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

        # Try Cohere format if no content yet
        if not content and chat_response:
            cohere_text = chat_response.text
            if cohere_text:
                content = str(cohere_text)

        # Ensure content is never None
        if content is None:
            content = ""

        # Extract usage from OCI response
        usage = None
        try:
            usage_data = response.data.chat_response.usage
            usage = Usage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens
            )
        except AttributeError:
            # If unable to extract usage, continue without it (usage remains None)
            pass

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )
