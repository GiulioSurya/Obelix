# src/client_adapters/ibm_provider.py
import asyncio
from typing import List, Dict, Any, Optional

from src.client_adapters.llm_abstraction import AbstractLLMProvider
from src.client_adapters._legacy_mapping_mixin import LegacyMappingMixin
from src.obelix_types.assistant_message import AssistantMessage
from src.obelix_types.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import IBMConnection
from src.logging_config import get_logger

try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
except ImportError:
    raise ImportError(
        "ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai"
    )

# Logger for IBM Watson X provider
logger = get_logger(__name__)

class IBMWatsonXLLm(LegacyMappingMixin, AbstractLLMProvider):
    """Provider for IBM Watson X with configurable parameters"""

    @property
    def provider_type(self) -> Providers:
        return Providers.IBM_WATSON

    def __init__(self,
                 connection: Optional[IBMConnection] = None,
                 model_id: str = "meta-llama/llama-3-3-70b-instruct",
                 max_tokens: int = 3000,
                 temperature: float = 0.3,
                 top_p: Optional[float] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 logprobs: Optional[bool] = None,
                 top_logprobs: Optional[int] = None,
                 n: Optional[int] = None,
                 logit_bias: Optional[Dict[int, float]] = None):
        """
        Initialize the IBM Watson X provider with dependency injection of connection

        Args:
            connection: IBMConnection singleton (default: None, reuse from GlobalConfig if provider matches)
            model_id: Model ID (default: "meta-llama/llama-3-3-70b-instruct")
            max_tokens: Maximum number of tokens (default: 3000)
            temperature: Sampling temperature (default: 0.3)
            top_p: Top-p sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            frequency_penalty: Token frequency penalty (default: None)
            presence_penalty: Token presence penalty (default: None)
            logprobs: Return log probabilities (default: None)
            top_logprobs: Number of top log probabilities (default: None)
            n: Number of completions to generate (default: None)
            logit_bias: Bias for specific tokens (default: None)

        Raises:
            ValueError: If connection=None and GlobalConfig does not have IBM_WATSON set
        """
        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.IBM_WATSON,
                "IBMWatsonXLLm"
            )

        self.connection = connection

        # Save model_id
        self.model_id = model_id

        # Build parameters
        params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if top_p is not None:
            params_dict["top_p"] = top_p
        if seed is not None:
            params_dict["seed"] = seed
        if stop is not None:
            params_dict["stop"] = stop
        if frequency_penalty is not None:
            params_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params_dict["presence_penalty"] = presence_penalty
        if logprobs is not None:
            params_dict["logprobs"] = logprobs
        if top_logprobs is not None:
            params_dict["top_logprobs"] = top_logprobs
        if n is not None:
            params_dict["n"] = n
        if logit_bias is not None:
            params_dict["logit_bias"] = logit_bias

        # Create ModelInference using credentials from connection
        credentials = self.connection.get_client()
        self.client = ModelInference(
            model_id=model_id,
            params=TextChatParameters(**params_dict),
            credentials=credentials,
            project_id=self.connection.get_project_id()
        )

    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the IBM Watson model with standardized obelix_types and tools (async).

        Uses asyncio.to_thread() to run the sync IBM SDK client without
        blocking the event loop.
        """
        logger.debug(f"IBM Watson invoke: model={self.model_id}, obelix_types={len(messages)}, tools={len(tools)}")

        # 1. Convert obelix_types and tools to IBM format (use base class methods)
        ibm_messages = self._convert_messages_to_provider_format(messages)
        ibm_tools = self._convert_tools_to_provider_format(tools)

        # 2. Call IBM Watson via thread pool to avoid blocking event loop
        # NOTE: tool_choice_option should be passed ONLY if tools are defined
        try:
            if ibm_tools:
                response = await asyncio.to_thread(
                    self.client.chat,
                    messages=ibm_messages,
                    tools=ibm_tools,
                    tool_choice_option="auto"
                )
            else:
                response = await asyncio.to_thread(
                    self.client.chat,
                    messages=ibm_messages
                )

            logger.info(f"IBM Watson chat completed: {self.model_id}")

            # Log usage if available
            usage = response.get("usage", {})
            if usage:
                logger.debug(f"IBM Watson tokens: input={usage.get('prompt_tokens')}, output={usage.get('completion_tokens')}, total={usage.get('total_tokens')}")

        except Exception as e:
            logger.error(f"IBM Watson request failed: {e}")
            raise

        # 3. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert IBM Watson response to standardized AssistantMessage
        """
        # Extract tool_calls using centralized method
        tool_calls = self._extract_tool_calls(response)

        # Extract text content
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        logger.debug(f"IBM Watson response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
