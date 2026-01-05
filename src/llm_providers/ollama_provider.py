# src/llm_providers/ollama_provider.py
from typing import List, Optional

try:
    from ollama import Client
except ImportError:
    raise ImportError(
        "ollama is not installed. Install with: pip install ollama"
    )

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.logging_config import get_logger

# Logger for Ollama provider
logger = get_logger(__name__)


class OllamaProvider(AbstractLLMProvider):
    """Provider for Ollama with configurable parameters"""

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
        Initialize the Ollama provider

        Args:
            model_id: Ollama model ID (default: "a-kore/Arctic-Text2SQL-R1-7B")
            base_url: Base URL of Ollama server (default: None = http://localhost:11434)
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum number of tokens (default: None)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            keep_alive: Keep model in memory (default: None)
        """
        self.model_id = model_id

        # Initialize Ollama client
        if base_url:
            self.client = Client(host=base_url)
        else:
            self.client = Client()

        # Build options dict with only non-None parameters
        self.options = {}
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

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the Ollama model with standardized messages and tools
        """
        logger.debug(f"Ollama invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}")

        # 1. Convert messages and tools to Ollama format (use base class methods)
        ollama_messages = self._convert_messages_to_provider_format(messages)
        ollama_tools = self._convert_tools_to_provider_format(tools)

        # 2. Build call parameters
        chat_params = {
            "model": self.model_id,
            "messages": ollama_messages,
        }

        if ollama_tools:
            chat_params["tools"] = ollama_tools

        if self.options:
            chat_params["options"] = self.options

        if self.keep_alive is not None:
            chat_params["keep_alive"] = self.keep_alive

        # 3. Call Ollama
        try:
            response = self.client.chat(**chat_params)

            logger.info(f"Ollama chat completed: {self.model_id}")

            # Log usage if available
            if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
                logger.debug(f"Ollama tokens: prompt={response.prompt_eval_count}, completion={response.eval_count}")

        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise

        # 4. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert Ollama response to standardized AssistantMessage
        """
        # Extract tool_calls using centralized method
        tool_calls = self._extract_tool_calls(response)

        # Extract text content
        content = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content or ""

        logger.debug(f"Ollama response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
