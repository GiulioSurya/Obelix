# src/connections/llm_connection/anthropic_connection.py
import threading
from typing import Optional

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection


class AnthropicConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to Anthropic Claude.

    The connection is shared among all Anthropic providers using the same API key.
    Client is lazy initialized on first access.
    """

    _instance: Optional['AnthropicConnection'] = None
    _lock = threading.Lock()
    _client = None
    _client_lock = threading.Lock()
    _initialized = False

    def __new__(cls, api_key: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: str):
        if self._initialized:
            return
        self._api_key = api_key
        self._initialized = True

    def get_client(self):
        """
        Returns the configured async Anthropic client.
        Lazy initialization: creates client only on first access.

        Returns:
            Configured AsyncAnthropic client

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self):
        """
        Creates and configures the async Anthropic client.

        Returns:
            Configured AsyncAnthropic client

        Raises:
            ImportError: If anthropic library is not installed
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic is not installed. Install with: pip install anthropic"
            )

        return AsyncAnthropic(api_key=self._api_key)