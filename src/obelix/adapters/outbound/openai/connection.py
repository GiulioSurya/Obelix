# src/connections/llm_connection/openai_connection.py
import threading
from typing import Optional

from obelix.ports.outbound.llm_connection import AbstractLLMConnection


class OpenAIConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to OpenAI or OpenAI-compatible APIs.

    The connection is shared among all OpenAI providers using the same API key.
    Client is lazy initialized on first access.

    Supports custom base_url for OpenAI-compatible providers:
    - Anthropic: https://api.anthropic.com/v1/
    - Azure OpenAI: https://<resource>.openai.azure.com/
    - Local LLMs: http://localhost:8000/v1/
    """

    _instance: Optional["OpenAIConnection"] = None
    _lock = threading.Lock()
    _client = None
    _client_lock = threading.Lock()
    _initialized = False

    def __new__(cls, api_key: str, base_url: str | None = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: str, base_url: str | None = None):
        if self._initialized:
            return
        self._api_key = api_key
        self._base_url = base_url
        self._initialized = True

    def get_client(self):
        """
        Returns the configured async OpenAI client.
        Lazy initialization: creates client only on first access.

        Returns:
            Configured AsyncOpenAI client

        Raises:
            ValueError: If OPENAI_API_KEY is not configured
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self):
        """
        Creates and configures the async OpenAI client.

        Returns:
            Configured AsyncOpenAI client

        Raises:
            ImportError: If openai library is not installed
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Install with: pip install openai"
            )

        if self._base_url:
            return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return AsyncOpenAI(api_key=self._api_key)
