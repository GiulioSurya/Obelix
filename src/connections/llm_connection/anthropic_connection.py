# src/connections/llm_connection/anthropic_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


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

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self):
        """
        Returns the configured Anthropic client.
        Lazy initialization: creates client only on first access.

        Returns:
            Configured Anthropic client

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
        Creates and configures the Anthropic client.

        Returns:
            Configured Anthropic client

        Raises:
            ValueError: If API key is missing
            ImportError: If anthropic library is not installed
        """
        # Import here to avoid circular dependencies
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic is not installed. Install with: pip install anthropic"
            )

        # Validate API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not configured. "
                "Set the environment variable ANTHROPIC_API_KEY"
            )

        return Anthropic(api_key=api_key)