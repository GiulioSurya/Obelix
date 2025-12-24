# src/connections/llm_connection/anthropic_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


class AnthropicConnection(AbstractLLMConnection):
    """
    Singleton thread-safe per gestire la connessione a Anthropic Claude.

    La connection è condivisa tra tutti i provider Anthropic che usano la stessa API key.
    Il client viene inizializzato lazy al primo accesso.
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
        Ritorna il client Anthropic configurato.
        Lazy initialization: crea il client solo al primo accesso.

        Returns:
            Anthropic client configurato

        Raises:
            ValueError: Se ANTHROPIC_API_KEY non è configurata
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self):
        """
        Crea e configura il client Anthropic.

        Returns:
            Anthropic client configurato

        Raises:
            ValueError: Se API key mancante
            ImportError: Se libreria anthropic non installata
        """
        # Import qui per evitare dipendenze circolari
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic is not installed. Install with: pip install anthropic"
            )

        # Validazione API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY non configurata. "
                "Imposta la variabile d'ambiente ANTHROPIC_API_KEY"
            )

        return Anthropic(api_key=api_key)