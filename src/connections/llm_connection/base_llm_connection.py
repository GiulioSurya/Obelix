# src/connections/llm_connection/base_llm_connection.py
from abc import ABC, abstractmethod
from typing import Any


class AbstractLLMConnection(ABC):
    """
    Base interface for singleton connections to LLM providers.

    Each provider implements its own connection that manages:
    - Lazy client initialization
    - Thread-safety via singleton pattern
    - Credential validation
    """

    @abstractmethod
    def get_client(self) -> Any:
        """
        Returns the configured client for the LLM provider.
        Lazy initialization: client is created on first access.

        Returns:
            Provider SDK client (type depends on implementation)

        Raises:
            ValueError: If credentials are missing or invalid
        """
        pass