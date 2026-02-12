"""
Outbound ports - Abstract interfaces for external dependencies.
"""

from src.ports.outbound.llm_provider import AbstractLLMProvider
from src.ports.outbound.llm_connection import AbstractLLMConnection
from src.ports.outbound.embedding_provider import AbstractEmbeddingProvider

__all__ = [
    "AbstractLLMProvider",
    "AbstractLLMConnection",
    "AbstractEmbeddingProvider",
]