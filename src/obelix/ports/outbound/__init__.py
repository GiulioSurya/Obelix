"""
Outbound ports - Abstract interfaces for external dependencies.
"""

from obelix.ports.outbound.llm_provider import AbstractLLMProvider
from obelix.ports.outbound.llm_connection import AbstractLLMConnection
from obelix.ports.outbound.embedding_provider import AbstractEmbeddingProvider

__all__ = [
    "AbstractLLMProvider",
    "AbstractLLMConnection",
    "AbstractEmbeddingProvider",
]
