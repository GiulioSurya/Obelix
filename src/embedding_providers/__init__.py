# src/embedding_providers/__init__.py
"""
Modulo per provider di embedding semantici.

Exports:
    AbstractEmbeddingProvider: Classe base astratta per embedding providers
    OCIEmbeddingProvider: Implementazione per OCI Cohere Embed v4
"""

from src.embedding_providers.abstract_embedding_provider import (
    AbstractEmbeddingProvider,
    SingletonEmbeddingMeta
)
from src.embedding_providers.oci_embedding_provider import OCIEmbeddingProvider

__all__ = [
    "AbstractEmbeddingProvider",
    "SingletonEmbeddingMeta",
    "OCIEmbeddingProvider"
]