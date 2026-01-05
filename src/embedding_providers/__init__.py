# src/embedding_providers/__init__.py
"""
Module for semantic embedding providers.

Exports:
    AbstractEmbeddingProvider: Abstract base class for embedding providers
    OCIEmbeddingProvider: Implementation for OCI Cohere Embed v4
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