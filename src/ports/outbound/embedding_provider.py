# src/embedding_providers/abstract_embedding_provider.py
from abc import ABC, abstractmethod, ABCMeta
from typing import List, Union
import numpy as np
import threading


class SingletonEmbeddingMeta(ABCMeta):
    """
    Thread-safe singleton metaclass to ensure a single instance
    of each Embedding Provider class in the system.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonEmbeddingMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractEmbeddingProvider(ABC, metaclass=SingletonEmbeddingMeta):
    """
    Abstract base class for embedding providers.

    Each provider implements its own semantic encoding logic
    while maintaining a uniform interface to generate numerical vectors
    from natural language text.

    The only mandatory public interface is embed().
    """

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generates embeddings for one or more texts.

        Args:
            texts: Single text (str) or list of texts (List[str])

        Returns:
            - If input is str: np.ndarray with shape (embedding_dim,)
            - If input is List[str]: List[np.ndarray] with len = len(texts)

        Raises:
            ValueError: If batch size exceeds provider limits
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Returns the dimensionality of generated embeddings.

        Returns:
            int: Number of dimensions in the embedding vector
        """
        pass