# src/embedding_providers/abstract_embedding_provider.py
from abc import ABC, abstractmethod, ABCMeta
from typing import List, Union
import numpy as np
import threading


class SingletonEmbeddingMeta(ABCMeta):
    """
    Metaclasse singleton thread-safe per garantire una sola istanza
    di ogni classe Embedding Provider nel sistema.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Pattern di double-checked locking
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonEmbeddingMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractEmbeddingProvider(ABC, metaclass=SingletonEmbeddingMeta):
    """
    Classe base astratta per i provider di embedding.

    Ogni provider implementa la propria logica di encoding semantico
    mantenendo un'interfaccia uniforme per generare vettori numerici
    da testo in linguaggio naturale.

    L'unica interfaccia pubblica obbligatoria è embed().
    """

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Genera embeddings per uno o più testi.

        Args:
            texts: Singolo testo (str) o lista di testi (List[str])

        Returns:
            - Se input è str: np.ndarray con shape (embedding_dim,)
            - Se input è List[str]: List[np.ndarray] con len = len(texts)

        Raises:
            ValueError: Se il batch size eccede i limiti del provider
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Restituisce la dimensionalità degli embedding generati.

        Returns:
            int: Numero di dimensioni del vettore embedding
        """
        pass