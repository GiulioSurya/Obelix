# src/connections/llm_connection/base_llm_connection.py
from abc import ABC, abstractmethod
from typing import Any


class AbstractLLMConnection(ABC):
    """
    Interfaccia base per connessioni singleton ai provider LLM.

    Ogni provider implementa la propria connection che gestisce:
    - Inizializzazione lazy del client
    - Thread-safety tramite singleton pattern
    - Validazione credenziali
    """

    @abstractmethod
    def get_client(self) -> Any:
        """
        Ritorna il client configurato per il provider LLM.
        Lazy initialization: il client viene creato al primo accesso.

        Returns:
            Client SDK del provider (tipo dipende dall'implementazione)

        Raises:
            ValueError: Se credenziali mancanti o invalide
        """
        pass