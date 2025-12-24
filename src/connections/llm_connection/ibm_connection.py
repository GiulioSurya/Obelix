# src/connections/llm_connection/ibm_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


class IBMConnection(AbstractLLMConnection):
    """
    Singleton thread-safe per gestire la connessione a IBM Watson X.

    La connection è condivisa tra tutti i provider IBM che usano le stesse credenziali.
    Le credenziali vengono validate al primo accesso.
    """

    _instance: Optional['IBMConnection'] = None
    _lock = threading.Lock()
    _credentials = None
    _credentials_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self):
        """
        Ritorna le credenziali IBM Watson X configurate.

        NOTA: IBM Watson X non ha un client singleton vero e proprio.
        Ogni ModelInference viene creato con model_id specifico,
        ma condivide le stesse credenziali.

        Returns:
            Credentials IBM Watson X

        Raises:
            ValueError: Se le credenziali IBM non sono configurate
        """
        if self._credentials is None:
            with self._credentials_lock:
                if self._credentials is None:
                    self._credentials = self._create_credentials()
        return self._credentials

    def _create_credentials(self):
        """
        Crea e valida le credenziali IBM Watson X.

        Returns:
            Credentials IBM Watson X configurate

        Raises:
            ValueError: Se le credenziali mancanti
            ImportError: Se libreria ibm-watsonx-ai non installata
        """
        # Import qui per evitare dipendenze circolari
        try:
            from ibm_watsonx_ai import Credentials
        except ImportError:
            raise ImportError(
                "ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai"
            )

        # Validazione credenziali
        api_key = os.getenv("IBM_WATSONX_API_KEY")
        project_id = os.getenv("IBM_WATSONX_PROJECT_ID")

        if not api_key or not project_id:
            raise ValueError(
                "Credenziali IBM Watson X mancanti. "
                "Richieste: IBM_WATSONX_API_KEY, IBM_WATSONX_PROJECT_ID"
            )

        return Credentials(
            url="https://eu-de.ml.cloud.ibm.com",
            api_key=api_key,
        )

    def get_project_id(self) -> str:
        """
        Ritorna il project_id IBM Watson X.

        Returns:
            Project ID configurato

        Raises:
            ValueError: Se project_id non configurato
        """
        project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
        if not project_id:
            raise ValueError("IBM_WATSONX_PROJECT_ID non configurato")
        return project_id