# src/connections/llm_connection/ibm_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


class IBMConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to IBM Watson X.

    The connection is shared among all IBM providers using the same credentials.
    Credentials are validated on first access.
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
        Returns the configured IBM Watson X credentials.

        NOTE: IBM Watson X does not have a true singleton client.
        Each ModelInference is created with a specific model_id,
        but shares the same credentials.

        Returns:
            IBM Watson X Credentials

        Raises:
            ValueError: If IBM credentials are not configured
        """
        if self._credentials is None:
            with self._credentials_lock:
                if self._credentials is None:
                    self._credentials = self._create_credentials()
        return self._credentials

    def _create_credentials(self):
        """
        Creates and validates IBM Watson X credentials.

        Returns:
            Configured IBM Watson X Credentials

        Raises:
            ValueError: If credentials are missing
            ImportError: If ibm-watsonx-ai library is not installed
        """
        # Import here to avoid circular dependencies
        try:
            from ibm_watsonx_ai import Credentials
        except ImportError:
            raise ImportError(
                "ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai"
            )

        # Validate credentials
        api_key = os.getenv("IBM_WATSONX_API_KEY")
        project_id = os.getenv("IBM_WATSONX_PROJECT_ID")

        if not api_key or not project_id:
            raise ValueError(
                "IBM Watson X credentials missing. "
                "Required: IBM_WATSONX_API_KEY, IBM_WATSONX_PROJECT_ID"
            )

        return Credentials(
            url="https://eu-de.ml.cloud.ibm.com",
            api_key=api_key,
        )

    def get_project_id(self) -> str:
        """
        Returns the IBM Watson X project_id.

        Returns:
            Configured Project ID

        Raises:
            ValueError: If project_id is not configured
        """
        project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
        if not project_id:
            raise ValueError("IBM_WATSONX_PROJECT_ID not configured")
        return project_id