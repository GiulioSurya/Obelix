# src/connections/llm_connection/ibm_connection.py
import threading
from typing import Optional

from src.ports.outbound.llm_connection import AbstractLLMConnection


class IBMConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to IBM watsonx.ai.

    The connection is shared among all IBM providers using the same credentials.
    Credentials are validated on first access.

    Args:
        api_key: IBM Cloud API key (find at https://cloud.ibm.com/iam/apikeys)
        project_id: watsonx.ai project ID (find in project settings on watsonx.ai)
        url: Regional endpoint URL. Available regions:
            - https://us-south.ml.cloud.ibm.com
            - https://eu-de.ml.cloud.ibm.com
            - https://eu-gb.ml.cloud.ibm.com
            - https://jp-tok.ml.cloud.ibm.com
    """

    _instance: Optional['IBMConnection'] = None
    _lock = threading.Lock()
    _credentials = None
    _credentials_lock = threading.Lock()
    _initialized = False

    def __new__(cls, api_key: str, project_id: str, url: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: str, project_id: str, url: str):
        if self._initialized:
            return
        self._api_key = api_key
        self._project_id = project_id
        self._url = url
        self._initialized = True

    def get_client(self):
        """
        Returns the configured IBM watsonx.ai credentials.

        NOTE: IBM watsonx.ai does not have a true singleton client.
        Each ModelInference is created with a specific model_id,
        but shares the same credentials.

        Returns:
            IBM watsonx.ai Credentials
        """
        if self._credentials is None:
            with self._credentials_lock:
                if self._credentials is None:
                    self._credentials = self._create_credentials()
        return self._credentials

    def _create_credentials(self):
        """
        Creates IBM watsonx.ai credentials.

        Returns:
            Configured IBM watsonx.ai Credentials

        Raises:
            ImportError: If ibm-watsonx-ai library is not installed
        """
        try:
            from ibm_watsonx_ai import Credentials
        except ImportError:
            raise ImportError("ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai")

        return Credentials(
            url=self._url,
            api_key=self._api_key,
        )

    def get_project_id(self) -> str:
        """
        Returns the IBM watsonx.ai project_id.

        Returns:
            Configured Project ID
        """
        return self._project_id