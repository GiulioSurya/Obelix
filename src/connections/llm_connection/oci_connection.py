# src/connections/llm_connection/oci_connection.py
import os
import threading
from typing import Optional, Union

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

DEFAULT_OCI_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oci", "config")


class OCIConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to OCI Generative AI.

    The connection is shared among all OCI providers using the same credentials.
    Client is lazy initialized on first access.

    Args:
        oci_config: OCI configuration. Can be:
            - None: loads from default path (~/.oci/config)
            - str: path to OCI config file
            - dict: configuration dictionary with keys: user, fingerprint, key_content/key_file, tenancy, region
    """

    _instance: Optional['OCIConnection'] = None
    _lock = threading.Lock()
    _client = None
    _client_lock = threading.Lock()
    _initialized = False

    def __new__(cls, oci_config: Optional[Union[dict, str]] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, oci_config: Optional[Union[dict, str]] = None):
        if self._initialized:
            return
        self._oci_config = self._resolve_config(oci_config)
        self._initialized = True

    def _resolve_config(self, oci_config: Optional[Union[dict, str]]) -> dict:
        """
        Resolves the OCI configuration from various input types.

        Args:
            oci_config: Config dict, file path, or None for default

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If config cannot be resolved
        """
        if isinstance(oci_config, dict):
            return oci_config

        config_path = oci_config if isinstance(oci_config, str) else DEFAULT_OCI_CONFIG_PATH

        if not os.path.exists(config_path):
            raise ValueError(
                f"OCI config file not found at '{config_path}'. "
                "Please provide either:\n"
                "  - A configuration dictionary with keys: user, fingerprint, key_content/key_file, tenancy, region\n"
                "  - A valid path to an OCI config file\n"
                f"  - Or create a config file at the default location: {DEFAULT_OCI_CONFIG_PATH}"
            )

        try:
            from oci.config import from_file
        except ImportError:
            raise ImportError("oci is not installed. Install with: pip install oci")

        return from_file(config_path, "DEFAULT")

    def get_client(self):
        """
        Returns the configured OCI Generative AI client.
        Lazy initialization: creates client only on first access.

        Returns:
            Configured GenerativeAiInferenceClient

        Raises:
            ValueError: If OCI credentials are not configured
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self):
        """
        Creates and configures the OCI Generative AI client.

        Returns:
            Configured GenerativeAiInferenceClient

        Raises:
            ImportError: If oci library is not installed
        """
        try:
            from oci.generative_ai_inference import GenerativeAiInferenceClient
        except ImportError:
            raise ImportError("oci is not installed. Install with: pip install oci")

        return GenerativeAiInferenceClient(self._oci_config)