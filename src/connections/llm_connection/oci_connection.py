# src/connections/llm_connection/oci_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


class OCIConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to OCI Generative AI.

    The connection is shared among all OCI providers using the same credentials.
    Client is lazy initialized on first access.
    """

    _instance: Optional['OCIConnection'] = None
    _lock = threading.Lock()
    _client = None
    _client_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

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
            ValueError: If credentials are missing
            ImportError: If oci library is not installed
        """
        # Import here to avoid circular dependencies
        try:
            from oci.generative_ai_inference import GenerativeAiInferenceClient
        except ImportError:
            raise ImportError(
                "oci is not installed. Install with: pip install oci"
            )

        from src.k8s_config import YamlConfig
        import os

        # Read complete OCI configuration from infrastructure.yaml (includes private_key_content)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_provider_config = infra_config.get("llm_providers.oci")

        # Validate presence of private key
        if not oci_provider_config.get("private_key_content"):
            raise ValueError(
                "Credential private_key_content missing in infrastructure.yaml. "
                "This key must be configured in the Kubernetes ConfigMap or Secrets."
            )

        # OCI configuration
        oci_config = {
            'user': oci_provider_config["user_id"],
            'fingerprint': oci_provider_config["fingerprint"],
            'key_content': oci_provider_config["private_key_content"],
            'tenancy': oci_provider_config["tenancy"],
            'region': oci_provider_config["region"]
        }

        return GenerativeAiInferenceClient(oci_config)