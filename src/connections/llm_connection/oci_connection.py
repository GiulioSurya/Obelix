# src/connections/llm_connection/oci_connection.py
import os
import threading
from typing import Optional
from dotenv import load_dotenv

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection

load_dotenv()


class OCIConnection(AbstractLLMConnection):
    """
    Singleton thread-safe per gestire la connessione a OCI Generative AI.

    La connection è condivisa tra tutti i provider OCI che usano le stesse credenziali.
    Il client viene inizializzato lazy al primo accesso.
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
        Ritorna il client OCI Generative AI configurato.
        Lazy initialization: crea il client solo al primo accesso.

        Returns:
            GenerativeAiInferenceClient configurato

        Raises:
            ValueError: Se le credenziali OCI non sono configurate
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self):
        """
        Crea e configura il client OCI Generative AI.

        Returns:
            GenerativeAiInferenceClient configurato

        Raises:
            ValueError: Se le credenziali mancanti
            ImportError: Se libreria oci non installata
        """
        # Import qui per evitare dipendenze circolari
        try:
            from oci.generative_ai_inference import GenerativeAiInferenceClient
        except ImportError:
            raise ImportError(
                "oci is not installed. Install with: pip install oci"
            )

        from src.k8s_config import YamlConfig
        import os

        # Leggi configurazione OCI completa da infrastructure.yaml (include private_key_content)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_provider_config = infra_config.get("llm_providers.oci")

        # Validazione presenza chiave privata
        if not oci_provider_config.get("private_key_content"):
            raise ValueError(
                "Credenziale private_key_content mancante in infrastructure.yaml. "
                "Questa chiave deve essere configurata nel ConfigMap o Secrets di Kubernetes."
            )

        # Configurazione OCI
        oci_config = {
            'user': oci_provider_config["user_id"],
            'fingerprint': oci_provider_config["fingerprint"],
            'key_content': oci_provider_config["private_key_content"],
            'tenancy': oci_provider_config["tenancy"],
            'region': oci_provider_config["region"]
        }

        return GenerativeAiInferenceClient(oci_config)