from typing import Dict, Any
from src.providers import Providers


class GlobalConfig:
    """
    Singleton per gestire configurazione globale LLM providers.

    Gestisce:
    - Provider corrente
    - Connection singleton per ogni provider (lazy init)
    - Factory per creare provider instances con parametri custom
    """
    _instance = None
    _current_provider = None
    _connections: Dict[Providers, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._connections = {}  # Initialize connections dict
        return cls._instance

    def set_provider(self, provider: Providers):
        """
        Imposta il provider corrente.

        Args:
            provider: Provider da usare (OCI, IBM, Anthropic, etc.)
        """
        self._current_provider = provider

    def get_current_provider(self) -> Providers:
        """
        Ritorna il provider corrente.

        Returns:
            Provider enum corrente

        Raises:
            ValueError: Se provider non impostato
        """
        if self._current_provider is None:
            raise ValueError("Provider non impostato. Usa GlobalConfig().set_provider()")
        return self._current_provider

    def get_current_provider_instance(self, **kwargs):
        """
        Crea un'istanza del provider corrente con parametri personalizzabili.

        Usa connection singleton appropriate (lazy init).
        Permette di personalizzare model_id, temperature, max_tokens, etc.

        Args:
            **kwargs: Parametri specifici del provider (model_id, temperature, etc.)

        Returns:
            Istanza del provider configurato

        Raises:
            ValueError: Se provider non impostato o non supportato

        Examples:
            >>> config = GlobalConfig()
            >>> config.set_provider(Providers.OCI_GENERATIVE_AI)
            >>> provider = config.get_current_provider_instance(model_id="meta.llama-3.3-70b-instruct", temperature=0.5)
        """
        if self._current_provider is None:
            raise ValueError("Provider non impostato. Usa GlobalConfig().set_provider()")

        # Lazy init della connection (singleton)
        if self._current_provider not in self._connections:
            self._connections[self._current_provider] = self._create_connection(self._current_provider)

        connection = self._connections[self._current_provider]

        # Crea provider con connection + kwargs personalizzati
        return self._create_provider_instance(self._current_provider, connection, **kwargs)

    def _create_connection(self, provider: Providers):
        """
        Crea la connection singleton appropriata per il provider.

        Args:
            provider: Provider enum

        Returns:
            Connection singleton (OCIConnection, IBMConnection, etc.)
        """
        if provider == Providers.OCI_GENERATIVE_AI:
            from src.connections.llm_connection import OCIConnection
            return OCIConnection()
        elif provider == Providers.IBM_WATSON:
            from src.connections.llm_connection import IBMConnection
            return IBMConnection()
        elif provider == Providers.ANTHROPIC:
            from src.connections.llm_connection import AnthropicConnection
            return AnthropicConnection()
        elif provider == Providers.OLLAMA:
            # Ollama non ha connection singleton (usa client locale)
            return None
        elif provider == Providers.VLLM:
            # vLLM non ha connection singleton (carica modello in-process)
            return None
        else:
            raise ValueError(f"Provider {provider} non supportato")

    def _create_provider_instance(self, provider: Providers, connection, **kwargs):
        """
        Crea istanza del provider con connection e parametri custom.

        Args:
            provider: Provider enum
            connection: Connection singleton (o None)
            **kwargs: Parametri custom per il provider

        Returns:
            Istanza del provider
        """
        if provider == Providers.OCI_GENERATIVE_AI:
            from src.llm_providers.oci_provider import OCILLm
            return OCILLm(connection, **kwargs)
        elif provider == Providers.IBM_WATSON:
            from src.llm_providers.ibm_provider import IBMWatsonXLLm
            return IBMWatsonXLLm(connection, **kwargs)
        elif provider == Providers.ANTHROPIC:
            from src.llm_providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(connection, **kwargs)
        elif provider == Providers.OLLAMA:
            from src.llm_providers.ollama_provider import OllamaProvider
            return OllamaProvider(**kwargs)
        elif provider == Providers.VLLM:
            from src.llm_providers.vllm_provider import VLLMProvider
            return VLLMProvider(**kwargs)
        else:
            raise ValueError(f"Provider {provider} non supportato")


