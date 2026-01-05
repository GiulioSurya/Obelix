from typing import Dict, Any
from src.providers import Providers


class GlobalConfig:
    """
    Singleton to manage global LLM provider configuration.

    Manages:
    - Current provider
    - Singleton connection for each provider (lazy init)
    - Factory to create provider instances with custom parameters
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
        Set the current provider.

        Args:
            provider: Provider to use (OCI, IBM, Anthropic, etc.)
        """
        self._current_provider = provider

    def get_current_provider(self) -> Providers:
        """
        Return the current provider.

        Returns:
            Current provider enum

        Raises:
            ValueError: If provider is not set
        """
        if self._current_provider is None:
            raise ValueError("Provider not set. Use GlobalConfig().set_provider()")
        return self._current_provider

    def get_current_provider_instance(self, **kwargs):
        """
        Create an instance of the current provider with customizable parameters.

        Uses appropriate singleton connections (lazy init).
        Allows customization of model_id, temperature, max_tokens, etc.

        Args:
            **kwargs: Provider-specific parameters (model_id, temperature, etc.)

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not set or not supported

        Examples:
            >>> config = GlobalConfig()
            >>> config.set_provider(Providers.OCI_GENERATIVE_AI)
            >>> provider = config.get_current_provider_instance(model_id="meta.llama-3.3-70b-instruct", temperature=0.5)
        """
        if self._current_provider is None:
            raise ValueError("Provider not set. Use GlobalConfig().set_provider()")

        # Lazy init of connection (singleton)
        if self._current_provider not in self._connections:
            self._connections[self._current_provider] = self._create_connection(self._current_provider)

        connection = self._connections[self._current_provider]

        # Create provider with connection + custom kwargs
        return self._create_provider_instance(self._current_provider, connection, **kwargs)

    def _create_connection(self, provider: Providers):
        """
        Create the appropriate singleton connection for the provider.

        Args:
            provider: Provider enum

        Returns:
            Singleton connection (OCIConnection, IBMConnection, etc.)
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
            # Ollama does not have singleton connection (uses local client)
            return None
        elif provider == Providers.VLLM:
            # vLLM does not have singleton connection (loads model in-process)
            return None
        else:
            raise ValueError(f"Provider {provider} not supported")

    def _create_provider_instance(self, provider: Providers, connection, **kwargs):
        """
        Create provider instance with connection and custom parameters.

        Args:
            provider: Provider enum
            connection: Singleton connection (or None)
            **kwargs: Custom parameters for the provider

        Returns:
            Provider instance
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
            raise ValueError(f"Provider {provider} not supported")


