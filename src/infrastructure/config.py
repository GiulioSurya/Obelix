from typing import Dict, Any, Optional
from src.infrastructure.providers import Providers
from src.ports.outbound.llm_connection import AbstractLLMConnection


class GlobalConfig:
    """
    Singleton to manage global LLM provider configuration.

    Manages:
    - Current provider
    - Singleton connection for each provider (passed explicitly)
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

    def set_provider(self, provider: Providers, connection: Optional[AbstractLLMConnection] = None):
        """
        Set the current provider and its connection.

        Args:
            provider: Provider to use (OCI, IBM, Anthropic, etc.)
            connection: Initialized connection for the provider.
                       If None, the connection must be passed before using the provider.
        """
        self._current_provider = provider
        if connection is not None:
            self._connections[provider] = connection

    def get_current_provider(self) -> Providers:
        """
        Return the current provider.

        Returns:
            Current provider enum

        Raises:
            ValueError: If provider is not set
        """
        if self._current_provider is None:
            raise ValueError(
                "Provider not set. Either use GlobalConfig().set_provider() "
                "or pass a provider directly to the agent constructor."
            )
        return self._current_provider

    def get_current_provider_instance(self, **kwargs):
        """
        Create an instance of the current provider with customizable parameters.

        Uses the connection passed to set_provider().
        Allows customization of model_id, temperature, max_tokens, etc.

        Args:
            **kwargs: Provider-specific parameters (model_id, temperature, etc.)

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not set or connection is not provided

        Examples:
            >>> config = GlobalConfig()
            >>> connection = OCIConnection(oci_config)
            >>> config.set_provider(Providers.OCI_GENERATIVE_AI, connection=connection)
            >>> provider = config.get_current_provider_instance(model_id="meta.llama-3.3-70b-instruct", temperature=0.5)
        """
        if self._current_provider is None:
            raise ValueError(
                "Provider not set. Either use GlobalConfig().set_provider() "
                "or pass a provider directly to the agent constructor."
            )

        if self._current_provider not in self._connections:
            raise ValueError(
                f"Connection for {self._current_provider.value} not initialized. "
                f"Pass the connection to set_provider()."
            )

        connection = self._connections[self._current_provider]

        # Create provider with connection + custom kwargs
        return self._create_provider_instance(self._current_provider, connection, **kwargs)

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
            from src.adapters.outbound.oci.provider import OCILLm
            return OCILLm(connection, **kwargs)
        elif provider == Providers.IBM_WATSON:
            from src.adapters.outbound.ibm.provider import IBMWatsonXLLm
            return IBMWatsonXLLm(connection, **kwargs)
        elif provider == Providers.ANTHROPIC:
            from src.adapters.outbound.anthropic.provider import AnthropicProvider
            return AnthropicProvider(connection, **kwargs)
        elif provider == Providers.OLLAMA:
            from src.adapters.outbound.ollama.provider import OllamaProvider
            return OllamaProvider(**kwargs)
        elif provider == Providers.VLLM:
            from src.adapters.outbound.vllm.provider import VLLMProvider
            return VLLMProvider(**kwargs)
        elif provider == Providers.OPENAI:
            from src.adapters.outbound.openai.provider import OpenAIProvider
            return OpenAIProvider(connection, **kwargs)
        else:
            raise ValueError(f"Provider {provider} not supported")


