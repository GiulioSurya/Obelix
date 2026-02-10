from enum import Enum
from src.client_adapters.llm_abstraction import AbstractLLMProvider


class Providers(Enum):
    ANTHROPIC = "anthropic"
    IBM_WATSON = "ibm_watson"
    OCI_GENERATIVE_AI = "oci"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"

    def create_instance(self) -> AbstractLLMProvider:
        """Factory method to create provider instance"""
        if self == Providers.ANTHROPIC:
            from src.client_adapters.anthropic_provider import AnthropicProvider
            return AnthropicProvider()
        elif self == Providers.IBM_WATSON:
            from src.client_adapters.ibm_provider import IBMWatsonXLLm
            return IBMWatsonXLLm()
        elif self == Providers.OCI_GENERATIVE_AI:
            from src.client_adapters.oci_provider import OCILLm
            return OCILLm(is_parallel_tool_calls=False)
        elif self == Providers.OLLAMA:
            from src.client_adapters.ollama_provider import OllamaProvider
            return OllamaProvider()
        elif self == Providers.VLLM:
            from src.client_adapters.vllm_provider import VLLMProvider
            return VLLMProvider()
        elif self == Providers.OPENAI:
            from src.client_adapters.openai_provider import OpenAIProvider
            return OpenAIProvider()
        raise ValueError(f"Provider {self} not supported")


class ProviderRegistry:
    _mappings = {}

    @classmethod
    def register(cls, provider: Providers, mapping):
        cls._mappings[provider.value] = mapping

    @classmethod
    def get_mapping(cls, provider: Providers):
        return cls._mappings[provider.value]