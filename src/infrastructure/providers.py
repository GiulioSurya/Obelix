from enum import Enum
from src.ports.outbound.llm_provider import AbstractLLMProvider


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
            from src.adapters.outbound.anthropic.provider import AnthropicProvider
            return AnthropicProvider()
        elif self == Providers.IBM_WATSON:
            from src.adapters.outbound.ibm.provider import IBMWatsonXLLm
            return IBMWatsonXLLm()
        elif self == Providers.OCI_GENERATIVE_AI:
            from src.adapters.outbound.oci.provider import OCILLm
            return OCILLm(is_parallel_tool_calls=False)
        elif self == Providers.OLLAMA:
            from src.adapters.outbound.ollama.provider import OllamaProvider
            return OllamaProvider()
        elif self == Providers.VLLM:
            from src.adapters.outbound.vllm.provider import VLLMProvider
            return VLLMProvider()
        elif self == Providers.OPENAI:
            from src.adapters.outbound.openai.provider import OpenAIProvider
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