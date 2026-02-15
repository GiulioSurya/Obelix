from enum import Enum

from obelix.ports.outbound.llm_provider import AbstractLLMProvider


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
            from obelix.adapters.outbound.anthropic.provider import AnthropicProvider

            return AnthropicProvider()
        elif self == Providers.IBM_WATSON:
            from obelix.adapters.outbound.ibm.provider import IBMWatsonXLLm

            return IBMWatsonXLLm()
        elif self == Providers.OCI_GENERATIVE_AI:
            from obelix.adapters.outbound.oci.provider import OCILLm

            return OCILLm(is_parallel_tool_calls=False)
        elif self == Providers.OLLAMA:
            from obelix.adapters.outbound.ollama.provider import OllamaProvider

            return OllamaProvider()
        elif self == Providers.VLLM:
            from obelix.adapters.outbound.vllm.provider import VLLMProvider

            return VLLMProvider()
        elif self == Providers.OPENAI:
            from obelix.adapters.outbound.openai.provider import OpenAIProvider

            return OpenAIProvider()
        raise ValueError(f"Provider {self} not supported")
