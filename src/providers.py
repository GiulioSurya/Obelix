from enum import Enum
from src.llm_providers.llm_abstraction import AbstractLLMProvider


class Providers(Enum):
    IBM_WATSON = "ibm_watson"
    OCI_GENERATIVE_AI = "oci"

    def create_instance(self) -> AbstractLLMProvider:
        """Factory method per creare l'istanza del provider"""
        if self == Providers.IBM_WATSON:
            from src.llm_providers.ibm_provider import IBMWatsonXLLm
            return IBMWatsonXLLm()
        elif self == Providers.OCI_GENERATIVE_AI:
            from src.llm_providers.oci_provider import OCILLm
            return OCILLm()
        raise ValueError(f"Provider {self} non supportato")


class ProviderRegistry:
    _mappings = {}

    @classmethod
    def register(cls, provider: Providers, mapping):
        cls._mappings[provider.value] = mapping

    @classmethod
    def get_mapping(cls, provider: Providers):
        return cls._mappings[provider.value]