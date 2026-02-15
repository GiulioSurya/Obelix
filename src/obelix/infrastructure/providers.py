from enum import Enum


class Providers(Enum):
    ANTHROPIC = "anthropic"
    IBM_WATSON = "ibm_watson"
    OCI_GENERATIVE_AI = "oci"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"