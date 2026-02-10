# src/client_adapters/oci_strategies/__init__.py
"""
OCI Request Strategy Module

This module implements the Strategy pattern for handling different OCI Generative AI API formats.

Strategies:
- GenericRequestStrategy: For Meta Llama, Google Gemini, xAI Grok, OpenAI models
- CohereRequestStrategy: For Cohere Command models

Usage:
    from src.client_adapters.oci_provider import OCILLm

    # Auto-detection based on model_id
    provider = OCILLm(model_id="meta.llama-3.3-70b-instruct")

    # Cohere model with specific parameters
    provider = OCILLm(
        model_id="cohere.command-r-plus",
        preamble_override="Custom system instructions",
        safety_mode="STRICT"
    )

    # Llama model with reasoning_effort
    provider = OCILLm(
        model_id="meta.llama-4-maverick",
        reasoning_effort="HIGH"
    )
"""

from src.client_adapters.oci_strategies.base_strategy import OCIRequestStrategy
from src.client_adapters.oci_strategies.generic_strategy import GenericRequestStrategy
from src.client_adapters.oci_strategies.cohere_strategy import CohereRequestStrategy

__all__ = [
    "OCIRequestStrategy",
    "GenericRequestStrategy",
    "CohereRequestStrategy",
]