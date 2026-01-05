"""
Mapping module - Maps message formats between different LLM provider APIs.

NOTE: We don't import provider_mapping here to avoid circular imports.
Use direct imports when needed:
    from src.mapping.provider_mapping import ProviderRegistry
    import src.mapping.provider_mapping  # to register the mappings
"""

# Import only utility functions that have no circular dependencies
from src.mapping.tool_extr_fall_back import (
    _extract_tool_calls_generic,
    _extract_tool_calls_cohere,
    _extract_tool_calls_ibm_watson_hybrid,
)

__all__ = [
    "_extract_tool_calls_generic",
    "_extract_tool_calls_cohere",
    "_extract_tool_calls_ibm_watson_hybrid",
]