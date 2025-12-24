"""
Mapping module - Mappatura tra formati di messaggi di diversi provider LLM.

NOTA: Non importiamo provider_mapping qui per evitare import circolari.
Usare import diretti quando necessario:
    from src.mapping.provider_mapping import ProviderRegistry
    import src.mapping.provider_mapping  # per registrare i mapping
"""

# Importiamo solo le funzioni di utility che non hanno dipendenze circolari
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