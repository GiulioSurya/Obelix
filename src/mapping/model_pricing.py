"""
Mapping dei modelli LLM ai costi per token (input/output) in euro.
"""

from typing import Dict


class ModelPricing:
    """
    Catalogo dei prezzi per i modelli LLM supportati.

    I costi sono espressi in euro per token (input/output).
    Per ora tutti i modelli hanno un costo di default di 0.1 € per token.
    """

    # Costi per 1M di token in euro (prezzi ufficiali Oracle Cloud Infrastructure)
    PRICING: Dict[str, Dict[str, float]] = {
        "meta.llama-3.3-70b-instruct": {
            "input_per_token": 0.17 / 1_000_000,  # 0.17€ per 1M token
            "output_per_token": 0.17 / 1_000_000,  # 0.17€ per 1M token
        },
        "openai.gpt-oss-120b": {
            "input_per_token": 0.1395 / 1_000_000,  # 0.1395€ per 1M token
            "output_per_token": 0.558 / 1_000_000,   # 0.558€ per 1M token
        },
    }

    # Costo di default per modelli non mappati
    DEFAULT_PRICING = {
        "input_per_token": 0.1 / 1_000_000,
        "output_per_token": 0.1 / 1_000_000,
    }

    @classmethod
    def get_input_cost(cls, model_id: str) -> float:
        """
        Restituisce il costo per token di input per un modello specifico.

        Args:
            model_id: ID del modello (es. "meta.llama-3.3-70b-instruct")

        Returns:
            Costo in euro per token di input
        """
        pricing = cls.PRICING.get(model_id, cls.DEFAULT_PRICING)
        return pricing["input_per_token"]

    @classmethod
    def get_output_cost(cls, model_id: str) -> float:
        """
        Restituisce il costo per token di output per un modello specifico.

        Args:
            model_id: ID del modello (es. "meta.llama-3.3-70b-instruct")

        Returns:
            Costo in euro per token di output
        """
        pricing = cls.PRICING.get(model_id, cls.DEFAULT_PRICING)
        return pricing["output_per_token"]

    @classmethod
    def get_pricing(cls, model_id: str) -> Dict[str, float]:
        """
        Restituisce il dizionario completo dei prezzi per un modello.

        Args:
            model_id: ID del modello

        Returns:
            Dizionario con chiavi 'input_per_token' e 'output_per_token'
        """
        return cls.PRICING.get(model_id, cls.DEFAULT_PRICING)
