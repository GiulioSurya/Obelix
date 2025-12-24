"""
Query Complexity Classifier Tool
=================================

Tool per classificare la complessità di una query SQL in base a criteri come:
- Numero di join necessari
- Presenza di aggregazioni complesse
- Richieste di calcoli temporali/window functions
- Subquery o CTE
- Ambiguità semantica
- Numero di tabelle coinvolte
"""

from pydantic import Field
from typing import Literal
from enum import Enum

from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase


class ComplexityLevel(str, Enum):
    """Livelli di complessità query"""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


@tool(
    name="classify_query_complexity",
    description=(
        "Classifies the complexity of a natural language query into EASY, MEDIUM, or HARD "
        "based on the analytical articulation required to answer the question."
    )
)
class QueryComplexityClassifier(ToolBase):
    """
    Tool per classificare la complessità di query SQL.
    Analizza la query dell'utente per determinare
    se richiede pipeline leggera (EASY/MEDIUM) o pesante (HARD).
    """

    # Schema - campi popolati automaticamente dal decoratore
    complexity_level: Literal["EASY", "MEDIUM", "HARD"] = Field(
        ...,
        description=(
            "The complexity level of the query:\n"
            "- EASY: Single straightforward question with basic filters\n"
            "- MEDIUM: Dual objectives or single aggregation analysis on one dimension\n"
            "- HARD: Multiple concurrent analyses, cross-dimensional aggregations, "
            "multi-period comparisons, or nested analytical layers"
        )
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the classification (0.0 to 1.0). "
            "Lower confidence (<0.7) suggests ambiguous queries that may need HARD classification."
        )
    )

    async def execute(self) -> dict:
        """
        Esegue la classificazione della complessità.

        Note:
            Questo tool è pensato per essere chiamato dall'LLM, che analizza
            la query e lo schema per determinare la complessità.
            Il tool stesso valida solo lo schema di output.

        Returns:
            dict con complexity_level e confidence
        """
        # self.complexity_level e self.confidence già popolati dal decoratore!
        return {
            "complexity_level": self.complexity_level,
            "confidence": self.confidence
        }
