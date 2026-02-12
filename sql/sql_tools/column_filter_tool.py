# src/tools/column_filter_tool.py
"""
Tool per selezionare colonne rilevanti dalle tabelle dello schema.
"""
from typing import List, Optional, Dict
from pydantic import Field

from src.core.tool import ToolBase,tool
from sql.database.schema.generated.schema_literals import (
    VistaBilancioEntrataAiColumns,
    VistaBilancioSpesaAiColumns,
)


@tool(
    name="column_filter",
    description="""
    Seleziona le tabelle e le colonne relative alla query dell'utente, utilizzalo in modo permissimo e non restrittivo
    Esempio: {"VISTA_BILANCIO_SPESA_AI": ["CAPITOLO", "DESCRIZIONE_CAP", "STN_DEFINITIVO_CO"]}'
    """
)
class ColumnFilterTool(ToolBase):
    VISTA_BILANCIO_ENTRATA_AI: Optional[List[VistaBilancioEntrataAiColumns]] = Field(
        default=None,
        description='Array di nomi colonna dalla vista bilancio entrata. '
                    'Esempio: ["CAPITOLO", "DESCRIZIONE_CAP", "TOT_ACCERTATO_COMP"]',
    )
    VISTA_BILANCIO_SPESA_AI: Optional[List[VistaBilancioSpesaAiColumns]] = Field(
        default=None,
        description='Array di nomi colonna dalla vista bilancio spesa. '
                    'Esempio: ["CAPITOLO", "DESCRIZIONE_CAP", "TOT_IMPEGNATO_COMP"]',
    )

    def execute(self) -> Dict[str, List[str]]:
        result = {}
        if self.VISTA_BILANCIO_ENTRATA_AI:
            result["VISTA_BILANCIO_ENTRATA_AI"] = list(self.VISTA_BILANCIO_ENTRATA_AI)
        if self.VISTA_BILANCIO_SPESA_AI:
            result["VISTA_BILANCIO_SPESA_AI"] = list(self.VISTA_BILANCIO_SPESA_AI)
        return result
