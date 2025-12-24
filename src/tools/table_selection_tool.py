"""
Table Selection Tool
====================

Tool per la selezione intelligente delle tabelle database tramite LLM.
Valida i nomi delle tabelle selezionate contro lo schema cache.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic import Field

from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase


# Path hardcodato della cache schema
SCHEMA_CACHE_PATH = Path(__file__).parent.parent / "database" / "cache" / "schema_cache" / "bilancio_schema.json"


@tool(
    name="select_tables",
    description=(
        "Seleziona le tabelle del database Oracle più rilevanti per rispondere alla query utente. "
        "IMPORTANTE: compila PRIMA il campo 'reasoning' con il tuo ragionamento, POI 'selected_tables'."
    )
)
class SelectTablesTool(ToolBase):
    """
    Tool per selezionare tabelle database tramite LLM.

    Carica lo schema cache dal path hardcodato e valida che i nomi
    delle tabelle selezionate dall'LLM esistano effettivamente.

    Se l'LLM usa un nome sbagliato, il tool ritorna un errore con la lista
    delle tabelle disponibili, permettendo all'LLM di correggersi nel retry.
    """

    # Schema - campi popolati automaticamente dal decoratore
    # L'ordine dei campi è importante: reasoning PRIMA di selected_tables
    # per forzare l'LLM a ragionare prima di decidere.
    reasoning: str = Field(
        ...,
        description=(
            "Ragionamento dettagliato su quale/quali tabella/e selezionare. "
            "Spiega PERCHÉ queste tabelle sono rilevanti per la query dell'utente. "
            "Questo campo è OBBLIGATORIO e deve essere compilato PRIMA di selected_tables."
        )
    )

    selected_tables: List[str] = Field(
        ...,
        description=(
            "Lista dei nomi ESATTI delle tabelle selezionate. "
            "Usa i nomi esattamente come forniti nelle tabelle disponibili (case-sensitive). "
            "Esempio: ['VISTA_BILANCIO_ENTRATA_AI'] oppure ['VISTA_BILANCIO_ENTRATA_AI', 'VISTA_BILANCIO_SPESA_AI']"
        ),
        min_length=1
    )

    def __init__(self):
        """
        Inizializza il tool caricando lo schema cache.

        Raises:
            FileNotFoundError: Se il file schema cache non esiste
            json.JSONDecodeError: Se il file JSON non è valido
        """
        self._schema_data = self._load_schema()
        self._available_tables = self._extract_tables_info()

    def _load_schema(self) -> Dict[str, Any]:
        """
        Carica lo schema JSON dalla cache.

        Returns:
            Dict con i dati dello schema

        Raises:
            FileNotFoundError: Se il file non esiste
            json.JSONDecodeError: Se il JSON non è valido
        """
        if not SCHEMA_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Schema cache non trovata: {SCHEMA_CACHE_PATH}\n"
                f"Assicurati che il file esista."
            )

        with open(SCHEMA_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_tables_info(self) -> Dict[str, Dict[str, str]]:
        """
        Estrae informazioni sulle tabelle dallo schema.

        Returns:
            Dict {table_name: {"description": ..., "type": ...}}
        """
        tables = {}
        for table in self._schema_data.get("tables", []):
            table_name = table.get("table_name")
            if table_name:
                tables[table_name] = {
                    "description": table.get("description", ""),
                    "type": table.get("type", "TABLE")
                }
        return tables

    def get_tables_info_formatted(self) -> str:
        """
        Formatta le informazioni delle tabelle per il system prompt.

        Returns:
            Stringa formattata con nome e descrizione di ogni tabella
        """
        lines = []
        for i, (table_name, info) in enumerate(self._available_tables.items(), 1):
            lines.append(f"{i}. **{table_name}**")
            lines.append(f"   Tipo: {info['type']}")
            lines.append(f"   Descrizione: {info['description']}")
            lines.append("")

        return "\n".join(lines)

    def get_available_table_names(self) -> List[str]:
        """
        Ritorna la lista dei nomi delle tabelle disponibili.

        Returns:
            Lista di nomi tabelle
        """
        return list(self._available_tables.keys())

    async def execute(self) -> dict:
        """
        Esegue la validazione delle tabelle selezionate.

        Valida che tutti i nomi in selected_tables esistano nello schema.
        Se qualche nome è invalido, solleva eccezione per permettere
        all'LLM di correggersi nel retry.

        Returns:
            dict con tabelle validate, reasoning e count

        Raises:
            ValueError: Se qualche tabella non esiste nello schema
        """
        # self.reasoning e self.selected_tables già popolati dal decoratore!

        # Valida che le tabelle esistano
        valid_tables = []
        invalid_tables = []

        for table_name in self.selected_tables:
            if table_name in self._available_tables:
                valid_tables.append(table_name)
            else:
                invalid_tables.append(table_name)

        # Se ci sono tabelle invalide, solleva errore
        if invalid_tables:
            available_names = self.get_available_table_names()
            raise ValueError(
                f"Tabelle non trovate: {invalid_tables}. "
                f"Tabelle disponibili: {available_names}. "
                f"Riprova usando i nomi esatti delle tabelle disponibili."
            )

        # Successo: ritorna tabelle validate
        return {
            "tables": valid_tables,
            "reasoning": self.reasoning,
            "table_count": len(valid_tables)
        }
