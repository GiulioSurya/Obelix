# src/tools/sql_query_executor_tool.py
"""Tool per eseguire query SQL su Oracle database."""
from pydantic import Field

from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase
from src.connections.db_connection.oracle_connection import OracleConnection


@tool(
    name="sql_query_executor",
    description="Esegue una query SQL su database Oracle e restituisce i risultati, deve essere usato solo per la query finale"
)
class SqlQueryExecutorTool(ToolBase):
    """Tool per eseguire query SQL su Oracle database"""

    # Schema - campi popolati automaticamente dal decoratore
    error_analysis: str = Field(
        default="",
        description="Analisi dell'errore SQL dal precedente tentativo di query fallito. Spiega cosa è andato storto e come lo stai correggendo."
    )
    sql_query: str = Field(
        ...,
        description="Query SQL valida senza caratteri speciali, sequenze di escape o formattazione aggiuntiva."
    )

    def __init__(self, oracle_conn: OracleConnection):
        """
        Inizializza il tool con connessione Oracle.

        Args:
            oracle_conn: Connessione Oracle già inizializzata (OBBLIGATORIA).
        """
        self._oracle_conn = oracle_conn

    async def execute(self) -> dict:
        """
        Esegue la query SQL sul database Oracle.

        Returns:
            dict con columns, data, row_count e sql_query
        """
        # self.sql_query già popolato dal decoratore!
        results, cursor_description = self._oracle_conn.execute_query(self.sql_query)

        # I risultati sono già tuple da fetchall()
        result_data = results if results else []

        # Estrai nomi colonne da cursor_description
        columns = [
            desc[0]  # Nome della colonna
            for desc in cursor_description
        ] if cursor_description else []

        # Output strutturato per conversione a DataFrame pandas
        return {
            "columns": columns,
            "data": result_data,
            "row_count": len(result_data),
            "sql_query": self.sql_query
        }
