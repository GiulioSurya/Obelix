# src/tools/sql_query_executor_tool.py
"""Tool per eseguire query SQL su Oracle database (async)."""
import oracledb
from pydantic import Field

from obelix.core.tool import tool,ToolBase
from sql.connections.db_connection import execute_query


@tool(
    name="sql_query_executor",
    description="Esegue una query SQL su database Oracle e restituisce i risultati"
)
class SqlQueryExecutorTool(ToolBase):
    """Tool per eseguire query SQL su Oracle database."""

    sql_query: str = Field(
        ...,
        description="Query SQL valida senza caratteri speciali, sequenze di escape o formattazione aggiuntiva."
    )

    def __init__(self, oracle_pool: oracledb.ConnectionPool):
        """
        Inizializza il tool con pool Oracle.

        Args:
            oracle_pool: Pool connessioni Oracle async (iniettato via DI).
        """
        self._pool = oracle_pool

    async def execute(self) -> dict:
        """
        Esegue la query SQL sul database Oracle in modo async.

        Returns:
            dict con columns, data, row_count e sql_query
        """
        # self.sql_query già popolato dal decoratore
        results, cursor_description = await execute_query(self._pool, self.sql_query)

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
