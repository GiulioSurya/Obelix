# src/tools/sql_query_executor_tool.py
"""Tool for executing SQL queries on any supported database."""
from pydantic import Field

from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase
from src.connections.db_connection import AbstractDatabaseConnection


@tool(
    name="sql_query_executor",
    description="Executes a SQL query on the configured database and returns the results. Use only for the final query."
)
class SqlQueryExecutorTool(ToolBase):
    """
    Tool for executing SQL queries on any database.

    Works with any AbstractDatabaseConnection implementation (Oracle, PostgreSQL, etc.).
    The connection's security mode controls which operations are allowed.
    """

    # Schema fields - populated automatically by decorator
    error_analysis: str = Field(
        default="",
        description="Analysis of SQL error from previous failed query attempt. Explains what went wrong and how you are correcting it."
    )
    sql_query: str = Field(
        ...,
        description="Valid SQL query without special characters, escape sequences, or additional formatting."
    )

    def __init__(self, db_conn: AbstractDatabaseConnection):
        """
        Initialize the tool with a database connection.

        Args:
            db_conn: Any database connection (Oracle, PostgreSQL, etc.)
        """
        self._db_conn = db_conn

    async def execute(self) -> dict:
        """
        Execute the SQL query on the database.

        Returns:
            dict with columns, data, row_count, and sql_query
        """
        # Execute query - returns QueryResult
        result = self._db_conn.execute_query(self.sql_query)

        if not result.is_success:
            # Return error information
            return {
                "error": result.error,
                "sql_query": self.sql_query,
                "success": False
            }

        # Return structured output for DataFrame conversion
        return {
            "columns": result.column_names,
            "data": result.rows,
            "row_count": result.row_count,
            "sql_query": self.sql_query,
            "success": True
        }
