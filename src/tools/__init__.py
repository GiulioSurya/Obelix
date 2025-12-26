"""
Tools module - Strumenti per agenti (tool calling).

Uso:
    from src.tools import tool, ToolBase

    @tool(name="my_tool", description="Descrizione del tool")
    class MyTool(ToolBase):
        param: str = Field(...)

        async def execute(self) -> dict:
            return {"result": self.param}
"""

from src.tools.tool_base import ToolBase
from src.tools.tool_decorator import tool
from src.tools.sql_query_executor_tool import SqlQueryExecutorTool

__all__ = [
    "ToolBase",
    "tool",
    "SqlQueryExecutorTool",
]
