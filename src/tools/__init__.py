"""
Tools module - Tools for agents (tool calling).

Usage:
    from src.tools import tool, ToolBase

    @tool(name="my_tool", description="Description of the tool")
    class MyTool(ToolBase):
        param: str = Field(...)

        async def execute(self) -> dict:
            return {"result": self.param}
"""

from src.tools.tool_base import ToolBase
from src.tools.tool_decorator import tool

__all__ = [
    "ToolBase",
    "tool",
]
