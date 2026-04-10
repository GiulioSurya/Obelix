# src/tools/tool_base.py
"""
Tool Protocol - structural typing contract for all tools.

All tools must satisfy this protocol:
- tool_name: str
- tool_description: str
- execute(tool_call: ToolCall) -> ToolResult
- create_schema() -> MCPToolSchema

The @tool decorator is the primary way to create tools:
    from obelix.core.tool.tool_decorator import tool

    @tool(name="my_tool", description="Description")
    class MyTool:
        param: str = Field(...)

        def execute(self) -> dict:  # can be sync or async
            return {"result": self.param}

The decorator wraps execute() to match the Protocol signature automatically.
SubAgentWrapper and MCPToolAdapter satisfy the Protocol via structural typing.
"""

from typing import Protocol, runtime_checkable

from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Contract that all tools must satisfy (structural typing)."""

    tool_name: str
    tool_description: str
    is_deferred: bool

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
    def create_schema(self) -> MCPToolSchema: ...


# Behavioral annotations — set on the Protocol class itself so they are
# available for type-checking and documentation, but NOT included in
# __protocol_attrs__ (which would break isinstance() for tools that don't
# declare them).  The @tool decorator sets these on each decorated class.
Tool.read_only = None  # type: ignore[attr-defined]  # bool | None
Tool.destructive = None  # type: ignore[attr-defined]  # bool | None
Tool.idempotent = None  # type: ignore[attr-defined]  # bool | None
Tool.open_world = None  # type: ignore[attr-defined]  # bool | None
