# src/tools/tool_base.py
"""
Base class for tools - simplified version.

Use together with the @tool decorator:
    from src.core.tool.tool_decorator import tool
    from src.core.tool.tool_base import ToolBase

    @tool(name="my_tool", description="Description")
    class MyTool(ToolBase):
        param: str = Field(...)

        def execute(self) -> dict:  # can be sync or async
            return {"result": self.param}

The @tool decorator automatically handles both sync and async methods:
- If execute is sync: executed in a separate thread (asyncio.to_thread)
- If execute is async: executed directly with await
"""
from abc import ABC, abstractmethod
from typing import Any


class ToolBase(ABC):
    """
    Abstract base class for all tools.

    Attributes populated by the @tool decorator:
    - tool_name: str - Unique tool name
    - tool_description: str - Description for LLM
    - _input_schema: Type[BaseModel] - Pydantic schema for validation

    The @tool decorator handles:
    - Validate required name/description
    - Create Pydantic schema from Fields
    - Wrap execute() for automatic error handling
    - Populate attributes before execute()
    """

    # Attributes populated by the @tool decorator
    tool_name: str = None
    tool_description: str = None

    @classmethod
    def create_schema(cls):
        """
        Generate MCP schema for the tool.

        This method is overridden by the @tool decorator.
        If called without decorator, raises an error.
        """
        raise NotImplementedError(
            f"Class {cls.__name__} must use the @tool decorator. "
            f"Example: @tool(name='...', description='...')"
        )

    @abstractmethod
    def execute(self) -> Any:
        """
        Execute the tool's logic.

        Can be defined as sync or async method:
        - sync: def execute(self) -> Any
        - async: async def execute(self) -> Any

        The @tool decorator automatically detects the type and handles execution:
        - If sync: executed in separate thread (asyncio.to_thread) to avoid blocking the event loop
        - If async: executed directly with await

        Field attributes are already populated by the decorator before the call.
        No need to handle ToolCall or ToolResult - the decorator wraps everything.

        Returns:
            Any: Result of execution (automatically wrapped in ToolResult)

        Raises:
            Exception: Any exception is caught and converted to ToolResult with status=ERROR
        """
        pass
