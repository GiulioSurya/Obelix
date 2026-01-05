# src/tools/tool_base.py
"""
Base class for tools - simplified version.

Use together with @tool decorator:
    from src.tools.tool_decorator import tool
    from src.tools.tool_base import ToolBase

    @tool(name="my_tool", description="Description")
    class MyTool(ToolBase):
        param: str = Field(...)

        async def execute(self) -> dict:
            return {"result": self.param}
"""
from abc import ABC, abstractmethod
from typing import Any


class ToolBase(ABC):
    """
    Abstract base class for all tools.

    Attributes populated by @tool decorator:
    - tool_name: str - Unique name of the tool
    - tool_description: str - Description for LLM
    - _input_schema: Type[BaseModel] - Pydantic schema for validation

    The @tool decorator handles:
    - Validating required name/description
    - Creating Pydantic schema from Fields
    - Wrapping execute() for automatic error handling
    - Populating attributes before execute()
    """

    # Attributes populated by @tool decorator
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
            f"Class {cls.__name__} must use @tool decorator. "
            f"Example: @tool(name='...', description='...')"
        )

    @abstractmethod
    async def execute(self) -> Any:
        """
        Execute the tool logic.

        Field attributes are already populated by the decorator before calling.
        No need to handle ToolCall or ToolResult - the decorator wraps everything.

        Returns:
            Any: Result of execution (automatically wrapped in ToolResult)

        Raises:
            Exception: Any exception is caught and converted to ToolResult with status=ERROR
        """
        pass
