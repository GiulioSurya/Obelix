# src/tools/tool_decorator.py
"""
@tool decorator for defining tools in a declarative way.

Usage example:
    @tool(name="sql_query_executor", description="Execute SQL queries")
    class SqlQueryExecutor(ToolBase):
        sql_query: str = Field(..., description="Valid SQL query")

        def __init__(self, oracle_conn: OracleConnection):
            self.oracle_conn = oracle_conn

        async def execute(self) -> dict:
            # self.sql_query already populated!
            return self.oracle_conn.execute_query(self.sql_query)
"""
import copy
import time
from typing import Type, Any, get_type_hints

from pydantic import create_model
from pydantic.fields import FieldInfo

from src.messages.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema


def tool(name: str = None, description: str = None):
    """
    Decorator for defining tools in a declarative way.

    Args:
        name: Name of the tool (REQUIRED)
        description: Description of the tool (REQUIRED)

    Raises:
        ValueError: If name or description are missing (at import time)

    Returns:
        Decorator that transforms the class into a complete tool
    """
    def decorator(cls: Type) -> Type:
        # 1. Mandatory validation - fail-fast at import time
        if not name:
            raise ValueError(
                f"Tool {cls.__name__}: 'name' is required in @tool decorator. "
                f"Usage: @tool(name='my_tool', description='...')"
            )
        if not description:
            raise ValueError(
                f"Tool {cls.__name__}: 'description' is required in @tool decorator. "
                f"Usage: @tool(name='my_tool', description='...')"
            )

        # 2. Add class attributes for direct access
        cls.tool_name = name
        cls.tool_description = description

        # 3. Extract Fields from class to create Pydantic schema
        cls._input_schema = _create_input_schema(cls, name)

        # 4. Wrap the original execute method
        original_execute = cls.execute

        async def wrapped_execute(self, tool_call: ToolCall) -> ToolResult:
            """Wrapped execute that handles validation and errors automatically"""
            start_time = time.time()
            try:
                # Create isolated copy for parallel execution and thread-safety
                instance = copy.copy(self)

                # Validate arguments and populate attributes on the COPY
                validated = instance._input_schema(**tool_call.arguments)
                for field_name in validated.model_fields:
                    setattr(instance, field_name, getattr(validated, field_name))

                # Call original execute on the COPY (isolated)
                result = await original_execute(instance)

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=result,
                    status=ToolStatus.SUCCESS,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=None,
                    status=ToolStatus.ERROR,
                    error=str(e),
                    execution_time=time.time() - start_time
                )

        cls.execute = wrapped_execute

        # 5. Add create_schema() as a class method
        @classmethod
        def create_schema(cls_inner) -> MCPToolSchema:
            """Generate MCP schema from internal Pydantic model"""
            return MCPToolSchema(
                name=cls_inner.tool_name,
                description=cls_inner.tool_description,
                inputSchema=cls_inner._input_schema.model_json_schema(),
                outputSchema={"type": "object", "additionalProperties": True}
            )

        cls.create_schema = create_schema

        return cls

    return decorator


def _create_input_schema(cls: Type, tool_name: str) -> Type:
    """
    Create dynamic Pydantic model from Fields annotated on the class.

    Extracts all class attributes that have:
    - Type annotation (e.g. sql_query: str)
    - Default that is a FieldInfo (e.g. = Field(...))

    Args:
        cls: The tool class to analyze
        tool_name: Name of the tool for model naming

    Returns:
        Dynamic Pydantic model with extracted fields
    """
    fields = {}

    # Get type hints from class
    try:
        hints = get_type_hints(cls)
    except Exception:
        # Fallback if get_type_hints fails (e.g. forward references)
        hints = getattr(cls, '__annotations__', {})

    # Extract fields with FieldInfo
    for attr_name, attr_type in hints.items():
        # Skip internal attributes and base class
        if attr_name.startswith('_'):
            continue

        # Get default from class
        default = getattr(cls, attr_name, ...)

        # If it's a FieldInfo, include it in schema
        if isinstance(default, FieldInfo):
            fields[attr_name] = (attr_type, default)

    # Create dynamic Pydantic model
    schema_class_name = f"{tool_name.title().replace('_', '')}Schema"
    return create_model(schema_class_name, **fields)
