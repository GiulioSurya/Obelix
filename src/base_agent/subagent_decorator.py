# src/base_agent/subagent_decorator.py
"""
@subagent decorator for exposing BaseAgent as a registrable sub-agent.

Usage:
    @subagent(name="sql_analyzer", description="Analyzes SQL errors")
    class SQLAnalyzerAgent(BaseAgent):
        error_context: str = Field(default="", description="Error context")

        def __init__(self):
            super().__init__(system_message="You are a SQL expert...")

The decorated agent can then be registered by an @orchestrator agent:
    coordinator.register_agent(SQLAnalyzerAgent())
"""
from typing import Type, get_type_hints, Dict, Any

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from src.messages.tool_message import MCPToolSchema


def subagent(name: str = None, description: str = None):
    """
    Decorator that marks a BaseAgent as registrable by orchestrators.

    Adds metadata and schema generation capabilities without modifying
    the agent's core behavior. The actual wrapping happens in
    @orchestrator.register_agent().

    Args:
        name: Unique name for the sub-agent (REQUIRED)
        description: Description of what the sub-agent does (REQUIRED)

    Raises:
        ValueError: If name or description are missing (at import time)

    Returns:
        Decorated class with subagent metadata
    """
    def decorator(cls: Type) -> Type:
        # 1. Mandatory validation - fail-fast at import time
        if not name:
            raise ValueError(
                f"SubAgent {cls.__name__}: 'name' is required in @subagent decorator. "
                f"Usage: @subagent(name='my_agent', description='...')"
            )
        if not description:
            raise ValueError(
                f"SubAgent {cls.__name__}: 'description' is required in @subagent decorator. "
                f"Usage: @subagent(name='my_agent', description='...')"
            )

        # 2. Add subagent metadata
        cls.subagent_name = name
        cls.subagent_description = description

        # 3. Extract Fields from class and add hardcoded 'query'
        cls._subagent_fields = _extract_subagent_fields(cls)

        # 4. Create input schema for MCP compatibility
        cls._subagent_input_schema = _create_input_schema(cls._subagent_fields, name)

        # 5. Add schema generation method
        @classmethod
        def create_subagent_schema(cls_inner) -> MCPToolSchema:
            """Generate MCP schema for the sub-agent"""
            return MCPToolSchema(
                name=cls_inner.subagent_name,
                description=cls_inner.subagent_description,
                inputSchema=cls_inner._subagent_input_schema.model_json_schema(),
                outputSchema={
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string"},
                        "content": {"type": "string"},
                        "tool_results": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "error": {"type": "string", "nullable": True}
                    }
                }
            )

        cls.create_subagent_schema = create_subagent_schema

        return cls

    return decorator


def _extract_subagent_fields(cls: Type) -> Dict[str, tuple]:
    """
    Extract Field definitions from the class.

    Automatically adds 'query' field if not present, always as first parameter.

    Args:
        cls: The subagent class to analyze

    Returns:
        Dict mapping field names to (type, FieldInfo) tuples, with 'query' first
    """
    other_fields = {}

    # Get type hints from class
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = getattr(cls, '__annotations__', {})

    # Extract fields with FieldInfo (excluding 'query' to add it manually later)
    for attr_name, attr_type in hints.items():
        # Skip internal attributes
        if attr_name.startswith('_'):
            continue

        # Skip if it's 'query' (we'll add it first at the end)
        if attr_name == 'query':
            continue

        # Get default from class
        default = getattr(cls, attr_name, ...)

        # If it's a FieldInfo, include it
        if isinstance(default, FieldInfo):
            other_fields[attr_name] = (attr_type, default)

    # Build final dict with 'query' ALWAYS as first field
    fields = {
        'query': (str, Field(..., description="Query for the sub-agent")),
        **other_fields
    }

    return fields


def _create_input_schema(fields: Dict[str, tuple], name: str) -> Type:
    """
    Create dynamic Pydantic model from extracted fields.

    Args:
        fields: Dict mapping field names to (type, FieldInfo) tuples
        name: Sub-agent name for schema class naming

    Returns:
        Dynamic Pydantic model class
    """
    schema_class_name = f"{name.title().replace('_', '')}SubAgentSchema"
    return create_model(schema_class_name, **fields)
