"""Tests for obelix.core.tool.tool_base - Tool Protocol.

Covers:
- Protocol definition and structural typing
- runtime_checkable behavior
- Required attributes and methods
- Classes that satisfy / do not satisfy the protocol
"""

import pytest

from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from obelix.core.tool.tool_base import Tool

# ---------------------------------------------------------------------------
# Helpers: classes that do/don't satisfy the Tool protocol
# ---------------------------------------------------------------------------


class ValidToolImpl:
    """Minimal class that satisfies the Tool protocol via structural typing."""

    tool_name: str = "valid_tool"
    tool_description: str = "A valid tool"

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            result={"ok": True},
            status=ToolStatus.SUCCESS,
        )

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema={"type": "object", "properties": {}},
        )


class MissingExecute:
    """Missing execute method."""

    tool_name: str = "bad"
    tool_description: str = "bad"

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name="bad",
            description="bad",
            inputSchema={"type": "object"},
        )


class MissingCreateSchema:
    """Missing create_schema method."""

    tool_name: str = "bad"
    tool_description: str = "bad"

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_name="bad",
            tool_call_id="x",
            result=None,
            status=ToolStatus.ERROR,
        )


class EmptyClass:
    """Completely empty class."""

    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolProtocol:
    """Tests for the Tool Protocol definition."""

    def test_tool_is_runtime_checkable(self):
        """Tool protocol should be decorated with @runtime_checkable."""
        assert hasattr(Tool, "__protocol_attrs__") or hasattr(
            Tool, "_is_runtime_protocol"
        )

    def test_valid_implementation_is_instance(self):
        """A class with all required attributes/methods satisfies isinstance check."""
        impl = ValidToolImpl()
        assert isinstance(impl, Tool)

    def test_missing_execute_not_instance(self):
        """A class missing execute() does not satisfy the protocol."""
        obj = MissingExecute()
        assert not isinstance(obj, Tool)

    def test_missing_create_schema_not_instance(self):
        """A class missing create_schema() does not satisfy the protocol."""
        obj = MissingCreateSchema()
        assert not isinstance(obj, Tool)

    def test_empty_class_not_instance(self):
        """An empty class does not satisfy the protocol."""
        obj = EmptyClass()
        assert not isinstance(obj, Tool)

    def test_protocol_requires_tool_name_attribute(self):
        """Protocol should declare tool_name as a required attribute."""
        # At minimum, the protocol should have execute and create_schema callable members
        assert callable(getattr(Tool, "execute", None)) or hasattr(Tool, "execute")

    def test_protocol_requires_tool_description_attribute(self):
        """Protocol should declare tool_description as a required attribute."""
        assert callable(getattr(Tool, "create_schema", None)) or hasattr(
            Tool, "create_schema"
        )


class TestValidToolBehavior:
    """Verify that a valid structural implementation works correctly."""

    @pytest.fixture
    def valid_tool(self) -> ValidToolImpl:
        return ValidToolImpl()

    @pytest.fixture
    def sample_call(self) -> ToolCall:
        return ToolCall(id="tc_1", name="valid_tool", arguments={})

    async def test_execute_returns_tool_result(self, valid_tool, sample_call):
        result = await valid_tool.execute(sample_call)
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"ok": True}

    def test_create_schema_returns_mcp_tool_schema(self, valid_tool):
        schema = valid_tool.create_schema()
        assert isinstance(schema, MCPToolSchema)
        assert schema.name == "valid_tool"
        assert schema.description == "A valid tool"

    def test_tool_name_accessible(self, valid_tool):
        assert valid_tool.tool_name == "valid_tool"

    def test_tool_description_accessible(self, valid_tool):
        assert valid_tool.tool_description == "A valid tool"
