"""Tests for obelix.plugins.builtin.bash_tool - BashTool (deferred).

Covers:
- is_deferred flag
- OutputSchema detection and validation
- Input schema (required/optional fields, defaults)
- Output schema in MCPToolSchema
- execute() returns None (deferred semantics)
"""

import pytest
from pydantic import BaseModel

from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolStatus
from obelix.plugins.builtin.bash_tool import BashTool

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBashToolAttributes:
    """Tests for BashTool class-level attributes set by @tool decorator."""

    def test_bash_tool_is_deferred(self):
        """BashTool must be marked as deferred (client-side execution)."""
        assert BashTool.is_deferred is True

    def test_bash_tool_has_output_schema(self):
        """BashTool._output_schema should reference the inner OutputSchema class."""
        assert BashTool._output_schema is not None
        assert BashTool._output_schema is BashTool.OutputSchema

    def test_bash_tool_output_schema_is_base_model(self):
        """OutputSchema must be a Pydantic BaseModel subclass."""
        assert issubclass(BashTool.OutputSchema, BaseModel)

    def test_bash_tool_name(self):
        """Tool name should be 'bash'."""
        assert BashTool.tool_name == "bash"

    def test_bash_tool_description_present(self):
        """Tool description should be a non-empty string."""
        assert isinstance(BashTool.tool_description, str)
        assert len(BashTool.tool_description) > 0


class TestBashToolInputSchema:
    """Tests for BashTool input schema (MCP inputSchema)."""

    @pytest.fixture
    def schema(self) -> MCPToolSchema:
        return BashTool.create_schema()

    def test_schema_is_mcp_tool_schema(self, schema: MCPToolSchema):
        assert isinstance(schema, MCPToolSchema)

    def test_schema_name(self, schema: MCPToolSchema):
        assert schema.name == "bash"

    def test_input_schema_has_command(self, schema: MCPToolSchema):
        props = schema.inputSchema["properties"]
        assert "command" in props

    def test_input_schema_has_description(self, schema: MCPToolSchema):
        props = schema.inputSchema["properties"]
        assert "description" in props

    def test_input_schema_has_timeout(self, schema: MCPToolSchema):
        props = schema.inputSchema["properties"]
        assert "timeout" in props

    def test_input_schema_has_working_directory(self, schema: MCPToolSchema):
        props = schema.inputSchema["properties"]
        assert "working_directory" in props

    def test_input_schema_command_required(self, schema: MCPToolSchema):
        required = schema.inputSchema.get("required", [])
        assert "command" in required

    def test_input_schema_description_required(self, schema: MCPToolSchema):
        required = schema.inputSchema.get("required", [])
        assert "description" in required

    def test_input_schema_timeout_has_default(self, schema: MCPToolSchema):
        props = schema.inputSchema["properties"]
        assert props["timeout"].get("default") == 120

    def test_input_schema_timeout_not_required(self, schema: MCPToolSchema):
        required = schema.inputSchema.get("required", [])
        assert "timeout" not in required

    def test_input_schema_working_directory_not_required(self, schema: MCPToolSchema):
        required = schema.inputSchema.get("required", [])
        assert "working_directory" not in required


class TestBashToolOutputSchema:
    """Tests for BashTool output schema (MCP outputSchema)."""

    @pytest.fixture
    def schema(self) -> MCPToolSchema:
        return BashTool.create_schema()

    def test_output_schema_present(self, schema: MCPToolSchema):
        """outputSchema should be populated from OutputSchema class."""
        assert schema.outputSchema is not None

    def test_output_schema_has_stdout(self, schema: MCPToolSchema):
        props = schema.outputSchema["properties"]
        assert "stdout" in props

    def test_output_schema_has_stderr(self, schema: MCPToolSchema):
        props = schema.outputSchema["properties"]
        assert "stderr" in props

    def test_output_schema_has_exit_code(self, schema: MCPToolSchema):
        props = schema.outputSchema["properties"]
        assert "exit_code" in props

    def test_output_schema_matches_model_json_schema(self, schema: MCPToolSchema):
        """outputSchema should match OutputSchema.model_json_schema() exactly."""
        expected = BashTool.OutputSchema.model_json_schema()
        assert schema.outputSchema == expected


class TestBashToolOutputSchemaValidation:
    """Tests for BashTool.OutputSchema as a standalone Pydantic model."""

    def test_output_schema_full_construction(self):
        result = BashTool.OutputSchema(stdout="hello", stderr="", exit_code=0)
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_output_schema_defaults(self):
        """All fields have defaults: empty strings and 0."""
        result = BashTool.OutputSchema()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_output_schema_error_case(self):
        result = BashTool.OutputSchema(
            stdout="", stderr="command not found", exit_code=127
        )
        assert result.stderr == "command not found"
        assert result.exit_code == 127

    def test_output_schema_serialization_round_trip(self):
        original = BashTool.OutputSchema(stdout="ok", stderr="warn", exit_code=1)
        data = original.model_dump()
        restored = BashTool.OutputSchema(**data)
        assert restored == original


class TestBashToolExecute:
    """Tests for BashTool.execute() deferred behavior."""

    @pytest.fixture
    def tool_instance(self) -> BashTool:
        return BashTool()

    @pytest.fixture
    def sample_call(self) -> ToolCall:
        return ToolCall(
            id="tc_bash_1",
            name="bash",
            arguments={
                "command": "ls -la",
                "description": "List files in current directory",
            },
        )

    async def test_bash_tool_execute_returns_none(
        self, tool_instance: BashTool, sample_call: ToolCall
    ):
        """execute() should return ToolResult with result=None (deferred)."""
        result = await tool_instance.execute(sample_call)
        assert result.result is None

    async def test_bash_tool_execute_status_success(
        self, tool_instance: BashTool, sample_call: ToolCall
    ):
        """Even though result is None, status should be SUCCESS."""
        result = await tool_instance.execute(sample_call)
        assert result.status == ToolStatus.SUCCESS

    async def test_bash_tool_execute_preserves_tool_call_id(
        self, tool_instance: BashTool, sample_call: ToolCall
    ):
        result = await tool_instance.execute(sample_call)
        assert result.tool_call_id == "tc_bash_1"

    async def test_bash_tool_execute_with_all_args(self, tool_instance: BashTool):
        call = ToolCall(
            id="tc_bash_2",
            name="bash",
            arguments={
                "command": "git status",
                "description": "Check git status",
                "timeout": 30,
                "working_directory": "/tmp",
            },
        )
        result = await tool_instance.execute(call)
        assert result.result is None
        assert result.status == ToolStatus.SUCCESS

    async def test_bash_tool_execute_validation_error_missing_command(
        self, tool_instance: BashTool
    ):
        """Missing required 'command' field should produce an error ToolResult."""
        call = ToolCall(
            id="tc_bad",
            name="bash",
            arguments={"description": "no command provided"},
        )
        result = await tool_instance.execute(call)
        assert result.status == ToolStatus.ERROR

    async def test_bash_tool_execute_validation_error_missing_description(
        self, tool_instance: BashTool
    ):
        """Missing required 'description' field should produce an error ToolResult."""
        call = ToolCall(
            id="tc_bad",
            name="bash",
            arguments={"command": "ls"},
        )
        result = await tool_instance.execute(call)
        assert result.status == ToolStatus.ERROR
