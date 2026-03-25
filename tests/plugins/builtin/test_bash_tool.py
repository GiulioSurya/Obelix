"""Tests for obelix.plugins.builtin.bash_tool - BashTool.

Covers:
- is_deferred flag (based on executor.is_remote)
- OutputSchema detection and validation
- Input schema (required/optional fields, defaults)
- Output schema in MCPToolSchema
- execute() with local executor (returns result)
- execute() with remote executor (returns None / deferred)
- system_prompt_fragment() with populated/empty shell_info
"""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from obelix.adapters.outbound.shell.client_executor import ClientShellExecutor
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolStatus
from obelix.plugins.builtin.bash_tool import BashTool

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def remote_executor() -> ClientShellExecutor:
    """A ClientShellExecutor (remote / deferred)."""
    return ClientShellExecutor()


@pytest.fixture
def local_executor():
    """A mock local executor (not remote)."""
    executor = AsyncMock()
    executor.is_remote = False
    executor.shell_info = {"platform": "Linux", "shell_name": "bash"}
    executor.execute = AsyncMock(
        return_value={"stdout": "hello\n", "stderr": "", "exit_code": 0}
    )
    return executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBashToolAttributes:
    """Tests for BashTool class-level attributes set by @tool decorator."""

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


class TestBashToolDeferredFlag:
    """Tests for is_deferred based on executor type."""

    def test_with_remote_executor_is_deferred(self, remote_executor):
        """BashTool with ClientShellExecutor should be deferred."""
        tool = BashTool(executor=remote_executor)
        assert tool.is_deferred is True

    def test_with_local_executor_not_deferred(self, local_executor):
        """BashTool with local executor should NOT be deferred."""
        tool = BashTool(executor=local_executor)
        assert tool.is_deferred is False

    def test_class_level_is_deferred_true(self):
        """Class-level is_deferred should remain True (decorator default)."""
        assert BashTool.is_deferred is True

    def test_executor_required(self):
        """BashTool() without executor should raise TypeError."""
        with pytest.raises(TypeError):
            BashTool()


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


class TestBashToolExecuteDeferred:
    """Tests for BashTool.execute() with remote executor (deferred)."""

    @pytest.fixture
    def tool_instance(self, remote_executor) -> BashTool:
        return BashTool(executor=remote_executor)

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


class TestBashToolWithLocalExecutor:
    """Tests for BashTool with a local executor (server-side execution)."""

    def test_with_executor_not_deferred(self, local_executor):
        """BashTool with local executor should NOT be deferred."""
        tool = BashTool(executor=local_executor)
        assert tool.is_deferred is False

    async def test_execute_with_executor_returns_result(self, local_executor):
        """With local executor, execute() should return the executor's result."""
        tool = BashTool(executor=local_executor)
        call = ToolCall(
            id="tc_exec_1",
            name="bash",
            arguments={"command": "echo hello", "description": "Print hello"},
        )
        result = await tool.execute(call)
        assert result.result == {"stdout": "hello\n", "stderr": "", "exit_code": 0}
        assert result.status == ToolStatus.SUCCESS

    async def test_execute_with_executor_passes_arguments(self, local_executor):
        """Executor should receive command, timeout, working_directory."""
        tool = BashTool(executor=local_executor)
        call = ToolCall(
            id="tc_exec_2",
            name="bash",
            arguments={
                "command": "ls -la",
                "description": "List files",
                "timeout": 30,
                "working_directory": "/tmp",
            },
        )
        await tool.execute(call)
        local_executor.execute.assert_called_once_with(
            command="ls -la",
            timeout=30,
            working_directory="/tmp",
        )


class TestBashToolSystemPromptFragment:
    """Tests for system_prompt_fragment() with different executor states."""

    def test_fragment_none_when_no_shell_info(self, remote_executor):
        """Returns None when executor has no shell_info."""
        tool = BashTool(executor=remote_executor)
        assert tool.system_prompt_fragment() is None

    def test_fragment_with_shell_info(self, remote_executor):
        """Returns fragment when executor has shell_info."""
        remote_executor.set_shell_info(
            {
                "platform": "Linux",
                "os_version": "Linux-6.1",
                "shell_name": "bash",
                "cwd": "/home/user",
            }
        )
        tool = BashTool(executor=remote_executor)
        fragment = tool.system_prompt_fragment()
        assert fragment is not None
        assert "## Shell Environment" in fragment
        assert "Linux" in fragment
        assert "bash" in fragment
        assert "/home/user" in fragment

    def test_fragment_with_local_executor(self, local_executor):
        """Local executor with shell_info produces a fragment."""
        tool = BashTool(executor=local_executor)
        fragment = tool.system_prompt_fragment()
        assert fragment is not None
        assert "Linux" in fragment

    def test_fragment_windows_syntax_hint(self, remote_executor):
        """Windows platform should include syntax hint."""
        remote_executor.set_shell_info(
            {
                "platform": "Windows",
                "shell_name": "bash",
            }
        )
        tool = BashTool(executor=remote_executor)
        fragment = tool.system_prompt_fragment()
        assert "Unix shell syntax" in fragment

    def test_fragment_with_mounts(self, remote_executor):
        """Drive mounts should appear in the fragment."""
        remote_executor.set_shell_info(
            {
                "platform": "Windows",
                "shell_name": "bash",
                "mounts": {"C:": "/c/", "D:": "/d/"},
            }
        )
        tool = BashTool(executor=remote_executor)
        fragment = tool.system_prompt_fragment()
        assert "C: -> /c/" in fragment
        assert "D: -> /d/" in fragment
        assert "Unix-style paths" in fragment
