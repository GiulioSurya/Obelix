"""Tests for MCPToolAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolStatus
from obelix.core.tool.tool_base import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "read_file",
    description: str | None = "Read a file",
    input_schema: dict | None = None,
    annotations: object | None = None,
) -> MagicMock:
    """Create a fake MCP SDK Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    tool.annotations = annotations
    return tool


def _make_annotations(
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
) -> MagicMock:
    ann = MagicMock()
    ann.readOnlyHint = read_only
    ann.destructiveHint = destructive
    ann.idempotentHint = idempotent
    ann.openWorldHint = open_world
    return ann


def _make_call_result(
    texts: list[str] | None = None,
    is_error: bool = False,
    structured_content: dict | None = None,
) -> MagicMock:
    """Create a fake MCP CallToolResult."""
    result = MagicMock()
    blocks = []
    for t in texts or []:
        block = MagicMock()
        block.text = t
        blocks.append(block)
    result.content = blocks
    result.isError = is_error
    result.structuredContent = structured_content
    return result


def _tool_call(tool_name: str = "mcp__srv__read_file") -> ToolCall:
    return ToolCall(id="call_1", name=tool_name, arguments={"path": "/tmp/x"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPToolAdapterInit:
    def test_tool_name_has_prefix(self):
        adapter = MCPToolAdapter("myserver", _make_mcp_tool(), MagicMock())
        assert adapter.tool_name == "mcp__myserver__read_file"

    def test_description_defaults_to_empty(self):
        adapter = MCPToolAdapter("s", _make_mcp_tool(description=None), MagicMock())
        assert adapter.tool_description == ""

    def test_is_deferred_always_false(self):
        adapter = MCPToolAdapter("s", _make_mcp_tool(), MagicMock())
        assert adapter.is_deferred is False

    def test_satisfies_tool_protocol(self):
        adapter = MCPToolAdapter("s", _make_mcp_tool(), MagicMock())
        assert isinstance(adapter, Tool)

    def test_annotations_mapped(self):
        ann = _make_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        )
        adapter = MCPToolAdapter("s", _make_mcp_tool(annotations=ann), MagicMock())
        assert adapter.read_only is True
        assert adapter.destructive is False
        assert adapter.idempotent is True
        assert adapter.open_world is False

    def test_annotations_default_none_when_no_annotations(self):
        adapter = MCPToolAdapter("s", _make_mcp_tool(annotations=None), MagicMock())
        assert adapter.read_only is None
        assert adapter.destructive is None
        assert adapter.idempotent is None
        assert adapter.open_world is None


class TestCreateSchema:
    def test_returns_mcp_tool_schema(self):
        mcp_tool = _make_mcp_tool()
        adapter = MCPToolAdapter("srv", mcp_tool, MagicMock())
        schema = adapter.create_schema()

        assert isinstance(schema, MCPToolSchema)
        assert schema.name == "mcp__srv__read_file"
        assert schema.description == "Read a file"
        assert schema.inputSchema == mcp_tool.inputSchema


class TestExecute:
    @pytest.mark.asyncio
    async def test_success(self):
        group = MagicMock()
        group.call_tool = AsyncMock(
            return_value=_make_call_result(texts=["hello world"])
        )
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), group)
        tc = _tool_call()

        result = await adapter.execute(tc)

        group.call_tool.assert_awaited_once_with("read_file", {"path": "/tmp/x"})
        assert result.status == ToolStatus.SUCCESS
        assert result.result == "hello world"
        assert result.error is None
        assert result.execution_time is not None
        assert result.execution_time >= 0

    @pytest.mark.asyncio
    async def test_error_result(self):
        group = MagicMock()
        group.call_tool = AsyncMock(
            return_value=_make_call_result(texts=["something broke"], is_error=True)
        )
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), group)

        result = await adapter.execute(_tool_call())

        assert result.status == ToolStatus.ERROR
        assert result.error == "something broke"

    @pytest.mark.asyncio
    async def test_structured_content(self):
        sc = {"rows": [{"a": 1}]}
        group = MagicMock()
        group.call_tool = AsyncMock(
            return_value=_make_call_result(texts=["ok"], structured_content=sc)
        )
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), group)

        result = await adapter.execute(_tool_call())

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"text": "ok", "structuredContent": sc}

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        group = MagicMock()
        group.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), group)

        result = await adapter.execute(_tool_call())

        assert result.status == ToolStatus.ERROR
        assert "connection lost" in result.error
        assert result.execution_time is not None

    @pytest.mark.asyncio
    async def test_multiple_content_blocks_joined(self):
        group = MagicMock()
        group.call_tool = AsyncMock(
            return_value=_make_call_result(texts=["line1", "line2", "line3"])
        )
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), group)

        result = await adapter.execute(_tool_call())

        assert result.result == "line1\nline2\nline3"
