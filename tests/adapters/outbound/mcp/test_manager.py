"""Tests for MCPManager — adapter implementing AbstractMCPProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.adapters.outbound.mcp.manager import MCPManager
from obelix.adapters.outbound.mcp.resource_adapter import MCPResourceAdapter
from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter


def _make_config(
    name: str = "test-server", transport: str = "stdio"
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=transport,
        command="node" if transport == "stdio" else None,
        args=["server.js"] if transport == "stdio" else [],
        url="http://localhost:8080" if transport == "streamable-http" else None,
    )


def _make_mock_group(
    tools: dict | None = None,
    resources: dict | None = None,
) -> MagicMock:
    """Create a mock ClientSessionGroup that works as an async context manager."""
    group = MagicMock()
    group.__aenter__ = AsyncMock(return_value=group)
    group.__aexit__ = AsyncMock(return_value=False)
    group.connect_to_server = AsyncMock()
    group.tools = tools or {}
    group.resources = resources or {}
    return group


def _make_mock_mcp_tool(
    name: str = "search", description: str = "Search tool"
) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {"type": "object", "properties": {"query": {"type": "string"}}}
    tool.annotations = None
    return tool


def _make_mock_mcp_resource(
    name: str = "readme",
    uri: str = "file:///readme.md",
    description: str = "Readme file",
) -> MagicMock:
    resource = MagicMock()
    resource.name = name
    resource.uri = uri
    resource.description = description
    resource.mimeType = "text/markdown"
    return resource


class TestMCPManagerInitialState:
    def test_initial_state(self):
        cfg = _make_config()
        manager = MCPManager(config=[cfg])

        assert manager.is_connected() is False
        assert manager.get_resources() == {}


class TestMCPManagerConnect:
    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self):
        cfg = _make_config()
        mock_tool = _make_mock_mcp_tool("search", "Search the web")
        mock_group = _make_mock_group(tools={"search": mock_tool})

        manager = MCPManager(config=[cfg])
        with patch.object(manager, "_create_group", return_value=mock_group):
            tools = await manager.connect()

        assert len(tools) == 1
        assert isinstance(tools[0], MCPToolAdapter)
        assert tools[0].tool_name == "mcp__test-server__search"
        assert manager.is_connected() is True
        mock_group.connect_to_server.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_discovers_resources(self):
        cfg = _make_config()
        mock_resource = _make_mock_mcp_resource()
        mock_group = _make_mock_group(resources={"readme": mock_resource})

        manager = MCPManager(config=[cfg])
        with patch.object(manager, "_create_group", return_value=mock_group):
            await manager.connect()

        resources = manager.get_resources()
        assert "test-server" in resources
        assert len(resources["test-server"]) == 1
        assert isinstance(resources["test-server"][0], MCPResourceAdapter)
        assert resources["test-server"][0].name == "readme"

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self):
        cfg = _make_config()
        mock_group = _make_mock_group()

        manager = MCPManager(config=[cfg])
        with patch.object(manager, "_create_group", return_value=mock_group):
            await manager.connect()
            # Second connect should return empty and skip
            tools = await manager.connect()

        assert tools == []
        assert manager.is_connected() is True

    @pytest.mark.asyncio
    async def test_connect_multiple_servers(self):
        cfg1 = _make_config("server-a")
        cfg2 = _make_config("server-b")
        tool_a = _make_mock_mcp_tool("tool_a", "Tool A")
        tool_b = _make_mock_mcp_tool("tool_b", "Tool B")
        mock_group = _make_mock_group(
            tools={"tool_a": tool_a, "tool_b": tool_b},
        )

        manager = MCPManager(config=[cfg1, cfg2])
        with patch.object(manager, "_create_group", return_value=mock_group):
            tools = await manager.connect()

        assert len(tools) == 2
        assert mock_group.connect_to_server.await_count == 2

    @pytest.mark.asyncio
    async def test_connect_server_failure_propagates(self):
        cfg = _make_config()
        mock_group = _make_mock_group()
        mock_group.connect_to_server = AsyncMock(side_effect=ConnectionError("refused"))

        manager = MCPManager(config=[cfg])
        with (
            patch.object(manager, "_create_group", return_value=mock_group),
            pytest.raises(ConnectionError, match="refused"),
        ):
            await manager.connect()


class TestMCPManagerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self):
        cfg = _make_config()
        mock_group = _make_mock_group()

        manager = MCPManager(config=[cfg])
        with patch.object(manager, "_create_group", return_value=mock_group):
            await manager.connect()
            assert manager.is_connected() is True

            await manager.disconnect()
            assert manager.is_connected() is False
            assert manager.get_resources() == {}

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        cfg = _make_config()
        manager = MCPManager(config=[cfg])
        # Should not raise
        await manager.disconnect()
        assert manager.is_connected() is False
