"""Tests for BaseAgent MCP integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)

_MCP_CONFIG_MODULE = "obelix.adapters.outbound.mcp.config"
_MCP_MANAGER_MODULE = "obelix.adapters.outbound.mcp.manager"


class FakeTool:
    """Minimal tool satisfying the Tool protocol."""

    def __init__(self, name="fake"):
        self.tool_name = name
        self.tool_description = "Fake"
        self.is_deferred = False
        self.read_only = None
        self.destructive = None
        self.idempotent = None
        self.open_world = None

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            result="ok",
            status=ToolStatus.SUCCESS,
        )

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema={"type": "object", "properties": {}},
        )


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.model_id = "test-model"
    provider.provider_type = "test"
    return provider


class TestBaseAgentMCPConfig:
    def test_no_mcp_config_by_default(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        assert agent._mcp_manager is None

    def test_mcp_config_creates_manager(self, mock_provider):
        with (
            patch(f"{_MCP_CONFIG_MODULE}.parse_mcp_config") as mock_parse,
            patch(f"{_MCP_MANAGER_MODULE}.MCPManager"),
        ):
            mock_parse.return_value = [MagicMock()]
            agent = BaseAgent(  # noqa: F841
                system_message="test",
                provider=mock_provider,
                mcp_config="fake.json",
            )
            mock_parse.assert_called_once_with("fake.json")
            assert agent._mcp_manager is not None


class TestBaseAgentMCPContextManager:
    @pytest.mark.asyncio
    async def test_aenter_connects_and_registers_tools(self, mock_provider):
        fake_tool = FakeTool("mcp__srv__search")
        mock_manager = MagicMock()
        mock_manager.connect = AsyncMock(return_value=[fake_tool])
        mock_manager.is_connected.return_value = False

        with (
            patch(f"{_MCP_CONFIG_MODULE}.parse_mcp_config", return_value=[MagicMock()]),
            patch(f"{_MCP_MANAGER_MODULE}.MCPManager", return_value=mock_manager),
        ):
            agent = BaseAgent(
                system_message="test",
                provider=mock_provider,
                mcp_config="fake.json",
            )

        async with agent:
            assert any(
                t.tool_name == "mcp__srv__search" for t in agent.registered_tools
            )
            mock_manager.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_disconnects(self, mock_provider):
        mock_manager = MagicMock()
        mock_manager.connect = AsyncMock(return_value=[])
        mock_manager.disconnect = AsyncMock()
        mock_manager.is_connected.return_value = True

        with (
            patch(f"{_MCP_CONFIG_MODULE}.parse_mcp_config", return_value=[MagicMock()]),
            patch(f"{_MCP_MANAGER_MODULE}.MCPManager", return_value=mock_manager),
        ):
            agent = BaseAgent(
                system_message="test",
                provider=mock_provider,
                mcp_config="fake.json",
            )

        async with agent:
            pass
        mock_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_aenter_without_mcp_is_noop(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        async with agent:
            assert agent._mcp_manager is None


class TestBaseAgentMCPWarning:
    def test_mcp_config_logs_info(self, mock_provider):
        """Verify MCP config init logs info (loguru-based, check via sink)."""
        from loguru import logger as loguru_logger

        messages: list[str] = []
        handler_id = loguru_logger.add(lambda m: messages.append(m.record["message"]))
        try:
            with (
                patch(
                    f"{_MCP_CONFIG_MODULE}.parse_mcp_config", return_value=[MagicMock()]
                ),
                patch(f"{_MCP_MANAGER_MODULE}.MCPManager"),
            ):
                BaseAgent(
                    system_message="test",
                    provider=mock_provider,
                    mcp_config="fake.json",
                )
            assert any("MCP config detected" in m for m in messages)
        finally:
            loguru_logger.remove(handler_id)
