"""MCP adapter — client for connecting to MCP servers."""

from obelix.adapters.outbound.mcp.config import MCPServerConfig, parse_mcp_config
from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter

__all__ = ["MCPServerConfig", "MCPToolAdapter", "parse_mcp_config"]
