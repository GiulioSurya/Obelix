"""MCPManager — adapter implementing AbstractMCPProvider.

Thin wrapper over the MCP SDK's ClientSessionGroup. Handles:
- Converting MCPServerConfig to SDK transport parameters
- Wrapping discovered tools in MCPToolAdapter
- Storing discovered resources for future skill bridge
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.adapters.outbound.mcp.resource_adapter import MCPResourceAdapter
from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter
from obelix.core.tool.tool_base import Tool
from obelix.infrastructure.logging import get_logger
from obelix.ports.outbound.mcp_provider import AbstractMCPProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class MCPPrompt:
    """Lightweight DTO: a prompt offered by an MCP server.

    Produced by MCPManager.list_prompts() and consumed by MCPSkillProvider.
    `arguments` holds the SDK-provided argument descriptors (typically
    `mcp.types.PromptArgument` — only the `.name` attribute is read).
    `template` is the prompt body when materialized; `None` means the
    SDK exposed the prompt but didn't materialize its template (v1: always None).
    """

    name: str
    description: str
    arguments: list[Any]
    server_name: str
    template: str | None = None


class MCPManager(AbstractMCPProvider):
    """MCP client manager using the official SDK's ClientSessionGroup."""

    def __init__(self, config: list[MCPServerConfig]):
        self._config = config
        self._group: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._connected = False
        self._resources: dict[str, list[MCPResourceAdapter]] = {}

    async def connect(self) -> list[Tool]:
        """Connect to all configured servers and return discovered tools."""
        if self._connected:
            logger.warning("MCPManager already connected — skipping")
            return []

        group = self._create_group()
        self._exit_stack = AsyncExitStack()
        self._group = await self._exit_stack.enter_async_context(group)

        # Connect each server
        for cfg in self._config:
            params = _config_to_params(cfg)
            logger.info(f"Connecting to MCP server '{cfg.name}' ({cfg.transport})")
            try:
                await self._group.connect_to_server(params)
                logger.info(f"Connected to MCP server '{cfg.name}'")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{cfg.name}': {e}")
                raise

        # Wrap discovered tools
        tools: list[Tool] = []
        for tool_name, mcp_tool in self._group.tools.items():
            server_name = self._resolve_server_name(tool_name)
            adapter = MCPToolAdapter(server_name, mcp_tool, self._group)
            tools.append(adapter)
            logger.info(f"Discovered MCP tool: {adapter.tool_name}")

        # Store discovered resources
        for resource_name, mcp_resource in self._group.resources.items():
            server_name = self._resolve_server_name(resource_name)
            adapter = MCPResourceAdapter(server_name, mcp_resource)
            self._resources.setdefault(server_name, []).append(adapter)

        self._connected = True
        logger.info(
            f"MCP connected: {len(tools)} tool(s), "
            f"{sum(len(v) for v in self._resources.values())} resource(s)"
        )
        return tools

    async def disconnect(self) -> None:
        """Close all MCP server connections."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._group = None
        self._connected = False
        self._resources = {}
        logger.info("MCP disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def list_prompts(self) -> list[MCPPrompt]:
        """Return all prompts exposed by connected servers.

        Returns [] when not connected, when _group is missing, or when the
        SDK session group does not expose a prompts attribute.
        """
        if not self._connected or self._group is None:
            return []
        prompts_dict = getattr(self._group, "prompts", None) or {}
        out: list[MCPPrompt] = []
        for name, prompt in prompts_dict.items():
            server_name = self._resolve_server_name(name)
            # `None` and missing attribute both coerce to the respective empty default.
            description = getattr(prompt, "description", None) or ""
            arguments = list(getattr(prompt, "arguments", None) or [])
            out.append(
                MCPPrompt(
                    name=name,
                    description=description,
                    arguments=arguments,
                    server_name=server_name,
                )
            )
        return out

    def get_resources(self) -> dict[str, list[MCPResourceAdapter]]:
        return self._resources

    def _create_group(self) -> Any:
        """Create a ClientSessionGroup. Separated for testability."""
        from mcp import ClientSessionGroup

        return ClientSessionGroup()

    def _resolve_server_name(self, tool_or_resource_name: str) -> str:
        """Resolve which server a tool/resource came from.

        ClientSessionGroup may prefix names. For now, if we have a
        single server, use its name. With multiple, use first as fallback.
        """
        if len(self._config) == 1:
            return self._config[0].name
        return self._config[0].name


def _config_to_params(cfg: MCPServerConfig) -> Any:
    """Convert MCPServerConfig to SDK transport parameters."""
    if cfg.transport == "stdio":
        from mcp.client.session_group import StdioServerParameters

        return StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env or None,
        )
    elif cfg.transport == "streamable-http":
        from mcp.client.session_group import StreamableHttpParameters

        return StreamableHttpParameters(url=cfg.url)
    else:
        raise ValueError(f"Unsupported transport: {cfg.transport}")
