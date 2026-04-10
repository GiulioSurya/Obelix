# MCP Client Redesign — Design Spec

**Date**: 2026-04-10
**Status**: Approved
**SDK Version**: mcp >= 1.25, < 2 (current: 1.27.0)

---

## Goal

Rewrite the MCP client in Obelix from scratch. The old implementation (`plugins/mcp/`) is 6 months old, poorly structured, and not integrated with BaseAgent. The new implementation leverages the official MCP Python SDK's `ClientSessionGroup` for multi-server management and follows Obelix's hexagonal architecture.

## Scope

- **In scope**: MCP tool discovery + execution, MCP resource discovery (for future skill bridge), config parsing (`.mcp.json` + programmatic), BaseAgent integration with lifecycle management, Tool protocol annotations
- **Out of scope**: MCP prompts, MCP skill bridge (future), authentication/OAuth

## Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Capabilities | Tool + resources, no prompts | Prompts rarely used; resources needed for future skill bridge |
| 2 | Annotations | On Obelix Tool protocol (optional) | Coherence between native and MCP tools; enables generic policies |
| 3 | Code location | `adapters/outbound/mcp/` | Hexagonal: MCP is an outbound adapter to external services |
| 4 | Config passing | `mcp_config` param accepts str/Path/MCPServerConfig/list | File for common case, programmatic for tests/advanced |
| 5 | Lifecycle | `async with` optional, lazy connect by default with warning | Zero breaking change, control for those who want it |
| 6 | Tool naming | `mcp__{server}__{tool}` always | Predictable, no collisions, same as Claude Code |
| 7 | Old code | Delete `plugins/mcp/` entirely | Clean break, no users in production |
| 8 | Deferred | MCP tools always `is_deferred=False` | MCP tools execute on their server, not client-side |
| 9 | Cohesion | MCPManager separate from BaseAgent | BaseAgent imports interface only, no MCP SDK knowledge |

---

## Architecture

### File Structure

```
ports/outbound/
  mcp_provider.py              # AbstractMCPProvider (ABC)

adapters/outbound/mcp/
  __init__.py                  # Public exports
  config.py                    # MCPServerConfig dataclass, parse_mcp_config()
  manager.py                   # MCPManager implements AbstractMCPProvider
  tool_adapter.py              # MCPToolAdapter: mcp.types.Tool -> Obelix Tool protocol
  resource_adapter.py          # MCPResourceAdapter (future skill bridge placeholder)

core/tool/
  tool_base.py                 # Tool protocol updated with annotations

core/agent/
  base_agent.py                # +mcp_config, +__aenter__/__aexit__, +lazy connect
```

### Responsibility Matrix

| File | Knows | Does NOT Know |
|------|-------|---------------|
| `mcp_provider.py` | Contract: connect -> tools, disconnect | SDK, transports, config format |
| `config.py` | How to parse `.mcp.json` and MCPServerConfig | Connections, tools, SDK |
| `manager.py` | ClientSessionGroup, transports, discovery | BaseAgent, how tools are used |
| `tool_adapter.py` | How to convert mcp.types.Tool -> Tool protocol | Servers, connections |
| `resource_adapter.py` | How to convert mcp.types.Resource -> internal format | Servers, connections |
| `base_agent.py` | That MCPManager has connect/disconnect | MCP SDK, transports, config parsing |

---

## Component Details

### 1. Port — AbstractMCPProvider

```python
# ports/outbound/mcp_provider.py

from abc import ABC, abstractmethod
from typing import Any
from obelix.core.tool.tool_base import Tool

class AbstractMCPProvider(ABC):
    @abstractmethod
    async def connect(self) -> list[Tool]:
        """Connect to all configured MCP servers.
        Returns discovered tools converted to Obelix Tool protocol."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close all MCP server connections."""
        ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_resources(self) -> dict[str, Any]:
        """Return discovered resources keyed by server name.
        Placeholder for future skill bridge."""
        ...
```

### 2. Config

```python
# adapters/outbound/mcp/config.py

@dataclass
class MCPServerConfig:
    name: str
    transport: str  # "stdio" | "streamable-http"

    # stdio fields
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # streamable-http fields
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def parse_mcp_config(
    config: str | Path | MCPServerConfig | list[str | Path | MCPServerConfig],
) -> list[MCPServerConfig]:
    """Resolve any mix of file paths and config objects into a flat list."""
    # - str/Path: read JSON file, parse each entry under "mcpServers"
    # - MCPServerConfig: use directly
    # - Supports ${ENV_VAR} substitution in string values
```

**.mcp.json format** (standard MCP client config):
```json
{
  "mcpServers": {
    "server-name": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "my_server"],
      "env": {"KEY": "value"}
    }
  }
}
```

### 3. MCPManager

```python
# adapters/outbound/mcp/manager.py

class MCPManager(AbstractMCPProvider):
    def __init__(self, config: list[MCPServerConfig]):
        self._config = config
        self._group: ClientSessionGroup | None = None
        self._connected = False
        self._resources: dict[str, list] = {}

    async def connect(self) -> list[Tool]:
        # 1. Create ClientSessionGroup (SDK)
        # 2. For each MCPServerConfig:
        #    - stdio -> StdioServerParameters(command, args, env)
        #    - streamable-http -> StreamableHttpParameters(url)
        #    - await group.connect_to_server(params)
        # 3. Iterate group.tools -> wrap each in MCPToolAdapter
        # 4. Store group.resources in self._resources
        # 5. Return list of MCPToolAdapter

    async def disconnect(self) -> None:
        # Close ClientSessionGroup

    def is_connected(self) -> bool:
        return self._connected

    def get_resources(self) -> dict[str, list]:
        return self._resources
```

### 4. MCPToolAdapter

```python
# adapters/outbound/mcp/tool_adapter.py

class MCPToolAdapter:
    """Wraps an MCP SDK Tool to satisfy Obelix Tool protocol."""

    def __init__(self, server_name: str, mcp_tool: mcp.types.Tool, group: ClientSessionGroup):
        self.tool_name = f"mcp__{server_name}__{mcp_tool.name}"
        self.tool_description = mcp_tool.description or ""
        self.is_deferred = False
        self._mcp_tool = mcp_tool
        self._original_name = mcp_tool.name  # for calling the server
        self._group = group

        # Annotations (optional)
        ann = mcp_tool.annotations
        self.read_only = getattr(ann, 'readOnlyHint', None)
        self.destructive = getattr(ann, 'destructiveHint', None)
        self.idempotent = getattr(ann, 'idempotentHint', None)
        self.open_world = getattr(ann, 'openWorldHint', None)

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            input_schema=self._mcp_tool.inputSchema,
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        # 1. Call group.call_tool(self._original_name, tool_call.arguments)
        # 2. Convert CallToolResult -> ToolResult:
        #    - Extract text from result.content blocks
        #    - result.isError -> ToolStatus.ERROR
        #    - result.structuredContent -> include in result metadata
```

### 5. Tool Protocol Update

```python
# core/tool/tool_base.py

@runtime_checkable
class Tool(Protocol):
    tool_name: str
    tool_description: str
    is_deferred: bool

    # New — optional annotations, default None
    read_only: bool | None
    destructive: bool | None
    idempotent: bool | None
    open_world: bool | None

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
    def create_schema(self) -> MCPToolSchema: ...
```

The `@tool` decorator gains optional annotation parameters:
```python
@tool(name="search", description="...", read_only=True, open_world=False)
```

All annotations default to `None` (not declared) for existing tools — zero breaking change.

### 6. BaseAgent Changes

**Constructor** — new `mcp_config` parameter:
```python
def __init__(self, ..., mcp_config=None):
    self._mcp_manager: AbstractMCPProvider | None = None
    if mcp_config:
        from obelix.adapters.outbound.mcp.config import parse_mcp_config
        from obelix.adapters.outbound.mcp.manager import MCPManager
        configs = parse_mcp_config(mcp_config)
        self._mcp_manager = MCPManager(configs)
        logger.info(
            f"[{self.__class__.__name__}] MCP config detected with "
            f"{len(configs)} server(s). Use 'async with agent:' for "
            "persistent connections."
        )
```

**Context manager**:
```python
async def __aenter__(self):
    if self._mcp_manager:
        tools = await self._mcp_manager.connect()
        for tool in tools:
            self.register_tool(tool)
    return self

async def __aexit__(self, *exc):
    if self._mcp_manager and self._mcp_manager.is_connected():
        await self._mcp_manager.disconnect()
```

**Lazy connect** in `_async_execute_query()`:
```python
# Before LLM loop starts:
if self._mcp_manager and not self._mcp_manager.is_connected():
    logger.warning(
        f"[{self.__class__.__name__}] MCP lazy connect — "
        "use 'async with agent:' for persistent connections."
    )
    tools = await self._mcp_manager.connect()
    for tool in tools:
        self.register_tool(tool)
    self._lazy_mcp = True

# In finally block:
if getattr(self, '_lazy_mcp', False):
    await self._mcp_manager.disconnect()
    self._lazy_mcp = False
```

**Remove MCPTool check** from `register_tool()` — delete the `isinstance(tool, MCPTool)` block with conditional import. The method accepts any `Tool` protocol, period.

---

## Deletions

Delete entirely:
- `src/obelix/plugins/mcp/mcp_client_manager.py`
- `src/obelix/plugins/mcp/mcp_tool.py`
- `src/obelix/plugins/mcp/mcp_validator.py`
- `src/obelix/plugins/mcp/run_time_manager.py`
- `src/obelix/plugins/mcp/__init__.py`

---

## Usage Examples

### Simple (lazy, syncrono)
```python
agent = BaseAgent(
    system_message="You have access to external tools.",
    provider=provider,
    mcp_config=".mcp.json",
)
response = agent.execute_query("What traces do we have?")
```

### Explicit lifecycle (async, production)
```python
async with BaseAgent(
    system_message="You have access to external tools.",
    provider=provider,
    mcp_config=".mcp.json",
) as agent:
    r1 = await agent.async_execute_query("What traces do we have?")
    r2 = await agent.async_execute_query("Show me the latest one")
```

### Programmatic config
```python
from obelix.adapters.outbound.mcp.config import MCPServerConfig

agent = BaseAgent(
    system_message="...",
    provider=provider,
    mcp_config=[
        MCPServerConfig(name="tracer", transport="stdio",
                        command="uv", args=["run", "python", "-m", "obelix_tracer_mcp"]),
        MCPServerConfig(name="api", transport="streamable-http",
                        url="http://localhost:8000/mcp"),
    ],
)
```

### Mixed config
```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    mcp_config=[".mcp.json", MCPServerConfig(name="extra", transport="stdio", command="...")],
)
```

---

## Dependencies

- `mcp>=1.25,<2` (already in pyproject.toml, update upper bound)
- No new dependencies — mcp SDK brings httpx, anyio, pydantic (all already used)

## Testing Strategy

- Unit tests for `parse_mcp_config()` — file parsing, env var substitution, mixed input
- Unit tests for `MCPToolAdapter` — schema conversion, annotation mapping, execute result conversion
- Integration test with a minimal stdio MCP server (test fixture)
- BaseAgent test: verify MCP tools appear in `registered_tools` after `__aenter__`
- BaseAgent test: verify lazy connect works and warning is logged
