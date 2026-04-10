# MCP Client Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the MCP client from scratch, integrating it with BaseAgent via hexagonal architecture and the official MCP Python SDK's `ClientSessionGroup`.

**Architecture:** Port ABC in `ports/outbound/`, adapter in `adapters/outbound/mcp/` using SDK's `ClientSessionGroup` for multi-server management, `MCPToolAdapter` wraps MCP tools into Obelix's `Tool` protocol, BaseAgent gains `mcp_config` parameter with optional `async with` lifecycle.

**Tech Stack:** mcp SDK >= 1.25 < 2, Python 3.13, pydantic, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-10-mcp-client-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/obelix/core/tool/tool_base.py` | Modify | Add annotation fields to Tool protocol |
| `src/obelix/core/tool/tool_decorator.py` | Modify | Add annotation params to @tool decorator |
| `src/obelix/ports/outbound/mcp_provider.py` | Create | AbstractMCPProvider ABC |
| `src/obelix/adapters/outbound/mcp/__init__.py` | Create | Public exports |
| `src/obelix/adapters/outbound/mcp/config.py` | Create | MCPServerConfig + parse_mcp_config() |
| `src/obelix/adapters/outbound/mcp/tool_adapter.py` | Create | MCPToolAdapter wrapping MCP tools |
| `src/obelix/adapters/outbound/mcp/resource_adapter.py` | Create | MCPResourceAdapter placeholder |
| `src/obelix/adapters/outbound/mcp/manager.py` | Create | MCPManager implements AbstractMCPProvider |
| `src/obelix/core/agent/base_agent.py` | Modify | +mcp_config, +__aenter__/__aexit__, +lazy connect, remove MCPTool check |
| `src/obelix/plugins/mcp/` | Delete | Old MCP implementation (entire directory) |
| `tests/core/tool/test_tool_annotations.py` | Create | Tests for annotation fields |
| `tests/adapters/outbound/mcp/test_config.py` | Create | Tests for config parsing |
| `tests/adapters/outbound/mcp/test_tool_adapter.py` | Create | Tests for tool adapter |
| `tests/adapters/outbound/mcp/test_manager.py` | Create | Tests for MCPManager |
| `tests/core/agent/test_base_agent_mcp.py` | Create | Tests for BaseAgent MCP integration |

---

## Task 1: Add annotations to Tool protocol

**Files:**
- Modify: `src/obelix/core/tool/tool_base.py:31-39`
- Create: `tests/core/tool/test_tool_annotations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/tool/test_tool_annotations.py
"""Tests for Tool protocol annotation fields."""
import pytest
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult, ToolStatus


class FakeToolWithAnnotations:
    """A tool that declares annotations."""

    tool_name = "annotated_tool"
    tool_description = "A tool with annotations"
    is_deferred = False
    read_only = True
    destructive = False
    idempotent = True
    open_world = False

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


class FakeToolWithoutAnnotations:
    """A tool that does NOT declare annotations (backwards compat)."""

    tool_name = "plain_tool"
    tool_description = "A plain tool"
    is_deferred = False

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


def test_tool_with_annotations_satisfies_protocol():
    from obelix.core.tool.tool_base import Tool

    tool = FakeToolWithAnnotations()
    assert isinstance(tool, Tool)
    assert tool.read_only is True
    assert tool.destructive is False
    assert tool.idempotent is True
    assert tool.open_world is False


def test_tool_without_annotations_does_not_satisfy_protocol():
    """Tools without annotation fields should NOT satisfy the updated protocol."""
    from obelix.core.tool.tool_base import Tool

    tool = FakeToolWithoutAnnotations()
    # This will fail until we make annotations optional in the protocol
    # or add defaults. We need to decide: does the protocol require them?
    # Per spec: annotations are OPTIONAL with default None.
    # runtime_checkable Protocol checks only method signatures, not attributes.
    # So this will still pass isinstance check.
    assert isinstance(tool, Tool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/tool/test_tool_annotations.py -v`
Expected: FAIL — `Tool` protocol doesn't have `read_only`, `destructive`, `idempotent`, `open_world` yet.

- [ ] **Step 3: Update Tool protocol with annotations**

Replace content of `src/obelix/core/tool/tool_base.py`:

```python
# src/tools/tool_base.py
"""
Tool Protocol - structural typing contract for all tools.

All tools must satisfy this protocol:
- tool_name: str
- tool_description: str
- execute(tool_call: ToolCall) -> ToolResult
- create_schema() -> MCPToolSchema

Optional annotation fields (default None):
- read_only: bool | None — no side effects
- destructive: bool | None — irreversible modifications
- idempotent: bool | None — repeatable without additional effect
- open_world: bool | None — interacts with external systems

The @tool decorator is the primary way to create tools:
    from obelix.core.tool.tool_decorator import tool

    @tool(name="my_tool", description="Description")
    class MyTool:
        param: str = Field(...)

        def execute(self) -> dict:  # can be sync or async
            return {"result": self.param}

The decorator wraps execute() to match the Protocol signature automatically.
MCPTool and SubAgentWrapper satisfy the Protocol via structural typing.
"""

from typing import Protocol, runtime_checkable

from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Contract that all tools must satisfy (structural typing)."""

    tool_name: str
    tool_description: str
    is_deferred: bool

    # Behavioral annotations (optional, default None = not declared)
    read_only: bool | None
    destructive: bool | None
    idempotent: bool | None
    open_world: bool | None

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
    def create_schema(self) -> MCPToolSchema: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/tool/test_tool_annotations.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All existing tests pass. If any `isinstance(x, Tool)` checks fail because existing tools lack annotation attributes, we fix them in the next step.

- [ ] **Step 6: Add annotation defaults to @tool decorator**

In `src/obelix/core/tool/tool_decorator.py`, update the `tool()` function signature and the decorator body.

Change the function signature (line 42):
```python
def tool(
    name: str = None,
    description: str = None,
    is_deferred: bool = False,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
):
```

Inside `def decorator(cls: type) -> type:`, after line 76 (`cls.is_deferred = is_deferred`), add:
```python
        cls.read_only = read_only
        cls.destructive = destructive
        cls.idempotent = idempotent
        cls.open_world = open_world
```

- [ ] **Step 7: Run full test suite again**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/obelix/core/tool/tool_base.py src/obelix/core/tool/tool_decorator.py tests/core/tool/test_tool_annotations.py
git commit -m "feat(tool): add behavioral annotations to Tool protocol and @tool decorator"
```

---

## Task 2: Delete old MCP plugin

**Files:**
- Delete: `src/obelix/plugins/mcp/` (entire directory)
- Modify: `src/obelix/core/agent/base_agent.py:120-139` (remove MCPTool check)

- [ ] **Step 1: Remove MCPTool check from BaseAgent.register_tool()**

In `src/obelix/core/agent/base_agent.py`, replace the `register_tool` method (lines 120-156) with:

```python
    def register_tool(self, tool: Tool):
        """
        Registers a tool for the agent.
        """
        if tool not in self.registered_tools:
            self.registered_tools.append(tool)
            tool_name = getattr(tool, "tool_name", None) or tool.__class__.__name__
            logger.info(
                f"[{self.__class__.__name__}] Tool registered | tool={tool_name}"
            )

            # Auto-inject system prompt fragment if the tool provides one
            fragment_fn = getattr(tool, "system_prompt_fragment", None)
            if callable(fragment_fn):
                fragment = fragment_fn()
                if fragment:
                    self.system_message.content += fragment
                    logger.info(
                        f"[{self.__class__.__name__}] System prompt enriched by tool={tool_name}"
                    )
```

- [ ] **Step 2: Delete the old MCP plugin directory**

```bash
rm -rf src/obelix/plugins/mcp/
```

- [ ] **Step 3: Check for remaining references to old MCP imports**

Search for `from obelix.plugins.mcp` in the codebase. The only remaining reference should be in `skills/creating-agents/SKILL.md` (documentation) — update that later when we update docs.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS. If any test imported from `obelix.plugins.mcp`, it will fail — delete or update those tests.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor(mcp): delete old plugins/mcp implementation, clean register_tool()"
```

---

## Task 3: Create AbstractMCPProvider port

**Files:**
- Create: `src/obelix/ports/outbound/mcp_provider.py`

- [ ] **Step 1: Create the ABC**

```python
# src/obelix/ports/outbound/mcp_provider.py
"""
Abstract MCP Provider — contract for MCP client implementations.

Follows the same pattern as AbstractLLMProvider: the port defines
the contract, the adapter (adapters/outbound/mcp/) implements it.
"""

from abc import ABC, abstractmethod
from typing import Any

from obelix.core.tool.tool_base import Tool


class AbstractMCPProvider(ABC):
    """Contract for MCP client implementations."""

    @abstractmethod
    async def connect(self) -> list[Tool]:
        """Connect to all configured MCP servers.

        Discovers tools on each server and returns them converted
        to the Obelix Tool protocol, ready for BaseAgent.register_tool().

        Returns:
            List of Tool-protocol-compatible objects.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close all MCP server connections and release resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the provider currently has active connections."""
        ...

    @abstractmethod
    def get_resources(self) -> dict[str, Any]:
        """Return discovered MCP resources, keyed by server name.

        Placeholder for the future skill bridge. Returns an empty
        dict until resource consumption is implemented.
        """
        ...
```

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/obelix/ports/outbound/mcp_provider.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/obelix/ports/outbound/mcp_provider.py
git commit -m "feat(mcp): add AbstractMCPProvider port"
```

---

## Task 4: Create MCPServerConfig and parse_mcp_config()

**Files:**
- Create: `src/obelix/adapters/outbound/mcp/__init__.py`
- Create: `src/obelix/adapters/outbound/mcp/config.py`
- Create: `tests/adapters/outbound/mcp/__init__.py`
- Create: `tests/adapters/outbound/mcp/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/adapters/outbound/mcp/__init__.py
```

```python
# tests/adapters/outbound/mcp/test_config.py
"""Tests for MCP config parsing."""
import json
import os
from pathlib import Path

import pytest

from obelix.adapters.outbound.mcp.config import MCPServerConfig, parse_mcp_config


class TestMCPServerConfig:
    def test_stdio_config(self):
        config = MCPServerConfig(
            name="tracer",
            transport="stdio",
            command="uv",
            args=["run", "python", "-m", "obelix_tracer_mcp"],
        )
        assert config.name == "tracer"
        assert config.transport == "stdio"
        assert config.command == "uv"

    def test_http_config(self):
        config = MCPServerConfig(
            name="remote",
            transport="streamable-http",
            url="http://localhost:8000/mcp",
        )
        assert config.name == "remote"
        assert config.transport == "streamable-http"
        assert config.url == "http://localhost:8000/mcp"


class TestParseMcpConfig:
    def test_parse_single_config_object(self):
        config = MCPServerConfig(name="test", transport="stdio", command="echo")
        result = parse_mcp_config(config)
        assert len(result) == 1
        assert result[0].name == "test"

    def test_parse_list_of_config_objects(self):
        configs = [
            MCPServerConfig(name="a", transport="stdio", command="echo"),
            MCPServerConfig(name="b", transport="streamable-http", url="http://x"),
        ]
        result = parse_mcp_config(configs)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_parse_json_file(self, tmp_path):
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {
                "tracer": {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["run", "server"],
                    "env": {"KEY": "val"},
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer tok"},
                },
            }
        }))
        result = parse_mcp_config(str(mcp_json))
        assert len(result) == 2
        tracer = next(c for c in result if c.name == "tracer")
        assert tracer.transport == "stdio"
        assert tracer.command == "uv"
        assert tracer.args == ["run", "server"]
        assert tracer.env == {"KEY": "val"}
        remote = next(c for c in result if c.name == "remote")
        assert remote.transport == "streamable-http"
        assert remote.url == "http://localhost:8000/mcp"
        assert remote.headers == {"Authorization": "Bearer tok"}

    def test_parse_json_file_with_path_object(self, tmp_path):
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {"s": {"type": "stdio", "command": "echo"}}
        }))
        result = parse_mcp_config(Path(mcp_json))
        assert len(result) == 1

    def test_parse_mixed_list(self, tmp_path):
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {"from_file": {"type": "stdio", "command": "echo"}}
        }))
        result = parse_mcp_config([
            str(mcp_json),
            MCPServerConfig(name="inline", transport="stdio", command="echo"),
        ])
        assert len(result) == 2
        names = {c.name for c in result}
        assert names == {"from_file", "inline"}

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {
                "api": {
                    "type": "streamable-http",
                    "url": "http://localhost/mcp",
                    "headers": {"Authorization": "Bearer ${MY_TOKEN}"},
                }
            }
        }))
        result = parse_mcp_config(str(mcp_json))
        assert result[0].headers["Authorization"] == "Bearer secret123"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_mcp_config("/nonexistent/.mcp.json")

    def test_invalid_transport_raises(self):
        with pytest.raises(ValueError, match="transport"):
            parse_mcp_config(MCPServerConfig(name="bad", transport="ftp", command="x"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/outbound/mcp/test_config.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Create the __init__.py**

```python
# src/obelix/adapters/outbound/mcp/__init__.py
"""MCP adapter — client for connecting to MCP servers."""

from obelix.adapters.outbound.mcp.config import MCPServerConfig, parse_mcp_config

__all__ = ["MCPServerConfig", "parse_mcp_config"]
```

- [ ] **Step 4: Implement config.py**

```python
# src/obelix/adapters/outbound/mcp/config.py
"""MCP server configuration and .mcp.json parser."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

VALID_TRANSPORTS = {"stdio", "streamable-http"}


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

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
    """Resolve any mix of file paths and config objects into a flat list.

    Args:
        config: A single config, file path, or list mixing both.

    Returns:
        Flat list of validated MCPServerConfig.

    Raises:
        FileNotFoundError: If a config file path doesn't exist.
        ValueError: If a config has an invalid transport type.
    """
    if not isinstance(config, list):
        config = [config]

    result: list[MCPServerConfig] = []
    for item in config:
        if isinstance(item, MCPServerConfig):
            _validate_config(item)
            result.append(item)
        elif isinstance(item, (str, Path)):
            result.extend(_parse_json_file(Path(item)))
        else:
            raise TypeError(
                f"Expected str, Path, or MCPServerConfig, got {type(item).__name__}"
            )

    logger.info(f"Parsed {len(result)} MCP server config(s): {[c.name for c in result]}")
    return result


def _parse_json_file(path: Path) -> list[MCPServerConfig]:
    """Parse a .mcp.json file into a list of MCPServerConfig."""
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    servers = data.get("mcpServers", {})
    configs: list[MCPServerConfig] = []

    for name, server_data in servers.items():
        transport = server_data.get("type", "stdio")

        config = MCPServerConfig(
            name=name,
            transport=transport,
            command=server_data.get("command"),
            args=server_data.get("args", []),
            env=_substitute_env_vars(server_data.get("env", {})),
            url=_substitute_env_str(server_data.get("url")),
            headers=_substitute_env_vars(server_data.get("headers", {})),
        )
        _validate_config(config)
        configs.append(config)

    return configs


def _validate_config(config: MCPServerConfig) -> None:
    """Validate a config has required fields for its transport type."""
    if config.transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid transport '{config.transport}' for server '{config.name}'. "
            f"Valid: {VALID_TRANSPORTS}"
        )
    if config.transport == "stdio" and not config.command:
        raise ValueError(
            f"Server '{config.name}' with transport 'stdio' requires 'command'."
        )
    if config.transport == "streamable-http" and not config.url:
        raise ValueError(
            f"Server '{config.name}' with transport 'streamable-http' requires 'url'."
        )


def _substitute_env_str(value: str | None) -> str | None:
    """Replace ${VAR} with os.environ[VAR] in a string."""
    if value is None:
        return None
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _substitute_env_vars(d: dict[str, str]) -> dict[str, str]:
    """Replace ${VAR} in all dict values."""
    return {k: _substitute_env_str(v) or v for k, v in d.items()}
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/adapters/outbound/mcp/test_config.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/outbound/mcp/__init__.py src/obelix/adapters/outbound/mcp/config.py tests/adapters/outbound/mcp/__init__.py tests/adapters/outbound/mcp/test_config.py
git commit -m "feat(mcp): add MCPServerConfig and .mcp.json parser"
```

---

## Task 5: Create MCPToolAdapter

**Files:**
- Create: `src/obelix/adapters/outbound/mcp/tool_adapter.py`
- Create: `tests/adapters/outbound/mcp/test_tool_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/adapters/outbound/mcp/test_tool_adapter.py
"""Tests for MCPToolAdapter."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter
from obelix.core.model.tool_message import ToolCall, ToolStatus
from obelix.core.tool.tool_base import Tool


def _make_mcp_tool(
    name="search",
    description="Search things",
    input_schema=None,
    annotations=None,
):
    """Create a fake mcp.types.Tool-like object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {"type": "object", "properties": {"q": {"type": "string"}}}
    tool.annotations = annotations
    return tool


def _make_annotations(read_only=None, destructive=None, idempotent=None, open_world=None):
    ann = MagicMock()
    ann.readOnlyHint = read_only
    ann.destructiveHint = destructive
    ann.idempotentHint = idempotent
    ann.openWorldHint = open_world
    return ann


class TestMCPToolAdapterProperties:
    def test_tool_name_with_prefix(self):
        mcp_tool = _make_mcp_tool(name="search")
        group = MagicMock()
        adapter = MCPToolAdapter("tracer", mcp_tool, group)
        assert adapter.tool_name == "mcp__tracer__search"

    def test_tool_description(self):
        mcp_tool = _make_mcp_tool(description="Find stuff")
        adapter = MCPToolAdapter("srv", mcp_tool, MagicMock())
        assert adapter.tool_description == "Find stuff"

    def test_description_defaults_to_empty(self):
        mcp_tool = _make_mcp_tool(description=None)
        adapter = MCPToolAdapter("srv", mcp_tool, MagicMock())
        assert adapter.tool_description == ""

    def test_is_deferred_always_false(self):
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), MagicMock())
        assert adapter.is_deferred is False

    def test_satisfies_tool_protocol(self):
        adapter = MCPToolAdapter("srv", _make_mcp_tool(), MagicMock())
        assert isinstance(adapter, Tool)


class TestMCPToolAdapterAnnotations:
    def test_annotations_mapped(self):
        ann = _make_annotations(read_only=True, destructive=False, idempotent=True, open_world=False)
        adapter = MCPToolAdapter("srv", _make_mcp_tool(annotations=ann), MagicMock())
        assert adapter.read_only is True
        assert adapter.destructive is False
        assert adapter.idempotent is True
        assert adapter.open_world is False

    def test_no_annotations_defaults_to_none(self):
        adapter = MCPToolAdapter("srv", _make_mcp_tool(annotations=None), MagicMock())
        assert adapter.read_only is None
        assert adapter.destructive is None
        assert adapter.idempotent is None
        assert adapter.open_world is None


class TestMCPToolAdapterSchema:
    def test_create_schema(self):
        input_schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        mcp_tool = _make_mcp_tool(name="search", description="Search", input_schema=input_schema)
        adapter = MCPToolAdapter("tracer", mcp_tool, MagicMock())
        schema = adapter.create_schema()
        assert schema.name == "mcp__tracer__search"
        assert schema.description == "Search"
        assert schema.inputSchema == input_schema


class TestMCPToolAdapterExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        mcp_tool = _make_mcp_tool(name="search")
        group = MagicMock()
        # group.call_tool is async
        call_result = MagicMock()
        call_result.content = [MagicMock(text="found it", type="text")]
        call_result.isError = False
        call_result.structuredContent = None
        group.call_tool = AsyncMock(return_value=call_result)

        adapter = MCPToolAdapter("srv", mcp_tool, group)
        tool_call = ToolCall(id="tc_1", name="mcp__srv__search", arguments={"q": "hello"})
        result = await adapter.execute(tool_call)

        assert result.status == ToolStatus.SUCCESS
        assert "found it" in result.result
        group.call_tool.assert_called_once_with("search", {"q": "hello"})

    @pytest.mark.asyncio
    async def test_execute_error(self):
        mcp_tool = _make_mcp_tool(name="fail_tool")
        group = MagicMock()
        call_result = MagicMock()
        call_result.content = [MagicMock(text="something went wrong", type="text")]
        call_result.isError = True
        call_result.structuredContent = None
        group.call_tool = AsyncMock(return_value=call_result)

        adapter = MCPToolAdapter("srv", mcp_tool, group)
        tool_call = ToolCall(id="tc_2", name="mcp__srv__fail_tool", arguments={})
        result = await adapter.execute(tool_call)

        assert result.status == ToolStatus.ERROR

    @pytest.mark.asyncio
    async def test_execute_with_structured_content(self):
        mcp_tool = _make_mcp_tool(name="structured")
        group = MagicMock()
        call_result = MagicMock()
        call_result.content = [MagicMock(text="ok", type="text")]
        call_result.isError = False
        call_result.structuredContent = {"key": "value"}
        group.call_tool = AsyncMock(return_value=call_result)

        adapter = MCPToolAdapter("srv", mcp_tool, group)
        tool_call = ToolCall(id="tc_3", name="mcp__srv__structured", arguments={})
        result = await adapter.execute(tool_call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result["structuredContent"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        mcp_tool = _make_mcp_tool(name="crash")
        group = MagicMock()
        group.call_tool = AsyncMock(side_effect=Exception("connection lost"))

        adapter = MCPToolAdapter("srv", mcp_tool, group)
        tool_call = ToolCall(id="tc_4", name="mcp__srv__crash", arguments={})
        result = await adapter.execute(tool_call)

        assert result.status == ToolStatus.ERROR
        assert "connection lost" in result.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/outbound/mcp/test_tool_adapter.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement tool_adapter.py**

```python
# src/obelix/adapters/outbound/mcp/tool_adapter.py
"""MCPToolAdapter — wraps an MCP SDK Tool into the Obelix Tool protocol."""

import time
from typing import Any

from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MCPToolAdapter:
    """Wraps an MCP SDK Tool to satisfy the Obelix Tool protocol.

    Translates between MCP SDK types and Obelix types:
    - mcp.types.Tool → Tool protocol attributes
    - group.call_tool() → ToolResult
    """

    def __init__(self, server_name: str, mcp_tool: Any, group: Any):
        """
        Args:
            server_name: Name of the MCP server (used for prefixing).
            mcp_tool: An mcp.types.Tool object from the SDK.
            group: The ClientSessionGroup for calling tools.
        """
        self.tool_name = f"mcp__{server_name}__{mcp_tool.name}"
        self.tool_description = mcp_tool.description or ""
        self.is_deferred = False
        self._mcp_tool = mcp_tool
        self._original_name = mcp_tool.name
        self._server_name = server_name
        self._group = group

        # Behavioral annotations (optional)
        ann = mcp_tool.annotations
        self.read_only: bool | None = getattr(ann, "readOnlyHint", None)
        self.destructive: bool | None = getattr(ann, "destructiveHint", None)
        self.idempotent: bool | None = getattr(ann, "idempotentHint", None)
        self.open_world: bool | None = getattr(ann, "openWorldHint", None)

    def create_schema(self) -> MCPToolSchema:
        """Generate MCP schema from the discovered tool."""
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema=self._mcp_tool.inputSchema,
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute the tool via the MCP server.

        Calls the server using the original (unprefixed) tool name.
        """
        start_time = time.time()
        try:
            result = await self._group.call_tool(
                self._original_name, tool_call.arguments
            )

            elapsed = time.time() - start_time

            if result.isError:
                error_text = _extract_text(result.content)
                logger.warning(
                    f"MCP tool error | tool={self.tool_name} | error={error_text[:200]}"
                )
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=None,
                    status=ToolStatus.ERROR,
                    error=error_text,
                    execution_time=elapsed,
                )

            text = _extract_text(result.content)
            result_data: dict[str, Any] | str = text
            if result.structuredContent:
                result_data = {
                    "text": text,
                    "structuredContent": result.structuredContent,
                }

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=result_data,
                status=ToolStatus.SUCCESS,
                execution_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"MCP tool exception | tool={self.tool_name} | error={e}"
            )
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=elapsed,
            )


def _extract_text(content_blocks: list) -> str:
    """Extract text from MCP content blocks."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(str(block.data))
    return "\n".join(parts) if parts else ""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/adapters/outbound/mcp/test_tool_adapter.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/mcp/tool_adapter.py tests/adapters/outbound/mcp/test_tool_adapter.py
git commit -m "feat(mcp): add MCPToolAdapter wrapping MCP SDK tools to Obelix protocol"
```

---

## Task 6: Create MCPResourceAdapter placeholder

**Files:**
- Create: `src/obelix/adapters/outbound/mcp/resource_adapter.py`

- [ ] **Step 1: Create the placeholder**

```python
# src/obelix/adapters/outbound/mcp/resource_adapter.py
"""MCPResourceAdapter — placeholder for future MCP resource → skill bridge.

MCP servers can expose resources (files, data) that will be used
by the future skill system. This module will convert mcp.types.Resource
into an internal representation consumable by the skill loader.

Not implemented yet — resources are discovered and stored by MCPManager
but not consumed.
"""

from typing import Any

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MCPResourceAdapter:
    """Placeholder: converts MCP resources for future skill bridge."""

    def __init__(self, server_name: str, mcp_resource: Any):
        self.server_name = server_name
        self.uri = str(getattr(mcp_resource, "uri", ""))
        self.name = getattr(mcp_resource, "name", "")
        self.description = getattr(mcp_resource, "description", "")
        self.mime_type = getattr(mcp_resource, "mimeType", None)
```

- [ ] **Step 2: Commit**

```bash
git add src/obelix/adapters/outbound/mcp/resource_adapter.py
git commit -m "feat(mcp): add MCPResourceAdapter placeholder for future skill bridge"
```

---

## Task 7: Create MCPManager

**Files:**
- Create: `src/obelix/adapters/outbound/mcp/manager.py`
- Create: `tests/adapters/outbound/mcp/test_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/adapters/outbound/mcp/test_manager.py
"""Tests for MCPManager."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.adapters.outbound.mcp.manager import MCPManager
from obelix.core.tool.tool_base import Tool


def _make_mcp_tool(name="search", description="Search", annotations=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {"type": "object", "properties": {}}
    tool.annotations = annotations
    return tool


def _make_mcp_resource(name="readme", uri="file://readme.md"):
    resource = MagicMock()
    resource.name = name
    resource.uri = uri
    resource.description = "A resource"
    resource.mimeType = "text/markdown"
    return resource


class TestMCPManagerLifecycle:
    def test_initial_state(self):
        configs = [MCPServerConfig(name="test", transport="stdio", command="echo")]
        manager = MCPManager(configs)
        assert manager.is_connected() is False
        assert manager.get_resources() == {}

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self):
        configs = [MCPServerConfig(name="srv", transport="stdio", command="echo")]
        manager = MCPManager(configs)

        mock_group = MagicMock()
        mock_group.tools = {"search": _make_mcp_tool("search")}
        mock_group.resources = {}
        mock_group.connect_to_server = AsyncMock()

        with patch.object(manager, "_create_group", return_value=mock_group):
            tools = await manager.connect()

        assert len(tools) == 1
        assert tools[0].tool_name == "mcp__srv__search"
        assert isinstance(tools[0], Tool)
        assert manager.is_connected() is True

    @pytest.mark.asyncio
    async def test_connect_discovers_resources(self):
        configs = [MCPServerConfig(name="srv", transport="stdio", command="echo")]
        manager = MCPManager(configs)

        mock_group = MagicMock()
        mock_group.tools = {}
        mock_group.resources = {"readme": _make_mcp_resource()}
        mock_group.connect_to_server = AsyncMock()

        with patch.object(manager, "_create_group", return_value=mock_group):
            await manager.connect()

        resources = manager.get_resources()
        assert "srv" in resources
        assert len(resources["srv"]) == 1

    @pytest.mark.asyncio
    async def test_disconnect(self):
        configs = [MCPServerConfig(name="srv", transport="stdio", command="echo")]
        manager = MCPManager(configs)

        mock_group = MagicMock()
        mock_group.tools = {}
        mock_group.resources = {}
        mock_group.connect_to_server = AsyncMock()
        mock_group.__aexit__ = AsyncMock()

        with patch.object(manager, "_create_group", return_value=mock_group):
            await manager.connect()
            await manager.disconnect()

        assert manager.is_connected() is False

    @pytest.mark.asyncio
    async def test_connect_multiple_servers(self):
        configs = [
            MCPServerConfig(name="a", transport="stdio", command="echo"),
            MCPServerConfig(name="b", transport="stdio", command="echo"),
        ]
        manager = MCPManager(configs)

        tool_a = _make_mcp_tool("search")
        tool_b = _make_mcp_tool("query")
        mock_group = MagicMock()
        mock_group.tools = {"search": tool_a, "query": tool_b}
        mock_group.resources = {}
        mock_group.connect_to_server = AsyncMock()

        with patch.object(manager, "_create_group", return_value=mock_group):
            tools = await manager.connect()

        assert len(tools) == 2
        names = {t.tool_name for t in tools}
        # ClientSessionGroup aggregates tools — we can't know which server each came from
        # without inspecting the group's internals. For now, we use a generic server name.
        assert all("mcp__" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/outbound/mcp/test_manager.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement manager.py**

```python
# src/obelix/adapters/outbound/mcp/manager.py
"""MCPManager — adapter implementing AbstractMCPProvider.

Thin wrapper over the MCP SDK's ClientSessionGroup. Handles:
- Converting MCPServerConfig to SDK transport parameters
- Wrapping discovered tools in MCPToolAdapter
- Storing discovered resources for future skill bridge
"""

from contextlib import AsyncExitStack
from typing import Any

from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.adapters.outbound.mcp.resource_adapter import MCPResourceAdapter
from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter
from obelix.core.tool.tool_base import Tool
from obelix.infrastructure.logging import get_logger
from obelix.ports.outbound.mcp_provider import AbstractMCPProvider

logger = get_logger(__name__)


class MCPManager(AbstractMCPProvider):
    """MCP client manager using the official SDK's ClientSessionGroup."""

    def __init__(self, config: list[MCPServerConfig]):
        self._config = config
        self._group: Any = None  # ClientSessionGroup (lazy import)
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

    def get_resources(self) -> dict[str, list[MCPResourceAdapter]]:
        return self._resources

    def _create_group(self) -> Any:
        """Create a ClientSessionGroup. Separated for testability."""
        from mcp import ClientSessionGroup

        return ClientSessionGroup()

    def _resolve_server_name(self, tool_or_resource_name: str) -> str:
        """Resolve which server a tool/resource came from.

        ClientSessionGroup may prefix names with server info via
        component_name_hook. For now, if we have a single server,
        use its name. With multiple servers, use the first config name
        as fallback (the group doesn't expose server→tool mapping directly).
        """
        # TODO: improve server name resolution when SDK exposes
        # tool→server mapping. For now, use single server name or first.
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/adapters/outbound/mcp/test_manager.py -v`
Expected: All PASS.

- [ ] **Step 5: Update __init__.py exports**

Update `src/obelix/adapters/outbound/mcp/__init__.py`:

```python
# src/obelix/adapters/outbound/mcp/__init__.py
"""MCP adapter — client for connecting to MCP servers."""

from obelix.adapters.outbound.mcp.config import MCPServerConfig, parse_mcp_config
from obelix.adapters.outbound.mcp.manager import MCPManager
from obelix.adapters.outbound.mcp.tool_adapter import MCPToolAdapter

__all__ = ["MCPManager", "MCPServerConfig", "MCPToolAdapter", "parse_mcp_config"]
```

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/outbound/mcp/manager.py src/obelix/adapters/outbound/mcp/__init__.py tests/adapters/outbound/mcp/test_manager.py
git commit -m "feat(mcp): add MCPManager implementing AbstractMCPProvider"
```

---

## Task 8: Integrate MCP into BaseAgent

**Files:**
- Modify: `src/obelix/core/agent/base_agent.py`
- Create: `tests/core/agent/test_base_agent_mcp.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/agent/test_base_agent_mcp.py
"""Tests for BaseAgent MCP integration."""
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult, ToolStatus


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
            tool_name=tool_call.name, tool_call_id=tool_call.id,
            result="ok", status=ToolStatus.SUCCESS,
        )

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name=self.tool_name, description=self.tool_description,
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
        with patch("obelix.core.agent.base_agent.parse_mcp_config") as mock_parse, \
             patch("obelix.core.agent.base_agent.MCPManager") as mock_manager_cls:
            mock_parse.return_value = [MagicMock()]
            agent = BaseAgent(
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

        with patch("obelix.core.agent.base_agent.parse_mcp_config", return_value=[MagicMock()]), \
             patch("obelix.core.agent.base_agent.MCPManager", return_value=mock_manager):
            agent = BaseAgent(
                system_message="test",
                provider=mock_provider,
                mcp_config="fake.json",
            )

        async with agent:
            assert any(t.tool_name == "mcp__srv__search" for t in agent.registered_tools)
            mock_manager.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_disconnects(self, mock_provider):
        mock_manager = MagicMock()
        mock_manager.connect = AsyncMock(return_value=[])
        mock_manager.disconnect = AsyncMock()
        mock_manager.is_connected.return_value = True

        with patch("obelix.core.agent.base_agent.parse_mcp_config", return_value=[MagicMock()]), \
             patch("obelix.core.agent.base_agent.MCPManager", return_value=mock_manager):
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
    def test_mcp_config_logs_info(self, mock_provider, caplog):
        with patch("obelix.core.agent.base_agent.parse_mcp_config", return_value=[MagicMock()]), \
             patch("obelix.core.agent.base_agent.MCPManager"):
            with caplog.at_level(logging.INFO):
                agent = BaseAgent(
                    system_message="test",
                    provider=mock_provider,
                    mcp_config="fake.json",
                )
            assert "MCP config detected" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/agent/test_base_agent_mcp.py -v`
Expected: FAIL — BaseAgent doesn't accept `mcp_config` yet.

- [ ] **Step 3: Modify BaseAgent constructor**

In `src/obelix/core/agent/base_agent.py`, update the `__init__` signature (line 54-65) to add `mcp_config`:

```python
    def __init__(
        self,
        system_message: str,
        provider: AbstractLLMProvider,
        max_iterations: int = 15,
        tools: Tool | type[Tool] | list[type[Tool] | Tool] | None = None,
        tool_policy: list[ToolRequirement] | None = None,
        exit_on_success: list[str] | None = None,
        response_schema: type[BaseModel] | None = None,
        tracer: "Tracer | None" = None,
        planning: bool = False,
        mcp_config: "str | Path | MCPServerConfig | list | None" = None,
    ):
```

At the end of `__init__` (after line 117 `register_memory_hooks(self)`), add:

```python
        # MCP integration (optional)
        self._mcp_manager = None
        self._lazy_mcp = False
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

Add the `TYPE_CHECKING` import for type annotation at the top of the file (inside the existing `if TYPE_CHECKING:` block):

```python
if TYPE_CHECKING:
    from pathlib import Path

    from obelix.adapters.outbound.mcp.config import MCPServerConfig
    from obelix.core.agent.shared_memory import SharedMemoryGraph
    from obelix.core.tracer.tracer import Tracer
```

- [ ] **Step 4: Add __aenter__ and __aexit__**

After the `register_memory_hooks(self)` and MCP init block, add these methods (before `register_tool`):

```python
    async def __aenter__(self):
        """Enter async context — connects MCP servers if configured."""
        if self._mcp_manager:
            tools = await self._mcp_manager.connect()
            for t in tools:
                self.register_tool(t)
            logger.info(
                f"[{self.__class__.__name__}] MCP connected, "
                f"{len(tools)} tool(s) registered"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context — disconnects MCP servers."""
        if self._mcp_manager and self._mcp_manager.is_connected():
            await self._mcp_manager.disconnect()
        return False
```

- [ ] **Step 5: Add lazy connect in _execute_loop**

In `_execute_loop()` (line 316), add lazy MCP connect after the trace setup and before the loop. Find this line (around line 370):

```python
        try:
            if not resume:
                query_text = (
```

Insert before `try:`:

```python
        # Lazy MCP connect (if not using async with)
        if self._mcp_manager and not self._mcp_manager.is_connected():
            logger.warning(
                f"[{self.__class__.__name__}] MCP lazy connect — "
                "use 'async with agent:' for persistent connections"
            )
            mcp_tools = await self._mcp_manager.connect()
            for t in mcp_tools:
                self.register_tool(t)
            self._lazy_mcp = True
```

In the `finally` block at the end of `_execute_loop`, add:

```python
            # Lazy MCP disconnect
            if self._lazy_mcp and self._mcp_manager:
                await self._mcp_manager.disconnect()
                self._lazy_mcp = False
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/core/agent/test_base_agent_mcp.py -v`
Expected: All PASS.

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/obelix/core/agent/base_agent.py tests/core/agent/test_base_agent_mcp.py
git commit -m "feat(mcp): integrate MCP into BaseAgent with async context manager and lazy connect"
```

---

## Task 9: Update pyproject.toml and run final validation

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update MCP dependency upper bound**

In `pyproject.toml`, find the mcp dependency line and update to pin `< 2`:

```
mcp = ["mcp>=1.25.0,<2"]
```

Also update the line in the `serve` extras if present.

- [ ] **Step 2: Run linter on all new files**

```bash
uv run ruff check src/obelix/adapters/outbound/mcp/ src/obelix/ports/outbound/mcp_provider.py src/obelix/core/tool/tool_base.py src/obelix/core/agent/base_agent.py
uv run ruff format src/obelix/adapters/outbound/mcp/ src/obelix/ports/outbound/mcp_provider.py src/obelix/core/tool/tool_base.py src/obelix/core/agent/base_agent.py
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: pin mcp SDK to >=1.25,<2"
```

---

## Summary

| Task | Description | Est. Files |
|------|-------------|-----------|
| 1 | Tool protocol annotations | 3 |
| 2 | Delete old MCP plugin | 2 |
| 3 | AbstractMCPProvider port | 1 |
| 4 | MCPServerConfig + parser | 4 |
| 5 | MCPToolAdapter | 2 |
| 6 | MCPResourceAdapter placeholder | 1 |
| 7 | MCPManager | 3 |
| 8 | BaseAgent integration | 2 |
| 9 | pyproject.toml + validation | 1 |

**Total: 9 tasks, ~19 files created/modified**
