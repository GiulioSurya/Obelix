"""MCP server configuration and .mcp.json parser."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

_VALID_TRANSPORTS = {"stdio", "streamable-http"}
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: str  # "stdio" or "streamable-http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _substitute_env_vars(value: str) -> str:
    """Replace ``${ENV_VAR}`` placeholders with their values.

    If the environment variable is not set the placeholder is left as-is.
    """

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    return _ENV_VAR_RE.sub(_replace, value)


def _substitute_dict(d: dict[str, str]) -> dict[str, str]:
    return {k: _substitute_env_vars(v) for k, v in d.items()}


def _validate(cfg: MCPServerConfig) -> None:
    if cfg.transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid transport {cfg.transport!r} for server {cfg.name!r}. "
            f"Must be one of {_VALID_TRANSPORTS}"
        )
    if cfg.transport == "stdio" and not cfg.command:
        raise ValueError(f"Server {cfg.name!r} uses stdio transport but has no command")
    if cfg.transport == "streamable-http" and not cfg.url:
        raise ValueError(
            f"Server {cfg.name!r} uses streamable-http transport but has no url"
        )


def _parse_file(path: Path) -> list[MCPServerConfig]:
    """Parse a .mcp.json file into a list of server configs."""
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers", {})

    configs: list[MCPServerConfig] = []
    for name, entry in servers.items():
        transport = entry.get("type", "stdio")
        cfg = MCPServerConfig(
            name=name,
            transport=transport,
            command=entry.get("command"),
            args=entry.get("args", []),
            env=_substitute_dict(entry.get("env", {})),
            url=_substitute_env_vars(entry["url"]) if "url" in entry else None,
            headers=_substitute_dict(entry.get("headers", {})),
        )
        _validate(cfg)
        configs.append(cfg)
        logger.debug(f"Loaded MCP server config: {name} ({transport})")

    return configs


def parse_mcp_config(
    config: str | Path | MCPServerConfig | list[str | Path | MCPServerConfig],
) -> list[MCPServerConfig]:
    """Parse MCP configuration into a list of :class:`MCPServerConfig`.

    Accepts:
    - A single :class:`MCPServerConfig` instance
    - A ``str`` or :class:`Path` pointing to a ``.mcp.json`` file
    - A list mixing any of the above
    """
    if isinstance(config, MCPServerConfig):
        return [config]

    if isinstance(config, (str, Path)):
        return _parse_file(Path(config))

    if isinstance(config, list):
        result: list[MCPServerConfig] = []
        for item in config:
            result.extend(parse_mcp_config(item))
        return result

    raise TypeError(f"Unsupported config type: {type(config)}")
