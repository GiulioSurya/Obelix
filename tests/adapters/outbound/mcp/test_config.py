"""Tests for MCPServerConfig and parse_mcp_config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obelix.adapters.outbound.mcp.config import MCPServerConfig, parse_mcp_config

# ---------------------------------------------------------------------------
# MCPServerConfig construction
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="my-server",
            transport="stdio",
            command="uv",
            args=["run", "python", "-m", "my_server"],
            env={"KEY": "value"},
        )
        assert cfg.name == "my-server"
        assert cfg.transport == "stdio"
        assert cfg.command == "uv"
        assert cfg.args == ["run", "python", "-m", "my_server"]
        assert cfg.env == {"KEY": "value"}
        assert cfg.url is None
        assert cfg.headers == {}

    def test_http_config(self):
        cfg = MCPServerConfig(
            name="remote",
            transport="streamable-http",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.transport == "streamable-http"
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.headers == {"Authorization": "Bearer tok"}
        assert cfg.command is None
        assert cfg.args == []

    def test_defaults(self):
        cfg = MCPServerConfig(name="x", transport="stdio", command="cmd")
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.headers == {}


# ---------------------------------------------------------------------------
# parse_mcp_config — single MCPServerConfig
# ---------------------------------------------------------------------------


class TestParseSingleConfig:
    def test_single_config_passthrough(self):
        cfg = MCPServerConfig(name="s", transport="stdio", command="echo")
        result = parse_mcp_config(cfg)
        assert result == [cfg]


# ---------------------------------------------------------------------------
# parse_mcp_config — list of configs
# ---------------------------------------------------------------------------


class TestParseListConfig:
    def test_list_of_configs(self):
        a = MCPServerConfig(name="a", transport="stdio", command="a")
        b = MCPServerConfig(name="b", transport="streamable-http", url="http://x")
        result = parse_mcp_config([a, b])
        assert result == [a, b]


# ---------------------------------------------------------------------------
# parse_mcp_config — .mcp.json file
# ---------------------------------------------------------------------------


class TestParseMcpJsonFile:
    def test_parse_file_str(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "tracer": {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["run", "python", "-m", "tracer"],
                    "env": {"TRACER_URL": "http://localhost:8100"},
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "http://remote:9090/mcp",
                    "headers": {"X-Token": "abc"},
                },
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        result = parse_mcp_config(str(f))
        assert len(result) == 2

        by_name = {c.name: c for c in result}
        tracer = by_name["tracer"]
        assert tracer.transport == "stdio"
        assert tracer.command == "uv"
        assert tracer.args == ["run", "python", "-m", "tracer"]
        assert tracer.env == {"TRACER_URL": "http://localhost:8100"}

        remote = by_name["remote"]
        assert remote.transport == "streamable-http"
        assert remote.url == "http://remote:9090/mcp"
        assert remote.headers == {"X-Token": "abc"}

    def test_parse_path_object(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "srv": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["server.js"],
                }
            }
        }
        f = tmp_path / "config.mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        result = parse_mcp_config(f)
        assert len(result) == 1
        assert result[0].name == "srv"
        assert result[0].command == "node"


# ---------------------------------------------------------------------------
# parse_mcp_config — mixed list
# ---------------------------------------------------------------------------


class TestParseMixedList:
    def test_mixed_list(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "from-file": {
                    "type": "stdio",
                    "command": "python",
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        inline = MCPServerConfig(
            name="inline", transport="streamable-http", url="http://x"
        )
        result = parse_mcp_config([str(f), inline])
        assert len(result) == 2
        names = {c.name for c in result}
        assert names == {"from-file", "inline"}


# ---------------------------------------------------------------------------
# ${ENV_VAR} substitution
# ---------------------------------------------------------------------------


class TestEnvVarSubstitution:
    def test_env_substitution_in_env_field(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "s3cret")
        data = {
            "mcpServers": {
                "s": {
                    "type": "stdio",
                    "command": "cmd",
                    "env": {"TOKEN": "${MY_SECRET}"},
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        result = parse_mcp_config(str(f))
        assert result[0].env["TOKEN"] == "s3cret"

    def test_env_substitution_in_url(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MCP_HOST", "myhost.example.com")
        data = {
            "mcpServers": {
                "r": {
                    "type": "streamable-http",
                    "url": "http://${MCP_HOST}:8080/mcp",
                    "headers": {"X-Key": "${MCP_HOST}"},
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        result = parse_mcp_config(str(f))
        assert result[0].url == "http://myhost.example.com:8080/mcp"
        assert result[0].headers["X-Key"] == "myhost.example.com"

    def test_unset_env_var_left_as_is(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        data = {
            "mcpServers": {
                "s": {
                    "type": "stdio",
                    "command": "cmd",
                    "env": {"X": "${NONEXISTENT_VAR_XYZ}"},
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        result = parse_mcp_config(str(f))
        assert result[0].env["X"] == "${NONEXISTENT_VAR_XYZ}"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_mcp_config("/nonexistent/path/.mcp.json")

    def test_invalid_transport(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "bad": {
                    "type": "grpc",
                    "command": "x",
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="transport"):
            parse_mcp_config(str(f))

    def test_stdio_missing_command(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "bad": {
                    "type": "stdio",
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="command"):
            parse_mcp_config(str(f))

    def test_http_missing_url(self, tmp_path: Path):
        data = {
            "mcpServers": {
                "bad": {
                    "type": "streamable-http",
                }
            }
        }
        f = tmp_path / ".mcp.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="url"):
            parse_mcp_config(str(f))
