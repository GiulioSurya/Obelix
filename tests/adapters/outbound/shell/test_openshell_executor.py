"""Tests for OpenShellExecutor adapter.

All tests mock the openshell SDK — no real Gateway or sandbox is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from obelix.ports.outbound.shell_executor import AbstractShellExecutor

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeExecResult:
    """Mimics openshell ExecResult."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class FakeSandboxRef:
    """Mimics openshell SandboxRef."""

    id: str = "sb-123"
    name: str = "test-sandbox"


def _make_mock_client(
    exec_result: FakeExecResult | None = None,
    sandbox_ref: FakeSandboxRef | None = None,
) -> MagicMock:
    """Create a mock SandboxClient with sensible defaults."""
    client = MagicMock()
    client.exec.return_value = exec_result or FakeExecResult(
        exit_code=0, stdout="ok\n", stderr=""
    )
    ref = sandbox_ref or FakeSandboxRef()
    client.get.return_value = ref
    client.create.return_value = ref
    client.wait_ready.return_value = None
    client.delete.return_value = None
    client.health.return_value = True
    return client


def _patch_sdk(mock_client: MagicMock):
    """Patch the openshell SDK in the executor module."""
    mock_class = MagicMock()
    mock_class.return_value = mock_client
    mock_class.from_active_cluster.return_value = mock_client
    return patch(
        "obelix.adapters.outbound.shell.openshell_executor.SandboxClient",
        mock_class,
    )


# ---------------------------------------------------------------------------
# Import helper — the module checks for openshell at import time
# ---------------------------------------------------------------------------


def _import_executor():
    """Import OpenShellExecutor with SandboxClient mocked."""
    from obelix.adapters.outbound.shell.openshell_executor import (
        OpenShellExecutor,
    )

    return OpenShellExecutor


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestContract:
    """Verify the adapter satisfies the port contract."""

    def test_isinstance_abstract(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
        assert isinstance(executor, AbstractShellExecutor)

    def test_is_remote_false(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
        assert executor.is_remote is False


# ---------------------------------------------------------------------------
# Shell info tests
# ---------------------------------------------------------------------------


class TestShellInfo:
    """Tests for shell_info before and after initialization."""

    def test_shell_info_before_execute(self) -> None:
        """Before first execute, returns static Linux/bash info."""
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
        info = executor.shell_info
        assert info["platform"] == "Linux"
        assert info["shell_name"] == "bash"

    @pytest.mark.asyncio
    async def test_shell_info_after_execute(self) -> None:
        """After execute, shell_info includes cwd and home from probe."""
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(
            exit_code=0, stdout="CWD=/sandbox\nHOME=/home/user\n", stderr=""
        )
        exec_result = FakeExecResult(exit_code=0, stdout="hello\n", stderr="")
        mock_client = _make_mock_client()
        # First call is probe, second is actual execute
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            await executor.execute("echo hello")

        info = executor.shell_info
        assert info["cwd"] == "/sandbox"
        assert info["home"] == "/home/user"
        assert info["sandbox"] == "test"


# ---------------------------------------------------------------------------
# Execute with pre-existing sandbox
# ---------------------------------------------------------------------------


class TestExecutePreExisting:
    """Tests for executing commands in a pre-existing sandbox."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="file.txt\n", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="my-sb")
            result = await executor.execute("ls")

        assert result["stdout"] == "file.txt\n"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_connects_to_named_sandbox(self) -> None:
        """When sandbox_name is provided, client.get() is called, not create()."""
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="existing-sb")
            await executor.execute("echo ok")

        mock_client.get.assert_called_once_with(sandbox_name="existing-sb")
        mock_client.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_passes_timeout(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            await executor.execute("sleep 5", timeout=30)

        # Second call is the actual execute (first is probe)
        _, exec_call = mock_client.exec.call_args_list
        assert (
            exec_call.kwargs.get("timeout_seconds") == 30
            or exec_call[1].get("timeout_seconds") == 30
        )

    @pytest.mark.asyncio
    async def test_execute_with_working_directory(self) -> None:
        """working_directory prepends cd to the command."""
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            await executor.execute("ls", working_directory="/data")

        _, exec_call = mock_client.exec.call_args_list
        command_arg = exec_call.kwargs.get("command") or exec_call[1].get("command")
        # The command should be ["bash", "-c", "cd /data && ls"]
        assert "cd /data" in command_arg[-1]
        assert "ls" in command_arg[-1]


# ---------------------------------------------------------------------------
# Auto-create sandbox
# ---------------------------------------------------------------------------


class TestAutoCreateSandbox:
    """Tests for auto-created sandbox lifecycle."""

    @pytest.mark.asyncio
    async def test_auto_create_calls_create_and_wait(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()  # no sandbox_name = auto-create
            await executor.execute("echo ok")

        mock_client.create.assert_called_once()
        mock_client.wait_ready.assert_called_once()
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_destroys_auto_created(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
            await executor.execute("echo ok")
            await executor.close()

        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_not_destroy_pre_existing(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="keep-me")
            await executor.execute("echo ok")
            await executor.close()

        mock_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Lazy init
# ---------------------------------------------------------------------------


class TestLazyInit:
    """Verify sandbox is initialized only once."""

    @pytest.mark.asyncio
    async def test_second_execute_does_not_reinitialize(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            await executor.execute("echo 1")
            await executor.execute("echo 2")

        # client.get() called only once (during first execute's _ensure_sandbox)
        mock_client.get.assert_called_once()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestOutputTruncation:
    """Verify output is truncated at 50K chars."""

    @pytest.mark.asyncio
    async def test_long_output_truncated(self) -> None:
        OpenShellExecutor = _import_executor()
        long_output = "x" * 60_000
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout=long_output, stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            result = await executor.execute("cat bigfile")

        assert len(result["stdout"]) < 60_000
        assert "truncated" in result["stdout"]
        assert "60000 total" in result["stdout"]

    @pytest.mark.asyncio
    async def test_short_output_not_truncated(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="short", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            result = await executor.execute("echo short")

        assert result["stdout"] == "short"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error conditions."""

    @pytest.mark.asyncio
    async def test_gateway_unreachable(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        mock_client.get.side_effect = ConnectionError("Gateway unreachable")

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            result = await executor.execute("echo hello")

        assert result["exit_code"] == -1
        assert "Gateway unreachable" in result["stderr"]

    @pytest.mark.asyncio
    async def test_sandbox_not_found(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        mock_client.get.side_effect = Exception("Sandbox 'ghost' not found")

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="ghost")
            result = await executor.execute("echo hello")

        assert result["exit_code"] == -1
        assert "not found" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_after_close(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(sandbox_name="test")
            await executor.close()
            result = await executor.execute("echo hello")

        assert result["exit_code"] == -1
        assert "closed" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            async with OpenShellExecutor() as executor:
                result = await executor.execute("echo ok")

        assert result["exit_code"] == 0
        # Auto-created sandbox should be destroyed on exit
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        """Calling close() multiple times is safe."""
        OpenShellExecutor = _import_executor()
        probe_result = FakeExecResult(exit_code=0, stdout="CWD=/sandbox\n", stderr="")
        exec_result = FakeExecResult(exit_code=0, stdout="ok", stderr="")
        mock_client = _make_mock_client()
        mock_client.exec.side_effect = [probe_result, exec_result]

        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
            await executor.execute("echo ok")
            await executor.close()
            await executor.close()  # should not raise

        mock_client.delete.assert_called_once()


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Env var resolution
# ---------------------------------------------------------------------------


class TestEnvVarResolution:
    """Tests for gateway/tls_cert_dir env var fallback."""

    def test_gateway_from_explicit_param(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(gateway="explicit:443")
        assert executor._gateway == "explicit:443"

    def test_gateway_from_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENSHELL_GATEWAY", "env-gw:443")
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
        assert executor._gateway == "env-gw:443"

    def test_gateway_none_when_no_param_no_env(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENSHELL_GATEWAY", raising=False)
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
        assert executor._gateway is None

    def test_tls_cert_dir_from_explicit_param(self) -> None:
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor(tls_cert_dir="/explicit/certs")
        assert executor._tls_cert_dir == "/explicit/certs"

    def test_tls_cert_dir_from_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENSHELL_TLS_CERT_DIR", "/env/certs")
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
        assert executor._tls_cert_dir == "/env/certs"

    def test_tls_cert_dir_none_when_no_param_no_env(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENSHELL_TLS_CERT_DIR", raising=False)
        OpenShellExecutor = _import_executor()
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client):
            executor = OpenShellExecutor()
        assert executor._tls_cert_dir is None


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Verify behavior when openshell SDK is not installed."""

    def test_import_error_when_sdk_missing(self) -> None:
        with patch(
            "obelix.adapters.outbound.shell.openshell_executor.SandboxClient",
            None,
        ):
            OpenShellExecutor = _import_executor()
            with pytest.raises(ImportError, match="openshell"):
                OpenShellExecutor(sandbox_name="test")
