"""Tests for ClientShellExecutor adapter."""

from __future__ import annotations

import pytest

from obelix.adapters.outbound.shell import ClientShellExecutor
from obelix.ports.outbound.shell_executor import AbstractShellExecutor


class TestClientShellExecutorContract:
    """Verify the adapter satisfies the port contract."""

    def test_isinstance_abstract(self) -> None:
        executor = ClientShellExecutor()
        assert isinstance(executor, AbstractShellExecutor)

    def test_is_remote_true(self) -> None:
        """ClientShellExecutor is always remote."""
        executor = ClientShellExecutor()
        assert executor.is_remote is True


class TestShellInfoLifecycle:
    """Tests for shell_info population via set_shell_info."""

    def test_shell_info_initially_empty(self) -> None:
        executor = ClientShellExecutor()
        assert executor.shell_info == {}

    def test_shell_info_falsy_when_empty(self) -> None:
        executor = ClientShellExecutor()
        assert not executor.shell_info

    def test_set_shell_info(self) -> None:
        executor = ClientShellExecutor()
        info = {"platform": "Linux", "shell_name": "bash", "cwd": "/home/user"}
        executor.set_shell_info(info)
        assert executor.shell_info == info

    def test_set_shell_info_overwrites(self) -> None:
        executor = ClientShellExecutor()
        executor.set_shell_info({"platform": "Linux"})
        executor.set_shell_info({"platform": "Windows"})
        assert executor.shell_info["platform"] == "Windows"

    def test_shell_info_truthy_after_set(self) -> None:
        executor = ClientShellExecutor()
        executor.set_shell_info({"platform": "Linux"})
        assert executor.shell_info


class TestExecuteRaises:
    """ClientShellExecutor.execute() must not be called."""

    @pytest.mark.asyncio
    async def test_execute_raises_not_implemented(self) -> None:
        executor = ClientShellExecutor()
        with pytest.raises(NotImplementedError, match="deferred"):
            await executor.execute("ls -la")

    @pytest.mark.asyncio
    async def test_execute_raises_with_all_params(self) -> None:
        executor = ClientShellExecutor()
        with pytest.raises(NotImplementedError):
            await executor.execute("ls", timeout=30, working_directory="/tmp")


class TestLocalExecutorIsNotRemote:
    """Verify LocalShellExecutor.is_remote defaults to False."""

    def test_local_executor_is_remote_false(self) -> None:
        from obelix.adapters.outbound.shell import LocalShellExecutor

        executor = LocalShellExecutor()
        assert executor.is_remote is False
