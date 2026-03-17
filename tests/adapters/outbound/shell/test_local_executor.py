"""Tests for LocalShellExecutor adapter."""

from __future__ import annotations

import sys

import pytest

from obelix.adapters.outbound.shell import LocalShellExecutor
from obelix.adapters.outbound.shell.local_executor import (
    _MAX_OUTPUT_CHARS,
    _truncate,
)
from obelix.ports.outbound.shell_executor import AbstractShellExecutor


class TestLocalShellExecutorIsAbstract:
    """Verify the adapter satisfies the port contract."""

    def test_isinstance_abstract(self) -> None:
        executor = LocalShellExecutor()
        assert isinstance(executor, AbstractShellExecutor)


class TestExecuteCommands:
    """Integration tests — spawn real subprocesses."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self) -> None:
        executor = LocalShellExecutor()
        result = await executor.execute("python -c \"print('hello')\"")

        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert result["stderr"] == ""

    @pytest.mark.asyncio
    async def test_execute_stderr(self) -> None:
        executor = LocalShellExecutor()
        result = await executor.execute(
            "python -c \"import sys; sys.stderr.write('error_msg\\n')\""
        )

        assert result["exit_code"] == 0
        assert "error_msg" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_nonzero_exit(self) -> None:
        executor = LocalShellExecutor()
        result = await executor.execute('python -c "import sys; sys.exit(1)"')

        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        executor = LocalShellExecutor()
        result = await executor.execute(
            'python -c "import time; time.sleep(30)"',
            timeout=2,
        )

        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_execute_working_directory(self, tmp_path: object) -> None:
        executor = LocalShellExecutor()
        # Use Python to print cwd — works on all platforms.
        result = await executor.execute(
            'python -c "import os; print(os.getcwd())"',
            working_directory=str(tmp_path),
        )

        assert result["exit_code"] == 0
        # Normalize both paths for reliable comparison on Windows.
        import os

        expected = os.path.normcase(os.path.realpath(str(tmp_path)))
        actual = os.path.normcase(result["stdout"].strip())
        assert actual == expected

    @pytest.mark.asyncio
    async def test_execute_invalid_working_directory(self) -> None:
        executor = LocalShellExecutor()
        bogus_dir = "/no/such/directory/exists_12345"
        if sys.platform == "win32":
            bogus_dir = r"C:\no_such_dir_12345\child"

        result = await executor.execute(
            'python -c "print(1)"',
            working_directory=bogus_dir,
        )

        assert result["exit_code"] == -1
        assert (
            "not found" in result["stderr"].lower()
            or "error" in result["stderr"].lower()
        )


class TestTruncate:
    """Unit tests for the _truncate helper."""

    def test_output_truncation_under_limit(self) -> None:
        short_text = "hello world"
        assert _truncate(short_text) == short_text

    def test_output_truncation_exact_limit(self) -> None:
        exact_text = "x" * _MAX_OUTPUT_CHARS
        assert _truncate(exact_text) == exact_text

    def test_output_truncation_over_limit(self) -> None:
        total = _MAX_OUTPUT_CHARS + 500
        long_text = "a" * total
        truncated = _truncate(long_text)

        assert truncated.startswith("a" * _MAX_OUTPUT_CHARS)
        assert "truncated" in truncated.lower()
        assert str(total) in truncated
        assert str(_MAX_OUTPUT_CHARS) in truncated
        # Original text should be longer than the truncated output
        # (we cut content but added a short message).
        assert len(truncated) < len(long_text)
