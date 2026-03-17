# src/obelix/adapters/outbound/shell/local_executor.py
"""Local shell executor — runs commands via asyncio subprocess.

Detects the best available shell (bash, zsh, sh) and uses it explicitly,
avoiding cmd.exe on Windows which lacks heredoc, pipes, and other Unix features.

Stateless: each call spawns a fresh process. No persistent session.
Thread-safe: asyncio subprocess is safe for concurrent use.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
from pathlib import Path

from obelix.infrastructure.logging import get_logger
from obelix.ports.outbound.shell_executor import AbstractShellExecutor

logger = get_logger(__name__)

_MAX_OUTPUT_CHARS = 50_000
_TRUNCATION_MSG = "\n\n... [output truncated at {limit} chars — {total} total]"


def _detect_shell() -> str:
    """Find the best available shell for command execution.

    Priority: bash > zsh > sh > cmd.exe (Windows fallback).
    On Windows with Git Bash installed, bash is preferred over cmd.exe
    because it supports heredoc, pipes, and standard Unix syntax.
    """
    for shell in ("bash", "zsh", "sh"):
        path = shutil.which(shell)
        if path:
            return path

    # Windows fallback — cmd.exe
    if platform.system() == "Windows":
        return "cmd.exe"

    return "/bin/sh"


class LocalShellExecutor(AbstractShellExecutor):
    """Executes shell commands locally via asyncio subprocess.

    Automatically detects the best available shell. Pass ``shell``
    to override (e.g. ``LocalShellExecutor(shell="/bin/zsh")``).

    The ``shell_info`` property returns a dict with platform and shell
    details — inject this into the agent's system message so the LLM
    generates compatible commands.
    """

    def __init__(self, shell: str | None = None):
        self._shell = shell or _detect_shell()
        self._platform = platform.system()
        logger.info(
            f"[Shell] Initialized | shell={self._shell} platform={self._platform}"
        )

    @property
    def shell_info(self) -> dict:
        """Platform and shell info for agent system message injection.

        Example usage:
            executor = LocalShellExecutor()
            system_msg += f"\\nShell: {executor.shell_info['shell']} on {executor.shell_info['platform']}"
        """
        return {
            "platform": self._platform,
            "shell": self._shell,
            "shell_name": Path(self._shell).stem,
        }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        working_directory: str | None = None,
    ) -> dict:
        """Execute a shell command in a subprocess.

        Uses the detected shell explicitly (e.g. ``bash -c "command"``),
        avoiding cmd.exe on Windows.

        Returns dict with stdout, stderr, exit_code.
        On timeout, kills the process and returns exit_code=-1.
        """
        cwd = str(Path(working_directory).resolve()) if working_directory else None

        logger.info(
            f"[Shell] Executing | shell={self._shell} "
            f"command={command!r} timeout={timeout}s cwd={cwd}"
        )

        try:
            process = await asyncio.create_subprocess_exec(
                self._shell,
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(
                    f"[Shell] Timeout | command={command!r} timeout={timeout}s"
                )
                return {
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s and was killed.",
                    "exit_code": -1,
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            logger.info(
                f"[Shell] Completed | exit_code={process.returncode} "
                f"stdout_len={len(stdout)} stderr_len={len(stderr)}"
            )

            return {
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "exit_code": process.returncode,
            }

        except FileNotFoundError:
            logger.error(f"[Shell] Working directory not found | cwd={cwd}")
            return {
                "stdout": "",
                "stderr": f"Working directory not found: {cwd}",
                "exit_code": -1,
            }
        except OSError as e:
            logger.error(f"[Shell] OS error | command={command!r} error={e}")
            return {
                "stdout": "",
                "stderr": f"OS error executing command: {e}",
                "exit_code": -1,
            }


def _truncate(text: str) -> str:
    """Truncate output that would bloat the context window."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + _TRUNCATION_MSG.format(
        limit=_MAX_OUTPUT_CHARS, total=len(text)
    )
