# src/obelix/adapters/outbound/shell/local_executor.py
"""Local shell executor — runs commands via asyncio subprocess.

Detects the best available shell (bash, zsh, sh) and uses it explicitly,
avoiding cmd.exe on Windows which lacks heredoc, pipes, and other Unix features.

Stateless: each call spawns a fresh process. No persistent session.
Thread-safe: asyncio subprocess is safe for concurrent use.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from obelix.infrastructure.logging import get_logger
from obelix.ports.outbound.shell_executor import AbstractShellExecutor

logger = get_logger(__name__)

_MAX_OUTPUT_CHARS = 50_000
_TRUNCATION_MSG = "\n\n... [output truncated at {limit} chars — {total} total]"

_MSYSTEM_EXPECTED = "MINGW64"
_MSYSTEM_BAD_VALUES = frozenset(
    {"MSYS", "MINGW32", "CLANG64", "CLANG32", "UCRT64", "CLANGARM64"}
)


class ShellEnvironmentError(RuntimeError):
    """Raised when the shell environment is misconfigured for safe execution."""


def _find_git_bash() -> str | None:
    """Find Git Bash on Windows.

    Checks standard installation paths. Returns None if not found.
    This is needed because ``shutil.which("bash")`` on Windows often
    resolves to the WSL bash proxy (WindowsApps/bash.EXE) which uses
    ``/mnt/c/`` paths instead of ``/c/`` — breaking MSYS2 path translation.
    """
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Git"
        / "bin"
        / "bash.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Git"
        / "bin"
        / "bash.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "Git"
        / "bin"
        / "bash.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _detect_shell() -> str:
    """Find the best available shell for command execution.

    On Windows: Git Bash > WSL bash > cmd.exe.
    On Unix: bash > zsh > sh.

    Git Bash is preferred on Windows because it uses MSYS2 path translation
    (C:/ → /c/) which is compatible with ``_build_env()`` and ``MSYSTEM``.
    WSL bash uses /mnt/c/ paths and a different environment model.
    """
    if platform.system() == "Windows":
        git_bash = _find_git_bash()
        if git_bash:
            return git_bash

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
        self._os_version = platform.platform()  # e.g. "Windows-11-10.0.26200-SP0"
        self._env = self._build_env()
        self._cwd: str | None = None
        self._home: str | None = None
        self._mounts: dict[str, str] = {}  # {"C:": "/c/", "P:": "/p/"}
        self._probe()
        logger.info(
            f"[Shell] Initialized | shell={self._shell} platform={self._platform} "
            f"cwd={self._cwd} home={self._home} mounts={self._mounts}"
        )

    def _build_env(self) -> dict[str, str] | None:
        """Build environment for subprocess, fixing Windows Git Bash issues.

        When Git Bash runs without MSYSTEM=MINGW64 (e.g. from PyCharm/uvicorn),
        it operates in "pure MSYS" mode where C:/ paths are not translated
        to /c/ and HOME points to /home/<user> instead of /c/Users/<user>.
        Setting MSYSTEM=MINGW64 enables proper Windows path translation.

        Returns None on non-Windows or non-bash shells (inherit parent env).

        Raises ShellEnvironmentError if MSYSTEM is set to an incompatible
        value — this means someone (or another tool) changed the MSYS2
        profile, which will break Windows path translation (C:/ → /c/).
        """
        if self._platform != "Windows":
            return None

        shell_name = Path(self._shell).stem.lower()
        if shell_name not in ("bash", "sh"):
            return None

        env = os.environ.copy()
        current_msystem = env.get("MSYSTEM")

        if not current_msystem:
            env["MSYSTEM"] = _MSYSTEM_EXPECTED
            logger.info("[Shell] Set MSYSTEM=MINGW64 for Git Bash path translation")
            return env

        if current_msystem == _MSYSTEM_EXPECTED:
            return None  # already correct, inherit parent env

        # MSYSTEM is set but to a non-standard value
        raise ShellEnvironmentError(
            f"MSYSTEM is set to '{current_msystem}', but LocalShellExecutor "
            f"requires '{_MSYSTEM_EXPECTED}' for correct Windows path translation "
            f"(C:/... → /c/...). "
            f"Current value will cause path resolution failures.\n\n"
            f"How to fix:\n"
            f"  1. Set MSYSTEM=MINGW64 in your environment before starting the server\n"
            f"     (e.g. in .env, systemd unit, or docker-compose)\n"
            f"  2. Or launch from a standard Git Bash terminal (which sets it automatically)\n"
            f"  3. Or pass shell='cmd.exe' to LocalShellExecutor to bypass MSYS entirely\n"
            f"     (but commands must then use Windows syntax, not Unix)\n\n"
            f"Why this matters: Git Bash uses MSYSTEM to decide how to translate paths.\n"
            f"With MSYSTEM={current_msystem}, bash runs in {current_msystem} mode which "
            f"uses different path prefixes and toolchain paths, causing 'No such file or "
            f"directory' errors on standard Windows paths like C:/Users/..."
        )

    def _probe(self) -> None:
        """Sync one-shot shell probe to discover environment details.

        Populates cwd, home, and drive mounts (Windows only).
        Called automatically in ``__init__`` — no need to call manually.
        Uses ``subprocess.run`` (sync) because it runs once at construction
        time (~100ms) and the result must be available before any async work.
        """
        probe_cmd = 'echo "CWD=$(pwd)" && echo "HOME=$HOME"'
        if self._platform == "Windows":
            probe_cmd += ' && mount | grep "^[A-Z]:"'

        try:
            result = subprocess.run(
                [self._shell, "-c", probe_cmd],
                capture_output=True,
                text=True,
                timeout=10,
                env=self._env,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"[Shell] Probe failed ({e}) — shell_info will be basic")
            return

        if result.returncode != 0:
            logger.warning("[Shell] Probe failed — shell_info will be basic")
            return

        for line in result.stdout.splitlines():
            if line.startswith("CWD="):
                self._cwd = line[4:]
            elif line.startswith("HOME="):
                self._home = line[5:]
            else:
                m = re.match(r"^([A-Z]:)\s+on\s+(/\S+)", line)
                if m:
                    mp = m.group(2)
                    if not mp.endswith("/"):
                        mp += "/"
                    self._mounts[m.group(1)] = mp

    @property
    def shell_info(self) -> dict:
        """Platform and shell details for system message injection.

        Includes cwd, home, and drive mounts (populated automatically
        at construction time via sync probe).
        """
        info = {
            "platform": self._platform,
            "os_version": self._os_version,
            "shell": self._shell,
            "shell_name": Path(self._shell).stem,
        }
        if self._cwd:
            info["cwd"] = self._cwd
        if self._home:
            info["home"] = self._home
        if self._mounts:
            info["mounts"] = self._mounts
        return info

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
                env=self._env,
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
