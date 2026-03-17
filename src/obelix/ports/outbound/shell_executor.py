# src/obelix/ports/outbound/shell_executor.py
"""Abstract interface for shell command execution.

The shell is treated as an external resource in the hexagonal architecture.
Concrete implementations (local subprocess, Docker, SSH) live in
adapters/outbound/shell/.
"""

from abc import ABC, abstractmethod


class AbstractShellExecutor(ABC):
    """Contract for executing shell commands."""

    @property
    def shell_info(self) -> dict:
        """Platform and shell details for system message injection.

        Returns dict with platform, shell (path), shell_name.
        Override in subclasses; default returns empty dict.
        """
        return {}

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: int = 120,
        working_directory: str | None = None,
    ) -> dict:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds (default 120).
            working_directory: Working directory for the command.
                If None, uses the process default.

        Returns:
            dict with keys:
                stdout (str): Standard output of the command.
                stderr (str): Standard error of the command.
                exit_code (int): Process exit code (0 = success, -1 = timeout/error).
        """
        pass
