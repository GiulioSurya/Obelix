# src/obelix/plugins/builtin/bash_tool.py
"""
Bash tool — executes shell commands locally or delegates to the client.

Behavior depends on whether an executor is provided:

- **With executor** (e.g. LocalShellExecutor): execute() runs the command
  directly via the executor. The tool is NOT deferred — the agent loop
  continues normally. Use this for local/server-side execution.

- **Without executor** (default): execute() returns None, which stops the
  agent loop and yields a StreamEvent with deferred_tool_calls. The consumer
  (A2A client, CLI runner, human) receives the command and handles execution.

The OutputSchema declares the expected response format so consumers and
executors produce consistent results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from obelix.core.tool.tool_decorator import tool

if TYPE_CHECKING:
    from obelix.ports.outbound.shell_executor import AbstractShellExecutor


@tool(
    name="bash",
    description=(
        "Execute a shell command. "
        "Provide a clear description of what the command does. "
        "Prefer simple, composable commands. Use pipes for chaining."
    ),
    is_deferred=True,
)
class BashTool:
    """Shell command tool with pluggable execution backend.

    Without executor: deferred (client executes).
    With executor: local (server executes via AbstractShellExecutor).
    """

    command: str = Field(
        ...,
        description="The shell command to execute (e.g. 'ls -la', 'git status')",
    )
    description: str = Field(
        ...,
        description=(
            "Human-readable explanation of what this command does. "
            "Required for audit and client review before execution."
        ),
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Maximum execution time in seconds (1-600, default 120)",
    )
    working_directory: str | None = Field(
        default=None,
        description="Working directory for the command. If omitted, uses default.",
    )

    class OutputSchema(BaseModel):
        """Expected response format after command execution."""

        stdout: str = Field(default="", description="Standard output of the command")
        stderr: str = Field(default="", description="Standard error of the command")
        exit_code: int = Field(default=0, description="Process exit code (0 = success)")

    def __init__(self, executor: AbstractShellExecutor | None = None):
        self._executor = executor
        # Override the class-level is_deferred based on executor presence
        self.is_deferred = executor is None

    async def execute(self) -> dict | None:
        """Execute the command or defer to client.

        With executor: runs the command and returns OutputSchema-compatible dict.
        Without executor: returns None (deferred — client must execute).
        """
        if self._executor is None:
            return None

        return await self._executor.execute(
            command=self.command,
            timeout=self.timeout,
            working_directory=self.working_directory,
        )
