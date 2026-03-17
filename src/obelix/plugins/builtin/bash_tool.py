# src/obelix/plugins/builtin/bash_tool.py
"""
Deferred bash tool — delegates shell command execution to the client.

The tool never executes commands itself. When the LLM invokes it, execute()
returns None, which stops the agent loop and yields a StreamEvent with
deferred_tool_calls. The consumer (A2A executor, CLI runner, or human)
receives the command, executes it locally, and sends back the result.

The OutputSchema declares the expected response format so the executor
can validate and parse the client's response automatically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from obelix.core.tool.tool_decorator import tool


@tool(
    name="bash",
    description=(
        "Execute a shell command on the client environment. "
        "The command is delegated to the client for local execution. "
        "Provide a clear description of what the command does. "
        "Prefer simple, composable commands. Use pipes for chaining."
    ),
    is_deferred=True,
)
class BashTool:
    """Deferred tool that delegates shell command execution to the client."""

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
        description="Working directory for the command. If omitted, uses client default.",
    )

    class OutputSchema(BaseModel):
        """Expected response format from the client after command execution."""

        stdout: str = Field(default="", description="Standard output of the command")
        stderr: str = Field(default="", description="Standard error of the command")
        exit_code: int = Field(default=0, description="Process exit code (0 = success)")

    def execute(self) -> None:
        """Return None to signal deferred execution.

        The BaseAgent detects is_deferred=True + None result and stops
        the loop, yielding the tool call for the caller to handle.
        """
        return None
