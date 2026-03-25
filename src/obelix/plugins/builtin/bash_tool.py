# src/obelix/plugins/builtin/bash_tool.py
"""
Bash tool — executes shell commands locally or delegates to the client.

Behavior depends on the executor type:

- **LocalShellExecutor**: execute() runs the command directly on the server.
  The tool is NOT deferred — the agent loop continues normally.

- **ClientShellExecutor**: execute() is never called (deferred). The agent
  loop stops and yields a StreamEvent with deferred_tool_calls. The A2A
  client receives the command, executes it, and sends back the result.

Both executor types carry shell environment info (platform, shell, cwd)
via shell_info, which system_prompt_fragment() uses to enrich the agent's
system message.

The executor parameter is required — there is no default.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from obelix.core.tool.tool_decorator import tool
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

    Requires an executor (LocalShellExecutor or ClientShellExecutor).
    The executor determines whether commands run locally or are deferred
    to the client.
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

    def __init__(self, executor: AbstractShellExecutor):
        self._executor = executor
        self.is_deferred = executor.is_remote

    def system_prompt_fragment(self) -> str | None:
        """Build shell environment block for the agent's system prompt.

        Returns None if the executor has no shell_info yet (e.g.
        ClientShellExecutor before the client sends its environment).
        """
        info = self._executor.shell_info
        if not info:
            return None

        shell_name = info.get("shell_name", "sh")
        plat = info.get("platform", "Unknown")
        os_ver = info.get("os_version", "")

        lines = [
            "\n## Shell Environment",
            f"- Platform: {plat}",
        ]

        if os_ver:
            lines.append(f"- OS Version: {os_ver}")

        # Shell name + syntax hint
        syntax_hint = (
            " (use Unix shell syntax, not Windows "
            "-- e.g., /dev/null not NUL, forward slashes in paths)"
            if plat == "Windows"
            else ""
        )
        lines.append(f"- Shell: {shell_name}{syntax_hint}")

        if cwd := info.get("cwd"):
            lines.append(f"- Working directory: {cwd}")

        if mounts := info.get("mounts"):
            mount_str = ", ".join(
                f"{drive} -> {mp}" for drive, mp in sorted(mounts.items())
            )
            lines.append(f"- Drive mounts: {mount_str}")
            first_mp = next(iter(mounts.values()))
            lines.append(
                f"- Use Unix-style paths with forward slashes "
                f"(e.g. {first_mp}Users/... not C:\\Users\\...)"
            )

        return "\n".join(lines)

    async def execute(self) -> dict | None:
        """Execute the command or defer to client.

        Local executor: runs the command and returns OutputSchema-compatible dict.
        Remote executor: returns None (deferred — client must execute).
        """
        if self._executor.is_remote:
            return None

        return await self._executor.execute(
            command=self.command,
            timeout=self.timeout,
            working_directory=self.working_directory,
        )
