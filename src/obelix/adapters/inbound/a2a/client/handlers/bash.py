"""Handler for the ``bash`` deferred tool.

Renders the command for review, asks for confirmation (respecting the
permission policy), executes locally via ``LocalShellExecutor``, and
returns a dict matching ``BashTool.OutputSchema``.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from obelix.adapters.inbound.a2a.client.handlers.base import (
    BaseDeferredHandler,
    InputCallback,
    PermissionPolicy,
)

_DENIED_RESPONSE = {"stdout": "", "stderr": "Execution denied by user", "exit_code": -1}


class BashHandler(BaseDeferredHandler):
    """Execute shell commands locally on behalf of the remote agent."""

    tool_name = "bash"

    def __init__(self, permission: PermissionPolicy = PermissionPolicy.ALWAYS_ASK):
        self.permission = permission
        self._executor = None  # lazy init

    async def _get_executor(self):
        """Lazily create the shell executor (probe runs in constructor)."""
        if self._executor is None:
            from obelix.adapters.outbound.shell.local_executor import (
                LocalShellExecutor,
            )

            self._executor = LocalShellExecutor()
        return self._executor

    def render(self, args: dict, console: Console) -> None:
        command = args.get("command", "")
        description = args.get("description", "")
        timeout = args.get("timeout", 120)
        cwd = args.get("working_directory")

        body = Text()
        if description:
            body.append(description, style="dim")
            body.append("\n\n")
        body.append("  $ ", style="bold green")
        body.append(command, style="bold")

        footer_parts = [f"timeout: {timeout}s"]
        if cwd:
            footer_parts.append(f"cwd: {cwd}")
        body.append(f"\n\n  {' | '.join(footer_parts)}", style="dim")

        console.print()
        console.print(
            Panel(
                body,
                title="[bold yellow]bash[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    async def prompt_response(self, args: dict, input_fn: InputCallback) -> dict:
        # Check permission policy
        if self.permission == PermissionPolicy.ALWAYS_DENY:
            return _DENIED_RESPONSE

        if self.permission == PermissionPolicy.ALWAYS_ASK:
            answer = await input_fn()
            if answer.strip().lower() in ("n", "no"):
                return _DENIED_RESPONSE

        # Execute
        executor = await self._get_executor()
        return await executor.execute(
            command=args.get("command", ""),
            timeout=args.get("timeout", 120),
            working_directory=args.get("working_directory"),
        )
