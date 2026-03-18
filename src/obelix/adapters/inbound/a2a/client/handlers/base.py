"""Base handler and permission model for deferred A2A tool calls.

Each deferred tool (request_user_input, bash, ...) gets a handler that
knows how to render the tool call in the terminal and collect the
response to send back to the server.
"""

from __future__ import annotations

import json
from enum import Enum

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class PermissionPolicy(Enum):
    """Controls whether a tool handler asks for confirmation before acting."""

    ALWAYS_ASK = "always_ask"
    AUTO_APPROVE = "auto_approve"
    ALWAYS_DENY = "always_deny"


class BaseDeferredHandler:
    """Default handler for deferred tool calls.

    Renders arguments as pretty-printed JSON and prompts for free-form
    text input.  Acts as the **fallback** for unknown tool types.

    Subclass and override ``render()`` / ``prompt_response()`` for
    tool-specific behavior.  ``handle()`` is the entry point callers use.
    """

    tool_name: str = ""
    permission: PermissionPolicy = PermissionPolicy.ALWAYS_ASK

    def can_handle(self, name: str) -> bool:
        return self.tool_name == name

    def render(self, args: dict, console: Console) -> None:
        """Display the deferred tool call.  Override for custom UI."""
        console.print()
        console.print(
            Panel(
                Text(json.dumps(args, indent=2)),
                title=f"[bold yellow]{self.tool_name or 'unknown'}[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    async def prompt_response(self, args: dict, session: PromptSession) -> dict:
        """Collect the user's response as a dict for DataPart transport.

        Override for custom interaction.  The returned dict is sent to
        the server as a DataPart and validated against OutputSchema.
        """
        answer = await session.prompt_async(HTML("<b>> </b>"))
        return {"answer": answer.strip() or "proceed"}

    async def handle(
        self, args: dict, console: Console, session: PromptSession
    ) -> dict:
        """Render the tool call, then collect and return the response dict.

        This is the main entry point used by the client's input-required
        loop.  The returned dict is wrapped in a DataPart by the caller.
        """
        self.render(args, console)
        return await self.prompt_response(args, session)
