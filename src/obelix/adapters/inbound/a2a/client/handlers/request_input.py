"""Handler for the ``request_user_input`` deferred tool.

Renders a question with optional numbered choices and returns the
user's answer (selected label or free-form text) as a plain string.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from obelix.adapters.inbound.a2a.client.handlers.base import BaseDeferredHandler


class RequestInputHandler(BaseDeferredHandler):
    """Interactive handler for ``request_user_input`` tool calls."""

    tool_name = "request_user_input"

    def render(self, args: dict, console: Console) -> None:
        question = args.get("question", "")
        options: list[dict] = args.get("options", [])

        body = Text()
        body.append(question, style="bold")

        if options:
            body.append("\n")
            for i, opt in enumerate(options, 1):
                label = opt.get("label", "")
                desc = opt.get("description", "")
                body.append(f"\n  {i}. ", style="bold cyan")
                body.append(label)
                if desc:
                    body.append(f"\n     {desc}", style="dim")

        console.print()
        console.print(
            Panel(
                body,
                title="[bold yellow]request_user_input[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    async def prompt_response(self, args: dict, session: PromptSession) -> dict:
        options: list[dict] = args.get("options", [])

        console_hint = ""
        if options:
            console_hint = (
                f" <style fg='ansibrightblack'>[1-{len(options)} or free text]</style>"
            )

        answer = await session.prompt_async(
            HTML(f"<b>> </b>{console_hint} "),
        )
        answer = answer.strip()

        if not answer:
            return {"answer": "proceed"}

        # Number selection -> return the label
        if options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return {"answer": options[idx].get("label", answer)}

        return {"answer": answer}
