"""Handler for the ``request_user_input`` deferred tool.

Renders a question with optional numbered choices and returns the
user's answer (selected label or free-form text) as a plain string.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from obelix.adapters.inbound.a2a.client.handlers.base import (
    BaseDeferredHandler,
    InputCallback,
)


class RequestInputHandler(BaseDeferredHandler):
    """Interactive handler for ``request_user_input`` tool calls."""

    tool_name = "request_user_input"

    def get_input_hint(self, args: dict) -> str:
        options: list[dict] = args.get("options", [])
        if options:
            nums = ", ".join(str(i) for i in range(1, len(options) + 1))
            return f"Type {nums} or free text"
        return "Type your answer and press Enter"

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

    async def prompt_response(self, args: dict, input_fn: InputCallback) -> dict:
        options: list[dict] = args.get("options", [])

        answer = await input_fn()
        answer = answer.strip()

        if not answer:
            return {"answer": "proceed"}

        # Number selection -> return the label
        if options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return {"answer": options[idx].get("label", answer)}

        return {"answer": answer}
