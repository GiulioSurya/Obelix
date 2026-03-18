"""Interactive CLI client for A2A agents.

Connects to one or more A2A agent servers and provides a Rich-based
terminal UI with intelligent handling of deferred tool calls
(request_user_input, bash, etc.) via pluggable handlers.
"""

from __future__ import annotations

import json
import sys
import uuid

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from obelix.adapters.inbound.a2a.client.handlers import HandlerDispatcher
from obelix.infrastructure.logging import suppress_console

# -- Theme -------------------------------------------------------------------

_THEME = Theme(
    {
        "agent.name": "bold cyan",
        "agent.url": "dim",
        "user.prompt": "bold green",
        "status.ok": "bold green",
        "status.fail": "bold red",
        "status.wait": "bold yellow",
        "info": "dim cyan",
        "cmd": "bold magenta",
    }
)

LOGO = r"""[bold cyan]
     ___  _          _ _
    / _ \| |__   ___| (_)_  __
   | | | | '_ \ / _ \ | \ \/ /
   | |_| | |_) |  __/ | |>  <
    \___/|_.__/ \___|_|_/_/\_\
[/bold cyan][dim]    A2A Agent CLI[/dim]
"""


# -- Agent Connection --------------------------------------------------------


class AgentConnection:
    """A resolved connection to a remote A2A agent."""

    __slots__ = ("name", "description", "skills", "client", "url", "context_id")

    def __init__(
        self,
        name: str,
        description: str,
        skills: list[str],
        client,
        url: str,
    ):
        self.name = name
        self.description = description
        self.skills = skills
        self.client = client
        self.url = url
        self.context_id: str | None = None


# -- A2A message helpers -----------------------------------------------------


def _make_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )


def _extract_text(task) -> str:
    if not hasattr(task, "artifacts") or not task.artifacts:
        return ""
    parts = []
    for artifact in task.artifacts:
        for part in artifact.parts:
            if hasattr(part.root, "text") and part.root.text:
                parts.append(part.root.text)
    return "".join(parts)


def _extract_status_message(task) -> str:
    if not task.status or not task.status.message:
        return ""
    parts = []
    for part in task.status.message.parts:
        if hasattr(part.root, "text"):
            parts.append(part.root.text)
    return "".join(parts)


async def _collect_task(response_iter):
    """Consume the async iterator and return the final task."""
    task = None
    async for item in response_iter:
        if isinstance(item, tuple):
            task = item[0]
        elif isinstance(item, Message):
            return item
    return task


# -- CLIClient ---------------------------------------------------------------


class CLIClient:
    """Rich-based terminal client for A2A agents."""

    def __init__(
        self,
        dispatcher: HandlerDispatcher,
        console: Console | None = None,
    ):
        self.dispatcher = dispatcher
        self.console = console or Console(theme=_THEME, highlight=False)

    # -- UI helpers ----------------------------------------------------------

    def show_logo(self) -> None:
        self.console.print(LOGO)

    def show_agents_table(self, agents: list[AgentConnection], current: int) -> None:
        table = Table(
            title="Connected Agents",
            title_style="bold",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=3, justify="right")
        table.add_column("Agent", style="agent.name")
        table.add_column("Description")
        table.add_column("Skills", style="dim")
        table.add_column("URL", style="agent.url")

        for i, agent in enumerate(agents):
            marker = " *" if i == current else ""
            skills_str = ", ".join(agent.skills[:3]) if agent.skills else "-"
            desc = agent.description[:50] + (
                "..." if len(agent.description) > 50 else ""
            )
            table.add_row(
                f"{i + 1}{marker}",
                agent.name,
                desc,
                skills_str,
                agent.url,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_help(self) -> None:
        help_table = Table(show_header=False, box=None, padding=(0, 2))
        help_table.add_column("Command", style="cmd")
        help_table.add_column("Description")
        help_table.add_row("/agents", "List connected agents")
        help_table.add_row("/switch <n>", "Switch to agent n")
        help_table.add_row("/clear", "Clear conversation context")
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/quit", "Exit")
        self.console.print(Panel(help_table, title="Commands", border_style="dim"))

    def show_agent_response(self, agent_name: str, text: str) -> None:
        md = Markdown(text)
        self.console.print(
            Panel(
                md,
                title=f"[agent.name]{agent_name}[/agent.name]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def show_error(self, msg: str) -> None:
        self.console.print(f"[status.fail]  Error:[/status.fail] {msg}")

    def show_info(self, msg: str) -> None:
        self.console.print(f"[info]{msg}[/info]")

    # -- Connection ----------------------------------------------------------

    async def resolve_agents(
        self, urls: list[str], httpx_client: httpx.AsyncClient
    ) -> list[AgentConnection]:
        agents = []
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)

        for url in urls:
            try:
                with self.console.status(f"[info]Connecting to {url}...[/info]"):
                    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
                    card = await resolver.get_agent_card()
                    client = factory.create(card)
                skills = [s.name for s in card.skills] if card.skills else []
                agents.append(
                    AgentConnection(
                        name=card.name,
                        description=card.description or "",
                        skills=skills,
                        client=client,
                        url=url,
                    )
                )
                self.console.print(
                    f"  [status.ok]OK[/status.ok] {card.name} "
                    f"[agent.url]({url})[/agent.url]"
                )
            except Exception as e:
                self.console.print(f"  [status.fail]FAIL[/status.fail] {url} -- {e}")
        return agents

    # -- Messaging -----------------------------------------------------------

    async def send_message(
        self,
        agent: AgentConnection,
        text: str,
        session: PromptSession,
    ) -> None:
        """Send a message and handle the response, including input-required."""
        msg = _make_message(text, context_id=agent.context_id)

        with self.console.status("[info]Thinking...[/info]"):
            task = await _collect_task(agent.client.send_message(msg))

        if task is None:
            self.show_info("No response from agent")
            return

        if not hasattr(task, "status"):
            self.show_info(str(task))
            return

        agent.context_id = task.context_id

        # Handle input-required loop (deferred tools)
        while task.status.state == TaskState.input_required:
            status_msg = _extract_status_message(task)
            answer = await self._handle_deferred(status_msg, session)

            msg = _make_message(answer, context_id=agent.context_id)
            with self.console.status("[info]Thinking...[/info]"):
                task = await _collect_task(agent.client.send_message(msg))

            if task is None:
                self.show_error("No response from agent")
                return

        # Final result
        if task.status.state == TaskState.completed:
            text = _extract_text(task)
            if text:
                self.show_agent_response(agent.name, text)
            else:
                self.show_info(f"{agent.name}: completed (no content)")
        elif task.status.state == TaskState.failed:
            msg_text = _extract_status_message(task)
            self.show_error(f"{agent.name}: {msg_text}")
        else:
            self.show_info(f"{agent.name}: state={task.status.state}")

    async def _handle_deferred(self, status_msg: str, session: PromptSession) -> str:
        """Parse deferred tool calls and dispatch to the appropriate handler."""
        try:
            deferred_calls = json.loads(status_msg)
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON — show raw and ask for input
            self.console.print(
                Panel(
                    status_msg,
                    title="[status.wait]Input Required[/status.wait]",
                    border_style="yellow",
                )
            )
            self.console.print()
            answer = await session.prompt_async(HTML("<b>> </b>"))
            return answer.strip() or "proceed"

        if not isinstance(deferred_calls, list) or not deferred_calls:
            self.console.print()
            answer = await session.prompt_async(HTML("<b>> </b>"))
            return answer.strip() or "proceed"

        # Dispatch first call to the matching handler
        call = deferred_calls[0]
        tool_name = call.get("tool_name", "")
        args = call.get("arguments", {})

        handler = self.dispatcher.get(tool_name)
        return await handler.handle(args, self.console, session)

    # -- Main loop -----------------------------------------------------------

    async def run(self, urls: list[str]) -> None:
        if not urls:
            self.show_logo()
            self.console.print(
                "Usage: [cmd]uv run python demo_cli.py <url1> [url2] ...[/cmd]"
            )
            self.console.print(
                "Example: [cmd]uv run python demo_cli.py http://localhost:8002[/cmd]"
            )
            sys.exit(1)

        self.show_logo()
        suppress_console()  # hide loguru INFO/DEBUG — Rich UI only

        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as httpx_client:
            agents = await self.resolve_agents(urls, httpx_client)

            if not agents:
                self.show_error("No agents connected. Exiting.")
                sys.exit(1)

            current = 0
            self.show_agents_table(agents, current)
            self.show_help()
            self.console.print()

            session = PromptSession(history=InMemoryHistory())

            while True:
                try:
                    agent = agents[current]
                    prompt_text = (
                        f"<style fg='ansigreen' bg='' bold='true'>"
                        f"[{agent.name}]</style> <b>> </b>"
                    )
                    user_input = await session.prompt_async(HTML(prompt_text))
                except (EOFError, KeyboardInterrupt):
                    self.console.print("\n[info]Bye.[/info]")
                    break

                text = user_input.strip()
                if not text:
                    continue

                if text == "/quit":
                    self.console.print("[info]Bye.[/info]")
                    break

                if text == "/help":
                    self.show_help()
                    continue

                if text == "/agents":
                    self.show_agents_table(agents, current)
                    continue

                if text == "/clear":
                    agents[current].context_id = None
                    self.show_info(f"Context cleared for {agents[current].name}")
                    continue

                if text.startswith("/switch"):
                    parts = text.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        idx = int(parts[1]) - 1
                        if 0 <= idx < len(agents):
                            current = idx
                            self.show_info(f"Switched to: {agents[current].name}")
                        else:
                            self.show_error(f"Invalid. Use 1-{len(agents)}.")
                    else:
                        self.show_error("Usage: /switch <n>")
                    continue

                try:
                    await self.send_message(agents[current], text, session)
                except Exception as e:
                    self.show_error(str(e))
                self.console.print()
