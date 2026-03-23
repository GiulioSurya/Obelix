"""Interactive CLI client for A2A agents.

Connects to one or more A2A agent servers and provides a Rich-based
terminal UI with intelligent handling of deferred tool calls
(request_user_input, bash, etc.) via pluggable handlers.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    PushNotificationConfig,
    Role,
    TaskQueryParams,
    TextPart,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from obelix.adapters.inbound.a2a.client.handlers import HandlerDispatcher
from obelix.adapters.inbound.a2a.client.webhook_server import (
    TaskInfo,
    TaskTracker,
    WebhookServer,
)
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
        ██████╗ ██████╗ ███████╗██╗     ██╗██╗  ██╗
       ██╔═══██╗██╔══██╗██╔════╝██║     ██║╚██╗██╔╝
       ██║   ██║██████╔╝█████╗  ██║     ██║ ╚███╔╝
       ██║   ██║██╔══██╗██╔══╝  ██║     ██║ ██╔██╗
       ╚██████╔╝██████╔╝███████╗███████╗██║██╔╝ ██╗
        ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝[/bold cyan]

[dim]          ⚡ Multi-Agent LLM Framework ⚡
               A2A Protocol CLI[/dim]
"""


_COMMANDS = {
    "/agents": "List connected agents",
    "/switch": "Switch to agent n",
    "/tasks": "List background tasks and their status",
    "/task-info": "Show task details (last task or /task-info <id>)",
    "/clear": "Clear conversation context",
    "/help": "Show help",
    "/quit": "Exit",
}


class _SlashCompleter(Completer):
    """Auto-complete slash commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


# -- Agent Connection --------------------------------------------------------


class AgentConnection:
    """A resolved connection to a remote A2A agent."""

    __slots__ = (
        "name",
        "description",
        "skills",
        "client",
        "url",
        "context_id",
        "last_task_id",
    )

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
        self.last_task_id: str | None = None


# -- A2A message helpers -----------------------------------------------------


def _make_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )


def _make_data_message(
    data: dict[str, Any],
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    """Build a user message with a DataPart payload (for deferred tool responses)."""
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=DataPart(data=data))],
        context_id=context_id,
        task_id=task_id,
    )


# Regex: @<path> where path can be quoted ("@/path with spaces/f.pdf")
# or unquoted (@/simple/path.png).  Supports forward and back slashes.
_ATTACH_RE = re.compile(r'@"([^"]+)"|@(\S+)')


def _parse_attachments(text: str) -> tuple[str, list[Part]]:
    """Extract @path references from user input.

    Returns:
        (clean_text, file_parts) — text with @refs removed,
        and a list of FilePart wrapped in Part for each valid file.
    """
    file_parts: list[Part] = []
    matches = list(_ATTACH_RE.finditer(text))
    if not matches:
        return text, []

    for m in matches:
        raw_path = m.group(1) or m.group(2)
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue

        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "application/octet-stream"

        data = base64.b64encode(path.read_bytes()).decode("ascii")
        file_parts.append(
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=data,
                        mime_type=mime,
                        name=path.name,
                    )
                )
            )
        )

    # Remove @refs from the text
    clean = _ATTACH_RE.sub("", text).strip()
    # Collapse multiple spaces
    clean = re.sub(r"  +", " ", clean)
    return clean, file_parts


def _extract_text(task) -> str:
    if not hasattr(task, "artifacts") or not task.artifacts:
        return ""
    parts = []
    for artifact in task.artifacts:
        for part in artifact.parts:
            if hasattr(part.root, "text") and part.root.text:
                parts.append(part.root.text)
    return "".join(parts)


def _extract_status_data(task) -> dict | None:
    """Extract DataPart data from task status message, or None."""
    if not task.status or not task.status.message:
        return None
    for part in task.status.message.parts:
        if isinstance(part.root, DataPart):
            return part.root.data
    return None


def _extract_status_text(task) -> str:
    """Extract text from task status message."""
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
        self.tracker = TaskTracker()
        self._webhook_server: WebhookServer | None = None
        self._webhook_url: str | None = None

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
        for cmd, desc in _COMMANDS.items():
            label = f"{cmd} <n>" if cmd == "/switch" else cmd
            help_table.add_row(label, desc)
        help_table.add_row("", "")
        help_table.add_row("@<path>", "Attach a file (image, PDF, ...)")
        help_table.add_row('@"path with spaces"', "Attach a file with spaces in path")
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

    async def show_task_info(self, agent: AgentConnection) -> None:
        """Fetch the last task via tasks/get and display its details and messages."""
        if not agent.last_task_id:
            self.show_info("No task in this session yet.")
            return

        try:
            task = await agent.client.get_task(TaskQueryParams(id=agent.last_task_id))
        except Exception as e:
            self.show_error(f"tasks/get failed: {e}")
            return

        if not task.history:
            self.show_info("Task has no history.")
            return

        table = Table(
            title=f"Task {task.id} — Messages",
            title_style="bold",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=3, justify="right")
        table.add_column("Role", style="bold", width=8)
        table.add_column("Content")

        for i, msg in enumerate(task.history):
            role = msg.role if hasattr(msg, "role") else None
            role_label = role.value if role else "?"
            role_style = "green" if role == Role.user else "cyan"
            parts_text = []
            for part in msg.parts:
                if hasattr(part.root, "text") and part.root.text:
                    parts_text.append(part.root.text)
                elif isinstance(part.root, DataPart):
                    parts_text.append(f"[dim]DataPart: {part.root.data}[/dim]")
                else:
                    parts_text.append(f"[dim]{type(part.root).__name__}[/dim]")
            content = "\n".join(parts_text) or "[dim](empty)[/dim]"
            table.add_row(
                str(i + 1), f"[{role_style}]{role_label}[/{role_style}]", content
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_tasks(self) -> None:
        """Show all background tasks."""
        tasks = self.tracker.get_all()
        if not tasks:
            self.show_info("No background tasks.")
            return

        table = Table(
            title="Background Tasks",
            title_style="bold",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("Task ID", style="dim", max_width=12)
        table.add_column("Agent", style="agent.name")
        table.add_column("State", width=16)

        _state_styles = {
            "submitted": "dim",
            "working": "bold yellow",
            "input-required": "bold red",
            "completed": "bold green",
            "failed": "bold red",
            "canceled": "dim",
            "rejected": "bold red",
        }

        for t in tasks:
            style = _state_styles.get(t.state, "dim")
            state_label = t.state
            if t.state == "input-required":
                state_label = "⚠ input-required"
            elif t.state == "completed":
                state_label = "✓ completed"
            elif t.state == "failed":
                state_label = "✗ failed"
            table.add_row(
                t.task_id[:12],
                t.agent_name,
                f"[{style}]{state_label}[/{style}]",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _bottom_toolbar_html(self) -> HTML:
        """Build the bottom toolbar as HTML for prompt_toolkit.

        Groups tasks by agent — shows only the latest task per agent,
        so repeated messages to the same server don't pile up.
        """
        all_tasks = self.tracker.get_all()
        if not all_tasks:
            return HTML("<b> Tasks: none</b>")

        # Group by agent: keep only the most recent task per agent
        latest_by_agent: dict[str, TaskInfo] = {}
        for t in all_tasks:
            prev = latest_by_agent.get(t.agent_name)
            if prev is None or t.timestamp >= prev.timestamp:
                latest_by_agent[t.agent_name] = t

        active = [t for t in latest_by_agent.values() if not t.is_terminal]
        pending = [t for t in latest_by_agent.values() if t.state == "input-required"]

        segments = []

        if active:
            segments.append(f"{len(active)} active")
        if pending:
            segments.append(
                f'<style bg="ansired" fg="ansiwhite"> {len(pending)} input-required! </style>'
            )

        for t in latest_by_agent.values():
            if t.state == "input-required":
                segments.append(
                    f'<style fg="ansired"><b>[{t.agent_name}: ⚠ INPUT]</b></style>'
                )
            elif t.state == "working":
                segments.append(
                    f'<style fg="ansiyellow">[{t.agent_name}: working]</style>'
                )
            elif t.state == "completed":
                segments.append(f'<style fg="ansigreen">[{t.agent_name}: ✓]</style>')
            elif t.state == "failed":
                segments.append(f'<style fg="ansired">[{t.agent_name}: ✗]</style>')
            elif t.is_terminal:
                segments.append(f"[{t.agent_name}: {t.state}]")
            else:
                segments.append(f"[{t.agent_name}: {t.state}]")

        return HTML("<b> Tasks: </b>" + " | ".join(segments))

    def show_error(self, msg: str) -> None:
        self.console.print(f"[status.fail]  Error:[/status.fail] {msg}")

    def show_info(self, msg: str) -> None:
        self.console.print(f"[info]{msg}[/info]")

    # -- Connection ----------------------------------------------------------

    async def resolve_agents(
        self, urls: list[str], httpx_client: httpx.AsyncClient
    ) -> list[AgentConnection]:
        agents = []
        push_configs = []
        if self._webhook_url:
            push_configs = [PushNotificationConfig(url=self._webhook_url)]
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=False,
            polling=True,
            push_notification_configs=push_configs,
        )
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
        """Send a message (non-blocking) and register the task for push tracking."""
        # Parse @path attachments from the input
        clean_text, file_parts = _parse_attachments(text)
        if file_parts:
            self.show_info(f"Attached {len(file_parts)} file(s)")
            parts = []
            if clean_text:
                parts.append(Part(root=TextPart(text=clean_text)))
            parts.extend(file_parts)
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.user,
                parts=parts,
                context_id=agent.context_id,
            )
        else:
            msg = _make_message(clean_text or text, context_id=agent.context_id)

        # Non-blocking: send and get initial Task(working) back immediately
        task = await _collect_task(agent.client.send_message(msg))

        if task is None:
            self.show_info("No response from agent")
            return

        if not hasattr(task, "status"):
            self.show_info(str(task))
            return

        agent.context_id = task.context_id
        if hasattr(task, "id"):
            agent.last_task_id = task.id

        # Register in tracker for push notification updates
        await self.tracker.register(task.id, agent.name)
        await self.tracker.update(task.model_dump(mode="json", exclude_none=True))

        self.show_info(
            f"Task {task.id[:12]}... sent to {agent.name} "
            f"[status.wait][{task.status.state.value}][/status.wait]"
        )

    async def handle_pending_input(
        self,
        agent: AgentConnection,
        session: PromptSession,
    ) -> bool:
        """Check if agent has a pending input-required task and handle it.

        Returns True if a deferred interaction was handled.
        """
        pending = self.tracker.get_pending_input(agent.name)
        if not pending or not pending.task_data:
            return False

        task_data = pending.task_data
        status = task_data.get("status", {})
        message = status.get("message", {})
        parts = message.get("parts", [])

        # Extract DataPart from status message
        data = None
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "data":
                data = part.get("data")
                break
            if isinstance(part, dict) and "data" in part:
                data = part["data"]
                break

        if data is None:
            return False

        self.console.print(
            f"\n[status.wait]⚠ Task {pending.task_id[:12]}... needs input[/status.wait]"
        )

        result = await self._handle_deferred(data, session)

        # Send the deferred response back
        msg = _make_data_message(
            result,
            context_id=agent.context_id,
            task_id=pending.task_id,
        )

        # This resume call is also non-blocking — push will notify on completion
        task = await _collect_task(agent.client.send_message(msg))
        if task and hasattr(task, "id"):
            await self.tracker.update(task.model_dump(mode="json", exclude_none=True))
            self.show_info(
                f"Response sent for task {pending.task_id[:12]}... "
                f"[status.wait][{task.status.state.value}][/status.wait]"
            )

        return True

    def _show_completed_task(self, agent_name: str, task_data: dict) -> None:
        """Display a completed task result inline."""
        # Extract text from artifacts
        artifacts = task_data.get("artifacts", [])
        text_parts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if isinstance(part, dict):
                    root = part.get("root", part)
                    if "text" in root and root["text"]:
                        text_parts.append(root["text"])
                    elif "text" in part and part["text"]:
                        text_parts.append(part["text"])

        text = "".join(text_parts)
        if text:
            self.show_agent_response(agent_name, text)
        else:
            self.show_info(f"{agent_name}: completed (no content)")

    def _show_terminal_result(self, task_data: dict, agent_name: str) -> None:
        """Show result for a terminal-state task."""
        state = task_data.get("status", {}).get("state", "")
        if state == "completed":
            self._show_completed_task(agent_name, task_data)
        elif state == "rejected":
            status_msg = task_data.get("status", {}).get("message", {})
            parts = status_msg.get("parts", [])
            reason = ""
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    reason = p["text"]
                    break
            self.console.print(
                Panel(
                    reason or "No reason provided",
                    title=f"[status.fail]{agent_name} rejected[/status.fail]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
        elif state == "failed":
            status_msg = task_data.get("status", {}).get("message", {})
            parts = status_msg.get("parts", [])
            err_text = ""
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    err_text = p["text"]
                    break
            self.show_error(f"{agent_name}: {err_text}")

    async def _handle_deferred(self, data: dict | None, session: PromptSession) -> dict:
        """Dispatch deferred tool calls from DataPart to the appropriate handler."""
        if not data or "deferred_tool_calls" not in data:
            self.console.print(
                Panel(
                    str(data),
                    title="[status.wait]Input Required[/status.wait]",
                    border_style="yellow",
                )
            )
            self.console.print()
            answer = await session.prompt_async(HTML("<b>> </b>"))
            return {"answer": answer.strip() or "proceed"}

        deferred_calls = data["deferred_tool_calls"]
        if not isinstance(deferred_calls, list) or not deferred_calls:
            self.console.print()
            answer = await session.prompt_async(HTML("<b>> </b>"))
            return {"answer": answer.strip() or "proceed"}

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

        # Start webhook server for push notifications
        self._webhook_server = WebhookServer(self.tracker)
        await self._webhook_server.start()
        self._webhook_url = self._webhook_server.get_url()
        self.show_info(f"Webhook server listening on {self._webhook_url}")

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as httpx_client:
                agents = await self.resolve_agents(urls, httpx_client)

                if not agents:
                    self.show_error("No agents connected. Exiting.")
                    sys.exit(1)

                current = 0
                self.show_agents_table(agents, current)
                self.show_help()
                self.console.print()

                session = PromptSession(
                    history=InMemoryHistory(),
                    completer=_SlashCompleter(),
                    style=PTStyle.from_dict(
                        {
                            "completion-menu": "bg:#1a1a2e #e0e0e0",
                            "completion-menu.completion": "bg:#1a1a2e #e0e0e0",
                            "completion-menu.completion.current": "bg:#00d4aa #1a1a2e bold",
                            "completion-menu.meta": "bg:#1a1a2e #888888 italic",
                            "completion-menu.meta.current": "bg:#00d4aa #1a1a2e italic",
                        }
                    ),
                    bottom_toolbar=lambda: self._bottom_toolbar_html(),
                )

                # Background watcher: shows results as soon as push arrives
                _shown_results: set[str] = set()

                async def _result_watcher() -> None:
                    """Background task that shows results via run_in_terminal."""
                    while True:
                        await asyncio.sleep(0.5)
                        for t in self.tracker.get_all():
                            if (
                                t.is_terminal
                                and t.task_id not in _shown_results
                                and t.task_data
                            ):
                                _shown_results.add(t.task_id)
                                task_data = t.task_data
                                agent_name = t.agent_name

                                # Use run_in_terminal to safely print while
                                # prompt_toolkit is showing the prompt
                                def _print_result(_data=task_data, _name=agent_name):
                                    self._show_terminal_result(_data, _name)

                                try:
                                    await session.app.run_in_terminal_async(
                                        _print_result
                                    )
                                except Exception:
                                    # Fallback: just print directly
                                    _print_result()
                        # Refresh toolbar
                        try:
                            session.app.invalidate()
                        except Exception:
                            pass

                watcher_task = asyncio.create_task(_result_watcher())

                while True:
                    try:
                        agent = agents[current]

                        # Check for pending input-required on current agent
                        pending = self.tracker.get_pending_input(agent.name)
                        if pending:
                            handled = await self.handle_pending_input(agent, session)
                            if handled:
                                continue

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

                    if text == "/tasks":
                        self.show_tasks()
                        continue

                    if text.startswith("/task-info"):
                        parts_cmd = text.split()
                        if len(parts_cmd) == 2:
                            # /task-info <id> — look up by partial id
                            partial = parts_cmd[1]
                            info = None
                            for t in self.tracker.get_all():
                                if t.task_id.startswith(partial):
                                    info = t
                                    break
                            if info and info.task_data:
                                self._show_terminal_result(
                                    info.task_data, info.agent_name
                                )
                            elif info:
                                self.show_info(
                                    f"Task {info.task_id[:12]}... state={info.state}"
                                )
                            else:
                                # Fallback to tasks/get via API
                                await self.show_task_info(agents[current])
                        else:
                            await self.show_task_info(agents[current])
                        continue

                    if text == "/clear":
                        agents[current].context_id = None
                        agents[current].last_task_id = None
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
                watcher_task.cancel()
        finally:
            if self._webhook_server:
                await self._webhook_server.stop()
