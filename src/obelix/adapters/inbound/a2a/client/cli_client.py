"""Interactive CLI client for A2A agents.

Connects to one or more A2A agent servers and provides a Textual-based
full-screen terminal UI with intelligent handling of deferred tool calls
(request_user_input, bash, etc.) via pluggable handlers.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import platform
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from a2a.client import (
    A2ACardResolver,
    A2AClientHTTPError,
    A2AClientJSONError,
    A2AClientTimeoutError,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    PushNotificationConfig,
    Role,
    TaskIdParams,
    TaskQueryParams,
    TextPart,
)
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from obelix.adapters.inbound.a2a.client.handlers import HandlerDispatcher
from obelix.adapters.inbound.a2a.client.webhook_server import (
    TaskTracker,
    WebhookServer,
)
from obelix.infrastructure.logging import suppress_console

# -- Logo & commands ---------------------------------------------------------

LOGO = Text.from_markup(
    r"""[bold cyan]
        ██████╗ ██████╗ ███████╗██╗     ██╗██╗  ██╗
       ██╔═══██╗██╔══██╗██╔════╝██║     ██║╚██╗██╔╝
       ██║   ██║██████╔╝█████╗  ██║     ██║ ╚███╔╝
       ██║   ██║██╔══██╗██╔══╝  ██║     ██║ ██╔██╗
       ╚██████╔╝██████╔╝███████╗███████╗██║██╔╝ ██╗
        ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝[/bold cyan]

[dim]          ⚡ Multi-Agent LLM Framework ⚡
               A2A Protocol CLI[/dim]
"""
)

_COMMANDS = {
    "/agents": "List connected agents",
    "/switch <n>": "Switch to agent n",
    "/tasks": "List background tasks and their status",
    "/task-info": "Show task details (last task or /task-info <id>)",
    "/clear": "Clear conversation context",
    "/help": "Show help",
    "/quit": "Exit",
}

# Autocomplete data: (base_cmd, description, has_args)
_SLASH_COMPLETIONS: list[tuple[str, str, bool]] = [
    (full.split()[0], desc, "<" in full) for full, desc in _COMMANDS.items()
]


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


# -- Shell probe -------------------------------------------------------------


def _probe_shell_info() -> dict:
    """Detect the local shell environment for the server's system prompt.

    Returns a dict compatible with BashTool.system_prompt_fragment():
    platform, os_version, shell_name, cwd, and mounts (Windows only).
    """
    info: dict[str, Any] = {
        "platform": platform.system(),
        "os_version": platform.platform(),
    }

    # Find the best available shell
    shell_path = None
    if platform.system() == "Windows":
        # Prefer Git Bash
        for candidate in [
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ]:
            if candidate.is_file():
                shell_path = str(candidate)
                break
    if not shell_path:
        shell_path = shutil.which("bash") or shutil.which("sh") or "sh"

    info["shell"] = shell_path
    info["shell_name"] = Path(shell_path).stem

    # Probe cwd and mounts
    probe_cmd = 'echo "CWD=$(pwd)" && echo "HOME=$HOME"'
    if platform.system() == "Windows":
        probe_cmd += ' && mount | grep "^[A-Z]:"'

    try:
        result = subprocess.run(
            [shell_path, "-c", probe_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            mounts: dict[str, str] = {}
            for line in result.stdout.splitlines():
                if line.startswith("CWD="):
                    info["cwd"] = line[4:]
                elif line.startswith("HOME="):
                    info["home"] = line[5:]
                else:
                    m = re.match(r"^([A-Z]:)\s+on\s+(/\S+)", line)
                    if m:
                        mp = m.group(2)
                        if not mp.endswith("/"):
                            mp += "/"
                        mounts[m.group(1)] = mp
            if mounts:
                info["mounts"] = mounts
    except (subprocess.TimeoutExpired, OSError):
        pass  # probe failed — send what we have

    return info


# -- A2A message helpers -----------------------------------------------------


def _make_message(
    text: str,
    context_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
        metadata=metadata,
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
    """Extract @path references from user input."""
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
                    file=FileWithBytes(bytes=data, mime_type=mime, name=path.name)
                )
            )
        )

    clean = _ATTACH_RE.sub("", text).strip()
    clean = re.sub(r"  +", " ", clean)
    return clean, file_parts


async def _collect_task(response_iter):
    """Consume the async iterator and return the final task."""
    task = None
    async for item in response_iter:
        if isinstance(item, tuple):
            task = item[0]
        elif isinstance(item, Message):
            return item
    return task


# -- Rich renderable helpers -------------------------------------------------


def _build_agents_table(agents: list[AgentConnection], current: int) -> Table:
    table = Table(
        title="Connected Agents", title_style="bold", show_lines=False, padding=(0, 1)
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Agent", style="bold cyan")
    table.add_column("Description")
    table.add_column("Skills", style="dim")
    table.add_column("URL", style="dim")

    for i, agent in enumerate(agents):
        marker = " *" if i == current else ""
        skills_str = ", ".join(agent.skills[:3]) if agent.skills else "-"
        desc = agent.description[:50] + ("..." if len(agent.description) > 50 else "")
        table.add_row(f"{i + 1}{marker}", agent.name, desc, skills_str, agent.url)
    return table


def _build_help_table() -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="bold magenta")
    table.add_column("Description")
    for cmd, desc in _COMMANDS.items():
        table.add_row(cmd, desc)
    table.add_row("", "")
    table.add_row("@<path>", "Attach a file (image, PDF, ...)")
    table.add_row('@"path with spaces"', "Attach a file with spaces in path")
    table.add_row("", "")
    table.add_row("ESC", "Cancel the active task for the current agent")
    return table


def _extract_artifact_text(task_data: dict) -> str:
    """Extract text from task artifacts (push notification JSON format)."""
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
    return "".join(text_parts)


# -- Console adapter for handlers -------------------------------------------


class _RichLogConsole:
    """Adapter so handler.render(args, console) writes to a RichLog widget."""

    def __init__(self, richlog: RichLog):
        self._log = richlog

    def print(self, *args, **kwargs):
        if not args:
            self._log.write("")
            return
        for renderable in args:
            self._log.write(renderable)


# -- Textual App -------------------------------------------------------------


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class CLIClient(App):
    """Full-screen Textual client for A2A agents."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #chat {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #status {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    #autocomplete {
        height: auto;
        max-height: 8;
        display: none;
        background: $surface;
    }
    #input {
        height: auto;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        dispatcher: HandlerDispatcher,
        urls: list[str] | None = None,
        webhook_host: str | None = None,
    ):
        super().__init__()
        self.dispatcher = dispatcher
        self.urls = urls or []
        self._webhook_host = webhook_host
        self.tracker = TaskTracker()
        self.agents: list[AgentConnection] = []
        self.current = 0
        self._webhook_server: WebhookServer | None = None
        self._webhook_url: str | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._shown_results: set[str] = set()
        self._poll_warned: set[str] = set()  # task IDs already warned about
        self._last_poll: dict[str, float] = {}  # task_id -> last poll timestamp
        self._unseen_by_agent: dict[str, list[str]] = {}
        self._input_future: asyncio.Future | None = None
        self._handling_deferred: bool = False
        self._canceling: bool = False
        self._ac_suppress: bool = False
        self._agent_select_mode: bool = False
        self._agent_select_idx: int = 0
        self._shell_info: dict = _probe_shell_info()

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> CLIClient:
        """Create a CLIClient from command-line arguments.

        Args:
            argv: Argument list (defaults to ``sys.argv[1:]``).

        Usage::

            CLIClient.from_cli().run()
        """
        import argparse
        import sys

        from obelix.adapters.inbound.a2a.client.handlers import (
            default_dispatcher,
        )

        parser = argparse.ArgumentParser(
            description="Interactive CLI client for A2A agents",
        )
        parser.add_argument(
            "urls",
            nargs="*",
            help="A2A server URLs (e.g. http://localhost:8002)",
        )
        parser.add_argument(
            "--webhook-host",
            default=None,
            help=(
                "Hostname for the push notification webhook. "
                "Use when the server can't reach 127.0.0.1 "
                "(e.g. server in Docker/K8s). "
                "Also settable via OBELIX_WEBHOOK_HOST env var."
            ),
        )
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])

        if not args.urls:
            parser.print_help()
            sys.exit(1)

        return cls(
            dispatcher=default_dispatcher(),
            urls=args.urls,
            webhook_host=args.webhook_host,
        )

    # -- Layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat", markup=True, highlight=False, wrap=True)
        yield Static("Starting...", id="status")
        yield OptionList(id="autocomplete")
        yield Input(id="input", placeholder="Type a message or /help")

    # -- Lifecycle -----------------------------------------------------------

    async def on_mount(self) -> None:
        suppress_console()
        chat = self.query_one("#chat", RichLog)
        chat.write(LOGO)

        if not self.urls:
            chat.write(
                Text("Usage: uv run python examples/cli_client.py <url>", style="bold")
            )
            return

        self._connect_agents()

    @work(exclusive=True, group="setup")
    async def _connect_agents(self) -> None:
        chat = self.query_one("#chat", RichLog)

        # Webhook server
        self._webhook_server = WebhookServer(
            self.tracker, webhook_host=self._webhook_host
        )
        await self._webhook_server.start()
        self._webhook_url = self._webhook_server.get_url()
        chat.write(Text(f"  Webhook: {self._webhook_url}", style="dim cyan"))

        # Resolve agents
        self._httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        push_configs = [PushNotificationConfig(url=self._webhook_url)]
        config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=False,
            polling=True,
            push_notification_configs=push_configs,
        )
        factory = ClientFactory(config)

        for url in self.urls:
            try:
                resolver = A2ACardResolver(
                    httpx_client=self._httpx_client, base_url=url
                )
                card = await resolver.get_agent_card()
                client = factory.create(card)
                skills = [s.name for s in card.skills] if card.skills else []
                self.agents.append(
                    AgentConnection(
                        name=card.name,
                        description=card.description or "",
                        skills=skills,
                        client=client,
                        url=url,
                    )
                )
                chat.write(Text(f"  ✓ {card.name} ({url})", style="bold green"))
            except Exception as e:
                chat.write(Text(f"  ✗ {url} — {e}", style="bold red"))

        if not self.agents:
            chat.write(Text("\nNo agents connected.", style="bold red"))
            return

        chat.write("")
        chat.write(_build_agents_table(self.agents, self.current))
        chat.write("")
        chat.write(Panel(_build_help_table(), title="Commands", border_style="dim"))
        chat.write("")

        # Update input placeholder with agent name
        self._update_input_placeholder()

        # Start background task watcher
        self.set_interval(0.15, self._poll_tasks)

    # -- Background task polling ---------------------------------------------

    def _poll_tasks(self) -> None:
        """Called periodically. Updates status bar and shows completed results.

        Includes a **polling fallback**: if a task has been non-terminal for
        more than 1 second (suggesting push notification didn't arrive),
        actively polls the server via ``get_task()`` to retrieve the current
        state.  This ensures the client always gets results, even when the
        server can't reach the webhook endpoint (e.g. Docker networking).
        """
        # Update status bar
        self._update_status_bar()

        # Polling fallback for tasks stuck in non-terminal state
        now = time.time()
        _POLL_AFTER_SECONDS = 1.0
        _POLL_INTERVAL_SECONDS = 1.0
        for t in self.tracker.get_active():
            if now - t.timestamp > _POLL_AFTER_SECONDS:
                last = self._last_poll.get(t.task_id, 0.0)
                if now - last >= _POLL_INTERVAL_SECONDS:
                    self._last_poll[t.task_id] = now
                    self._polling_fallback(t.task_id, t.agent_name)

        # Show completed results for current agent
        current_name = self.agents[self.current].name if self.agents else ""
        for t in self.tracker.get_all():
            if t.is_terminal and t.task_id not in self._shown_results and t.task_data:
                self._shown_results.add(t.task_id)
                if t.agent_name == current_name:
                    self._show_result(t.task_data, t.agent_name)
                else:
                    self._unseen_by_agent.setdefault(t.agent_name, []).append(t.task_id)

        # Check for input-required on current agent
        if self.agents and not self._handling_deferred and not self._canceling:
            pending = self.tracker.get_pending_input(current_name)
            if pending and pending.task_data:
                self._handling_deferred = True
                self._handle_pending_input(pending)

    @work(exclusive=False, group="polling")
    async def _polling_fallback(self, task_id: str, agent_name: str) -> None:
        """Poll the server for a task's current state (fallback for failed push).

        Called when a task has been non-terminal for longer than expected,
        suggesting the push notification didn't reach the client.
        """
        # Find the agent connection
        agent = None
        for a in self.agents:
            if a.name == agent_name:
                agent = a
                break
        if not agent:
            return

        chat = self.query_one("#chat", RichLog)
        try:
            task = await agent.client.get_task(TaskQueryParams(id=task_id))
        except A2AClientHTTPError as exc:
            if exc.status_code >= 500:
                return  # retriable — server error, will retry next cycle
            # 4xx = client error, stop retrying this task
            chat.write(
                Text(f"  [poll] HTTP {exc.status_code}: {exc.message}", style="dim red")
            )
            self._last_poll[task_id] = float("inf")  # never retry
            return
        except A2AClientTimeoutError:
            return  # retriable — will retry next cycle
        except A2AClientJSONError as exc:
            chat.write(Text(f"  [poll] bad response: {exc.message}", style="dim red"))
            self._last_poll[task_id] = float("inf")  # never retry
            return
        except Exception as exc:
            chat.write(Text(f"  [poll] unexpected error: {exc}", style="dim red"))
            self._last_poll[task_id] = float("inf")
            return

        if task is None:
            return

        task_data = task.model_dump(mode="json", exclude_none=True)
        status = task_data.get("status", {})
        state = status.get("state", "unknown")

        # Only update if the server has a newer state
        tracked = self.tracker.get(task_id)
        if tracked and tracked.state == state:
            return  # No change

        # Log warning on first poll recovery for this task
        if task_id not in self._poll_warned:
            self._poll_warned.add(task_id)
            chat = self.query_one("#chat", RichLog)
            chat.write(
                Text(
                    f"  ⚠ Push notification not received for task {task_id[:8]}… "
                    f"— recovered via polling (state: {state})",
                    style="dim yellow",
                )
            )

        await self.tracker.update(task_data)

    def _update_status_bar(self) -> None:
        if self._agent_select_mode:
            return  # Don't overwrite agent selector
        all_tasks = self.tracker.get_all()
        status = self.query_one("#status", Static)

        if not all_tasks:
            status.update("Ready")
            return

        latest: dict[str, Any] = {}
        for t in all_tasks:
            prev = latest.get(t.agent_name)
            if prev is None or t.timestamp >= prev.timestamp:
                latest[t.agent_name] = t

        frame = _SPINNER[int(time.time() * 8) % len(_SPINNER)]
        segments = []

        for t in latest.values():
            unseen = len(self._unseen_by_agent.get(t.agent_name, []))
            if t.state == "input-required":
                segments.append(f"[bold red]⚠ {t.agent_name}: INPUT[/]")
            elif t.state in ("working", "submitted"):
                segments.append(f"[yellow]{frame} {t.agent_name} is thinking...[/]")
            elif t.state == "completed":
                if unseen > 0:
                    segments.append(f"[bold green]{t.agent_name}: {unseen} new[/]")
                else:
                    segments.append(f"[green]✓ {t.agent_name}[/]")
            elif t.state == "failed":
                if unseen > 0:
                    segments.append(f"[bold red]{t.agent_name}: {unseen} new[/]")
                else:
                    segments.append(f"[red]✗ {t.agent_name}[/]")
            else:
                segments.append(f"{t.agent_name}: {t.state}")

        status.update(Text.from_markup(" | ".join(segments)))

    # -- Display helpers -----------------------------------------------------

    def _show_result(self, task_data: dict, agent_name: str) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.auto_scroll = True
        state = task_data.get("status", {}).get("state", "")

        if state == "completed":
            text = _extract_artifact_text(task_data)
            if text:
                chat.write(
                    Panel(
                        Markdown(text),
                        title=f"[bold cyan]{agent_name}[/]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
            else:
                chat.write(
                    Text(f"{agent_name}: completed (no content)", style="dim cyan")
                )
        elif state == "rejected":
            reason = self._extract_status_text(task_data)
            chat.write(
                Panel(
                    reason or "No reason",
                    title=f"[bold red]{agent_name} rejected[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
        elif state == "failed":
            err = self._extract_status_text(task_data)
            chat.write(Text(f"✗ {agent_name}: {err}", style="bold red"))

        chat.scroll_end(animate=False)

    @staticmethod
    def _extract_status_text(task_data: dict) -> str:
        status_msg = task_data.get("status", {}).get("message", {})
        parts = status_msg.get("parts", [])
        for p in parts:
            if isinstance(p, dict) and "text" in p:
                return p["text"]
        return ""

    def _update_input_placeholder(self) -> None:
        if self.agents:
            agent = self.agents[self.current]
            inp = self.query_one("#input", Input)
            inp.placeholder = f"[{agent.name}] Type a message or /help"

    # -- Autocomplete --------------------------------------------------------

    @on(Input.Changed, "#input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide slash-command autocomplete as the user types."""
        if self._ac_suppress:
            self._ac_suppress = False
            return

        ac = self.query_one("#autocomplete", OptionList)
        text = event.value

        if text.startswith("/") and " " not in text:
            prefix = text.lower()
            matches = [
                (base, desc)
                for base, desc, _ in _SLASH_COMPLETIONS
                if base.startswith(prefix)
            ]
            ac.clear_options()
            for base, desc in matches:
                ac.add_option(Option(f"{base}  {desc}", id=base))
            ac.display = bool(matches)
            if matches:
                ac.highlighted = 0
        else:
            ac.display = False

    def on_key(self, event) -> None:
        """Route keys to autocomplete or agent selector when active."""
        # -- Autocomplete (highest priority) --
        ac = self.query_one("#autocomplete", OptionList)
        if ac.display:
            if event.key == "up":
                if ac.highlighted is not None and ac.highlighted > 0:
                    ac.highlighted -= 1
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                if ac.highlighted is not None and ac.highlighted < ac.option_count - 1:
                    ac.highlighted += 1
                event.prevent_default()
                event.stop()
            elif event.key == "tab":
                self._accept_autocomplete()
                event.prevent_default()
                event.stop()
            elif event.key == "escape":
                ac.display = False
                event.prevent_default()
                event.stop()
            return

        # -- Agent selector mode --
        if self._agent_select_mode:
            if event.key == "left":
                if self._agent_select_idx > 0:
                    self._agent_select_idx -= 1
                    self._render_agent_selector()
                event.prevent_default()
                event.stop()
            elif event.key == "right":
                if self._agent_select_idx < len(self.agents) - 1:
                    self._agent_select_idx += 1
                    self._render_agent_selector()
                event.prevent_default()
                event.stop()
            elif event.key == "enter":
                self._switch_to_agent(self._agent_select_idx)
                self._agent_select_mode = False
                event.prevent_default()
                event.stop()
            elif event.key in ("escape", "down"):
                self._agent_select_mode = False
                self._update_status_bar()
                event.prevent_default()
                event.stop()
            return

        # -- ESC: cancel current task --
        if event.key == "escape":
            if self._canceling:
                return
            # Set canceling flag immediately to prevent _poll_tasks from
            # re-dispatching the deferred panel during the cancel window.
            self._canceling = True
            # If a deferred handler is waiting for input, cancel it
            if self._input_future and not self._input_future.done():
                self._input_future.set_result(None)  # sentinel: canceled
                event.prevent_default()
                event.stop()
            # Cancel active (non-terminal) task for current agent
            if self.agents:
                agent = self.agents[self.current]
                if agent.last_task_id:
                    info = self.tracker.get(agent.last_task_id)
                    if info and not info.is_terminal:
                        self._cancel_current_task()
                        event.prevent_default()
                        event.stop()
                        return
            # No task to cancel — clear the flag
            self._canceling = False
            return

        # -- Enter agent selector (up arrow, empty input, multiple agents) --
        if event.key == "up" and len(self.agents) > 1:
            inp = self.query_one("#input", Input)
            if not inp.value:
                self._agent_select_mode = True
                self._agent_select_idx = self.current
                self._render_agent_selector()
                event.prevent_default()
                event.stop()

    def _render_agent_selector(self) -> None:
        """Render the status bar as an agent picker."""
        status = self.query_one("#status", Static)
        segments = []
        for i, agent in enumerate(self.agents):
            if i == self._agent_select_idx:
                segments.append(f"[bold reverse] {agent.name} [/]")
            elif i == self.current:
                segments.append(f"[bold cyan]{agent.name}[/]")
            else:
                segments.append(f"[dim]{agent.name}[/]")
        hint = "[dim italic]  ◀▶ navigate  Enter switch  ↓ cancel[/]"
        status.update(Text.from_markup(" | ".join(segments) + hint))

    def _accept_autocomplete(self) -> None:
        """Fill input with the selected autocomplete option."""
        ac = self.query_one("#autocomplete", OptionList)
        if ac.highlighted is not None:
            option = ac.get_option_at_index(ac.highlighted)
            base = str(option.id)
            has_arg = any(b == base and a for b, _, a in _SLASH_COMPLETIONS)
            inp = self.query_one("#input", Input)
            self._ac_suppress = True
            inp.value = f"{base} " if has_arg else base
            inp.cursor_position = len(inp.value)
        ac.display = False

    # -- Input handling ------------------------------------------------------

    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()

        # If a deferred handler is waiting for input, resolve the future
        if self._input_future and not self._input_future.done():
            self._input_future.set_result(text or "proceed")
            return

        if not text:
            return

        chat = self.query_one("#chat", RichLog)

        # Echo user input
        if not text.startswith("/"):
            agent_name = self.agents[self.current].name if self.agents else "?"
            chat.write(Text(f"[{agent_name}] > {text}", style="bold green"))

        # -- Slash commands --
        if text == "/quit":
            self.exit()
            return

        if text == "/help":
            chat.write(Panel(_build_help_table(), title="Commands", border_style="dim"))
            chat.scroll_end(animate=False)
            return

        if text == "/agents":
            if self.agents:
                chat.write(_build_agents_table(self.agents, self.current))
            chat.scroll_end(animate=False)
            return

        if text == "/tasks":
            self._show_tasks()
            return

        if text.startswith("/task-info"):
            self._show_task_info(text)
            return

        if text == "/clear":
            if self.agents:
                self.agents[self.current].context_id = None
                self.agents[self.current].last_task_id = None
                chat.write(
                    Text(
                        f"Context cleared for {self.agents[self.current].name}",
                        style="dim cyan",
                    )
                )
            return

        if text.startswith("/switch"):
            self._handle_switch(text)
            return

        if text.startswith("/"):
            chat.write(Text(f"Unknown command: {text}. Type /help", style="bold red"))
            return

        # -- Send message to agent --
        if not self.agents:
            chat.write(Text("No agents connected.", style="bold red"))
            return

        self._send_message(text)

    @work(thread=False, group="cancel")
    async def _cancel_current_task(self) -> None:
        """Cancel the active task for the current agent via A2A cancel_task."""
        if not self.agents:
            return
        agent = self.agents[self.current]
        task_id = agent.last_task_id
        if not task_id:
            return

        self._canceling = True
        chat = self.query_one("#chat", RichLog)
        chat.write(Text(f"  Canceling task {task_id[:12]}...", style="yellow italic"))

        try:
            await agent.client.cancel_task(TaskIdParams(id=task_id))
            self.tracker.force_terminal(task_id, "canceled")
            chat.write(Text("  Task canceled.", style="yellow"))
            # NOTE: do NOT reset _handling_deferred here — the running
            # _handle_pending_input work resets it in its finally block.
            # Resetting early causes _poll_tasks to re-dispatch the panel.
        except A2AClientHTTPError as e:
            chat.write(Text(f"  Cancel failed: HTTP {e.status_code}", style="red bold"))
        except (A2AClientTimeoutError, A2AClientJSONError) as e:
            chat.write(Text(f"  Cancel failed: {e}", style="red bold"))
        except Exception as e:
            chat.write(Text(f"  Cancel failed: {e}", style="red bold"))
        finally:
            self._canceling = False

        self._update_status_bar()

    @work(exclusive=True, group="send")
    async def _send_message(self, text: str) -> None:
        agent = self.agents[self.current]
        chat = self.query_one("#chat", RichLog)

        # First message to this agent: attach client shell info as metadata
        metadata = None
        if agent.context_id is None and self._shell_info:
            metadata = {"client_info": self._shell_info}

        clean_text, file_parts = _parse_attachments(text)
        if file_parts:
            chat.write(Text(f"  Attached {len(file_parts)} file(s)", style="dim cyan"))
            parts = []
            if clean_text:
                parts.append(Part(root=TextPart(text=clean_text)))
            parts.extend(file_parts)
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.user,
                parts=parts,
                context_id=agent.context_id,
                metadata=metadata,
            )
        else:
            msg = _make_message(
                clean_text or text,
                context_id=agent.context_id,
                metadata=metadata,
            )

        try:
            task = await _collect_task(agent.client.send_message(msg))
        except A2AClientHTTPError as e:
            chat.write(Text(f"✗ HTTP {e.status_code}: {e.message}", style="bold red"))
            return
        except A2AClientTimeoutError:
            chat.write(Text("✗ Request timed out — try again", style="bold red"))
            return
        except A2AClientJSONError as e:
            chat.write(
                Text(f"✗ Bad response from server: {e.message}", style="bold red")
            )
            return
        except Exception as e:
            chat.write(Text(f"✗ Error: {e}", style="bold red"))
            return

        if task is None:
            chat.write(Text("No response from agent", style="dim"))
            return

        if not hasattr(task, "status"):
            chat.write(Text(str(task), style="dim"))
            return

        agent.context_id = task.context_id
        if hasattr(task, "id"):
            agent.last_task_id = task.id

        await self.tracker.register(task.id, agent.name)
        await self.tracker.update(task.model_dump(mode="json", exclude_none=True))

    # -- Deferred tool handling ----------------------------------------------

    @work(exclusive=True, group="deferred")
    async def _handle_pending_input(self, pending) -> None:
        """Handle an input-required task from the current agent."""
        try:
            task_data = pending.task_data
            status = task_data.get("status", {})
            message = status.get("message", {})
            parts = message.get("parts", [])

            data = None
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "data":
                    data = part.get("data")
                    break
                if isinstance(part, dict) and "data" in part:
                    data = part["data"]
                    break

            if data is None:
                return

            chat = self.query_one("#chat", RichLog)
            agent = self.agents[self.current]
            inp = self.query_one("#input", Input)

            # Show contextual hint in the input bar
            hint = self._get_deferred_hint(data)
            inp.placeholder = hint

            try:
                result = await self._dispatch_deferred(data, chat)
            except asyncio.CancelledError:
                # User pressed ESC during deferred input — task cancel
                # already handled by _cancel_current_task
                return
            except Exception as e:
                chat.write(Text(f"✗ Handler error: {e}", style="bold red"))
                return

            # Guard: don't send if the task was canceled while we were waiting
            task_info = self.tracker.get(pending.task_id)
            if task_info and task_info.is_terminal:
                return

            msg = _make_data_message(
                result,
                context_id=agent.context_id,
                task_id=pending.task_id,
            )
            try:
                task = await _collect_task(agent.client.send_message(msg))
                if task and hasattr(task, "id"):
                    await self.tracker.update(
                        task.model_dump(mode="json", exclude_none=True)
                    )
            except A2AClientHTTPError as e:
                chat.write(
                    Text(f"✗ HTTP {e.status_code}: {e.message}", style="bold red")
                )
            except A2AClientTimeoutError:
                chat.write(Text("✗ Deferred response timed out", style="bold red"))
            except A2AClientJSONError as e:
                chat.write(
                    Text(f"✗ Bad server response: {e.message}", style="bold red")
                )
            except Exception as e:
                chat.write(
                    Text(f"✗ Error sending deferred response: {e}", style="bold red")
                )
        finally:
            self._handling_deferred = False
            self._update_input_placeholder()

    def _get_deferred_hint(self, data: dict | None) -> str:
        """Get the input hint for the current deferred tool call."""
        if not data or "deferred_tool_calls" not in data:
            return "Type your response and press Enter"
        calls = data.get("deferred_tool_calls", [])
        if not calls:
            return "Type your response and press Enter"
        call = calls[0]
        tool_name = call.get("tool_name", "")
        args = call.get("arguments", {})
        handler = self.dispatcher.get(tool_name)
        return handler.get_input_hint(args)

    async def _dispatch_deferred(self, data: dict | None, chat: RichLog) -> dict:
        if not data or "deferred_tool_calls" not in data:
            chat.write(
                Panel(
                    str(data),
                    title="[bold yellow]Input Required[/]",
                    border_style="yellow",
                )
            )
            return {"answer": await self._wait_for_input()}

        deferred_calls = data["deferred_tool_calls"]
        if not isinstance(deferred_calls, list) or not deferred_calls:
            return {"answer": await self._wait_for_input()}

        call = deferred_calls[0]
        tool_name = call.get("tool_name", "")
        args = call.get("arguments", {})

        handler = self.dispatcher.get(tool_name)
        console_adapter = _RichLogConsole(chat)
        return await handler.handle(args, console_adapter, self._wait_for_input)

    async def _wait_for_input(self) -> str:
        """Wait for the next Input submission. Used by deferred handlers.

        Returns the user's input string, or raises asyncio.CancelledError
        if the user pressed ESC to cancel the task.
        """
        loop = asyncio.get_event_loop()
        self._input_future = loop.create_future()
        result = await self._input_future
        self._input_future = None
        if result is None:
            # Sentinel from ESC cancel — abort deferred handler
            raise asyncio.CancelledError("User canceled via ESC")
        return result

    # -- Command handlers ----------------------------------------------------

    def _show_tasks(self) -> None:
        chat = self.query_one("#chat", RichLog)
        tasks = self.tracker.get_all()
        if not tasks:
            chat.write(Text("No background tasks.", style="dim cyan"))
            return

        table = Table(
            title="Background Tasks",
            title_style="bold",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("Task ID", style="dim", max_width=12)
        table.add_column("Agent", style="bold cyan")
        table.add_column("State", width=16)

        styles = {
            "submitted": "dim",
            "working": "bold yellow",
            "input-required": "bold red",
            "completed": "bold green",
            "failed": "bold red",
            "canceled": "dim",
            "rejected": "bold red",
        }

        for t in tasks:
            style = styles.get(t.state, "dim")
            label = t.state
            if t.state == "input-required":
                label = "⚠ input-required"
            elif t.state == "completed":
                label = "✓ completed"
            elif t.state == "failed":
                label = "✗ failed"
            table.add_row(t.task_id[:12], t.agent_name, f"[{style}]{label}[/{style}]")

        chat.write(table)
        chat.scroll_end(animate=False)

    def _show_task_info(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        parts_cmd = text.split()
        if len(parts_cmd) == 2:
            partial = parts_cmd[1]
            for t in self.tracker.get_all():
                if t.task_id.startswith(partial):
                    if t.task_data:
                        self._show_result(t.task_data, t.agent_name)
                    else:
                        chat.write(
                            Text(
                                f"Task {t.task_id[:12]}... state={t.state}", style="dim"
                            )
                        )
                    return
        # Show last task info
        if self.agents:
            agent = self.agents[self.current]
            if agent.last_task_id:
                t = self.tracker.get(agent.last_task_id)
                if t and t.task_data:
                    self._show_result(t.task_data, t.agent_name)
                    return
        chat.write(Text("No task found.", style="dim"))

    def _switch_to_agent(self, idx: int) -> None:
        """Switch to agent at index, show unseen results, update UI."""
        chat = self.query_one("#chat", RichLog)
        self.current = idx
        unseen = self._unseen_by_agent.pop(self.agents[idx].name, [])
        for tid in unseen:
            t = self.tracker.get(tid)
            if t and t.task_data:
                self._show_result(t.task_data, t.agent_name)
        chat.write(Text(f"Switched to: {self.agents[idx].name}", style="dim cyan"))
        self._update_input_placeholder()
        self._update_status_bar()

    def _handle_switch(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        parts = text.split()
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(self.agents):
                self._switch_to_agent(idx)
            else:
                chat.write(
                    Text(f"Invalid. Use 1-{len(self.agents)}.", style="bold red")
                )
        else:
            chat.write(Text("Usage: /switch <n>", style="bold red"))

    # -- Cleanup -------------------------------------------------------------

    def action_quit_app(self) -> None:
        self.exit()

    async def on_unmount(self) -> None:
        if self._httpx_client:
            await self._httpx_client.aclose()
        if self._webhook_server:
            await self._webhook_server.stop()
