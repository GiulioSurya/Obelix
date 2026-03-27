"""Embedded webhook server for receiving A2A push notifications.

Runs a lightweight Starlette app in a background asyncio task.
The server A2A POSTs Task JSON to /webhook on every state change;
the TaskTracker keeps the latest state for each task.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


@dataclass
class TaskInfo:
    """Tracks a single background task."""

    task_id: str
    agent_name: str
    state: str = "submitted"
    timestamp: float = field(default_factory=time.time)
    task_data: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in ("completed", "failed", "canceled", "rejected")


class TaskTracker:
    """Thread-safe tracker for background A2A tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._on_update: asyncio.Event = asyncio.Event()

    async def register(self, task_id: str, agent_name: str) -> None:
        """Register a new task before sending."""
        async with self._lock:
            self._tasks[task_id] = TaskInfo(task_id=task_id, agent_name=agent_name)

    async def update(self, task_data: dict[str, Any]) -> None:
        """Update a task from a push notification payload."""
        task_id = task_data.get("id", "")
        status = task_data.get("status", {})
        state = status.get("state", "unknown")

        async with self._lock:
            if task_id in self._tasks:
                info = self._tasks[task_id]
                info.state = state
                info.timestamp = time.time()
                info.task_data = task_data
            else:
                # Push arrived before register — create entry
                self._tasks[task_id] = TaskInfo(
                    task_id=task_id,
                    agent_name="unknown",
                    state=state,
                    task_data=task_data,
                )
        # Wake up anyone waiting (set then immediately replace the event
        # so the next waiter gets a fresh, unset event).
        self._on_update.set()
        self._on_update = asyncio.Event()

    def get_all(self) -> list[TaskInfo]:
        """Return all tracked tasks (snapshot, no lock needed for read)."""
        return list(self._tasks.values())

    def get_active(self) -> list[TaskInfo]:
        """Return non-terminal tasks."""
        return [t for t in self._tasks.values() if not t.is_terminal]

    def get_by_agent(self, agent_name: str) -> list[TaskInfo]:
        """Return tasks for a specific agent."""
        return [t for t in self._tasks.values() if t.agent_name == agent_name]

    def get_pending_input(self, agent_name: str) -> TaskInfo | None:
        """Return the first input-required task for an agent, or None."""
        for t in self._tasks.values():
            if t.agent_name == agent_name and t.state == "input-required":
                return t
        return None

    def get(self, task_id: str) -> TaskInfo | None:
        """Return a specific task."""
        return self._tasks.get(task_id)

    def force_terminal(self, task_id: str, state: str) -> None:
        """Force a task into a terminal state (e.g. after client-side cancel)."""
        info = self._tasks.get(task_id)
        if info:
            info.state = state
            info.timestamp = time.time()

    async def wait_for_terminal(
        self, task_id: str, timeout: float = 300.0
    ) -> TaskInfo | None:
        """Block until a task reaches a terminal or input-required state.

        Returns the TaskInfo, or None on timeout.
        """
        import time as _time

        deadline = _time.monotonic() + timeout
        while True:
            info = self._tasks.get(task_id)
            if info and (info.is_terminal or info.state == "input-required"):
                return info
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return info
            # Wait for the next update notification (or timeout)
            try:
                await asyncio.wait_for(
                    self._on_update.wait(), timeout=min(remaining, 1.0)
                )
            except TimeoutError:
                pass


def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WebhookServer:
    """Lightweight Starlette server for receiving push notifications.

    The webhook host (used in the URL sent to the A2A server) is resolved as:
        1. Explicit ``webhook_host`` parameter
        2. ``OBELIX_WEBHOOK_HOST`` environment variable
        3. ``127.0.0.1`` (default — assumes server is on the same host)

    The webhook port is resolved as:
        1. Explicit ``webhook_port`` parameter
        2. ``OBELIX_WEBHOOK_PORT`` environment variable
        3. Random free port (default)

    Use a fixed port when the client runs in a Kubernetes pod and the A2A server
    needs a stable Service endpoint to POST back to, or when a specific firewall
    rule must be opened (e.g. Docker Desktop on Windows).

    The server always binds on ``0.0.0.0`` so it accepts connections from any
    interface (needed when the A2A server rewrites the URL to a non-loopback IP).
    """

    def __init__(
        self,
        tracker: TaskTracker,
        webhook_host: str | None = None,
        webhook_port: int | None = None,
    ) -> None:
        self._tracker = tracker
        self._webhook_host = (
            webhook_host or os.environ.get("OBELIX_WEBHOOK_HOST") or "127.0.0.1"
        )
        _env_port = os.environ.get("OBELIX_WEBHOOK_PORT")
        self._fixed_port: int | None = webhook_port or (
            int(_env_port) if _env_port else None
        )
        self._port: int = 0
        self._task: asyncio.Task | None = None

        async def webhook_handler(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                await self._tracker.update(body)
            except Exception:
                pass
            return JSONResponse({"ok": True})

        self._app = Starlette(
            routes=[Route("/webhook", webhook_handler, methods=["POST"])],
        )

    @property
    def port(self) -> int:
        return self._port

    @property
    def webhook_host(self) -> str:
        return self._webhook_host

    def get_url(self) -> str:
        return f"http://{self._webhook_host}:{self._port}/webhook"

    async def start(self) -> None:
        """Start the webhook server in a background task."""
        import uvicorn

        self._port = self._fixed_port if self._fixed_port else _find_free_port()
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self._port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        self._task = asyncio.create_task(server.serve())

    async def stop(self) -> None:
        """Stop the webhook server."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
