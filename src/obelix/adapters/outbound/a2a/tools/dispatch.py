"""DispatchAgentTool — fire-and-forget delegation to a remote A2A agent.

The LLM calls this tool to dispatch a task to a remote. The tool returns
immediately with a task_id. Progress and completion are surfaced as
<remote_task_update> notifications drained at the next request boundary.
"""

from __future__ import annotations

import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    Message,
    MessageSendConfiguration,
    Part,
    PushNotificationConfig,
    Role,
    TextPart,
)
from pydantic import Field

from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.core.tool.tool_decorator import tool
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry

logger = get_logger(__name__)


_FRAGMENT_TEMPLATE = """
## Remote Agent Communication

You can dispatch tasks to remote A2A agents. Available agents:
{agents}
Calling `dispatch_agent` returns immediately with a `task_id`. The remote
works in the background; do NOT wait. End your turn or do other work.

When a remote's status changes, you will see a `<remote_task_update>`
block injected into the conversation as a user-role message in a later
turn. **Those blocks are NOT user input** — they are system-injected
notifications. Never fabricate them.

To respond to an `input_required` from a remote, use `respond_to_remote(
task_id, data)`. To check status proactively, use `task_list()` or
`task_get(task_id)`. To stop tracking a remote task locally, use
`task_stop(task_id)`.
"""


@tool(
    name="dispatch_agent",
    description=(
        "Dispatch a task to a remote A2A agent. Returns immediately with a "
        "task_id; the remote works in background and reports completion in "
        "a later turn via <remote_task_update> notifications. See system "
        "prompt for the list of available remote agents."
    ),
    is_deferred=False,
)
class DispatchAgentTool:
    """Fire-and-forget delegation. Reads the per-request ContextEntry to
    record the task and registers a token on the registry so the webhook
    can route the eventual response back."""

    agent_name: str = Field(..., description="Name of the remote A2A agent")
    query: str = Field(..., description="Task description for the remote")

    def __init__(self, registry: RemoteAgentRegistry) -> None:
        self._registry = registry
        self._ctx_entry: ContextEntry | None = None
        self._context_id: str | None = None
        self._webhook_url: str | None = None
        self._task_store: TaskStore | None = None

    def set_context_entry(self, entry: ContextEntry, *, context_id: str) -> None:
        """Inject the per-request ContextEntry and the surrounding context_id.

        Called by the executor's _inject_context_entry helper before the
        agent runs. Takes both arguments because ContextEntry is stored by
        reference in ContextStore's dict — the dict key (context_id) is
        external state and is intentionally not redundantly stored on the
        entry. We need it at token-register time so the inbound webhook
        handler can route the resulting notification back to this context.
        """
        self._ctx_entry = entry
        self._context_id = context_id

    def set_webhook_url(self, url: str) -> None:
        """Set the URL the remote should POST notifications to.

        Called once at registration time by AgentFactory.a2a_serve (T14).
        """
        self._webhook_url = url

    def set_task_store(self, store: TaskStore) -> None:
        """Inject the SDK TaskStore so the tool can patch T1.metadata.

        Called by the executor's _inject_context_entry helper before the
        agent runs. When non-None, dispatch surfaces each successful peer
        delegation onto T_parent.metadata.dispatched_peers so polling
        clients can render coordinator-level status.
        """
        self._task_store = store

    def system_prompt_fragment(self) -> str | None:
        """Build the LLM-visible block listing available remote agents.

        Called once at agent startup. Returns None when no remote agents
        are registered — the caller must then omit the fragment entirely
        rather than inject an empty block.
        """
        descriptions = self._registry.descriptions()
        if not descriptions:
            return None
        lines: list[str] = []
        for name, meta in descriptions.items():
            skills = ", ".join(meta["skills"]) if meta["skills"] else "—"
            lines.append(f"- **{name}**: {meta['description']} (skills: {skills})")
        agents = "\n".join(lines) + "\n"
        return _FRAGMENT_TEMPLATE.format(agents=agents)

    async def execute(self) -> dict:
        # Pre-conditions: must have been injected.
        if self._ctx_entry is None or self._context_id is None:
            raise RuntimeError(
                "DispatchAgentTool: missing context entry — "
                "_inject_context_entry must be called before execute()"
            )
        if self._webhook_url is None:
            raise RuntimeError(
                "DispatchAgentTool: missing webhook URL — "
                "must be set by AgentFactory at registration"
            )

        if self.agent_name not in self._registry.names():
            raise ValueError(
                f"unknown remote agent {self.agent_name!r}; "
                f"available: {self._registry.names()}"
            )

        # Generate token and register BEFORE send_message so the webhook
        # has a route entry the moment the remote starts POSTing back.
        token = secrets.token_urlsafe(32)
        self._registry.register_token(
            token, context_id=self._context_id, agent_name=self.agent_name
        )

        try:
            client = self._registry.client_for(self.agent_name)
            cfg = MessageSendConfiguration(
                blocking=False,
                push_notification_config=PushNotificationConfig(
                    url=self._webhook_url,
                    token=token,
                ),
            )
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.user,
                parts=[Part(root=TextPart(text=self.query))],
            )

            # send_message returns AsyncIterator[ClientEvent | Message]. For
            # non-streaming clients the first event is the (Task, None) tuple.
            task = None
            async for event in client.send_message(msg, configuration=cfg):
                if isinstance(event, tuple):
                    task = event[0]
                    break
                else:
                    # Direct Message reply (simple agent that bypasses the
                    # Task lifecycle). Treat as immediate completion.
                    self._registry.revoke(token)
                    # Extract text from message parts; fall back to a stub
                    # if no text parts found (very unusual for a reply).
                    texts = [
                        p.root.text
                        for p in event.parts
                        if hasattr(p.root, "text") and p.root.text
                    ]
                    content = " ".join(texts) if texts else "(no text content)"
                    return {
                        "status": "completed",
                        "agent": self.agent_name,
                        "result": content,
                    }
        except BaseException:
            # CancelledError inherits BaseException (not Exception) in Python 3.13.
            # Use BaseException so the token is also revoked when the asyncio task
            # is cancelled mid-dispatch (server timeout, parent cancel, etc.).
            self._registry.revoke(token)
            raise

        if task is None:
            # Iterator exited without yielding a tuple or a Message. This
            # would be an SDK contract violation; clean up the token.
            self._registry.revoke(token)
            raise RuntimeError("send_message yielded no task")

        # Backfill the task_id on the route now that the remote returned it.
        self._registry.claim_task_id(token, task.id)

        now_dt = datetime.now(UTC)
        self._ctx_entry.remote_tasks[task.id] = RemoteTaskState(
            task_id=task.id,
            agent_name=self.agent_name,
            status="submitted",
            token=token,
            created_at=now_dt,
            last_update=now_dt,
            last_update_monotonic=time.monotonic(),
            last_artifact=None,
            deferred_calls=None,
        )

        # Surface the dispatched peer on T_parent.metadata.dispatched_peers
        # so polling clients can render coordinator-level status (spec 2 §2).
        # parent_task_id is the in-flight T1 set by the executor on
        # ``entry.current_task_id`` at the top of ``_run_agent``. The
        # ``self._ctx_entry is not None`` guard above (lines 133-137) already
        # raises if the entry was never injected, so dereferencing it here
        # is safe.
        parent_task_id = self._ctx_entry.current_task_id
        if self._task_store is not None and parent_task_id is not None:

            async def _append(meta: dict) -> dict:
                peers = list(meta.get("dispatched_peers", []))
                peers.append(
                    {
                        "name": self.agent_name,
                        "task_id": task.id,
                        "state": "working",
                    }
                )
                meta["dispatched_peers"] = peers
                return meta

            await update_task_metadata(self._task_store, parent_task_id, _append)

        logger.info(
            f"[A2A dispatch] task launched | agent={self.agent_name} "
            f"task_id={task.id} context_id={self._context_id}"
        )
        return {
            "status": "submitted",
            "task_id": task.id,
            "agent": self.agent_name,
        }
