"""RespondToRemoteTool — answer an input_required emitted by a remote agent."""

from __future__ import annotations

import uuid
from contextlib import aclosing
from typing import TYPE_CHECKING

from a2a.types import (
    DataPart,
    Message,
    MessageSendConfiguration,
    Part,
    PushNotificationConfig,
    Role,
)
from pydantic import Field

from obelix.adapters.outbound.a2a.tools._base import _ContextAware
from obelix.core.tool.tool_decorator import tool
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.tasks.task_store import TaskStore

    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry

logger = get_logger(__name__)


@tool(
    name="respond_to_remote",
    description=(
        "Respond to an input_required notification from a remote agent. "
        "Provide the task_id and a data payload matching the remote tool's "
        "OutputSchema. Idempotent: a second respond within the same "
        "input_required cycle is rejected."
    ),
    is_deferred=False,
)
class RespondToRemoteTool(_ContextAware):
    """Answer an input_required from a remote. Reuses the original
    dispatch token so the remote's resume notifications still route to
    the same context."""

    task_id: str = Field(..., description="ID of the remote task awaiting input")
    data: dict = Field(
        ...,
        description=(
            "Response payload — must match the remote tool's OutputSchema. "
            "See the deferred_tool_calls in the latest <remote_task_update>."
        ),
    )

    def __init__(self, registry: RemoteAgentRegistry) -> None:
        self._registry = registry
        self._webhook_url: str | None = None
        self._task_store: TaskStore | None = None

    def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL passed to the remote in the continuation's
        push_notification_config (same URL as the original dispatch — must
        be set by AgentFactory at registration)."""
        self._webhook_url = url

    def set_task_store(self, store: TaskStore) -> None:
        """Inject the SDK TaskStore so the tool can mirror the local
        ``input_required → submitted`` transition onto T_parent.metadata.

        Called by the executor's ``_inject_context_entry`` helper before
        the agent runs (same wiring path as DispatchAgentTool). When None,
        respond still works — only the metadata mirror is skipped.
        """
        self._task_store = store

    async def execute(self) -> dict:
        ctx_entry = self._require_context("RespondToRemoteTool")
        if self._webhook_url is None:
            raise RuntimeError("RespondToRemoteTool: webhook URL not set")

        state = ctx_entry.remote_tasks.get(self.task_id)
        if state is None:
            raise ValueError(f"task {self.task_id!r} not found in this context")

        # Idempotency / cycle gate. Once we've responded (status flipped
        # to "submitted") or the task is terminal, reject further attempts
        # within the same input_required cycle. A new cycle (B emits a
        # FRESH input_required with new deferred_calls) restores the
        # status to "input_required" and re-allows respond.
        if state.status != "input_required" or state.deferred_calls is None:
            raise RuntimeError(
                f"task {self.task_id!r} is not in input_required state "
                f"(status={state.status!r}); already responded or never deferred"
            )

        client = self._registry.client_for(state.agent_name)

        # Reuse the existing token so resume notifications still route
        # to the same TokenRoute / ContextEntry.
        cfg = MessageSendConfiguration(
            blocking=False,
            push_notification_config=PushNotificationConfig(
                url=self._webhook_url,
                token=state.token,
            ),
        )
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[Part(root=DataPart(data=self.data))],
            task_id=self.task_id,
        )

        # Continuation: send_message in this case typically returns no
        # task tuple (the existing task continues). Drain the iterator
        # under aclosing() so the underlying async generator is properly
        # closed even if we break out early. Future-proofs against a
        # streaming=True client config.
        async with aclosing(client.send_message(msg, configuration=cfg)) as gen:
            async for _event in gen:
                break

        # Local state transition: input_required → submitted.
        state.status = "submitted"
        state.deferred_calls = None

        # Mirror onto T_parent.metadata so the CLI status bar reflects the
        # transition immediately (otherwise the segment would freeze on
        # input_required until the next webhook/polling tick).
        if self._task_store is not None and ctx_entry.current_task_id is not None:
            from obelix.adapters.inbound.a2a.server.metadata_patch import (
                update_dispatched_peer_state,
            )

            await update_dispatched_peer_state(
                self._task_store,
                ctx_entry.current_task_id,
                self.task_id,
                "submitted",
            )

        # Refresh the token's TTL: the input_required cycle may have lasted
        # hours waiting for human input, and the resume notifications need
        # the route to remain valid for at least another TTL window.
        self._registry.touch(state.token)

        logger.info(
            f"[A2A respond] reply sent | agent={state.agent_name} "
            f"task_id={self.task_id}"
        )
        return {"task_id": self.task_id, "status": "submitted"}
