"""Per-context state management for the A2A server.

Each A2A context_id maps to a _ContextEntry that holds conversation
history, idle gate (for serialization), and deferred tool state.
The ContextStore wraps the LRU-eviction OrderedDict.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.outbound.a2a.state import RemoteTaskState
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import HumanMessage, StandardMessage
    from obelix.core.model.tool_message import ToolCall

logger = get_logger(__name__)


class ContextEntry:
    """Holds the state for a single conversation context."""

    __slots__ = (
        "context_id",
        "history",
        "idle",
        "deferred_tool_calls",
        "deferred_tools",
        "trace_session",
        "trace_span",
        "deferred_wait_span_id",
        "active_agent",
        "client_info",
        "was_canceled",
        "was_rejected",
        "was_failed",
        "rejection_reason",
        "failure_error",
        "remote_tasks",
        "pending_notifications",
        "client_webhook_url",  # TEMP-PATCH-SPEC-1
        "client_webhook_token",  # TEMP-PATCH-SPEC-1
    )

    def __init__(self) -> None:
        # Populated by ``ContextStore.get_or_create``. Lets background
        # workers (polling, drainer) recover the context_id from an
        # entry alone, without scanning the store's keys.
        self.context_id: str | None = None
        self.history: list[StandardMessage] = []
        self.idle = asyncio.Event()
        self.idle.set()  # starts as idle (ready for new executions)
        self.deferred_tool_calls: list[ToolCall] | None = None
        self.deferred_tools: list | None = None  # tool snapshot for OutputSchema lookup
        self.trace_session = None  # TraceSession saved when loop stops for deferred
        self.trace_span = None  # Current span saved when loop stops for deferred
        # Span id of the open ``deferred_wait`` span that wraps the
        # input_required pause. Set when the executor suspends on a deferred
        # tool; cleared after the span is ended on resume (or on cancel).
        self.deferred_wait_span_id: str | None = None
        self.active_agent: BaseAgent | None = None  # ref to running agent for cancel
        self.client_info: dict | None = None  # client shell environment for BashTool
        # Flag set by ``cancel()`` (either via CancelledError in the agent
        # loop or via the input_required cancel path) so the outer
        # ``_run_agent`` finally closes the a2a_task span with
        # ``SpanStatus.canceled`` instead of the default ``ok``.
        self.was_canceled: bool = False
        # Flag set by the ``except TaskRejectedError`` handler in
        # ``_run_agent_impl`` so the outer ``_run_agent`` finally closes the
        # a2a_task span with ``SpanStatus.rejected`` and forwards the reason
        # as ``span.error``.
        self.was_rejected: bool = False
        # Flag set by the generic ``except Exception`` handler so the outer
        # finally closes the a2a_task span with ``SpanStatus.error`` and
        # forwards the exception message as ``span.error``.
        self.was_failed: bool = False
        # Rejection reason captured from ``TaskRejectedError.reason`` (or the
        # str(e) fallback); propagated onto span.error on close.
        self.rejection_reason: str | None = None
        # Failure error message captured from ``str(e)`` of the generic
        # Exception; propagated onto span.error on close.
        self.failure_error: str | None = None
        # Outbound A2A: tasks dispatched from this conversation, keyed by
        # remote task_id. Webhook handler and polling worker mutate this.
        self.remote_tasks: dict[str, RemoteTaskState] = {}
        # User-role messages built by webhook handler / polling worker for
        # terminal/input_required state changes. Drained at the start of
        # the next request on this context (executor._run_agent_impl).
        self.pending_notifications: list[HumanMessage] = []
        # TEMP-PATCH-SPEC-1: webhook URL + auth token sent by the CLI client
        # in the metadata of its first Message; used by the drain-spawn POST
        # in executor.py. Both removed when spec 2 (CLI streaming) lands.
        self.client_webhook_url: str | None = None
        self.client_webhook_token: str | None = None

    def is_evictable(self) -> bool:
        """LRU eviction guard. False if any non-terminal remote task
        is in flight — losing this context would silence its webhook
        returns (token revoked) and the user-visible result vanishes."""
        # If a future task adds another blocking condition (e.g. waiting on
        # a long-poll), extend by AND-ing additional checks here.
        return all(t.is_terminal for t in self.remote_tasks.values())


class ContextStore:
    """LRU context store with async-safe access.

    Thread safety: get_or_create() must be called under the caller's lock.
    """

    def __init__(self, max_contexts: int) -> None:
        self._max_contexts = max_contexts
        self._contexts: OrderedDict[str, ContextEntry] = OrderedDict()

    def get_or_create(self, context_id: str) -> ContextEntry:
        """Get or create a context entry, evicting oldest if over limit."""
        if context_id in self._contexts:
            self._contexts.move_to_end(context_id)
            return self._contexts[context_id]

        # Evict if at capacity. Protects non-evictable contexts but
        # enforces a hard cap at 2x max_contexts to prevent unbounded
        # growth under saturation.
        while len(self._contexts) >= self._max_contexts:
            if not self._evict_one():
                # Below 2x cap, but everyone is non-evictable: accept the
                # growth temporarily rather than infinite-loop.
                break

        entry = ContextEntry()
        entry.context_id = context_id
        self._contexts[context_id] = entry
        return entry

    def peek(self, context_id: str) -> ContextEntry | None:
        """Return an existing entry without creating one or promoting it
        in the LRU. Used by the webhook handler to locate a context for
        an inbound push notification without the side effect of phantom
        creation on stale tokens."""
        return self._contexts.get(context_id)

    def iter_entries(self) -> list[ContextEntry]:
        """Return a snapshot of all current ContextEntry instances.

        The caller owns the snapshot; concurrent mutations to the store
        do not affect the returned list. Used by background workers
        (e.g. polling) that need to iterate all contexts without
        creating phantom entries."""
        return list(self._contexts.values())

    def _evict_one(self) -> bool:
        """Evict one context. First pass: oldest evictable entry. Hard cap
        fallback: when at 2x max_contexts and everyone is non-evictable,
        force-evict the oldest with a warning so operators see saturation.

        Returns ``True`` when a context was evicted, ``False`` when no-op
        (so the caller can break out of its eviction loop).
        """
        for cid, entry in self._contexts.items():
            if entry.is_evictable():
                self._contexts.pop(cid)
                logger.debug(f"[A2A] Evicted context | context_id={cid}")
                return True
        if len(self._contexts) >= self._max_contexts * 2:
            oldest_id, _ = self._contexts.popitem(last=False)
            logger.warning(
                f"[A2A] Forced eviction | context_id={oldest_id} reason=hard_cap "
                f"cap={self._max_contexts * 2} consequence=webhooks_return_401"
            )
            return True
        return False
