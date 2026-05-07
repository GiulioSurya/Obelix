"""ObelixAgentExecutor: bridges the a2a-sdk AgentExecutor to Obelix BaseAgent.

Thread safety: each A2A context_id gets its own BaseAgent instance
(created via the agent_factory callable). Conversation history is
persisted per context_id so multi-turn conversations work correctly.
Concurrent requests on *different* contexts run in parallel without
interference. Concurrent requests on the *same* context are serialized
via a per-context asyncio.Event (idle gate).

Input-required flow (deferred tools): when the agent encounters a
tool with is_deferred=True that returns None, the loop stops and
yields a StreamEvent with deferred_tool_calls. The executor emits
TaskState.input_required with DataPart. On the next message/send for
the same contextId, the executor extracts the DataPart response,
injects it as a ToolMessage, and restarts the agent loop.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from obelix.adapters.inbound.a2a.part_converter import (
    a2a_parts_to_obelix,
    deferred_calls_to_a2a_parts,
    obelix_response_to_a2a_parts,
)
from obelix.adapters.inbound.a2a.server.context import ContextStore
from obelix.adapters.inbound.a2a.server.deferred import inject_deferred_response
from obelix.adapters.inbound.a2a.server.helpers import (
    DEFAULT_MAX_CONTEXTS,
    agent_message,
)
from obelix.core.agent.exceptions import TaskRejectedError
from obelix.core.model.assistant_message import AssistantResponse
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from obelix.core.tracer.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from obelix.core.tracer.models import SpanStatus, SpanType
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext

    from obelix.adapters.inbound.a2a.server.context import ContextEntry
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.tracer.models import Span
    from obelix.core.tracer.tracer import Tracer

logger = get_logger(__name__)


class ObelixAgentExecutor(AgentExecutor):
    """Executes an Obelix BaseAgent in response to A2A requests.

    Bridges the a2a-sdk execution model (RequestContext + EventQueue)
    to the Obelix agent model (execute_query_stream).

    Each request creates a fresh BaseAgent via the factory callable.
    Conversation history is stored per context_id and injected into
    the fresh agent before execution, then saved back afterwards.

    Supports the input-required A2A flow via deferred tools: when the
    agent yields deferred_tool_calls, the executor emits input-required.
    The next request on the same contextId injects the response as a
    ToolMessage and restarts the agent.
    """

    def __init__(
        self,
        agent_factory: Callable[[], BaseAgent],
        *,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
        tracer: Tracer | None = None,
        registry: RemoteAgentRegistry | None = None,
        context_store: ContextStore | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._store = (
            context_store if context_store is not None else ContextStore(max_contexts)
        )
        self._store_lock = asyncio.Lock()
        self._tracer = tracer
        self._registry = registry

    async def _emit_state(
        self,
        from_state: str | None,
        to_state: str,
        reason: str | None = None,
    ) -> None:
        """Emit an a2a.state_change tracer event on the a2a_task span.

        Temporarily re-pins the current span to the a2a_task span so the event
        attaches there (not to whatever nested span — agent/tool — is active).
        Restores the prior current span afterwards. No-op if no tracer is
        configured or no a2a_task span exists on the current trace.
        """
        if not self._tracer:
            return
        trace = get_current_trace()
        if trace is None:
            return
        a2a_span = next(
            (s for s in trace.spans if s.span_type == SpanType.a2a_task),
            None,
        )
        if a2a_span is None:
            return
        prior = get_current_span()
        set_current_span(a2a_span)
        try:
            await self._tracer.add_event(
                "a2a.state_change",
                {"from": from_state, "to": to_state, "reason": reason},
            )
        finally:
            set_current_span(prior)

    async def _emit_cancellation_event(
        self,
        source: str = "client",
        iteration: int | None = None,
        trace=None,
    ) -> None:
        """Emit a ``cancellation.requested`` event on the a2a_task span.

        Same pattern as :meth:`_emit_state`: pin the current span to the
        ``a2a_task`` root so the event attaches there (not to a nested
        agent/tool/deferred_wait span), then restore the prior current span.
        No-op if no tracer is configured, no trace available, or no a2a_task
        span exists on the trace.

        ``trace`` overrides the contextvar lookup. Used by ``cancel()`` to
        emit the event via the entry's saved ``trace_session`` when running
        in a different asyncio task (and therefore empty contextvars).
        """
        if not self._tracer:
            return
        resolved_trace = trace if trace is not None else get_current_trace()
        if resolved_trace is None:
            return
        a2a_span = next(
            (s for s in resolved_trace.spans if s.span_type == SpanType.a2a_task),
            None,
        )
        if a2a_span is None:
            return
        prior_trace = get_current_trace()
        prior_span = get_current_span()
        # Pin both trace and span for add_event to resolve correctly.
        set_current_trace(resolved_trace)
        set_current_span(a2a_span)
        try:
            attributes: dict[str, object] = {"source": source}
            if iteration is not None:
                attributes["iteration"] = iteration
            await self._tracer.add_event("cancellation.requested", attributes)
        finally:
            set_current_span(prior_span)
            set_current_trace(prior_trace)

    async def _open_deferred_wait_span(
        self,
        entry,
        deferred_tool_calls,
    ) -> None:
        """Open a ``deferred_wait`` span covering the input_required pause.

        Temporarily pins the current span to ``a2a_task`` so the new span
        becomes a direct child of the task root (sibling of the agent span).
        The span is left OPEN — it is closed on the resume path (or on
        cancel). Records ``tool_name``, ``tool_call_ids`` and a fixed
        ``suspend_reason="deferred_tool"`` in the span metadata.

        No-op if no tracer is configured, no current trace, or no a2a_task
        span is present on the current trace.
        """
        if not self._tracer:
            return
        trace = get_current_trace()
        if trace is None:
            return
        a2a_span = next(
            (s for s in trace.spans if s.span_type == SpanType.a2a_task),
            None,
        )
        if a2a_span is None:
            return
        prior = get_current_span()
        set_current_span(a2a_span)
        try:
            tool_name = deferred_tool_calls[0].name if deferred_tool_calls else None
            tool_call_ids = [c.id for c in deferred_tool_calls]
            span = await self._tracer.start_span(
                SpanType.deferred_wait,
                name="deferred_wait",
                metadata={
                    "tool_name": tool_name,
                    "tool_call_ids": tool_call_ids,
                    "suspend_reason": "deferred_tool",
                },
            )
            entry.deferred_wait_span_id = span.span_id
        finally:
            # Restore the prior current span — the ``deferred_wait`` span
            # stays open and the a2a_task / agent spans remain the logical
            # frames for the rest of the suspension window.
            set_current_span(prior)

    async def _close_deferred_wait_span(self, entry) -> None:
        """Close the open ``deferred_wait`` span (if any) saved on ``entry``.

        Pins the current span to the open ``deferred_wait`` span, ends it,
        then restores the prior current span. Clears
        ``entry.deferred_wait_span_id`` afterwards so no stale id remains.
        No-op if no tracer, no current trace, or no span id is recorded.
        """
        if not self._tracer or not entry.deferred_wait_span_id:
            return
        trace = get_current_trace()
        if trace is None:
            entry.deferred_wait_span_id = None
            return
        dw_span = next(
            (s for s in trace.spans if s.span_id == entry.deferred_wait_span_id),
            None,
        )
        if dw_span is None:
            entry.deferred_wait_span_id = None
            return
        prior = get_current_span()
        set_current_span(dw_span)
        try:
            await self._tracer.end_span()
        finally:
            # Restore whatever span was current before we touched context.
            set_current_span(prior)
            entry.deferred_wait_span_id = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        # Extract content from the incoming message (multi-part)
        message = context.message
        user_text, attachments = a2a_parts_to_obelix(message.parts)

        # Extract client metadata (shell environment, etc.) if present.
        # The client sends this via Message.metadata on the first message.
        # Stored in the ContextEntry so it persists across the conversation.
        if message.metadata and "client_info" in message.metadata:
            async with self._store_lock:
                entry = self._store.get_or_create(context_id)
            entry.client_info = message.metadata["client_info"]

        if not user_text and not attachments:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=agent_message("No user input provided"),
                    ),
                    final=True,
                )
            )
            # No a2a_task span is opened in this early-exit path, so the
            # emission is a no-op (add_event is guarded), but we still log
            # the transition for symmetry.
            await self._emit_state(
                from_state=None,
                to_state="failed",
                reason="no_input",
            )
            return

        # Get or create the context entry (LRU eviction under global lock)
        async with self._store_lock:
            entry = self._store.get_or_create(context_id)

        # Serialize requests on the same context
        await entry.idle.wait()
        entry.idle.clear()

        try:
            # RESUME PATH: if we have deferred tool calls, inject response
            is_resume = bool(entry.deferred_tool_calls)
            if is_resume:
                inject_deferred_response(entry, message)

            # Drain pending remote-task notifications BEFORE starting the
            # agent. Goes AFTER inject_deferred_response so the deferred
            # ToolMessage stays adjacent to its AssistantMessage; remote-task
            # notifications append after as fresh user-role messages. Runs
            # for both first-turn and resume paths — no-op when empty.
            if entry.pending_notifications:
                # Atomic-ish swap: any notification arriving from the webhook
                # between this read and the extend() goes onto the *new* empty
                # list and will be drained at the next request. Without the swap,
                # an extend()+clear() race could silently drop a notification.
                drained = entry.pending_notifications
                entry.pending_notifications = []
                entry.history.extend(drained)

            await self._run_agent(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                attachments=attachments,
                entry=entry,
                event_queue=event_queue,
                is_resume=is_resume,
            )
        finally:
            entry.active_agent = None
            entry.idle.set()

    async def _open_a2a_task_span(
        self,
        *,
        task_id: str,
        context_id: str,
        entry,
        is_resume: bool,
        is_drain_spawn: bool,
    ) -> tuple[Span | None, bool]:
        """Open the a2a_task root span for ``_run_agent`` and report ownership.

        Three branches, mutually exclusive:

        - ``is_resume=True``: the trace + a2a_task span are restored by
          ``_run_agent_impl`` from ``entry.trace_session`` / ``entry.trace_span``.
          Returns ``(None, False)``: caller did not open a span here, and the
          trace is owned by the original turn (will be closed by this same
          finally because ``is_resume`` flips the close-trace condition).
        - ``is_drain_spawn=True`` AND ``entry.trace_session`` non-None:
          reuse the existing trace, open a NEW ``a2a_task`` sibling span.
          Returns ``(span, False)``: the span belongs to this invocation
          and must be closed in finally, but the trace is owned by the
          context's first task and must NOT be closed here.
        - Else (user-triggered first turn, or drain-spawn fallback when no
          saved trace): open a fresh trace AND a fresh ``a2a_task`` span.
          Returns ``(span, True)``: this invocation owns both — close span
          and end trace in finally.

        No-op on tracer not configured: returns ``(None, False)``.
        """
        tracer = self._tracer
        if not tracer:
            return None, False

        if is_resume:
            # Existing path: deferred tool resume, trace already active.
            return None, False

        if is_drain_spawn and entry.trace_session is not None:
            # Drain-spawned task: reuse the context's existing trace so the
            # new a2a_task span shares trace_id with the previous task(s)
            # under the same context.
            set_current_trace(entry.trace_session)
            span = await tracer.start_span(
                SpanType.a2a_task,
                name=f"task {task_id[:8] if task_id else 'unknown'} (drain-spawn)",
                input={"context_id": context_id, "drain_spawn": True},
                metadata={
                    "task_id": task_id,
                    "context_id": context_id,
                    "drain_spawn": True,
                },
            )
            # entry.trace_session unchanged: entry already owns the trace.
            return span, False

        # Existing path: new user-triggered task (or drain-spawn fallback).
        await tracer.start_trace(
            name="a2a.task",
            metadata={"task_id": task_id, "context_id": context_id},
        )
        span = await tracer.start_span(
            SpanType.a2a_task,
            name=f"task {task_id[:8] if task_id else 'unknown'}",
            input={"context_id": context_id},
            metadata={"task_id": task_id, "context_id": context_id},
        )
        # Store the live trace on the entry so ``cancel()`` (which runs in a
        # different asyncio task and therefore has empty contextvars) can
        # find it and emit ``cancellation.requested`` on the a2a_task span.
        # The deferred-suspension path later overwrites this with the same
        # trace when it saves context for resume.
        entry.trace_session = get_current_trace()
        return span, True

    async def _run_agent(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        attachments: list,
        entry,
        event_queue: EventQueue,
        is_resume: bool = False,
        is_drain_spawn: bool = False,
    ) -> None:
        """Run the agent with isolated context and persist history.

        When a tracer is configured, opens an ``a2a_task`` span as the trace
        root before delegating to the agent. The agent's own span becomes a
        child of it. On deferred-tool suspension the trace and ``a2a_task``
        span are left open so ``resume_after_deferred`` can continue inside
        them; they are closed on any terminal outcome (completed / failed /
        rejected / canceled) or on the resume invocation that finishes them.

        Span/trace lifecycle (set by ``_open_a2a_task_span``):

        - User-triggered first turn: opens trace + span; finally closes both.
        - Resume: span/trace inherited; finally closes both (trace owned by
          the original turn, terminating now).
        - Drain-spawn (with saved trace_session): opens span, NOT trace;
          finally closes the span only, leaving the trace open for the
          context's other tasks.
        """

        tracer = self._tracer
        a2a_task_span, trace_opened_here = await self._open_a2a_task_span(
            task_id=task_id,
            context_id=context_id,
            entry=entry,
            is_resume=is_resume,
            is_drain_spawn=is_drain_spawn,
        )

        # Tracks whether the executor suspended for a deferred tool. When
        # True, the finally block leaves the trace + a2a_task span open so
        # ``resume_after_deferred`` can continue inside them.
        deferred_suspended = False

        try:
            deferred_suspended = await self._run_agent_impl(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                attachments=attachments,
                entry=entry,
                event_queue=event_queue,
                is_resume=is_resume,
                is_drain_spawn=is_drain_spawn,
            )
        finally:
            if tracer and not deferred_suspended:
                # Determine whether this invocation has a span to close.
                # Three sources:
                #  - First-turn / drain-spawn: ``a2a_task_span`` returned by
                #    ``_open_a2a_task_span``.
                #  - Resume: span/trace restored by ``_run_agent_impl``; we
                #    walk the trace to find the open ``a2a_task`` span.
                # Drain-spawn closes the span only; first-turn and resume
                # close the span AND end the trace.
                has_owned_span = a2a_task_span is not None
                close_trace_here = trace_opened_here or is_resume
                if has_owned_span or close_trace_here:
                    # Re-pin the a2a_task span as current before closing. On
                    # the initial invocation this is defensive (agent's
                    # generator finally should have already restored it); on
                    # resume we need to walk the trace to find it.
                    task_span = a2a_task_span
                    if task_span is None:
                        trace = get_current_trace()
                        if trace is not None:
                            task_span = next(
                                (
                                    s
                                    for s in trace.spans
                                    if s.span_type == SpanType.a2a_task
                                    and s.end_time is None
                                ),
                                None,
                            )
                    if task_span is not None:
                        set_current_span(task_span)
                    # Propagate terminal status onto the a2a_task span AND
                    # the trace itself (when we close the trace). Precedence:
                    # cancel > rejected > failed > ok. Cancel takes priority
                    # because ``_run_agent_impl`` may set ``was_canceled``
                    # alongside normal terminal flags if a cancel races with
                    # a response. ``was_rejected`` / ``was_failed`` are set
                    # by the corresponding ``except`` handlers; when neither
                    # fires we fall through to ``ok``. ``error`` (not passed
                    # on the cancel path) forwards the reason/exception
                    # message so consumers can render it in span views.
                    if entry.was_canceled:
                        status = SpanStatus.canceled
                        error: str | None = None
                    elif entry.was_rejected:
                        status = SpanStatus.rejected
                        error = entry.rejection_reason
                    elif entry.was_failed:
                        status = SpanStatus.error
                        error = entry.failure_error
                    else:
                        status = SpanStatus.ok
                        error = None
                    if task_span is not None:
                        await tracer.end_span(status=status, error=error)
                    if close_trace_here:
                        await tracer.end_trace(status=status, error=error)
                        # Clear the saved trace ref — the trace is now ended
                        # and any subsequent turn on this context will open
                        # a new one. Reset the terminal-state flags so the
                        # next turn on this context starts clean (a retry
                        # after rejection/failure must not be marked
                        # terminal by stale flags).
                        entry.trace_session = None
                        entry.was_canceled = False
                        entry.was_rejected = False
                        entry.was_failed = False
                        entry.rejection_reason = None
                        entry.failure_error = None

    async def _run_agent_impl(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        attachments: list,
        entry,
        event_queue: EventQueue,
        is_resume: bool = False,
        is_drain_spawn: bool = False,
    ) -> bool:
        """Inner agent runner. Returns True if suspended for a deferred tool."""

        logger.info(
            f"[A2A] Executing agent | task_id={task_id} context_id={context_id} "
            f"text_len={len(user_text)} history_len={len(entry.history)} "
            f"is_resume={is_resume}"
        )

        # Create a fresh agent for this request
        agent = self._agent_factory()
        entry.active_agent = agent

        # Inject client shell info into BashTool's ClientShellExecutor
        if entry.client_info:
            self._inject_client_info(agent, entry.client_info)

        # Inject the per-request ContextEntry into outbound A2A tools
        # (DispatchAgentTool, TaskListTool, etc.) so they can read/mutate
        # entry.remote_tasks and entry.pending_notifications.
        self._inject_context_entry(agent, entry, context_id)

        # Inject conversation history from this context
        if entry.history:
            agent.conversation_history = [agent.system_message, *entry.history]

        # Signal that the agent is working
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )
        # Initial entry: None -> working; resume: input_required -> working
        await self._emit_state(
            from_state="input_required" if is_resume else None,
            to_state="working",
            reason="resume" if is_resume else "execute_start",
        )

        stream = None
        # Captured on the success path to emit an ``assistant`` span as a
        # sibling of ``agent`` under ``a2a_task``. Stays ``None`` on
        # cancel / reject / failure / deferred suspension so those paths
        # do NOT emit an assistant span. Initialized BEFORE the try block so
        # that if any call below (e.g. tracer ``start_span`` on the human
        # span) raises, the ``except`` / ``finally`` / post-block
        # ``final_response is not None`` check does not hit
        # ``UnboundLocalError``.
        final_response: AssistantResponse | None = None
        try:
            # For resume: restore the trace context from the first invocation
            # so the resume appears under the same trace in the tracer UI.
            if is_resume:
                if entry.trace_session:
                    set_current_trace(entry.trace_session)
                    set_current_span(entry.trace_span)
                    entry.trace_session = None
                    entry.trace_span = None
                # Close the ``deferred_wait`` span that was opened at the
                # suspension point. Duration now reflects the wall-clock
                # time the task spent in ``input_required``.
                await self._close_deferred_wait_span(entry)
                stream = agent.resume_after_deferred()
            elif attachments:
                # Pass as HumanMessage with attachments for multimodal
                query = HumanMessage(content=user_text, attachments=attachments)
                stream = agent.execute_query_stream(query)
            else:
                stream = agent.execute_query_stream(user_text)

            # Emit human span as direct child of a2a_task on initial
            # invocation only. On resume the input is a DataPart (deferred
            # tool response) conceptually continuing the same turn, not a
            # new user query, so we skip emitting a second human span.
            if self._tracer and not is_resume:
                await self._tracer.start_span(
                    SpanType.human,
                    "human.input",
                    input=user_text,
                )
                await self._tracer.end_span(output=user_text)

            artifact_id = str(uuid.uuid4())
            first_chunk = True

            async for event in stream:
                # === Agent canceled by user ===
                if event.canceled:
                    entry.history = agent.conversation_history[1:]
                    # Propagate cancel intent to the outer finally so the
                    # a2a_task span is closed with SpanStatus.canceled.
                    entry.was_canceled = True
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.canceled,
                                message=agent_message("Task canceled by client"),
                            ),
                            final=True,
                        )
                    )
                    await self._emit_state(
                        from_state="working",
                        to_state="canceled",
                        reason="client_cancel",
                    )
                    logger.info(f"[A2A] Agent canceled by user | task_id={task_id}")
                    return False

                # === Deferred tool detected: emit input-required ===
                if event.deferred_tool_calls:
                    entry.history = agent.conversation_history[1:]
                    entry.deferred_tool_calls = event.deferred_tool_calls
                    entry.deferred_tools = list(agent.registered_tools)
                    # Save trace context so the resume continues the same trace
                    entry.trace_session = get_current_trace()
                    entry.trace_span = get_current_span()

                    # Open a ``deferred_wait`` span as a direct child of the
                    # ``a2a_task`` root, covering the input_required pause.
                    # The span stays OPEN across the suspension — it will be
                    # closed on the resume path (or on cancel).
                    await self._open_deferred_wait_span(
                        entry, event.deferred_tool_calls
                    )

                    # Build DataPart message from deferred tool calls
                    deferred_parts = deferred_calls_to_a2a_parts(
                        event.deferred_tool_calls
                    )

                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.input_required,
                                message=Message(
                                    role=Role.agent,
                                    parts=deferred_parts,
                                    message_id=str(uuid.uuid4()),
                                ),
                            ),
                            final=True,
                        )
                    )
                    await self._emit_state(
                        from_state="working",
                        to_state="input_required",
                        reason="deferred_tool",
                    )
                    logger.info(
                        f"[A2A] Input required | task_id={task_id} "
                        f"context_id={context_id} "
                        f"deferred_count={len(event.deferred_tool_calls)}"
                    )
                    return True

                # === Streaming token ===
                if event.token:
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            artifact=Artifact(
                                artifact_id=artifact_id,
                                parts=[Part(root=TextPart(text=event.token))],
                            ),
                            append=not first_chunk,
                            last_chunk=False,
                        )
                    )
                    first_chunk = False

                # === Final response ===
                if event.is_final and not event.deferred_tool_calls:
                    response = event.assistant_response
                    final_response = response
                    entry.history = agent.conversation_history[1:]

                    if first_chunk:
                        # Non-streaming: emit full response with multi-part
                        parts = (
                            obelix_response_to_a2a_parts(response)
                            if response
                            else [Part(root=TextPart(text=""))]
                        )
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=artifact_id,
                                    parts=parts,
                                ),
                                append=False,
                                last_chunk=True,
                            )
                        )
                    else:
                        # Streaming: text already sent, emit DataParts for tool results
                        final_parts = []
                        if response and response.tool_results:
                            for result in response.tool_results:
                                if isinstance(result.result, dict):
                                    final_parts.append(
                                        Part(
                                            root=DataPart(
                                                data=result.result,
                                                metadata={
                                                    "type": "tool_result",
                                                    "tool_name": result.tool_name,
                                                },
                                            )
                                        )
                                    )
                        if not final_parts:
                            final_parts = [Part(root=TextPart(text=""))]
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=artifact_id,
                                    parts=final_parts,
                                ),
                                append=True,
                                last_chunk=True,
                            )
                        )

                    # Place the agent response in a working status so
                    # the SDK TaskManager appends it to task.history.
                    history_parts = (
                        obelix_response_to_a2a_parts(response)
                        if response
                        else [Part(root=TextPart(text=""))]
                    )
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.working,
                                message=Message(
                                    role=Role.agent,
                                    parts=history_parts,
                                    message_id=str(uuid.uuid4()),
                                ),
                            ),
                            final=False,
                        )
                    )

                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(state=TaskState.completed),
                            final=True,
                        )
                    )
                    await self._emit_state(
                        from_state="working",
                        to_state="completed",
                        reason="final_response",
                    )
                    logger.info(f"[A2A] Agent completed | task_id={task_id}")
                    break

        except asyncio.CancelledError:
            entry.history = agent.conversation_history[1:]
            # Mark the context so the outer _run_agent finally closes the
            # a2a_task span with SpanStatus.canceled.
            entry.was_canceled = True
            # Sweep any in-flight remote tasks dispatched from this context
            # — silence late webhooks (token revoke) and flip status to
            # "killed". No wire call to the remote — Decision 7.
            self._revoke_in_flight_remote_tokens(entry, self._registry)
            logger.info(f"[A2A] Agent canceled | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=agent_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            await self._emit_state(
                from_state="working",
                to_state="canceled",
                reason="async_cancel",
            )
            raise

        except TaskRejectedError as e:
            entry.history = agent.conversation_history[1:]
            # Propagate rejection onto the outer finally so the a2a_task span
            # closes with ``SpanStatus.rejected`` and ``span.error`` carries
            # the reason (default fallback matches the log message below).
            entry.was_rejected = True
            entry.rejection_reason = e.reason or "Task rejected"
            logger.info(
                f"[A2A] Agent rejected task | task_id={task_id} reason={e.reason}"
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.rejected,
                        message=agent_message(e.reason),
                    ),
                    final=True,
                )
            )
            await self._emit_state(
                from_state="working",
                to_state="rejected",
                reason=e.reason,
            )

        except Exception as e:
            entry.history = agent.conversation_history[1:]
            # Propagate failure onto the outer finally so the a2a_task span
            # closes with ``SpanStatus.error`` and ``span.error`` carries the
            # exception message.
            entry.was_failed = True
            entry.failure_error = str(e) or type(e).__name__
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=agent_message(f"Execution failed: {e}"),
                    ),
                    final=True,
                )
            )
            await self._emit_state(
                from_state="working",
                to_state="failed",
                reason=str(e),
            )

        finally:
            # Always close the async generator to trigger cleanup
            # (tracer span closing, resource release) in base_agent's
            # _execute_loop finally block.
            if stream is not None:
                aclose = getattr(stream, "aclose", None)
                if aclose and callable(aclose):
                    try:
                        await aclose()
                    except TypeError:
                        pass  # not a real async generator (e.g. mock)

        # Emit assistant span on the success path only. ``final_response`` is
        # only set when the stream produced a terminal ``is_final`` event
        # without deferred tool calls, so rejection / cancellation / failure /
        # deferred suspension naturally skip this. We emit AFTER the stream's
        # ``aclose()`` so the BaseAgent ``agent`` span has already ended —
        # that way the ``assistant`` span becomes a direct child of
        # ``a2a_task`` (sibling of ``agent``), per spec §6.
        if self._tracer and final_response is not None:
            content = getattr(final_response, "content", None)
            await self._tracer.start_span(
                SpanType.assistant,
                "assistant.response",
                input={"has_tool_calls": False},
            )
            await self._tracer.end_span(output={"content": content})

        # Normal termination (completed / rejected / failed): not suspended
        # for deferred input, so the caller should close the trace.
        return False

    @staticmethod
    def _inject_client_info(agent: BaseAgent, client_info: dict) -> None:
        """Inject client shell environment into the agent's system message.

        Finds any BashTool with a ClientShellExecutor among the agent's
        registered tools, populates its shell_info, and appends the
        system_prompt_fragment to the agent's system message.
        """
        from obelix.adapters.outbound.shell.client_executor import ClientShellExecutor

        for tool in agent.registered_tools:
            executor = getattr(tool, "_executor", None)
            if isinstance(executor, ClientShellExecutor):
                if not executor.shell_info:
                    executor.set_shell_info(client_info)
                fragment = tool.system_prompt_fragment()
                if fragment and fragment not in agent.system_message.content:
                    agent.system_message.content += fragment
                    logger.info("[A2A] Injected client shell info into system message")

    @staticmethod
    def _inject_context_entry(
        agent: BaseAgent,
        entry: ContextEntry,
        context_id: str,
    ) -> None:
        """Inject the per-request ContextEntry (and surrounding context_id)
        into outbound A2A tools that opt in via ``set_context_entry``.

        Mirrors the precedent of ``_inject_client_info`` for BashTool but
        targets a different family of tools (the outbound A2A tools that
        track per-context remote_tasks and pending_notifications).

        Tools have heterogeneous signatures:
        - DispatchAgentTool: ``set_context_entry(entry, *, context_id)``
        - RespondToRemoteTool, TaskListTool, TaskGetTool, TaskStopTool:
          ``set_context_entry(entry)``

        We use ``inspect.signature`` to detect which form to invoke,
        so adding new tools with either signature is safe. When the setter
        accepts ``context_id``, it is passed as a keyword argument. If
        introspection fails (``TypeError`` or ``ValueError``, e.g., on
        MagicMock or C-extension callables), we fall back to the single-arg
        form ``setter(entry)``.
        """
        for tool in agent.registered_tools:
            setter = getattr(tool, "set_context_entry", None)
            if setter is None or not callable(setter):
                continue
            try:
                sig = inspect.signature(setter)
            except (TypeError, ValueError):
                # Some MagicMock / C-extension setters can't be introspected;
                # default to the single-arg form.
                setter(entry)
                continue
            if "context_id" in sig.parameters:
                setter(entry, context_id=context_id)
            else:
                setter(entry)

    @staticmethod
    def _revoke_in_flight_remote_tokens(
        entry: ContextEntry,
        registry: RemoteAgentRegistry | None,
    ) -> None:
        """On context cancel, silence late webhooks for non-terminal remote
        tasks by revoking their tokens locally and flipping status to
        ``"killed"``. NO wire call to the remote — per Decision 7, the
        remote owns its own lifecycle.

        Safe to call when ``registry`` is None (no remote_agents
        configured) — in that case it's a no-op.
        """
        if registry is None:
            return
        killed_ids: list[str] = []
        for state in list(entry.remote_tasks.values()):
            if state.is_terminal:
                continue
            registry.revoke(state.token)
            state.status = "killed"
            # Pair last_update with last_update_monotonic — same contract
            # as handler.py, dispatch.py, and task_ops.py: both must be
            # written together on every state change so task_list/task_get
            # don't surface a stale wall-clock timestamp to the LLM.
            state.last_update = datetime.now(UTC)
            state.last_update_monotonic = time.monotonic()
            killed_ids.append(state.task_id)
        if killed_ids:
            logger.info(
                f"[A2A] revoked in-flight remote tasks on cancel | "
                f"count={len(killed_ids)} task_ids={killed_ids}"
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        logger.info(f"[A2A] Cancel requested | task_id={task_id}")

        # Try to find the active agent for this context and signal cancel
        async with self._store_lock:
            entry = self._store.get_or_create(context_id)

        # Emit cancellation.requested on the a2a_task span BEFORE branching
        # into the in-flight / deferred paths. This way the event fires
        # regardless of which branch handles the cancel. We pass the entry's
        # saved trace explicitly because cancel() typically runs in a
        # different asyncio task than execute(), so contextvars are empty.
        # The helper is a no-op if no trace or no a2a_task span exists —
        # safe even if no trace was ever opened.
        await self._emit_cancellation_event(
            source="client",
            trace=entry.trace_session,
        )

        if entry.active_agent:
            # Agent is running — signal cooperative cancellation.
            # The _execute_loop will detect the flag, yield a canceled
            # StreamEvent, and _run_agent will emit TaskState.canceled.
            # _run_agent_impl will set entry.was_canceled so the outer
            # finally closes the a2a_task span with SpanStatus.canceled.
            entry.active_agent.cancel()
            logger.info(
                f"[A2A] Cancel signal sent to active agent | "
                f"task_id={task_id} context_id={context_id}"
            )
        else:
            # No active agent — task was likely in input_required (deferred).
            # Capture the "from" state before cleanup below clears the flag.
            was_deferred = bool(entry.deferred_tool_calls)
            saved_trace = entry.trace_session
            saved_span = entry.trace_span
            # Clean up: replace the null ToolMessage with a cancel message
            # so the LLM knows the tool was not executed.
            if entry.deferred_tool_calls and entry.history:
                # Deferred tools have NO ToolMessage in the history yet —
                # only the AssistantMessage with tool_calls was saved.
                # Inject a ToolMessage with cancel results so the LLM
                # sees a proper tool_use → tool_result sequence.
                cancel_results = [
                    ToolResult(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result="Execution canceled by user",
                        status=ToolStatus.ERROR,
                    )
                    for tc in entry.deferred_tool_calls
                ]
                entry.history.append(ToolMessage(tool_results=cancel_results))
                entry.deferred_tool_calls = None
                entry.deferred_tools = None
                entry.trace_session = None
                entry.trace_span = None

            # Sweep any other in-flight remote tasks (besides the deferred
            # one) — see Decision 7. NO wire call.
            self._revoke_in_flight_remote_tokens(entry, self._registry)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=agent_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            # Restore the trace context saved at suspension so tracer
            # operations (deferred_wait close, state_change event, a2a_task
            # close, trace end) attach to the original task. The
            # ``cancellation.requested`` event was already emitted at the top
            # of cancel(). Restore whatever was current before afterwards
            # (almost always None here).
            if self._tracer and saved_trace is not None:
                prior_trace = get_current_trace()
                prior_span = get_current_span()
                set_current_trace(saved_trace)
                set_current_span(saved_span)
                try:
                    # Close the open deferred_wait span first so its end_time
                    # is recorded BEFORE the a2a_task span closes — addresses
                    # the Task 19 leak (span previously stayed open on cancel).
                    await self._close_deferred_wait_span(entry)
                    await self._emit_state(
                        from_state="input_required" if was_deferred else "working",
                        to_state="canceled",
                        reason="cancel_request",
                    )
                    # Finally, close the a2a_task span with canceled status
                    # and end the trace with canceled status so consumers
                    # filtering by trace status see the cancellation. The
                    # suspended deferred path owns the open trace, so cancel
                    # is the only place it can be closed (no resume will come).
                    entry.was_canceled = True
                    a2a_span = next(
                        (
                            s
                            for s in saved_trace.spans
                            if s.span_type == SpanType.a2a_task and s.end_time is None
                        ),
                        None,
                    )
                    if a2a_span is not None:
                        set_current_span(a2a_span)
                        await self._tracer.end_span(status=SpanStatus.canceled)
                        await self._tracer.end_trace(status=SpanStatus.canceled)
                finally:
                    set_current_trace(prior_trace)
                    set_current_span(prior_span)
