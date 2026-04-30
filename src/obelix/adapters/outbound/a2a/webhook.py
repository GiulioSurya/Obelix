"""Starlette route factory for the inbound webhook receiving A2A push
notifications from remote agents we dispatched tasks to.

The handler closure built by ``make_webhook_handler`` is mounted on the
existing A2A server's FastAPI app at ``/webhook`` (see AgentFactory.a2a_serve
in T14). It validates the ``X-A2A-Notification-Token`` header against the
in-memory token map, locates the originating ContextEntry, and delegates
to ``handle_remote_update`` (T6).

Tracer events ``remote_task.update`` are attached here using the saved
``entry.trace_session`` so the event lands on the original a2a_task span
even though we run in a different asyncio task with empty contextvars
(mirror of executor._emit_cancellation_event).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from a2a.types import Task
from starlette.requests import Request
from starlette.responses import JSONResponse

from obelix.adapters.outbound.a2a.handler import handle_remote_update
from obelix.core.tracer.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from obelix.core.tracer.models import SpanType
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextStore
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
    from obelix.core.tracer.tracer import Tracer

logger = get_logger(__name__)

_HEADER = "X-A2A-Notification-Token"


def make_webhook_handler(
    registry: RemoteAgentRegistry,
    context_store: ContextStore,
    *,
    tracer: Tracer | None = None,
) -> Callable:
    """Build the /webhook handler closure.

    The closure validates the token, locates the ContextEntry, and
    delegates to handle_remote_update. Tracer event emission is wrapped
    around the delegate call when a tracer + saved trace_session are
    available.
    """

    async def webhook_handler(request: Request) -> JSONResponse:
        token = request.headers.get(_HEADER)
        if not token:
            logger.warning("[A2A webhook] missing token header")
            return JSONResponse({"error": "missing token"}, status_code=401)

        route = registry.lookup(token)
        if route is None:
            logger.warning(f"[A2A webhook] unknown token (len={len(token)})")
            return JSONResponse({"error": "unknown token"}, status_code=401)

        # Parse body before doing anything else (so race-fallback can read body.id).
        try:
            body = await request.json()
        except Exception as e:
            logger.warning(f"[A2A webhook] malformed JSON | error={e}")
            return JSONResponse({"error": "bad json"}, status_code=400)

        try:
            fresh = Task(**body)
        except Exception as e:
            logger.warning(f"[A2A webhook] body not a Task | error={e}")
            return JSONResponse({"error": "bad task"}, status_code=400)

        # Race fallback: if send_message hasn't returned yet, route.task_id
        # is None. Use body.id (which the sender always sets).
        task_id = route.task_id or fresh.id
        if route.task_id is None:
            registry.claim_task_id(token, fresh.id)

        # Locate context. If evicted (or never created), log + 200-OK,
        # no further side effects.
        entry = context_store._contexts.get(route.context_id)
        if entry is None:
            logger.warning(
                f"[A2A webhook] context not in store (evicted or never "
                f"existed) | context_id={route.context_id} task_id={task_id}; "
                f"dropping update"
            )
            return JSONResponse({"ok": True})

        # Tracer event across HTTP boundary: pin trace + a2a_task span
        # from entry.trace_session, emit remote_task.update, restore.
        # Pattern mirrors _emit_cancellation_event in executor.py.
        if tracer is not None and entry.trace_session is not None:
            a2a_span = next(
                (
                    s
                    for s in entry.trace_session.spans
                    if s.span_type == SpanType.a2a_task
                ),
                None,
            )
            if a2a_span is not None:
                prior_trace = get_current_trace()
                prior_span = get_current_span()
                set_current_trace(entry.trace_session)
                set_current_span(a2a_span)
                try:
                    prev_status = (
                        entry.remote_tasks[task_id].status
                        if task_id in entry.remote_tasks
                        else None
                    )
                    new_status = (
                        fresh.status.state.value
                        if hasattr(fresh.status.state, "value")
                        else str(fresh.status.state)
                    )
                    await tracer.add_event(
                        "remote_task.update",
                        {
                            "task_id": task_id,
                            "agent": route.agent_name,
                            "from": prev_status,
                            "to": new_status.replace("-", "_"),
                        },
                    )
                finally:
                    set_current_trace(prior_trace)
                    set_current_span(prior_span)

        handle_remote_update(
            entry=entry, task_id=task_id, fresh=fresh, registry=registry
        )
        return JSONResponse({"ok": True})

    return webhook_handler
