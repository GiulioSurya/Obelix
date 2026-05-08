"""Bug regression: when a server-spawned task (drain-spawn) posts to the
CLI webhook, the task_id is unknown to the TaskTracker (the CLI never
called register() for it). The current behavior labels the task as
agent_name='unknown', which surfaces in the status bar as
``unknown: 1 new`` and detaches the task from the originating agent.

Fix: TaskTracker accepts an optional ``context_resolver`` — a callable
``(context_id) -> agent_name | None``. On unknown task_id, the resolver
is consulted using the payload's ``contextId`` field. If it returns a
known agent_name, that's used instead of the literal 'unknown'.

This works because the CLI maintains a 1:1 map of context_id → agent
through ``AgentConnection.context_id`` set on first send_message.

Iron rule: NO mock; hand-written resolver lambda.
"""

from __future__ import annotations

import pytest

from obelix.adapters.inbound.a2a.client.webhook_server import TaskTracker


@pytest.mark.asyncio
async def test_unknown_task_with_no_resolver_falls_back_to_unknown():
    """Backward compat: without a resolver, behavior is the legacy 'unknown'."""
    tracker = TaskTracker()
    await tracker.update(
        {
            "id": "task-server-spawn-1",
            "contextId": "ctx-orchestrator",
            "kind": "task",
            "status": {
                "state": "completed",
                "timestamp": "2026-05-08T10:00:00+00:00",
            },
        }
    )
    info = tracker.get("task-server-spawn-1")
    assert info is not None
    assert info.agent_name == "unknown"


@pytest.mark.asyncio
async def test_unknown_task_with_resolver_uses_resolved_agent_name():
    """When context_resolver returns a known agent_name for the payload's
    contextId, the unknown task is attributed to that agent (not 'unknown').
    """
    # Hand-written resolver: simulates the CLI's agent connection map.
    context_to_agent = {"ctx-orch-42": "orchestrator"}

    def resolver(context_id: str) -> str | None:
        return context_to_agent.get(context_id)

    tracker = TaskTracker(context_resolver=resolver)

    await tracker.update(
        {
            "id": "task-server-spawn-2",
            "contextId": "ctx-orch-42",
            "kind": "task",
            "status": {
                "state": "completed",
                "timestamp": "2026-05-08T10:00:00+00:00",
            },
        }
    )

    info = tracker.get("task-server-spawn-2")
    assert info is not None
    assert info.agent_name == "orchestrator", (
        f"context_resolver should have attributed the task to 'orchestrator', "
        f"got agent_name={info.agent_name!r}."
    )


@pytest.mark.asyncio
async def test_unknown_task_with_resolver_returning_none_falls_back_to_unknown():
    """If the resolver doesn't recognize the context_id, fall back to 'unknown'."""
    tracker = TaskTracker(context_resolver=lambda cid: None)
    await tracker.update(
        {
            "id": "task-server-spawn-3",
            "contextId": "ctx-stranger",
            "kind": "task",
            "status": {"state": "completed"},
        }
    )
    info = tracker.get("task-server-spawn-3")
    assert info is not None
    assert info.agent_name == "unknown"


@pytest.mark.asyncio
async def test_resolver_not_consulted_for_known_task_id():
    """If the task_id was previously registered, the resolver MUST NOT be
    invoked — register() takes precedence and the original agent_name is
    preserved.
    """
    calls: list[str] = []

    def resolver(context_id: str) -> str | None:
        calls.append(context_id)
        return "should-not-be-used"

    tracker = TaskTracker(context_resolver=resolver)
    await tracker.register("task-known", agent_name="real_agent")

    await tracker.update(
        {
            "id": "task-known",
            "contextId": "ctx-known",
            "kind": "task",
            "status": {"state": "completed"},
        }
    )
    info = tracker.get("task-known")
    assert info.agent_name == "real_agent"
    assert calls == [], (
        f"resolver was unexpectedly invoked: calls={calls}. "
        "It must only be consulted for UNKNOWN task_ids."
    )
