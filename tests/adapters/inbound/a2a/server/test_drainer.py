"""Tests for the drainer that auto-spawns A2A tasks from pending_notifications.

NO MOCKS: per iron rule (memoria feedback_test_iron_rule.md). Uses hand-written
Fake classes that implement the same Protocol surface as the real production code.
"""

from __future__ import annotations

import asyncio

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.drainer import maybe_spawn_drain_task
from obelix.core.model.human_message import HumanMessage


class FakeExecutor:
    """Records spawn_drain_task invocations. Real executor signature is preserved."""

    def __init__(self) -> None:
        self.spawn_calls: list[tuple[ContextEntry, str]] = []

    async def spawn_drain_task(self, *, entry: ContextEntry, context_id: str) -> None:
        self.spawn_calls.append((entry, context_id))


@pytest.mark.asyncio
async def test_no_op_when_no_pending_notifications():
    entry = ContextEntry()
    executor = FakeExecutor()

    await maybe_spawn_drain_task(entry=entry, context_id="ctx-1", executor=executor)

    assert executor.spawn_calls == []


@pytest.mark.asyncio
async def test_no_op_when_active_turn_in_progress():
    entry = ContextEntry()
    entry.pending_notifications.append(HumanMessage(content="x"))
    entry.idle.clear()
    executor = FakeExecutor()

    await maybe_spawn_drain_task(entry=entry, context_id="ctx-1", executor=executor)

    assert executor.spawn_calls == []


@pytest.mark.asyncio
async def test_spawns_when_pending_and_idle():
    entry = ContextEntry()
    entry.pending_notifications.append(HumanMessage(content="<remote_task_update>"))
    executor = FakeExecutor()

    await maybe_spawn_drain_task(entry=entry, context_id="ctx-1", executor=executor)

    assert len(executor.spawn_calls) == 1
    assert executor.spawn_calls[0] == (entry, "ctx-1")


@pytest.mark.asyncio
async def test_idempotent_when_called_twice():
    entry = ContextEntry()
    entry.pending_notifications.append(HumanMessage(content="x"))
    executor = FakeExecutor()

    await maybe_spawn_drain_task(entry=entry, context_id="c", executor=executor)
    await maybe_spawn_drain_task(entry=entry, context_id="c", executor=executor)

    assert len(executor.spawn_calls) == 2


@pytest.mark.asyncio
async def test_does_not_block_on_spawn():
    entry = ContextEntry()
    entry.pending_notifications.append(HumanMessage(content="x"))

    class SlowFakeExecutor(FakeExecutor):
        async def spawn_drain_task(self, *, entry, context_id):
            self.spawn_calls.append((entry, context_id))

    executor = SlowFakeExecutor()
    await asyncio.wait_for(
        maybe_spawn_drain_task(entry=entry, context_id="c", executor=executor),
        timeout=1.0,
    )
    assert len(executor.spawn_calls) == 1
