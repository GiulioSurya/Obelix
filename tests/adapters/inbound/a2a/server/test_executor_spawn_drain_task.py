"""Test that spawn_drain_task schedules a background asyncio task and returns
immediately (fire-and-forget pattern).

NO MOCKS per iron rule: usa una subclass dell'executor con __init__ vuoto
che sovrascrive solo _run_drain_task per osservare la sequenza temporale.
"""

from __future__ import annotations

import asyncio

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


@pytest.mark.asyncio
async def test_spawn_drain_task_returns_immediately_and_schedules_background():
    """spawn_drain_task must NOT block on the agent run.

    Verifies:
    1. Calling spawn_drain_task returns in << seconds.
    2. A background coroutine has been scheduled (sleep starts running).
    3. The outer caller did not wait for the sleep to complete.
    """
    sleep_started = asyncio.Event()
    sleep_completed = asyncio.Event()

    class FastSpawnExecutor(ObelixAgentExecutor):
        def __init__(self):
            # Skip parent __init__ — only test spawn_drain_task here.
            # Race-safe metadata write in spawn_drain_task reads
            # ``self._task_store``; declare it as None since these tests
            # don't exercise the metadata-patch branch.
            self._task_store = None

        async def _run_drain_task(self, *, task_id, context_id, entry, message):
            sleep_started.set()
            await asyncio.sleep(0.5)
            sleep_completed.set()

    executor = FastSpawnExecutor()
    entry = ContextEntry()

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await executor.spawn_drain_task(entry=entry, context_id="ctx-test")
    elapsed = loop.time() - t0

    assert elapsed < 0.1, (
        f"spawn returned too slowly ({elapsed:.3f}s) — must be fire-and-forget"
    )

    # Background must have started
    await asyncio.wait_for(sleep_started.wait(), timeout=1.0)
    # Sleep is in progress, not yet completed
    assert not sleep_completed.is_set()

    # Wait for cleanup
    await asyncio.wait_for(sleep_completed.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_spawn_drain_task_passes_synthetic_message_with_empty_parts():
    """spawn_drain_task constructs a Message with parts=[], role=Role.user,
    correct context_id. We capture it via _run_drain_task to verify."""
    captured: dict = {}

    class CapturingExecutor(ObelixAgentExecutor):
        def __init__(self):
            # Bypass parent __init__; declare _task_store=None so the
            # spawn_drain_task metadata-patch branch is skipped.
            self._task_store = None

        async def _run_drain_task(self, *, task_id, context_id, entry, message):
            captured["task_id"] = task_id
            captured["context_id"] = context_id
            captured["entry"] = entry
            captured["message"] = message

    executor = CapturingExecutor()
    entry = ContextEntry()

    await executor.spawn_drain_task(entry=entry, context_id="ctx-42")
    # Wait briefly for the asyncio task to schedule
    await asyncio.sleep(0.05)

    assert captured["context_id"] == "ctx-42"
    assert captured["entry"] is entry
    msg = captured["message"]
    # Verify Message contract per research artifact
    assert msg.role.value == "user"
    assert msg.parts == []
    assert msg.context_id == "ctx-42"
    assert msg.message_id  # uuid
    # task_id is a fresh uuid
    assert captured["task_id"]
    assert isinstance(captured["task_id"], str)


@pytest.mark.asyncio
async def test_spawn_drain_task_does_not_propagate_exceptions():
    """If _run_drain_task raises, spawn_drain_task itself must NOT raise.
    The error is logged inside _run_drain_task by design (fire-and-forget).
    """

    class BoomExecutor(ObelixAgentExecutor):
        def __init__(self):
            # Bypass parent __init__; declare _task_store=None so the
            # spawn_drain_task metadata-patch branch is skipped.
            self._task_store = None

        async def _run_drain_task(self, *, task_id, context_id, entry, message):
            raise RuntimeError("boom")

    executor = BoomExecutor()
    entry = ContextEntry()

    # Must not raise
    await executor.spawn_drain_task(entry=entry, context_id="c")
    # Give time for the asyncio task to run and the exception to be raised inside it
    await asyncio.sleep(0.05)
    # If we reach here, the outer call did not propagate the exception.
