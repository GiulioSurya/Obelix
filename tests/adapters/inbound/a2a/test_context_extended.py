import logging
from datetime import datetime

import pytest
from loguru import logger as loguru_logger

from obelix.adapters.inbound.a2a.server.context import (
    ContextEntry,
    ContextStore,
)
from obelix.adapters.outbound.a2a.state import RemoteTaskState


@pytest.fixture
def caplog(caplog):
    """Bridge loguru -> stdlib logging so pytest's ``caplog`` can capture
    warnings emitted via ``obelix.infrastructure.logging.get_logger``.

    ``context.py`` routes through loguru, which does not propagate to the
    standard logging tree by default.
    """
    handler_id = loguru_logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    loguru_logger.remove(handler_id)


def _running_task() -> RemoteTaskState:
    return RemoteTaskState(
        task_id="t-1",
        agent_name="B",
        status="working",
        token="tok",
        created_at=datetime.now(),
        last_update=datetime.now(),
        last_update_monotonic=0.0,
        last_artifact=None,
        deferred_calls=None,
    )


def _completed_task() -> RemoteTaskState:
    s = _running_task()
    s.status = "completed"
    return s


def test_context_entry_has_remote_tasks_and_pending_notifications() -> None:
    entry = ContextEntry()
    assert entry.remote_tasks == {}
    assert entry.pending_notifications == []


def test_context_entry_slots_includes_new_fields() -> None:
    entry = ContextEntry()
    assert "remote_tasks" in ContextEntry.__slots__
    assert "pending_notifications" in ContextEntry.__slots__
    # No __dict__ leak (verifies slots)
    assert not hasattr(entry, "__dict__")


def test_is_evictable_empty_remote_tasks() -> None:
    entry = ContextEntry()
    assert entry.is_evictable() is True


def test_is_evictable_all_terminal() -> None:
    entry = ContextEntry()
    entry.remote_tasks["t-1"] = _completed_task()
    assert entry.is_evictable() is True


def test_is_evictable_one_running() -> None:
    entry = ContextEntry()
    entry.remote_tasks["t-1"] = _running_task()
    assert entry.is_evictable() is False


def test_store_evicts_evictable_first() -> None:
    store = ContextStore(max_contexts=2)
    e1 = store.get_or_create("ctx-1")
    _e2 = store.get_or_create("ctx-2")
    # ctx-1 is non-evictable
    e1.remote_tasks["t-1"] = _running_task()
    # Trigger eviction — should evict ctx-2 (evictable), keep ctx-1
    _e3 = store.get_or_create("ctx-3")
    assert "ctx-1" in store._contexts
    assert "ctx-2" not in store._contexts
    assert "ctx-3" in store._contexts


def test_store_force_evicts_at_2x_when_all_non_evictable(caplog) -> None:
    caplog.set_level(logging.WARNING)
    store = ContextStore(max_contexts=2)
    # Fill up to 2x with non-evictable
    for i in range(4):
        e = store.get_or_create(f"ctx-{i}")
        e.remote_tasks[f"t-{i}"] = _running_task()
    assert len(store._contexts) == 4
    # One more must force-evict the oldest with a warning
    store.get_or_create("ctx-extra")
    assert len(store._contexts) == 4
    assert "ctx-0" not in store._contexts
    assert any("Forced eviction" in r.message for r in caplog.records)
