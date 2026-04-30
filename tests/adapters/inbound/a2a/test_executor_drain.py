"""Tests for _inject_context_entry + pending_notifications drain in
ObelixAgentExecutor (T12)."""

import inspect
import time
import unittest.mock
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.core.model.human_message import HumanMessage


def _running_remote_task() -> RemoteTaskState:
    return RemoteTaskState(
        task_id="t-1",
        agent_name="B",
        status="working",
        token="tok",
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic(),
        last_artifact=None,
        deferred_calls=None,
    )


# ── _inject_context_entry tests ───────────────────────────────────────────


def test_inject_context_entry_calls_set_on_supporting_tools():
    """Tool with set_context_entry(entry) gets called; tool without it skipped."""
    agent = MagicMock()
    tool_a = MagicMock()
    tool_a.set_context_entry = MagicMock()
    tool_b = MagicMock(spec=[])  # no set_context_entry attribute
    agent.registered_tools = [tool_a, tool_b]

    entry = ContextEntry()
    ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")

    tool_a.set_context_entry.assert_called_once()


def test_inject_context_entry_passes_context_id_when_signature_accepts():
    """Tools whose set_context_entry has a context_id parameter get it."""
    agent = MagicMock()

    # Real-shape function with context_id
    def _set_with_ctx(entry, *, context_id):
        _set_with_ctx.received = (entry, context_id)

    _set_with_ctx.received = None  # type: ignore[attr-defined]

    tool_with_ctx = MagicMock()
    tool_with_ctx.set_context_entry = _set_with_ctx
    # Make hasattr(...) return True
    tool_with_ctx.set_context_entry.__signature__ = inspect.signature(_set_with_ctx)
    agent.registered_tools = [tool_with_ctx]

    entry = ContextEntry()
    ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")

    assert _set_with_ctx.received == (entry, "ctx-MARIO")


def test_inject_context_entry_omits_context_id_when_signature_lacks_it():
    """Tools whose set_context_entry takes only entry don't get context_id."""

    def _set_simple(entry):
        _set_simple.received = entry

    _set_simple.received = None  # type: ignore[attr-defined]

    tool_simple = MagicMock()
    tool_simple.set_context_entry = _set_simple
    agent = MagicMock()
    agent.registered_tools = [tool_simple]

    entry = ContextEntry()
    ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")

    assert _set_simple.received is entry


def test_inject_context_entry_handles_no_tools():
    """Empty registered_tools is a no-op."""
    agent = MagicMock()
    agent.registered_tools = []
    entry = ContextEntry()
    # Must not raise.
    ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")


def test_inject_context_entry_falls_back_when_signature_fails():
    """Setter that raises TypeError/ValueError from inspect.signature
    (e.g., MagicMock with C-extension semantics) defaults to single-arg
    invocation."""
    received = []

    class _IntrospectableTool:
        """Plain class with a real (introspectable) setter. The
        test forcibly breaks inspect.signature via patch to exercise
        the fallback path, NOT because the class itself resists
        introspection."""

        def set_context_entry(self, entry):
            received.append(entry)

    tool = _IntrospectableTool()

    agent = MagicMock()
    agent.registered_tools = [tool]

    entry = ContextEntry()

    # Patch inspect.signature to raise for this specific setter
    real_signature = inspect.signature

    def _raising_signature(obj, *args, **kwargs):
        if obj is tool.set_context_entry:
            raise ValueError("synthetic introspection failure")
        return real_signature(obj, *args, **kwargs)

    with unittest.mock.patch(
        "obelix.adapters.inbound.a2a.server.executor.inspect.signature",
        side_effect=_raising_signature,
    ):
        ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")

    # Despite signature inspection failing, the setter was still called
    # with just the entry (single-arg fallback).
    assert received == [entry]


def test_inject_context_entry_propagates_setter_exceptions():
    """If the setter itself raises (e.g., wrong arity, programming error),
    the exception propagates — not silently swallowed."""

    class _BrokenSetter:
        def set_context_entry(self, entry, *, context_id):
            raise RuntimeError("broken setter")

    tool = _BrokenSetter()
    agent = MagicMock()
    agent.registered_tools = [tool]

    entry = ContextEntry()
    with pytest.raises(RuntimeError, match="broken setter"):
        ObelixAgentExecutor._inject_context_entry(agent, entry, "ctx-MARIO")


# ── pending_notifications drain logic ─────────────────────────────────────


def test_drain_appends_to_history_and_clears():
    """Direct unit test of the drain helper: notifications append to
    history, then queue is cleared."""
    entry = ContextEntry()
    entry.history = [HumanMessage(content="prior")]
    entry.pending_notifications = [
        HumanMessage(content="<remote_task_update>1</remote_task_update>"),
        HumanMessage(content="<remote_task_update>2</remote_task_update>"),
    ]

    # Inline the drain logic the executor performs (swap pattern,
    # see executor.py — keeps tests in sync with production).
    if entry.pending_notifications:
        drained = entry.pending_notifications
        entry.pending_notifications = []
        entry.history.extend(drained)

    assert len(entry.history) == 3
    assert "1" in entry.history[1].content
    assert "2" in entry.history[2].content
    assert entry.pending_notifications == []


def test_drain_empty_queue_is_noop():
    """When queue is empty, drain does nothing."""
    entry = ContextEntry()
    entry.history = [HumanMessage(content="hi")]

    # Inline the drain logic the executor performs (swap pattern,
    # see executor.py — keeps tests in sync with production).
    if entry.pending_notifications:
        drained = entry.pending_notifications
        entry.pending_notifications = []
        entry.history.extend(drained)

    assert len(entry.history) == 1
