"""Regression tests for the drain-spawn path.

NOTE (spec 2, Task 5): These tests build the executor without a TaskStore
and call ``_run_drain_task`` directly, relying on the now-deleted
``_NullEventQueue`` to swallow events without persisting state. Task 5
rewires the drainer onto the SDK's normal ``EventQueue + EventConsumer +
TaskManager`` pipeline, which requires a real ``TaskStore``. These three
tests are therefore skipped here and rewritten in Task 16 against the
new pipeline (with a real ``InMemoryTaskStore`` so the agent run actually
persists artifact + status). The new equivalent end-to-end coverage lives
in ``test_drain_run_writes_to_store.py``.

Two structural bugs were discovered during the spec-1 smoke test and
fixed in the same commit as these tests:

Bug A: ``_run_drain_task`` bypassed ``execute()`` and therefore SKIPPED the
       drain of ``entry.pending_notifications`` (executor.py:334-341 lives
       only in ``execute()``). Notifications stayed in the queue forever,
       and the agent ran on stale history without ever seeing the
       ``<remote_task_update>`` HumanMessage.

Bug B: ``_run_drain_task`` called ``_run_agent(user_text="", ...)`` which
       eventually reached ``BaseAgent.execute_query_stream("")``. In
       ``_execute_loop`` (base_agent.py:493-494) the empty string was
       treated as a fresh user query and a ``HumanMessage(content="")``
       was appended to the conversation history. Anthropic rejected this
       with a 400 ``messages: text content blocks must be non-empty``
       error.

Fix:
- ``_run_drain_task`` now drains ``entry.pending_notifications`` into
  ``entry.history`` BEFORE running the agent (replicates execute():334-341).
- ``_run_agent_impl`` now treats ``is_drain_spawn=True`` like the existing
  ``is_resume`` branch: calls ``agent.resume_after_deferred()`` so the
  loop restarts on the existing history WITHOUT appending a synthetic
  empty HumanMessage.

These tests assert the post-fix invariants and lock in the new behavior.

Iron rule: NO unittest.mock / pytest-mock / monkeypatch. All Fakes are
hand-written and implement the exact contract documented in the
research artifact (.claude/spikes/a2a-types-spike.py + research doc).
"""

from __future__ import annotations

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextStore
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.usage import Usage
from obelix.infrastructure.providers import Providers

# Spec 2 / Task 5: drainer now drives a real SDK EventQueue +
# EventConsumer + TaskManager. The legacy assertions in this file inspect
# state that requires the drainer's old _NullEventQueue (deleted) and a
# missing TaskStore (now mandatory). Task 16 rewrites the equivalent
# coverage against the new pipeline with a real InMemoryTaskStore. The
# end-to-end shape is already exercised in test_drain_run_writes_to_store.
pytestmark = pytest.mark.skip(
    reason=(
        "Legacy drain-spawn assertions: rewritten in Task 16 against the "
        "new SDK pipeline. See test_drain_run_writes_to_store.py for the "
        "current end-to-end coverage."
    )
)


class _FakeProvider:
    """Scripted provider. Returns AssistantMessage in order from `responses`.

    Implements the AbstractLLMProvider contract: ``invoke``,
    ``invoke_stream`` (async generator that immediately raises
    NotImplementedError so BaseAgent falls back to ``invoke``), plus
    ``provider_type`` / ``model_id`` properties.
    """

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.invocations = 0
        self.last_messages_seen: list = []

    @property
    def provider_type(self):
        return Providers.ANTHROPIC

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def invoke(self, messages, tools, response_schema=None) -> AssistantMessage:
        self.last_messages_seen = list(messages)
        if self.invocations >= len(self._responses):
            r = AssistantMessage(
                content="(no more scripted responses)",
                tool_calls=[],
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        else:
            r = self._responses[self.invocations]
        self.invocations += 1
        return r

    async def invoke_stream(self, messages, tools, response_schema=None):
        # Falls through to invoke() because BaseAgent catches NotImplementedError
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator)


def _make_agent_factory(provider: _FakeProvider):
    """Returns a callable that builds a fresh BaseAgent on each call —
    same shape as ``ObelixAgentExecutor.__init__``'s agent_factory param.
    """

    def factory() -> BaseAgent:
        return BaseAgent(
            system_message="You are a test orchestrator.",
            provider=provider,
        )

    return factory


@pytest.mark.asyncio
async def test_drain_spawn_drains_pending_notifications():
    """``_run_drain_task`` must drain ``entry.pending_notifications`` exactly
    once and append the notifications to ``entry.history`` before running
    the agent.

    Contract:
    - pending_notifications is empty after the call (FIFO drained)
    - history contains every drained HumanMessage at its tail
    """
    provider = _FakeProvider(
        responses=[
            AssistantMessage(
                content="ack",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            ),
        ]
    )
    store = ContextStore(max_contexts=8)
    executor = ObelixAgentExecutor(
        agent_factory=_make_agent_factory(provider),
        context_store=store,
    )

    entry = store.get_or_create("ctx-fix-a")
    notification = HumanMessage(
        content=(
            "<remote_task_update>"
            "<task_id>t-coord</task_id>"
            "<agent>coordinator</agent>"
            "<status>completed</status>"
            "<result>staging area is empty</result>"
            "</remote_task_update>"
        )
    )
    entry.pending_notifications.append(notification)
    history_before = len(entry.history)

    from a2a.types import Message, Role

    synthetic = Message(
        message_id="m-1",
        role=Role.user,
        parts=[],
        context_id="ctx-fix-a",
    )
    await executor._run_drain_task(
        task_id="t-drain-fix-a",
        context_id="ctx-fix-a",
        entry=entry,
        message=synthetic,
    )

    assert entry.pending_notifications == [], (
        "Invariant violated: pending_notifications must be empty "
        "after drain-spawn, found "
        f"{len(entry.pending_notifications)} item(s) still queued."
    )
    history_added = entry.history[history_before:]
    assert any(
        isinstance(m, HumanMessage) and "<remote_task_update>" in m.content
        for m in history_added
    ), (
        "Invariant violated: the drained <remote_task_update> "
        "notification must appear in entry.history, but did not. "
        f"Tail of history: {history_added}"
    )


@pytest.mark.asyncio
async def test_drain_spawn_does_not_append_empty_human_message():
    """The LLM provider must NOT see a HumanMessage with empty content.
    The drain-spawn semantically resumes the loop on the existing history
    (drained notifications); it must NOT inject a synthetic empty user
    query.
    """
    provider = _FakeProvider(
        responses=[
            AssistantMessage(
                content="ack",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            ),
        ]
    )
    store = ContextStore(max_contexts=8)
    executor = ObelixAgentExecutor(
        agent_factory=_make_agent_factory(provider),
        context_store=store,
    )

    entry = store.get_or_create("ctx-fix-b")
    entry.pending_notifications.append(
        HumanMessage(
            content=(
                "<remote_task_update><status>completed</status></remote_task_update>"
            )
        )
    )

    from a2a.types import Message, Role

    synthetic = Message(
        message_id="m-1",
        role=Role.user,
        parts=[],
        context_id="ctx-fix-b",
    )
    await executor._run_drain_task(
        task_id="t-drain-fix-b",
        context_id="ctx-fix-b",
        entry=entry,
        message=synthetic,
    )

    seen = provider.last_messages_seen
    empty_humans = [m for m in seen if isinstance(m, HumanMessage) and m.content == ""]
    assert empty_humans == [], (
        "Invariant violated: provider received "
        f"{len(empty_humans)} HumanMessage(content='') item(s). "
        "The drain-spawn must not synthesize empty user messages."
    )


@pytest.mark.asyncio
async def test_drain_spawn_passes_remote_task_update_to_llm_call():
    """The LLM provider must receive the ``<remote_task_update>`` HumanMessage
    as part of the conversation — that's the content the agent uses to
    formulate its response.
    """
    provider = _FakeProvider(
        responses=[
            AssistantMessage(
                content="processed remote update",
                tool_calls=[],
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
        ]
    )
    store = ContextStore(max_contexts=8)
    executor = ObelixAgentExecutor(
        agent_factory=_make_agent_factory(provider),
        context_store=store,
    )

    entry = store.get_or_create("ctx-fix-c")
    entry.pending_notifications.append(
        HumanMessage(
            content=(
                "<remote_task_update>"
                "<task_id>t-coord</task_id>"
                "<agent>coordinator</agent>"
                "<status>completed</status>"
                "<result>marker-RESULT-XYZ-marker</result>"
                "</remote_task_update>"
            )
        )
    )

    from a2a.types import Message, Role

    synthetic = Message(
        message_id="m-1",
        role=Role.user,
        parts=[],
        context_id="ctx-fix-c",
    )
    await executor._run_drain_task(
        task_id="t-drain-fix-c",
        context_id="ctx-fix-c",
        entry=entry,
        message=synthetic,
    )

    assert provider.invocations >= 1, (
        "Invariant violated: LLM provider was never invoked "
        "(drain-spawn did not run the agent loop)."
    )
    seen = provider.last_messages_seen
    saw_marker = any(
        hasattr(m, "content")
        and isinstance(m.content, str)
        and "marker-RESULT-XYZ-marker" in m.content
        for m in seen
    )
    assert saw_marker, (
        "Invariant violated: the drained <remote_task_update> "
        "did not reach the LLM provider's input. Messages seen: "
        f"{[type(m).__name__ for m in seen]}"
    )
