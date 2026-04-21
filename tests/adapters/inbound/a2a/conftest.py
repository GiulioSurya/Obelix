"""Shared fixtures for ``tests/adapters/inbound/a2a``.

Provides the ``executor_with_tracer`` fixture used by the tracer
instrumentation tests. It wires an ``ObelixAgentExecutor`` to a minimal
``BaseAgent`` factory and a spy tracer that captures every completed span.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.usage import Usage
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.tracer import Tracer


class _ExecutorSpyExporter(NoOpExporter):
    """Capture every completed span for inspection by tests."""

    def __init__(self) -> None:
        self.spans: list = []

    async def export_span(self, span, service_name):  # type: ignore[override]
        # Only record completed spans (end_time is populated by Tracer.end_span).
        if span.end_time is not None:
            self.spans.append(span)


@dataclass
class _FakeRequestContext:
    """Minimal stand-in for a2a ``RequestContext``.

    Matches the shape consumed by ``ObelixAgentExecutor.execute`` — task_id,
    context_id, and a ``message`` with a single ``TextPart``.
    """

    task_id: str = "task-tracing-001"
    context_id: str | None = "ctx-tracing-001"
    text: str = "hello"

    def __post_init__(self) -> None:
        from a2a.types import Message, Part, Role, TextPart

        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=self.text))],
            message_id="msg-tracing-001",
        )

    def get_user_input(self) -> str | None:
        return self.text


class _FakeEventQueue:
    """Captures enqueued events so tests can inspect them if needed."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def executor_with_tracer():
    """Return ``(send_message, spy_exporter)``.

    Calling ``await send_message(text)`` drives ``ObelixAgentExecutor.execute``
    through a fake ``RequestContext`` + ``EventQueue``. The agent's mocked
    provider returns a plain assistant message so the agent loop closes on
    the first iteration, producing a clean span tree.
    """
    spy = _ExecutorSpyExporter()
    tracer = Tracer(exporter=spy)

    # Single-turn mock provider: one plain-text response closes the loop.
    provider = MagicMock()
    provider.provider_type = "mock"
    provider.model_id = "mock-model"
    provider.invoke = AsyncMock(
        side_effect=[
            AssistantMessage(
                content="done",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        ]
    )

    def _no_stream(*a, **kw):
        raise NotImplementedError

    provider.invoke_stream = MagicMock(side_effect=_no_stream)

    def agent_factory() -> BaseAgent:
        return BaseAgent(
            system_message="system",
            provider=provider,
            tracer=tracer,
            max_iterations=3,
        )

    executor = ObelixAgentExecutor(agent_factory, tracer=tracer)

    async def send_message(text: str) -> None:
        ctx = _FakeRequestContext(text=text)
        queue = _FakeEventQueue()
        await executor.execute(ctx, queue)

    return send_message, spy


@dataclass
class _FakeResumeContext:
    """Minimal stand-in for a2a ``RequestContext`` carrying a DataPart (resume).

    The A2A executor uses ``context_id`` to match the original task and
    ``message.parts`` to extract the deferred tool response. On resume, the
    message carries a DataPart with the structured answer, not a TextPart.
    """

    task_id: str = "task-tracing-resume"
    context_id: str | None = "ctx-deferred-001"
    data: dict | None = None

    def __post_init__(self) -> None:
        from a2a.types import DataPart, Message, Part, Role

        self.message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data=self.data or {"answer": "resumed"}))],
            message_id=f"msg-resume-{uuid.uuid4()}",
        )

    def get_user_input(self) -> str | None:
        return None


@pytest.fixture
def executor_with_deferred_tool():
    """Return ``(send_message, resume, spy_exporter)``.

    First call to ``send_message`` drives the executor with a mock agent whose
    first response yields ``deferred_tool_calls`` — the executor emits
    ``input_required`` and opens the ``deferred_wait`` span. The call to
    ``resume`` then delivers a DataPart response on the same ``context_id``,
    triggering the resume path which should close the ``deferred_wait`` span.

    The mocked agent carries an ``is_deferred=True`` tool (``ask_user``) so
    the deferred_tool_calls event matches the expected shape.
    """
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage
    from obelix.core.model.tool_message import ToolCall

    spy = _ExecutorSpyExporter()
    tracer = Tracer(exporter=spy)

    created: list[MagicMock] = []
    context_id = "ctx-deferred-001"

    # Build a minimal deferred tool descriptor (matches the Tool protocol
    # attrs the executor reads: ``tool_name``, ``is_deferred``, etc.).
    class _DeferredToolStub:
        tool_name = "ask_user"
        tool_description = "Ask the user a question (deferred)."
        is_deferred = True

        async def execute(self, tool_call):  # pragma: no cover - not used
            return None

        def create_schema(self):  # pragma: no cover - not used
            from obelix.core.model.tool_message import MCPToolSchema

            return MCPToolSchema(
                name=self.tool_name,
                description=self.tool_description,
                inputSchema={"type": "object", "properties": {}},
            )

    def factory() -> MagicMock:
        agent = MagicMock()
        agent.system_message = SystemMessage(content="You are a test agent.")
        agent.conversation_history = [agent.system_message]
        agent.registered_tools = [_DeferredToolStub()]
        agent._tracer = tracer

        call_count = len(created)
        if call_count == 0:
            # First invocation: yield deferred tool calls
            async def first_stream(query):
                yield StreamEvent(
                    deferred_tool_calls=[
                        ToolCall(
                            id="tc-deferred-1",
                            name="ask_user",
                            arguments={"question": "something?"},
                        )
                    ],
                    is_final=True,
                )

            agent.execute_query_stream = MagicMock(side_effect=first_stream)
        else:
            # Resume invocation: yield a final assistant response
            async def resume_stream():
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test_agent",
                        content="resumed done",
                    ),
                )

            agent.resume_after_deferred = MagicMock(side_effect=resume_stream)

        created.append(agent)
        return agent

    executor = ObelixAgentExecutor(factory, tracer=tracer)

    async def send_message(text: str) -> None:
        ctx = _FakeRequestContext(
            context_id=context_id,
            text=text,
            task_id="task-tracing-deferred",
        )
        queue = _FakeEventQueue()
        await executor.execute(ctx, queue)

    async def resume(data: dict) -> None:
        ctx = _FakeResumeContext(
            context_id=context_id,
            data=data,
            task_id="task-tracing-resume",
        )
        queue = _FakeEventQueue()
        await executor.execute(ctx, queue)

    return send_message, resume, spy
