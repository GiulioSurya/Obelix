"""Shared fixtures for ``tests/adapters/inbound/a2a``.

Provides the ``executor_with_tracer`` fixture used by the tracer
instrumentation tests. It wires an ``ObelixAgentExecutor`` to a minimal
``BaseAgent`` factory and a spy tracer that captures every completed span.
"""

from __future__ import annotations

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
