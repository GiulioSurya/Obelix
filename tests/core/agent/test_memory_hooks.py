"""Tests for memory hook tracer events."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.shared_memory import PropagationPolicy, SharedMemoryGraph
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.usage import Usage
from obelix.core.tracer.tracer import Tracer


@pytest.mark.asyncio
async def test_memory_pull_event_per_predecessor(spy_tracer_exporter):
    """Agent with N predecessors having data emits N memory.pull events."""
    tracer = Tracer(exporter=spy_tracer_exporter)

    graph = SharedMemoryGraph()
    graph.add_agent("src1")
    graph.add_agent("src2")
    graph.add_agent("current")
    graph.add_edge("src1", "current", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    graph.add_edge("src2", "current", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    await graph.publish("src1", "content A", kind="final")
    await graph.publish("src2", "content B", kind="final")

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

    agent = BaseAgent(
        system_message="test",
        provider=provider,
        tracer=tracer,
        max_iterations=3,
    )
    agent.memory_graph = graph
    agent.agent_id = "current"

    await agent.execute_query_async("hi")

    agent_spans = [s for s in spy_tracer_exporter.spans if s.span_type.value == "agent"]
    assert agent_spans
    pull_events = [e for e in agent_spans[0].events if e.name == "memory.pull"]
    assert len(pull_events) == 2
    sources = {e.attributes["from_agent"] for e in pull_events}
    assert sources == {"src1", "src2"}
    for e in pull_events:
        assert e.attributes["policy"] == PropagationPolicy.FINAL_RESPONSE_ONLY.value
        assert e.attributes["bytes"] > 0


@pytest.mark.asyncio
async def test_memory_pull_event_not_emitted_when_no_predecessors(spy_tracer_exporter):
    """Agent with no predecessors emits no memory.pull events."""
    tracer = Tracer(exporter=spy_tracer_exporter)

    graph = SharedMemoryGraph()
    graph.add_agent("current")  # standalone node, no predecessors

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

    agent = BaseAgent(
        system_message="test",
        provider=provider,
        tracer=tracer,
        max_iterations=3,
    )
    agent.memory_graph = graph
    agent.agent_id = "current"

    await agent.execute_query_async("hi")

    agent_spans = [s for s in spy_tracer_exporter.spans if s.span_type.value == "agent"]
    pull_events = [e for s in agent_spans for e in s.events if e.name == "memory.pull"]
    assert pull_events == []


@pytest.mark.asyncio
async def test_memory_pull_event_not_emitted_when_predecessor_has_no_data(
    spy_tracer_exporter,
):
    """Predecessors that haven't published data yet don't produce memory.pull events."""
    tracer = Tracer(exporter=spy_tracer_exporter)

    graph = SharedMemoryGraph()
    graph.add_agent("empty_src")
    graph.add_agent("current")
    graph.add_edge("empty_src", "current", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    # No publish() call on empty_src — last_final stays None

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

    agent = BaseAgent(
        system_message="test",
        provider=provider,
        tracer=tracer,
        max_iterations=3,
    )
    agent.memory_graph = graph
    agent.agent_id = "current"

    await agent.execute_query_async("hi")

    agent_spans = [s for s in spy_tracer_exporter.spans if s.span_type.value == "agent"]
    pull_events = [e for s in agent_spans for e in s.events if e.name == "memory.pull"]
    assert pull_events == []
