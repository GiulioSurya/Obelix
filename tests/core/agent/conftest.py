"""Shared fixtures for ``tests/core/agent``.

Provides a spy-tracer factory that wires a ``BaseAgent`` to a ``Tracer`` whose
exporter captures every completed span. Used by the Task 10 tests that verify
the per-call LLM spans have been replaced by aggregated ``llm_usage`` on the
enclosing ``agent`` span.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall, ToolResult, ToolStatus
from obelix.core.model.usage import Usage
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.tracer import Tracer


class _SpyTracerExporter(NoOpExporter):
    """Capture every completed span for inspection by tests."""

    def __init__(self) -> None:
        self.spans: list = []

    async def export_span(self, span, service_name):  # type: ignore[override]
        # Only record completed spans (end_time is populated by Tracer.end_span).
        if span.end_time is not None:
            self.spans.append(span)


def _mock_assistant_text(
    text: str, usage_in: int = 100, usage_out: int = 50
) -> AssistantMessage:
    """An ``AssistantMessage`` with text content and a populated ``usage``."""
    return AssistantMessage(
        content=text,
        tool_calls=[],
        usage=Usage(
            input_tokens=usage_in,
            output_tokens=usage_out,
            total_tokens=usage_in + usage_out,
        ),
    )


def _mock_assistant_with_tool_call(
    tool_name: str = "dummy",
    usage_in: int = 100,
    usage_out: int = 50,
) -> AssistantMessage:
    """An ``AssistantMessage`` that requests a single tool call."""
    return AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="tc1", name=tool_name, arguments={})],
        usage=Usage(
            input_tokens=usage_in,
            output_tokens=usage_out,
            total_tokens=usage_in + usage_out,
        ),
    )


@pytest.fixture
def spy_tracer_exporter() -> _SpyTracerExporter:
    """Expose the spy exporter class / instance for direct use in tests."""
    return _SpyTracerExporter()


@pytest.fixture
def make_agent_with_spy_tracer():
    """Factory producing a ``(BaseAgent, _SpyTracerExporter)`` pair.

    Parameters on the returned callable:
        responses: list of ``AssistantMessage`` returned in order by ``provider.invoke``.
        tool_results: optional ``dict[str, Any]`` mapping tool name -> result value.
            For each entry a minimal tool that returns the mapped result is
            registered on the agent. The tool name must match the name used in
            any ``tool_calls`` produced by ``responses``.
    """

    def _factory(
        responses: list[AssistantMessage],
        tool_results: dict | None = None,
    ) -> tuple[BaseAgent, _SpyTracerExporter]:
        exporter = _SpyTracerExporter()
        tracer = Tracer(exporter=exporter)

        provider = MagicMock()
        provider.provider_type = "mock"
        provider.model_id = "mock-model"
        provider.invoke = AsyncMock(side_effect=list(responses))

        def _no_stream(*a, **kw):
            raise NotImplementedError

        provider.invoke_stream = MagicMock(side_effect=_no_stream)

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tracer=tracer,
            max_iterations=5,
        )

        if tool_results:
            # Register one inline tool per entry. The tool returns the mapped
            # result payload wrapped in a ToolResult (SUCCESS).
            for tool_name, result_value in tool_results.items():
                inline_tool = _make_inline_tool(tool_name, result_value)
                agent.register_tool(inline_tool)

        return agent, exporter

    return _factory


def _make_inline_tool(tool_name: str, result_value):
    """Build a minimal object satisfying the ``Tool`` protocol.

    Avoids the @tool decorator to keep the fixture dependency-free and to
    produce a tool that always returns ``result_value`` without argument
    validation. BaseAgent only inspects ``tool_name``, ``is_deferred`` and
    ``execute()`` when dispatching.
    """

    class _InlineTool:
        def __init__(self) -> None:
            self.tool_name = tool_name
            self.tool_description = f"inline mock for {tool_name}"
            self.is_deferred = False

        async def execute(self, tool_call: ToolCall) -> ToolResult:
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                result=result_value,
                status=ToolStatus.SUCCESS,
            )

        def create_schema(self):  # pragma: no cover - unused by the tests
            from obelix.core.model.tool_message import MCPToolSchema

            return MCPToolSchema(
                name=tool_name,
                description=f"inline mock for {tool_name}",
                inputSchema={"type": "object", "properties": {}, "required": []},
            )

    return _InlineTool()
