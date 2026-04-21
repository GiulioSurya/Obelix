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


@pytest.fixture
def make_agent_with_skill_and_tracer(tmp_path):
    """Factory producing a ``(BaseAgent, _SpyTracerExporter)`` pair wired with a skill.

    Creates a temporary filesystem skill with the requested ``mode`` (context)
    and returns an agent whose mocked provider invokes the SkillTool on its
    first iteration, then returns plain text on the second. Only filesystem
    source is supported by this fixture (MCP-backed skills need an MCPManager).
    """

    def _factory(
        skill_name: str,
        mode: str = "inline",
        source: str = "filesystem",
    ) -> tuple[BaseAgent, _SpyTracerExporter]:
        if source != "filesystem":
            raise NotImplementedError(
                "Only source='filesystem' is supported by this fixture."
            )
        # Layout: tmp_path / <skill_name> / SKILL.md
        skill_dir = tmp_path / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            f"---\ndescription: Test skill '{skill_name}'\ncontext: {mode}\n---\n"
            f"Skill body for '{skill_name}'.\n",
            encoding="utf-8",
        )

        exporter = _SpyTracerExporter()
        tracer = Tracer(exporter=exporter)

        # Response 1: agent calls the Skill tool with name=skill_name.
        # Response 2: agent returns plain text, ending the loop.
        response_1 = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="skill-call-1",
                    name="Skill",
                    arguments={"name": skill_name, "args": ""},
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        response_2 = _mock_assistant_text("done")

        # If mode=fork, the inner sub-agent also calls provider.invoke exactly
        # once to close its own loop. Feed a plain-text response for it.
        invoke_side_effect: list[AssistantMessage]
        if mode == "fork":
            fork_final = _mock_assistant_text("fork-finished")
            invoke_side_effect = [response_1, fork_final, response_2]
        else:
            invoke_side_effect = [response_1, response_2]

        provider = MagicMock()
        provider.provider_type = "mock"
        provider.model_id = "mock-model"
        provider.invoke = AsyncMock(side_effect=invoke_side_effect)

        def _no_stream(*a, **kw):
            raise NotImplementedError

        provider.invoke_stream = MagicMock(side_effect=_no_stream)

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tracer=tracer,
            max_iterations=5,
            skills_config=str(tmp_path),
        )
        return agent, exporter

    return _factory


@pytest.fixture
def make_agent_with_sub_agent_and_tracer(spy_tracer_exporter):
    """Factory producing a ``(BaseAgent, _SpyTracerExporter)`` pair with a sub-agent.

    The parent agent's mocked provider invokes the registered sub-agent on the
    first iteration (via a tool call with the sub-agent's registered name),
    then returns plain text on the second iteration to close its loop. The
    child agent's mocked provider returns a single plain-text response so its
    own loop closes on the first iteration.

    Both agents share the same ``Tracer``, so completed spans from both
    surface on the same exporter. The child is registered via
    :meth:`BaseAgent.register_agent` with ``stateless=True`` to match the
    most common real-world usage (parallel-safe sub-agents).
    """

    def _factory(sub_agent_name: str) -> tuple[BaseAgent, _SpyTracerExporter]:
        exporter = spy_tracer_exporter
        tracer = Tracer(exporter=exporter)

        # Child agent: a single plain-text response closes its loop immediately.
        child_provider = MagicMock()
        child_provider.provider_type = "mock"
        child_provider.model_id = "mock-child-model"
        child_provider.invoke = AsyncMock(
            side_effect=[_mock_assistant_text("child response")]
        )

        def _child_no_stream(*a, **kw):
            raise NotImplementedError

        child_provider.invoke_stream = MagicMock(side_effect=_child_no_stream)

        child = BaseAgent(
            system_message="child system",
            provider=child_provider,
            tracer=tracer,
            max_iterations=3,
        )

        # Parent agent: first response invokes the sub-agent by its registered
        # name, second response is plain text that ends the parent loop.
        response_1 = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc-sub",
                    name=sub_agent_name,
                    arguments={"query": "go"},
                )
            ],
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        response_2 = _mock_assistant_text("done")

        parent_provider = MagicMock()
        parent_provider.provider_type = "mock"
        parent_provider.model_id = "mock-parent-model"
        parent_provider.invoke = AsyncMock(side_effect=[response_1, response_2])

        def _parent_no_stream(*a, **kw):
            raise NotImplementedError

        parent_provider.invoke_stream = MagicMock(side_effect=_parent_no_stream)

        parent = BaseAgent(
            system_message="parent system",
            provider=parent_provider,
            tracer=tracer,
            max_iterations=5,
        )
        parent.register_agent(
            child,
            name=sub_agent_name,
            description="child sub-agent",
            stateless=True,
        )
        return parent, exporter

    return _factory


@pytest.fixture
def make_agent_with_regular_tool_and_tracer():
    """Factory producing a ``(BaseAgent, _SpyTracerExporter)`` pair with a regular tool.

    The provider's first response invokes a tool by ``tool_name`` with empty
    arguments; its second response is plain text that closes the loop.
    """

    def _factory(tool_name: str) -> tuple[BaseAgent, _SpyTracerExporter]:
        exporter = _SpyTracerExporter()
        tracer = Tracer(exporter=exporter)

        response_1 = _mock_assistant_with_tool_call(tool_name=tool_name)
        response_2 = _mock_assistant_text("done")

        provider = MagicMock()
        provider.provider_type = "mock"
        provider.model_id = "mock-model"
        provider.invoke = AsyncMock(side_effect=[response_1, response_2])

        def _no_stream(*a, **kw):
            raise NotImplementedError

        provider.invoke_stream = MagicMock(side_effect=_no_stream)

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tracer=tracer,
            max_iterations=5,
        )
        agent.register_tool(_make_inline_tool(tool_name, {"ok": True}))
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
