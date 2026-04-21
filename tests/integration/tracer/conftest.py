"""Shared fixtures for ``tests/integration/tracer``.

Provides ``dev_workflow_agents_with_spy``: a self-contained reproduction of the
CoordinatorAgent + Reviewer + CommitAgent + SummaryAgent pipeline from
``examples/dev_workflow_server.py`` — WITHOUT the A2A/server layer — wired with
mocked LLM providers so the scenario replays a deterministic tool-call
sequence. The returned ``spy`` exporter captures every completed span so tests
can assert the full span taxonomy (agent / sub_agent / skill / human /
assistant) and memory events.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.shared_memory import PropagationPolicy, SharedMemoryGraph
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall
from obelix.core.model.usage import Usage
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.tracer import Tracer

# ---------------------------------------------------------------------------
# Spy exporter
# ---------------------------------------------------------------------------


class _SpyExporter(NoOpExporter):
    """Capture every completed span plus trace-end statuses.

    The ``spans`` list only contains spans that have ``end_time`` populated —
    partial spans emitted at ``start_span`` time are ignored so tests can
    count them without double-counting.
    """

    def __init__(self) -> None:
        self.spans: list = []
        self.trace_end_statuses: dict[str, str] = {}

    async def export_span(self, span, service_name):  # type: ignore[override]
        if span.end_time is not None:
            self.spans.append(span)

    async def end_trace(self, trace_id, status, end_time):  # type: ignore[override]
        self.trace_end_statuses[trace_id] = (
            status.value if hasattr(status, "value") else str(status)
        )


# ---------------------------------------------------------------------------
# Mock provider builder
# ---------------------------------------------------------------------------


def _msg_text(text: str, in_tok: int = 50, out_tok: int = 25) -> AssistantMessage:
    return AssistantMessage(
        content=text,
        tool_calls=[],
        usage=Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        ),
    )


def _msg_tool_call(
    tool_name: str,
    arguments: dict | None = None,
    tool_call_id: str = "tc1",
    in_tok: int = 50,
    out_tok: int = 25,
) -> AssistantMessage:
    return AssistantMessage(
        content="",
        tool_calls=[
            ToolCall(id=tool_call_id, name=tool_name, arguments=arguments or {})
        ],
        usage=Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        ),
    )


def _make_provider(responses: list[AssistantMessage]) -> MagicMock:
    """Build a MagicMock provider whose .invoke replays ``responses`` in order.

    ``invoke_stream`` raises NotImplementedError — the BaseAgent loop transparently
    falls back to ``invoke`` when that happens, so the tests do not need to
    script streaming shape.
    """
    p = MagicMock()
    p.provider_type = "mock"
    p.model_id = "mock-model"
    p.invoke = AsyncMock(side_effect=list(responses))

    def _no_stream(*a, **kw):
        raise NotImplementedError

    p.invoke_stream = MagicMock(side_effect=_no_stream)
    return p


# ---------------------------------------------------------------------------
# Agent classes (mirror of examples/dev_workflow_server.py, provider injected)
# ---------------------------------------------------------------------------


class _ReviewerAgent(BaseAgent):
    """Mirror of examples/dev_workflow_server.ReviewerAgent — provider injected."""

    def __init__(self, *, provider, skills_config, **kwargs):
        super().__init__(
            system_message=(
                "You are a senior code reviewer with a security background. "
                "Invoke 'code-review' and 'security-check' skills."
            ),
            provider=provider,
            skills_config=skills_config,
            **kwargs,
        )


class _CommitAgent(BaseAgent):
    """Mirror of examples/dev_workflow_server.CommitAgent — provider injected."""

    def __init__(self, *, provider, skills_config, **kwargs):
        super().__init__(
            system_message=(
                "You are a commit message specialist. Invoke 'commit-writer' skill."
            ),
            provider=provider,
            skills_config=skills_config,
            **kwargs,
        )


class _SummaryAgent(BaseAgent):
    """Mirror of examples/dev_workflow_server.SummaryAgent — provider injected."""

    def __init__(self, *, provider, **kwargs):
        super().__init__(
            system_message=(
                "You are a technical writer producing a developer-ready report."
            ),
            provider=provider,
            **kwargs,
        )


class _CoordinatorAgent(BaseAgent):
    """Mirror of examples/dev_workflow_server.CoordinatorAgent — provider injected."""

    def __init__(self, *, provider, **kwargs):
        super().__init__(
            system_message=(
                "You are the Dev Workflow Coordinator. "
                "Orchestrate reviewer -> commit_writer -> summary."
            ),
            provider=provider,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Skill fixture wiring
# ---------------------------------------------------------------------------


def _write_skill(skills_dir, name: str, context: str, description: str) -> None:
    """Create a minimal SKILL.md on disk for the given skill name/context.

    The body is intentionally trivial — the fork path requires only the body
    to become the inner agent's system message, and inline path returns it as
    the tool result string.
    """
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\ndescription: {description}\ncontext: {context}\n---\n"
        f"You are executing the '{name}' skill. Produce a concise report.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixture: full dev_workflow with spy tracer
# ---------------------------------------------------------------------------


@pytest.fixture
def dev_workflow_agents_with_spy(
    tmp_path,
) -> tuple[BaseAgent, _SpyExporter]:
    """Return ``(coordinator_agent, spy_exporter)``.

    Wires:
    - SharedMemoryGraph with reviewer -> commit_writer, reviewer -> summary,
      commit_writer -> summary edges (FINAL_RESPONSE_ONLY policy).
    - 3 skills on disk (code-review:fork, security-check:inline,
      commit-writer:fork).
    - 4 agents (Reviewer / Commit / Summary / Coordinator) created via
      AgentFactory with mocked providers that drive a predictable sequence
      of tool calls:
        * Coordinator calls reviewer, then commit_writer, then summary
        * Reviewer calls code-review (fork), then security-check (inline),
          then final text
        * CommitAgent calls commit-writer (fork), then final text
        * SummaryAgent emits final text directly

    All agents share the same tracer and memory graph. Calling
    ``await coordinator.execute_query_async(...)`` triggers the full pipeline
    and the spy accumulates every completed span.
    """
    # --- Skills on disk ---
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    _write_skill(skills_dir, "code-review", "fork", "Deep code review (forked)")
    _write_skill(skills_dir, "security-check", "inline", "Inline security scan")
    _write_skill(skills_dir, "commit-writer", "fork", "Commit message writer (forked)")

    # --- Spy tracer ---
    spy = _SpyExporter()
    tracer = Tracer(exporter=spy)

    # --- Shared memory graph (mirrors dev_workflow_server.create_factory) ---
    graph = SharedMemoryGraph()
    graph.add_agent("reviewer")
    graph.add_agent("commit_writer")
    graph.add_agent("summary")
    graph.add_edge(
        "reviewer", "commit_writer", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )
    graph.add_edge("reviewer", "summary", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    graph.add_edge(
        "commit_writer", "summary", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )

    # --- Mocked providers ---
    # Reviewer: calls code-review (fork), then security-check (inline), then text.
    # With fork, the INNER agent also calls invoke() once to produce its final
    # text (a concrete AssistantMessage with no tool_calls).
    reviewer_provider = _make_provider(
        [
            # Iteration 1: invoke code-review (fork)
            _msg_tool_call(
                "Skill",
                {"name": "code-review", "args": ""},
                tool_call_id="rev-sk-1",
            ),
            # Fork inner agent call (code-review): return final text immediately
            _msg_text("Code review: LGTM, minor style nit on line 42."),
            # Iteration 2: invoke security-check (inline)
            _msg_tool_call(
                "Skill",
                {"name": "security-check", "args": ""},
                tool_call_id="rev-sk-2",
            ),
            # Iteration 3: final text response (closes reviewer loop)
            _msg_text("Review complete: no security issues; one style nit."),
        ]
    )

    # Commit writer: calls commit-writer (fork), then final text.
    commit_provider = _make_provider(
        [
            # Iteration 1: invoke commit-writer (fork)
            _msg_tool_call(
                "Skill",
                {"name": "commit-writer", "args": ""},
                tool_call_id="cmt-sk-1",
            ),
            # Fork inner agent call (commit-writer): return final text
            _msg_text("feat: add widget — addresses style nit from review"),
            # Iteration 2: final text response (closes commit_writer loop)
            _msg_text("feat: add widget — addresses style nit from review"),
        ]
    )

    # Summary: just emits final text (no sub-tools). One invoke only.
    summary_provider = _make_provider(
        [_msg_text("## Code Review\nLGTM\n## Commit\nfeat: add widget\n")]
    )

    # Coordinator: calls reviewer, commit_writer, summary, then emits final text.
    coordinator_provider = _make_provider(
        [
            _msg_tool_call(
                "reviewer",
                {"query": "Review the staged diff"},
                tool_call_id="coord-rev",
            ),
            _msg_tool_call(
                "commit_writer",
                {"query": "Write the commit message"},
                tool_call_id="coord-cmt",
            ),
            _msg_tool_call(
                "summary",
                {"query": "Assemble the report"},
                tool_call_id="coord-sum",
            ),
            _msg_text("Done. Report ready."),
        ]
    )

    # --- Factory wiring ---
    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.with_memory_graph(graph)

    factory.register(
        name="reviewer",
        cls=_ReviewerAgent,
        subagent_description="Reviews staged git changes.",
        stateless=True,
        defaults={
            "provider": reviewer_provider,
            "skills_config": str(skills_dir),
        },
    )
    factory.register(
        name="commit_writer",
        cls=_CommitAgent,
        subagent_description="Writes a conventional commit message.",
        stateless=True,
        defaults={
            "provider": commit_provider,
            "skills_config": str(skills_dir),
        },
    )
    factory.register(
        name="summary",
        cls=_SummaryAgent,
        subagent_description="Assembles the final developer report.",
        stateless=True,
        defaults={"provider": summary_provider},
    )
    factory.register(
        name="coordinator",
        cls=_CoordinatorAgent,
        defaults={"provider": coordinator_provider},
    )

    coordinator = factory.create(
        "coordinator",
        subagents=["reviewer", "commit_writer", "summary"],
    )

    return coordinator, spy
