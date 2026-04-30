"""Shared fixtures for ``tests/integration/tracer``.

Provides:

* ``dev_workflow_agents_with_spy``: a self-contained reproduction of the
  CoordinatorAgent + Reviewer + CommitAgent + SummaryAgent pipeline from
  ``examples/dev_workflow_server.py`` — WITHOUT the A2A/server layer — wired
  with mocked LLM providers so the scenario replays a deterministic tool-call
  sequence. The returned ``spy`` exporter captures every completed span so
  tests can assert the full span taxonomy (agent / sub_agent / skill / human /
  assistant) and memory events.
* ``deferred_scenario_with_spy``: end-to-end reproduction of the A2A
  deferred-tool suspend/resume flow — mirror of
  ``executor_with_deferred_tool`` in ``tests/adapters/inbound/a2a/conftest.py``
  scoped to the tracer integration suite. Exercises the full ``a2a_task`` +
  ``deferred_wait`` lifecycle with mocked agents so tests can inspect the
  resulting span tree.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Deferred-tool scenario (mirror of executor_with_deferred_tool)
# ---------------------------------------------------------------------------


@dataclass
class _DeferredRequestContext:
    """Minimal stand-in for a2a ``RequestContext`` carrying an initial TextPart.

    Matches the shape consumed by ``ObelixAgentExecutor.execute`` — task_id,
    context_id, and a ``message`` with a single ``TextPart``.
    """

    task_id: str = "task-deferred-int-001"
    context_id: str | None = "ctx-deferred-int-001"
    text: str = "hello"

    def __post_init__(self) -> None:
        from a2a.types import Message, Part, Role, TextPart

        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=self.text))],
            message_id=f"msg-deferred-int-{uuid.uuid4()}",
        )

    def get_user_input(self) -> str | None:
        return self.text


@dataclass
class _DeferredResumeContext:
    """Minimal stand-in for a2a ``RequestContext`` carrying a DataPart (resume).

    On resume, the message carries a DataPart with the structured answer, not
    a TextPart. The executor uses ``context_id`` to match the original task.
    """

    task_id: str = "task-deferred-int-resume"
    context_id: str | None = "ctx-deferred-int-001"
    data: dict | None = None

    def __post_init__(self) -> None:
        from a2a.types import DataPart, Message, Part, Role

        self.message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data=self.data or {"answer": "resumed"}))],
            message_id=f"msg-resume-int-{uuid.uuid4()}",
        )

    def get_user_input(self) -> str | None:
        return None


class _DeferredEventQueue:
    """Captures enqueued events so tests can inspect them if needed."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def deferred_scenario_with_spy():
    """Return ``(send_message, resume, spy_exporter)``.

    Mirror of ``executor_with_deferred_tool`` from
    ``tests/adapters/inbound/a2a/conftest.py`` scoped to the tracer integration
    suite. First call to ``send_message`` drives the executor with a mock
    agent whose first response yields ``deferred_tool_calls`` — the executor
    emits ``input_required`` and opens the ``deferred_wait`` span. The call
    to ``resume`` then delivers a DataPart response on the same
    ``context_id``, triggering the resume path which closes the
    ``deferred_wait`` span.

    The mocked agent carries an ``is_deferred=True`` tool (``ask_user``) so
    the deferred_tool_calls event matches the expected shape.
    """
    from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage
    from obelix.core.model.tool_message import MCPToolSchema

    spy = _SpyExporter()
    tracer = Tracer(exporter=spy)

    created: list[MagicMock] = []
    context_id = "ctx-deferred-int-001"

    class _DeferredToolStub:
        tool_name = "ask_user"
        tool_description = "Ask the user a question (deferred)."
        is_deferred = True

        async def execute(self, tool_call):  # pragma: no cover - not used
            return None

        def create_schema(self):  # pragma: no cover - not used
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
                            id="tc-deferred-int-1",
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
        ctx = _DeferredRequestContext(
            context_id=context_id,
            text=text,
            task_id="task-deferred-int-send",
        )
        queue = _DeferredEventQueue()
        await executor.execute(ctx, queue)

    async def resume(data: dict) -> None:
        ctx = _DeferredResumeContext(
            context_id=context_id,
            data=data,
            task_id="task-deferred-int-resume",
        )
        queue = _DeferredEventQueue()
        await executor.execute(ctx, queue)

    return send_message, resume, spy


# ---------------------------------------------------------------------------
# Rejection scenario (real BaseAgent + A2A executor with a rejecting hook)
# ---------------------------------------------------------------------------


@dataclass
class _RejectionRequestContext:
    """Minimal stand-in for a2a ``RequestContext`` — rejection scenario.

    Mirror of ``_FakeRequestContext`` in ``tests/adapters/inbound/a2a/conftest.py``
    scoped to the tracer integration suite. Carries a single TextPart user
    message; the task_id/context_id drive the A2A executor bookkeeping.
    """

    task_id: str = "task-reject-int-001"
    context_id: str | None = "ctx-reject-int-001"
    text: str = "please review"

    def __post_init__(self) -> None:
        from a2a.types import Message, Part, Role, TextPart

        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=self.text))],
            message_id=f"msg-reject-int-{uuid.uuid4()}",
        )

    def get_user_input(self) -> str | None:
        return self.text


class _RejectionEventQueue:
    """Captures enqueued events so tests can inspect them if needed."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def rejection_scenario_with_spy():
    """Return ``(send_message, spy_exporter)``.

    Wires a real ``BaseAgent`` behind ``ObelixAgentExecutor`` with a
    ``BEFORE_LLM_CALL`` hook that rejects the task via ``.reject(reason)``.
    The provider's ``invoke`` raises if ever called — the rejection fires
    before the agent enters its LLM call, so no LLM call is expected.

    Asserting through the A2A executor (rather than invoking ``BaseAgent``
    directly) ensures the ``a2a_task`` root span is opened and the rejection
    propagates into ``TaskState.rejected`` + ``a2a.state_change`` + trace
    status, matching the shape seen by real A2A clients.
    """
    from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
    from obelix.core.agent.hooks import AgentEvent

    spy = _SpyExporter()
    tracer = Tracer(exporter=spy)

    # Provider — must never be called. ``invoke`` raises so a silent failure
    # surfaces loudly if the rejection path regresses.
    provider = MagicMock()
    provider.provider_type = "mock"
    provider.model_id = "mock-model"

    async def _boom(*a, **kw):
        raise AssertionError("provider.invoke must not be called after REJECT")

    provider.invoke = AsyncMock(side_effect=_boom)

    def _no_stream(*a, **kw):
        raise NotImplementedError

    provider.invoke_stream = MagicMock(side_effect=_no_stream)

    def agent_factory() -> BaseAgent:
        # BaseAgent coerces the string into a SystemMessage internally —
        # passing a plain str matches the shape used by
        # ``executor_with_rejecting_agent`` in the a2a conftest.
        agent = BaseAgent(
            system_message="test system",
            provider=provider,
            tracer=tracer,
            max_iterations=3,
        )
        # Register the rejecting hook — BEFORE_LLM_CALL fires before the
        # provider is invoked so the loop terminates with TaskRejectedError.
        agent.on(AgentEvent.BEFORE_LLM_CALL).reject("No input provided")
        return agent

    executor = ObelixAgentExecutor(agent_factory, tracer=tracer)

    context_id = f"ctx-reject-int-{uuid.uuid4()}"

    async def send_message(text: str) -> None:
        ctx = _RejectionRequestContext(
            context_id=context_id,
            text=text,
            task_id=f"task-reject-int-{uuid.uuid4()}",
        )
        queue = _RejectionEventQueue()
        await executor.execute(ctx, queue)

    return send_message, spy
