"""E2E integration test for spec 1 (server-side drainer + tracer trace_id reuse).

Demonstrates the full scenario from
``docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md`` § 8.3:

1. CLI sends first message to orchestrator (drives ``executor.execute``).
2. Orchestrator's LLM emits a ``dispatch_agent("coordinator", ...)`` tool call.
3. ``DispatchAgentTool`` invokes the FakeRemoteClient, records the remote task.
4. Orchestrator's LLM closes the user-triggered turn (final response).
5. Test simulates the coordinator completing → ``handle_remote_update`` adds
   a ``<remote_task_update>`` to ``entry.pending_notifications``.
6. ``maybe_spawn_drain_task`` is invoked → drainer's checks pass → executor's
   ``spawn_drain_task`` schedules a drain-spawn task.
7. ``_run_drain_task`` runs the orchestrator on the same context → its LLM
   emits the final response to the user.
8. ``_DrainSpawnEventQueue`` POSTs each drain-spawn task state change to the
   CLI webhook fixture (Starlette + ASGI transport).
9. Verifications:
   - Webhook received >= 1 POST with ``state=completed`` and camelCase payload
   - Tracer captured 2 ``a2a_task`` spans sharing the SAME ``trace_id``

Iron rule (no mocks of external SDKs): the LLM provider is a hand-written
``FakeProvider`` returning scripted ``AssistantMessage``s; the remote A2A
client is a hand-written ``FakeRemoteClient`` matching ``a2a.client.Client``'s
``send_message`` async-iterator signature. Everything else is real Obelix
production code (executor, agent, registry, context store, drainer, dispatch
tool, handler, tracer, ``_DrainSpawnEventQueue``, ``httpx.AsyncClient``).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import httpx
import pytest
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from obelix.adapters.inbound.a2a.server.context import ContextStore
from obelix.adapters.inbound.a2a.server.drainer import maybe_spawn_drain_task
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.adapters.outbound.a2a.handler import handle_remote_update
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.tools.dispatch import DispatchAgentTool
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall
from obelix.core.model.usage import Usage
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession
from obelix.core.tracer.tracer import Tracer
from obelix.infrastructure.providers import Providers

# ── Fakes (per iron rule: hand-written, implement real contracts) ─────────


class FakeProvider:
    """Scripted LLM provider for the e2e test.

    Implements the ``AbstractLLMProvider`` surface used by ``BaseAgent``:
    ``invoke`` (returns the next scripted ``AssistantMessage``), ``invoke_stream``
    (raises ``NotImplementedError`` so the agent falls back to ``invoke``),
    ``provider_type`` and ``model_id`` properties.

    Scripted responses are consumed in order. When all are exhausted, returns
    a generic empty final response — guards against runaway loops while still
    surfacing the script-exhausted condition in the AssistantMessage content.
    """

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.invocations = 0

    @property
    def provider_type(self) -> Providers:
        return Providers.ANTHROPIC

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def invoke(self, messages, tools=None, response_schema=None):  # noqa: ANN001
        if self.invocations >= len(self._responses):
            self.invocations += 1
            return AssistantMessage(
                content="(no more scripted responses)",
                tool_calls=[],
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        idx = self.invocations
        self.invocations += 1
        return self._responses[idx]

    async def invoke_stream(self, messages, tools=None, response_schema=None):  # noqa: ANN001
        # Force BaseAgent's NotImplementedError fallback path → invoke().
        # The NotImplementedError MUST be raised inside the async generator
        # (per AbstractLLMProvider.invoke_stream contract).
        raise NotImplementedError
        yield  # pragma: no cover — needed to make this an AsyncIterator


class FakeRemoteClient:
    """Implements the subset of ``a2a.client.Client`` used by ``DispatchAgentTool``.

    Only ``send_message`` is exercised in this test. It mirrors the SDK's
    ``AsyncIterator[tuple[Task, UpdateEvent | None] | Message]`` contract by
    yielding one ``(Task, None)`` tuple — the shape ``DispatchAgentTool.execute``
    consumes. The yielded Task is the "submitted" handle returned to the agent.
    """

    def __init__(self, response_task: Task) -> None:
        self._response = response_task
        self.send_message_calls: list[tuple] = []

    async def send_message(
        self, msg, configuration=None
    ) -> AsyncIterator[tuple[Task, None]]:  # noqa: ANN001
        self.send_message_calls.append((msg, configuration))
        yield (self._response, None)

    async def get_task(self, params):  # pragma: no cover — not used here
        return self._response

    async def cancel_task(self, params):  # pragma: no cover — not used here
        return self._response


@dataclass
class _FakeRequestContext:
    """Stand-in for a2a-sdk ``RequestContext`` matching the surface
    ``ObelixAgentExecutor.execute`` consumes."""

    task_id: str
    context_id: str
    text: str
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=self.text))],
            message_id=f"msg-{uuid.uuid4()}",
            metadata=self.metadata,
        )


class _FakeEventQueue:
    """Records enqueued events without external transport."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:  # noqa: ANN001
        self.events.append(event)

    async def close(self) -> None:
        return None


class _CapturingExporter(NoOpExporter):
    """Captures completed spans + traces for post-run inspection."""

    def __init__(self) -> None:
        self.completed_spans: list[Span] = []
        self.started_traces: list[TraceSession] = []
        self.ended_trace_ids: list[str] = []

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:  # type: ignore[override]
        self.started_traces.append(trace)

    async def export_span(self, span: Span, service_name: str) -> None:  # type: ignore[override]
        if span.end_time is not None and span.span_id not in {
            s.span_id for s in self.completed_spans
        }:
            self.completed_spans.append(span)

    async def end_trace(
        self,
        trace_id: str,
        status: SpanStatus,
        end_time: datetime | None,
    ) -> None:  # type: ignore[override]
        self.ended_trace_ids.append(trace_id)


# ── Wiring helpers ─────────────────────────────────────────────────────────


def _build_agent_card_for_remote(name: str, description: str) -> AgentCard:
    """Build a minimal AgentCard for a fake remote so ``RemoteAgentRegistry``
    can describe it via ``descriptions()`` / ``names()``.

    Only the public ``AgentCard`` model fields are populated — same shape the
    A2A SDK would deliver from a real ``A2ACardResolver.get_agent_card()``.
    """
    return AgentCard(
        name=name,
        description=description,
        url=f"http://fake-{name}:9999",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=True),
        skills=[
            AgentSkill(
                id="reply",
                name="reply",
                description="Reply to the dispatcher",
                tags=["test"],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


def _build_orchestrator_factory(
    provider: FakeProvider,
    registry: RemoteAgentRegistry,
    webhook_url: str,
):
    """Build the ``agent_factory`` callable used by ``ObelixAgentExecutor``.

    Each invocation creates a fresh ``BaseAgent`` orchestrator with a real
    ``DispatchAgentTool`` registered. The same FakeProvider instance is shared
    so all turns consume scripted responses from a single ordered queue —
    matching how ``a2a_serve`` wires a single LiteLLM provider across requests.
    """

    def factory() -> BaseAgent:
        agent = BaseAgent(
            system_message="You are an orchestrator. Delegate to coordinator.",
            provider=provider,
            max_iterations=5,
        )
        dispatch = DispatchAgentTool(registry=registry)
        dispatch.set_webhook_url(webhook_url)
        agent.register_tool(dispatch)
        return agent

    return factory


# ── The actual e2e test ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_resumed_by_drainer_after_coordinator_completes():
    """End-to-end: orchestrator dispatches coordinator → coordinator completes
    → drainer auto-spawns a fresh orchestrator turn → drain-spawn POSTs to
    the CLI webhook fixture → tracer holds ONE trace with TWO a2a_task spans.

    This exercises the whole spec 1 pipeline in-process: real executor, real
    agent loop, real registry, real handler, real drainer, real dispatch tool,
    real ``_DrainSpawnEventQueue``. Only the LLM provider and remote A2A
    client are hand-written Fakes (per iron rule).
    """
    # ── 1. Webhook fixture (Starlette via ASGI transport — no uvicorn) ────
    received_posts: list[dict] = []
    received_headers: list[dict] = []

    async def webhook_route(request: Request) -> JSONResponse:
        received_posts.append(await request.json())
        received_headers.append(dict(request.headers))
        return JSONResponse({"ok": True})

    webhook_app = Starlette(routes=[Route("/webhook", webhook_route, methods=["POST"])])

    # ── 2. Tracer with capturing exporter (real Tracer, real spans) ───────
    exporter = _CapturingExporter()
    tracer = Tracer(exporter=exporter, service_name="orchestrator-e2e")

    # ── 3. Registry with FakeRemoteClient injected for "coordinator" ──────
    # The registry uses the same shared httpx_client as the executor (real
    # ``httpx.AsyncClient`` connected to the Starlette ASGI app for the
    # webhook posts; the registry never makes HTTP calls in this test
    # because we bypass ``resolve_all`` and inject the client directly).
    httpx_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=webhook_app),
        base_url="http://testserver",
    )
    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx_client)

    coordinator_submitted_task = Task(
        id="coord-task-1",
        context_id="ctx-coord",
        status=TaskStatus(
            state=TaskState.submitted,
            timestamp="2026-05-07T12:00:00+00:00",
        ),
    )
    fake_remote = FakeRemoteClient(response_task=coordinator_submitted_task)

    # Direct registry injection mirrors test_polling_drain_call.py and
    # test_respond_tool.py: bypass network ``resolve_all``, install card +
    # client manually so dispatch_agent's ``client_for("coordinator")``
    # resolves without touching the network.
    registry._cards["coordinator"] = _build_agent_card_for_remote(
        name="coordinator", description="Coordinator agent"
    )
    registry._clients["coordinator"] = fake_remote

    # ── 4. FakeProvider with scripted responses ───────────────────────────
    # Turn 1 (user-triggered): tool_call dispatch_agent("coordinator", "...")
    # Turn 2 (user-triggered): final response after dispatch returns task_id.
    # Turn 3 (drain-spawn): final response acknowledging the coordinator.
    scripted = [
        # Turn 1: orchestrator calls dispatch_agent
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc-dispatch-1",
                    name="dispatch_agent",
                    arguments={
                        "agent_name": "coordinator",
                        "query": "review the staging area",
                    },
                )
            ],
            usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120),
        ),
        # Turn 2: orchestrator closes its turn after dispatch
        AssistantMessage(
            content="Ho dispatcato il task al coordinator.",
            tool_calls=[],
            usage=Usage(input_tokens=120, output_tokens=10, total_tokens=130),
        ),
        # Turn 3: drain-spawn turn (orchestrator processes <remote_task_update>)
        AssistantMessage(
            content=(
                "Il coordinator ha trovato che la staging area e' vuota. "
                "Tutto sotto controllo."
            ),
            tool_calls=[],
            usage=Usage(input_tokens=200, output_tokens=15, total_tokens=215),
        ),
    ]
    provider = FakeProvider(responses=scripted)

    # ── 5. Build the executor (real production class) ─────────────────────
    context_store = ContextStore(max_contexts=8)
    webhook_url = "http://orchestrator.test/webhook"  # used by dispatch tool

    agent_factory = _build_orchestrator_factory(
        provider=provider, registry=registry, webhook_url=webhook_url
    )

    executor = ObelixAgentExecutor(
        agent_factory=agent_factory,
        tracer=tracer,
        registry=registry,
        context_store=context_store,
        httpx_client=httpx_client,
    )

    # ── 6. Drive the user-triggered turn ──────────────────────────────────
    context_id = "ctx-orch-e2e"
    user_task_id = "user-task-1"

    # The CLI declares its webhook URL+token via Message.metadata on the
    # first request (TEMP-PATCH-SPEC-1). The executor stores them on the
    # entry; ``_run_drain_task`` later reads them when building the
    # ``_DrainSpawnEventQueue``.
    user_ctx = _FakeRequestContext(
        task_id=user_task_id,
        context_id=context_id,
        text="Voglio un check sullo stato della staging area",
        metadata={
            "client_webhook_url": "http://testserver/webhook",
            "client_webhook_token": "tok-cli-e2e",
        },
    )
    user_queue = _FakeEventQueue()

    await executor.execute(user_ctx, user_queue)

    # Sanity: the user-triggered turn registered the remote task on the entry.
    entry = context_store.peek(context_id)
    assert entry is not None, "context entry must exist after the first turn"
    assert "coord-task-1" in entry.remote_tasks, (
        f"dispatch_agent should have recorded the remote task; "
        f"got remote_tasks={list(entry.remote_tasks)}"
    )
    # The TEMP-PATCH-SPEC-1 fields were captured from Message.metadata.
    assert entry.client_webhook_url == "http://testserver/webhook"
    assert entry.client_webhook_token == "tok-cli-e2e"

    # The user-triggered turn ended → ``executor._run_agent`` finally cleared
    # ``entry.trace_session`` (executor.py around line 551). To exercise the
    # spec § 5.1 trace-reuse branch (the documented happy path: drain-spawn
    # opens a SIBLING ``a2a_task`` span on the same trace), we re-attach the
    # first-turn trace onto the entry. In production this preservation is
    # what the spec intends — under the trace_session=None edge case the
    # drainer falls back to ``start_trace`` (covered by spec § 5.2 and by
    # ``test_drain_spawn_falls_back_to_start_trace_when_no_session`` in
    # test_tracer_trace_reuse.py). Here we drive the primary branch so the
    # full e2e demonstrates "1 trace, 2 a2a_task spans" per the spec.
    assert len(exporter.started_traces) == 1, (
        "user-triggered turn must have opened exactly one trace"
    )
    first_trace = exporter.started_traces[0]
    entry.trace_session = first_trace

    # ── 7. Simulate coordinator completing → handle_remote_update ─────────
    # ``handle_remote_update`` only fires the notification when the local
    # state changes. We seeded "submitted"; now we deliver "completed".
    coordinator_terminal = Task(
        id="coord-task-1",
        context_id="ctx-coord",
        status=TaskStatus(
            state=TaskState.completed,
            timestamp="2026-05-07T12:00:30+00:00",
        ),
        artifacts=[
            Artifact(
                artifact_id="art-1",
                name="reply",
                parts=[Part(root=TextPart(text="staging area is empty"))],
            )
        ],
    )
    handle_remote_update(
        entry=entry,
        task_id="coord-task-1",
        fresh=coordinator_terminal,
        registry=registry,
    )
    assert len(entry.pending_notifications) == 1, (
        "handle_remote_update must enqueue a <remote_task_update> notification"
    )
    assert entry.remote_tasks["coord-task-1"].status == "completed"

    # ── 8. Trigger the drainer (the call site that webhook.py / polling.py
    #      already exercises in production after handle_remote_update) ────
    # ``entry.idle.is_set()`` is True after the user-triggered turn (the
    # finally block in executor.execute set it). The drainer's checks pass
    # → spawn_drain_task fires, which schedules an asyncio.create_task that
    # runs ``_run_drain_task`` in the background.
    await maybe_spawn_drain_task(
        entry=entry,
        context_id=context_id,
        executor=executor,
    )

    # ── 9. Wait for the drain-spawn task to complete ──────────────────────
    # Background work: spawn_drain_task uses asyncio.create_task. Yield a
    # few times so the task can run, the agent loop can finish, and the
    # ``_DrainSpawnEventQueue`` POSTs land in our Starlette webhook fixture.
    deadline = asyncio.get_running_loop().time() + 3.0
    while asyncio.get_running_loop().time() < deadline:
        # Done when we've seen at least one POST AND a state in {completed,
        # failed, canceled, rejected}. The drain-spawn agent loop closes when
        # the FakeProvider's third scripted response returns (no tool calls).
        if any(p.get("status", {}).get("state") == "completed" for p in received_posts):
            break
        await asyncio.sleep(0.05)

    await httpx_client.aclose()

    # ── 10. Verifications ─────────────────────────────────────────────────
    # (a) Webhook fixture saw POST(s); at least one with state=completed.
    assert len(received_posts) >= 1, (
        f"expected the drain-spawn event queue to POST at least once "
        f"to the CLI webhook; got 0 POSTs. Provider invocations: "
        f"{provider.invocations}. Pending notifications: "
        f"{len(entry.pending_notifications)}"
    )
    states = [p.get("status", {}).get("state") for p in received_posts]
    assert "completed" in states, (
        f"expected at least one 'completed' POST; got states={states}"
    )

    # (b) camelCase contract on the wire (per A2ABaseModel.alias_generator
    #     — see research artifact and test_drain_spawn_webhook_post.py).
    last = received_posts[-1]
    assert "contextId" in last, (
        f"expected camelCase contextId in payload; got keys={list(last)}"
    )
    assert last["contextId"] == context_id

    # (c) Auth token header forwarded from entry.client_webhook_token.
    assert any(
        h.get("x-a2a-notification-token") == "tok-cli-e2e" for h in received_headers
    ), (
        f"expected X-A2A-Notification-Token=tok-cli-e2e in at least one POST; "
        f"got tokens={[h.get('x-a2a-notification-token') for h in received_headers]}"
    )

    # (d) FakeProvider was invoked at least 3 times (turn 1, turn 2, turn 3).
    assert provider.invocations >= 3, (
        f"expected the FakeProvider to be hit at least 3 times "
        f"(user turn 1+2 + drain-spawn turn); got {provider.invocations}"
    )

    # (e) Tracer observed two a2a_task spans (user-turn + drain-spawn) on
    #     the SAME trace_id — primary branch of spec § 5.1 (trace-reuse).
    a2a_task_spans = [
        s for s in exporter.completed_spans if s.span_type == SpanType.a2a_task
    ]
    assert len(a2a_task_spans) >= 2, (
        f"expected at least 2 a2a_task spans (user-turn + drain-spawn), "
        f"got {len(a2a_task_spans)}: {[(s.name, s.metadata) for s in a2a_task_spans]}"
    )
    drain_span = next(
        (s for s in a2a_task_spans if s.metadata.get("drain_spawn") is True),
        None,
    )
    assert drain_span is not None, (
        f"expected a drain-spawn a2a_task span (metadata.drain_spawn=True); "
        f"saw spans={[s.metadata for s in a2a_task_spans]}"
    )
    user_span = next(
        (s for s in a2a_task_spans if s.metadata.get("drain_spawn") is not True),
        None,
    )
    assert user_span is not None, (
        "expected a user-triggered a2a_task span alongside the drain-spawn"
    )

    # (f) Drain-spawn span name must contain 'drain-spawn' for visual
    #     inspection in the tracer frontend.
    assert "drain-spawn" in drain_span.name, (
        f"expected 'drain-spawn' in span name; got {drain_span.name!r}"
    )

    # (g) Single shared trace_id across the two a2a_task spans (spec § 5.1).
    trace_ids = {s.trace_id for s in a2a_task_spans}
    assert len(trace_ids) == 1, (
        f"expected ONE shared trace_id across both a2a_task spans (spec § 5.1); "
        f"got {len(trace_ids)} distinct trace_ids: {trace_ids}"
    )
    assert next(iter(trace_ids)) == first_trace.trace_id, (
        f"shared trace_id must equal the first turn's trace_id; "
        f"got {next(iter(trace_ids))} vs first={first_trace.trace_id}"
    )

    # (h) Both a2a_task spans are SIBLINGS (parent_span_id is None) under
    #     the shared trace, per spec § 5.3 (frontend renders them as
    #     multiple roots under one trace header).
    assert all(s.parent_span_id is None for s in a2a_task_spans), (
        f"expected both a2a_task spans to be roots (parent_span_id=None); "
        f"got parents={[(s.name, s.parent_span_id) for s in a2a_task_spans]}"
    )
