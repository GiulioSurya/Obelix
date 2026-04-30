# Tracer Guide

The Obelix tracer is an agent-instrumented, A2A-aware observability SDK that emits structured spans and point-in-time events for every significant step of an agent's execution (LLM turns, tool calls, skill invocations, sub-agent dispatch, A2A task lifecycle, memory propagation, hook side effects). Its purpose is twofold: **dev debugging** (understanding what an agent did, in what order, with which data) and **production observability** (shipping traces to a backend for later inspection). It is **not** a full OpenTelemetry implementation, does not provide distributed tracing or context propagation across network boundaries out of the box, and does not instrument provider-internal stream chunks (text/reasoning/tool-use deltas) as individual events — those aggregate onto the enclosing agent span.

This guide covers:
- The span taxonomy (what kinds of spans the tracer emits and when)
- The event taxonomy (what structured events attach to spans)
- How LLM usage is aggregated onto agent spans
- How to enable tracing on an `AgentFactory`
- The three built-in exporters and how to write a custom one
- Span status semantics
- Links to the live frontend mockup and the design spec

---

## Quick Start

```python
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import ConsoleExporter

tracer = Tracer(
    exporter=ConsoleExporter(verbosity=2),
    service_name="my_service",
)

factory = AgentFactory().with_tracer(tracer)
factory.register("assistant", MyAssistantAgent)

agent = factory.create("assistant")
await agent.execute_query("hello")
```

Every agent created by the factory inherits the tracer. Spans and events are exported inline as they open and close, so the console shows activity in real time.

---

## Span Taxonomy

The tracer emits eight span types, each modeling a distinct unit of agent execution. Span types are defined in `obelix.core.tracer.models.SpanType`.

| Span type | Parent | Purpose |
|-----------|--------|---------|
| `a2a_task` | (root when served via A2A) | A2A turn root. Wraps one end-to-end A2A request/response cycle, including any suspension window for `input_required`. Emitted by the A2A server executor. |
| `agent` | `a2a_task` or root | One `BaseAgent` execution. Root when the agent runs standalone (not via A2A). Carries aggregated LLM usage in `metadata.llm_usage`. |
| `sub_agent` | `agent` | An invocation of a registered sub-agent (through `SubAgentWrapper`). Has a child `agent` span for the sub-agent's actual execution. |
| `skill` | `agent` | A skill execution via the built-in `SkillTool`. Carries `metadata.mode` (`fork` or `inline`) and `metadata.source`. `fork` mode also produces a child `agent` span for the forked execution. |
| `tool` | `agent` | A regular tool call (anything that is not a skill and not a sub-agent). |
| `deferred_wait` | `a2a_task` | Suspension window between A2A `input_required` (the agent emits a deferred tool) and resume (the client answers). `duration_ms` is the wall-clock time the task was suspended. Carries `metadata.tool_name`, `metadata.tool_call_ids`, `metadata.suspend_reason`. |
| `human` | root (`a2a_task` or `agent`) | The user input for this turn. Direct child of the root; its `input` holds the query text. |
| `assistant` | root (`a2a_task` or `agent`) | The agent's final natural-language response for this turn. Direct child of the root; its `output.content` holds the answer. |

**Why no `llm` span type?** Individual LLM invocations are considered internal mechanics of an agent turn, not click-worthy tree nodes. Their stats aggregate onto the enclosing `agent` span (see [Aggregated LLM Usage](#aggregated-llm-usage)). This keeps the tree compact and focuses attention on units of work that matter.

---

## Event Taxonomy

Events are point-in-time annotations attached to spans via `tracer.add_event(name, attributes)`. They record behavioral inflections (state changes, hook effects, memory transfers) without adding visual weight to the tree. Five event names are emitted by the framework.

| Event | Attached to | Attributes | Meaning |
|-------|-------------|------------|---------|
| `a2a.state_change` | `a2a_task` | `from`, `to`, `reason` | The A2A task transitioned between protocol states (`working` → `input_required`, `input_required` → `working`, etc.). The emitting code temporarily re-pins the current span to the `a2a_task` root so the event attaches there even if a nested agent/tool span is active. |
| `hook.fired` | whichever span owns the hook (`agent` or `tool`) | `event` (the `AgentEvent` name), `decision` (`RETRY` / `STOP` / `FAIL` / `REJECT`), `reason`, `effects_count` | A registered hook ran and changed behavior (decision ≠ `CONTINUE`, or effects were applied). Plain `CONTINUE` with no effects is suppressed to avoid noise. |
| `memory.pull` | `agent` | `from_agent`, `policy`, `bytes` | Shared memory was pulled into this agent's context from another agent in the `SharedMemoryGraph`. |
| `memory.publish` | `agent` | `kind` (`final` or `tool_result`), `bytes` | This agent published content into shared memory — either its final response or a tool result. |
| `cancellation.requested` | `a2a_task` | `source` (plus optional `iteration`) | A cancellation request was received for this A2A task. Emitted once at the top of `cancel()`, before branching into the in-flight or deferred-resume cancellation path, so it fires regardless of which path ultimately handles it. |

Event attributes are serialized through the same helper as span input/output, so Pydantic models, dicts, and primitives all round-trip safely.

---

## Aggregated LLM Usage

LLM calls do not emit their own spans. Instead, their stats aggregate onto the enclosing `agent` span in `metadata`. Each call updates:

| Key | Type | Description |
|-----|------|-------------|
| `metadata.llm_usage.calls` | int | Number of LLM calls made by this agent. |
| `metadata.llm_usage.input_tokens` | int | Cumulative input tokens across calls. |
| `metadata.llm_usage.output_tokens` | int | Cumulative output tokens across calls. |
| `metadata.llm_usage.total_tokens` | int | `input_tokens + output_tokens`. |
| `metadata.model_id` | str | Model used (first call wins — set with `setdefault`). |
| `metadata.provider_type` | str | Provider type (first call wins). |
| `metadata.reasoning` | list | One entry per call when the provider surfaces reasoning traces (omitted otherwise). |
| `metadata.llm_durations_ms` | list[float] | One entry per call with the wall-clock duration of that LLM round-trip. |

The `ConsoleExporter` uses `llm_usage` to render the `N calls  Xk->Yk tok` suffix on agent end lines, and a trace-level footer summarizes totals when the trace ends.

---

## How to Enable

A tracer is a `Tracer` instance wired to a `TracerExporter`. Attach it to the `AgentFactory` via `with_tracer()`, and every agent created from that factory becomes instrumented.

```python
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import ConsoleExporter, HTTPExporter

# Console output (local dev)
tracer = Tracer(
    exporter=ConsoleExporter(verbosity=2),
    service_name="my_service",
)

# Or: HTTP export to a remote backend
tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="my_service",
)

factory = AgentFactory().with_tracer(tracer)
```

`service_name` is attached to every span and trace record; use it to distinguish which service/agent-process produced the trace when multiple emitters share a backend.

When the factory composes an A2A server via `a2a_serve()`, the executor picks up the same tracer and wraps each inbound A2A turn with an `a2a_task` root span.

---

## Exporters

Three exporters are built in; all implement the abstract `TracerExporter` interface from `obelix.core.tracer.exporters`.

### `NoOpExporter`

```python
from obelix.core.tracer.exporters import NoOpExporter

tracer = Tracer(exporter=NoOpExporter())
```

Drops every span and event. Useful as a baseline in tests where instrumentation must remain active (so code paths are exercised) but output should be silenced.

### `ConsoleExporter`

```python
from obelix.core.tracer.exporters import ConsoleExporter

tracer = Tracer(exporter=ConsoleExporter(verbosity=2, use_color=True))
```

Pretty-prints spans and events inline as they complete, with ANSI color and short icons (`[AG]`, `[SA]`, `[SK]`, `[TL]`, `[DW]`, `[TK]`, `[H]`, `[A]`). Verbosity controls how much detail is shown:

| Verbosity | Contents |
|-----------|----------|
| `1` (minimal) | Agent headers; LLM / tool / sub-agent / skill with duration only. |
| `2` (standard, default) | Above plus token usage, tool arguments and results (truncated to ~120 chars), user/assistant content previews, and inline `hook.fired` / `memory.*` event lines. |
| `3` (debug) | Above plus reasoning, full tool arguments and results (truncated to ~500 chars), and full debug lines for tool/sub-agent spans. |

The exporter auto-detects stdout encoding. When the terminal cannot encode common Unicode box-drawing characters (e.g., `cp1252` on Windows consoles), it falls back to ASCII alternatives (`---` instead of `───`, `-` instead of `—`) so output never corrupts.

At trace start the exporter prints a header (`--- Trace: <name> [<id>] service=<service> ---`). At trace end it prints a footer summarizing the trace duration, status, LLM call count, aggregate token usage, and tool-call count.

### `HTTPExporter`

```python
from obelix.core.tracer.exporters import HTTPExporter

tracer = Tracer(
    exporter=HTTPExporter(
        endpoint="http://localhost:8100/api/v1/ingest",
        timeout=10.0,
    ),
)
```

Streams traces to a remote backend (for example, the `obelix-tracer` backend at `:8100`). The protocol is designed for realtime observation:

- `start_trace` posts the trace header (no spans) to `POST /ingest/trace`, creating the trace record immediately.
- `export_span` posts each span to `POST /ingest/span` twice — once at `on_start` with `end_time=None` (so frontends can render the span in progress) and once at `on_end` with the completed data.
- `on_event` is inherited from the base class default (no-op). Events travel with the span payload when it is re-sent at end-time; realtime event emission can be added by overriding `on_event` in a subclass.
- `end_trace` sends a `PATCH /ingest/trace/{trace_id}` to finalize the trace status and end time.
- `export` (batch mode) posts the complete `TraceSession` with all spans to `POST /ingest` — used only if you explicitly call it; the streaming path is the default.
- `shutdown` closes the underlying `httpx.AsyncClient`.

Failures on any transport call are logged and swallowed so tracer errors cannot break agent execution.

---

## Writing a Custom Exporter

Extend `TracerExporter` and implement the five abstract methods. `on_event` has a default no-op implementation and is the only optional override.

```python
from datetime import datetime

from obelix.core.tracer.exporters import TracerExporter
from obelix.core.tracer.models import Span, SpanEvent, SpanStatus, TraceSession


class MyExporter(TracerExporter):
    async def export(self, trace: TraceSession) -> None:
        # Batch export of a complete trace. Called only if something
        # invokes it explicitly — the streaming methods below are the
        # normal path.
        ...

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:
        # Called once when a trace opens. No spans yet.
        ...

    async def export_span(self, span: Span, service_name: str) -> None:
        # Called twice per span: at start (end_time=None) and at end.
        # Inspect span.end_time to tell them apart.
        ...

    async def end_trace(
        self,
        trace_id: str,
        status: SpanStatus,
        end_time: datetime | None,
    ) -> None:
        # Called once when a trace closes.
        ...

    async def on_event(
        self,
        span: Span,
        event: SpanEvent,
        service_name: str,
    ) -> None:
        # Optional override for realtime event notification.
        # Default: no-op. Events are also present in span.events at
        # end-time, so overriding this only matters for low-latency
        # observers (e.g., a live UI updating on event emission).
        ...

    async def shutdown(self) -> None:
        # Release any held resources (network clients, file handles).
        ...
```

Register the exporter the same way as the built-ins:

```python
tracer = Tracer(exporter=MyExporter(), service_name="my_service")
```

---

## Span Status Semantics

The `SpanStatus` enum in `obelix.core.tracer.models` defines five values:

| Status | Meaning |
|--------|---------|
| `ok` | Normal completion. Default. |
| `error` | An exception or unexpected failure closed the span. |
| `timeout` | The operation exceeded its time budget. |
| `rejected` | By convention, only on the `a2a_task` root span. Signals the A2A turn was rejected (for example, by a `REJECT` hook decision). |
| `canceled` | By convention, only on the `a2a_task` root span. Signals the A2A turn was canceled by client request. |

`rejected` and `canceled` are semantic markers for A2A-level outcomes. Lower-level spans (`agent`, `tool`, `skill`, etc.) use `ok`, `error`, or `timeout`; a cancellation propagated down through the tree typically surfaces as the task-level `canceled` on the root plus whatever state the nested spans were in when cancellation arrived.

---

## Visual Preview

For an interactive preview of how a backend can render traces produced by this tracer, see [`tracer_mockup.html`](tracer_mockup.html). The mockup illustrates the collapsible tree layout, color coding, and event chips.

---

## Design Spec

The full design rationale — why `a2a_task` became the new root, why LLM spans were removed in favor of aggregation, the event catalog decisions, and non-goals — is documented in the design spec:

- [`superpowers/specs/2026-04-21-tracer-refactor-design.md`](superpowers/specs/2026-04-21-tracer-refactor-design.md)
