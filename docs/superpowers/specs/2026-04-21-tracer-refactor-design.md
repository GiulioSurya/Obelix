# Tracer refactor — design spec

**Date:** 2026-04-21
**Status:** Approved — ready for implementation plan
**Author:** Giulio + Claude (brainstorming session)
**Related mockup:** `docs/tracer_mockup.html`

---

## 1. Goals and non-goals

### Goals

1. **Make A2A task lifecycle first-class in traces.** The current tracer was built before the A2A protocol; state transitions (`working` / `input_required` / `completed` / `rejected` / `failed` / `canceled`) are invisible today.
2. **Make skill executions first-class and distinguishable from tools and sub-agents.** Today a skill fork is indistinguishable from a regular tool call because the wrapper check `isinstance(tool, SubAgentWrapper)` walks over ephemeral wrappers that are not in `registered_tools`.
3. **Make hook effects and memory propagation observable** without creating span noise. Hooks that mutate behavior (`REJECT`, `RETRY`, `STOP`, `INJECT`) must be visible; pull/publish across `SharedMemoryGraph` must be visible.
4. **Reduce visual noise.** LLM calls are not interesting as individual tree nodes; they are internal mechanics of an agent. Aggregate their stats onto the agent span. Keep the tree compact — one node = one unit of execution worth clicking.
5. **Preserve realtime observability.** The backend at `:8100` receives partial spans at `on_start` (end_time=None) and completed spans at `on_end`. Dev debugging relies on seeing spans grow duration in real time. This protocol is preserved.
6. **Preserve live console UX.** Three verbosity levels (minimal / standard / debug) keep working with the new taxonomy.

### Non-goals

- No backward compatibility. We swap the model. The frontend (`C:\Users\GLoverde\PycharmProjects\obelix-tracer\frontend`) and the tracer backend (`C:\Users\GLoverde\PycharmProjects\obelix-tracer\obelix_tracer`) will adapt.
- No migration to OpenTelemetry. Evaluated and declined in favor of preserving the custom realtime protocol and avoiding a new dependency.
- No instrumentation of provider-internal stream chunks (text deltas, reasoning deltas, tool_use deltas) as events by default. Those remain aggregated into the agent span's `llm_usage`. A debug-mode flag may enable them later.

### Priority ordering

Dev debugging > production observability > replay. The design optimizes for the first while keeping the other two possible.

---

## 2. Conceptual model

### 2.1 Trace

A **trace** is a single logical unit of work.

- **Via A2A:** one trace = one A2A task (one turn). Multi-turn conversations on the same `context_id` produce distinct traces, correlated via a `context_id` attribute on the root span and on `TraceSession.metadata`.
- **Standalone:** one trace = one `execute_query()` / `execute_query_stream()` call.

`trace_id` identifies a turn, not a conversation.

### 2.2 Span

A **span** is something with a duration (start + end). The new taxonomy has **eight** types:

| Type | Badge | When opened | Root? |
|---|---|---|---|
| `a2a_task` | `TK` | By A2A executor, wraps the full turn | ✓ (via A2A) |
| `agent` | `AG` | By `BaseAgent._execute_loop` | ✓ (standalone) |
| `sub_agent` | `SA` | By BaseAgent at tool dispatch when `isinstance(tool, SubAgentWrapper)` | no |
| `skill` | `SK` | By BaseAgent at tool dispatch when `isinstance(tool, SkillTool)` | no |
| `tool` | `TL` | By BaseAgent at tool dispatch for all other tools (incl. MCP) | no |
| `deferred_wait` | `DW` | By A2A executor on `input_required`; closed on resume | no |
| `human` | `H` | Once per trace as direct child of the outermost span (first child) | no |
| `assistant` | `A` | Once per trace as direct child of the outermost span (last child), only for the **final** response | no |

Removed from the current taxonomy: `llm` (aggregated into `agent`), `memory` (becomes event), `hook` (becomes event).

### 2.3 Span event

An **event** is a point-in-time occurrence attached to the current span. It has a `name`, `timestamp`, and `attributes` dict. It does **not** have a duration.

Five event types:

| Event | Attached to | Attributes | When |
|---|---|---|---|
| `a2a.state_change` | `a2a_task` | `from`, `to`, `reason` | Every transition inside the A2A executor |
| `hook.fired` | `agent` (LLM/final/query hooks) or `tool` (tool-scoped hooks) | `event` (`BEFORE_LLM_CALL`, …), `decision` (`REJECT`/`RETRY`/`STOP`/`INJECT`/`CONTINUE`), `reason`, `effects_count`, `hook_id?` | Only when `decision ≠ CONTINUE` or `effects` non-empty |
| `memory.pull` | `agent` | `from_agent`, `policy`, `items_count`, `bytes` | Only when `items_count > 0` |
| `memory.publish` | `agent` | `kind` (`final` / `tool_result`), `bytes` | On `BEFORE_FINAL_RESPONSE` publish |
| `cancellation.requested` | `a2a_task` or `agent` | `source`, `iteration?` | On cancel signal received |

### 2.4 Aggregated stats on the agent span

The `agent` span carries aggregated LLM execution stats in its `metadata`, computed by the BaseAgent loop as it runs. This is what replaces the per-call `llm` span.

```json
{
  "agent_name": "ReviewerAgent",
  "model_id": "anthropic/claude-haiku-4-5-20251001",
  "provider_type": "litellm",
  "iterations": 3,
  "llm_usage": {
    "calls": 3,
    "input_tokens": 1500,
    "output_tokens": 250,
    "total_tokens": 1750
  },
  "tool_calls_count": 4,
  "system_prompt": "...",
  "finish_reason": "end_turn" | "tool_use" | "max_tokens" | "stop",
  "reasoning": ["...", "..."],         // per iteration, only when provider exposes it
  "conversation_history": [...]        // only at verbosity 3 / debug mode
}
```

Optional per-iteration detail for replay (off by default):

```json
"iterations_detail": [
  {"n": 1, "duration_ms": 600, "tokens_in": 800, "tokens_out": 120, "tool_calls": 1, "reasoning": "..."},
  {"n": 2, "duration_ms": 900, "tokens_in": 600, "tokens_out": 80,  "tool_calls": 0}
]
```

---

## 3. Layer separation — who opens what

Span ownership is **by layer**, not by registration. Three decision points in the entire codebase.

### 3.1 A2A Executor — `adapters/inbound/a2a/server/executor.py`

Responsibilities:
- Opens `a2a_task` span at the start of every A2A turn, as the root.
- Emits `human` span as the first child of `a2a_task` after parts are converted to `HumanMessage`.
- Emits `assistant` span as the last child of `a2a_task` before closing, carrying the final `AssistantResponse`.
- Emits `a2a.state_change` events on every transition (`working`, `input_required`, `completed`, `rejected`, `failed`, `canceled`, and `working` again on resume).
- Opens `deferred_wait` span when emitting `input_required`, closes it on the matching resume.
- Emits `cancellation.requested` when cancel is signalled by the client.
- Does **not** know about agent internals: it opens `a2a_task`, calls `agent.execute_query_async(...)`, the agent nests itself under the a2a_task via `contextvars` automatically.

### 3.2 BaseAgent — `core/agent/base_agent.py` + `core/agent/agent_tracing.py`

Responsibilities:
- Opens `agent` span at the start of every `execute_query_async` / `execute_query_stream`. If `get_current_trace()` is `None`, also opens the trace (standalone mode). Otherwise nests under whatever is current (typically an `a2a_task`).
- Emits `human` span as the first child of the `agent` span (or of the `a2a_task` if served via A2A — the emission point needs to move up; see §6).
- Emits `assistant` span as the last child of the outermost span, only for the **final** agent response (not for intermediate LLM outputs mid-loop).
- Tracks per-iteration stats and folds them into `agent.metadata.llm_usage` + optionally `iterations_detail`.
- At tool dispatch (`_process_tool_calls`), chooses the span type with the single decision branch:

```python
if isinstance(tool, SkillTool):
    span_type = SpanType.skill
elif isinstance(tool, SubAgentWrapper):
    span_type = SpanType.sub_agent
else:
    span_type = SpanType.tool
```

- Emits `memory.pull` and `memory.publish` events (driven from `memory_hooks.py`) on the current `agent` span.
- Emits `hook.fired` events from `_run_hooks()` on the current span, but only when `decision ≠ CONTINUE` or `effects` is non-empty.

### 3.3 Tracer library — `core/tracer/`

Pure data layer. Knows nothing about A2A, agents, skills. API:

```python
tracer.start_trace(name, metadata) -> TraceSession
tracer.end_trace(status, error)
tracer.start_span(span_type, name, input, metadata) -> Span
tracer.end_span(output, status, error)
tracer.add_event(name, attributes)                  # NEW — attached to current span
tracer.span_context(span_type, name, ...)           # async context manager (existing)
tracer.trace_context(name, ...)                     # async context manager (existing)
```

Context propagation stays on `contextvars` (async-safe).

---

## 4. Data model (Python)

### 4.1 `SpanType` enum

```python
class SpanType(StrEnum):
    a2a_task = "a2a_task"
    agent = "agent"
    sub_agent = "sub_agent"
    skill = "skill"
    tool = "tool"
    deferred_wait = "deferred_wait"
    human = "human"
    assistant = "assistant"
```

### 4.2 `SpanEvent` (new)

```python
class SpanEvent(BaseModel):
    name: str                                         # e.g. "hook.fired"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = Field(default_factory=dict)
```

### 4.3 `Span` (modified)

```python
class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    parent_span_id: str | None = None
    span_type: SpanType
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = None
    input: Any | None = None
    output: Any | None = None
    status: SpanStatus = SpanStatus.ok
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)       # NEW
```

### 4.4 `TraceSession` (unchanged structurally)

No change; `context_id` lives in `metadata` when relevant.

### 4.5 `SpanStatus` (extended)

Adds the A2A-aware terminal states on the root `a2a_task` span, propagated from the A2A task state:

```python
class SpanStatus(StrEnum):
    ok = "ok"                    # standalone completed, or a2a completed
    error = "error"              # runtime exception — a2a failed
    timeout = "timeout"          # max_iterations reached
    rejected = "rejected"        # a2a rejected (hook or TaskRejectedError)
    canceled = "canceled"        # a2a canceled
```

By convention child spans only use `ok` / `error` / `timeout`. Terminal A2A states (`rejected`, `canceled`) are only set on the `a2a_task` root (and mirrored on `TraceSession.status`). The type system does not enforce this — it is documented discipline.

---

## 5. Rendering model

The same data can be shown in three registers, progressively denser. Mockup: `docs/tracer_mockup.html`.

### 5.1 Tree (default, compact)

One line per span. Name on the left, stat chips in the middle, event counter chips at the right before duration.

| Pictogram | Meaning |
|---|---|
| `⚡N` | `hook.fired` events count |
| `⇩N` | `memory.pull` events count |
| `⇧N` | `memory.publish` events count |
| `⚠` | `cancellation.requested` or rejection on this span |
| `deferred` | tool with `deferred=true` |
| `canceled` | tool ended by cancellation |

Examples:

```
[AG] ReviewerAgent    2.7s   2 calls  1.5k→250 tok   ⚡1
[AG] CommitAgent      790ms  1 call   ⇩1
[AG] SummaryAgent     390ms  1 call   ⇩2
```

### 5.2 Tree expanded (toggle "show events")

Inline event rows appear under the emitting span, prefixed with `·`:

```
[AG] ReviewerAgent    2.7s   2 calls  ⚡1
│    · hook.fired  event=BEFORE_LLM_CALL  decision=REJECT  reason="…"
```

### 5.3 Detail panel (click on a span)

Per span: meta chips (Latency, Model, LLM calls, In tokens, Out tokens, Tool calls, Iterations), Events section as a list of cards, Input, Output, System Prompt (collapsible), Reasoning, full Metadata.

### 5.4 Console exporter verbosity levels

| Level | Shows |
|---|---|
| 1 minimal | Span rows with duration. No events. No token chips. |
| 2 standard | + event rows inline, token chips, tool args truncated. This is the default. |
| 3 debug | + full tool args/results, per-iteration reasoning, system prompt, conversation history. |

### 5.5 `deferred_wait` visualization

In the frontend tree, render as a full-width yellow dashed divider between the span that deferred and the first span after resume:

```
⏸ deferred_wait — 7.0s
```

In console (verbosity 2+):

```
├─── SUSPEND  7.0s  tool=bash ───
```

Stored in the data model as a regular span; only the renderer chooses the divider representation.

---

## 6. Human and Assistant span placement

Today `emit_human_span` / `emit_assistant_span` are called inside BaseAgent. In the new model:

- **Standalone:** the agent is root. Human/Assistant spans are emitted as children of the agent, as today. First child = human, last child = final assistant.
- **Via A2A:** the A2A task is root. Human/Assistant should be direct children of the `a2a_task`, not buried inside the agent.

Implementation: move the emission to the outermost layer. The A2A executor emits `human` when it receives the `SendMessage` (after parts are converted to `HumanMessage`), and emits `assistant` when the agent produces the final `AssistantResponse`, before closing the `a2a_task`. BaseAgent only emits human/assistant when it is the root (i.e. when `get_current_trace()` was None at entry).

No intermediate assistant spans mid-loop. LLM replies between tool calls are tracked only via `agent.metadata.iterations_detail` (optional) and the final one as the `assistant` span.

---

## 7. Hook event emission

Inside `_run_hooks()` in `base_agent.py`:

```python
outcome = await hook.evaluate(status)
if outcome.decision != HookDecision.CONTINUE or outcome.effects:
    await self._tracer.add_event("hook.fired", {
        "event": event.value,                          # "BEFORE_LLM_CALL", …
        "decision": outcome.decision.value,            # "REJECT", …
        "reason": outcome.reason,                      # only for REJECT, else None
        "effects_count": len(outcome.effects or []),
        "hook_id": hook.id if hasattr(hook, "id") else None,
    })
# proceed with outcome handling
```

The event attaches to the current span, which is:
- `agent` for `BEFORE_LLM_CALL`, `AFTER_LLM_CALL`, `BEFORE_FINAL_RESPONSE`, `QUERY_END`
- `tool` for `BEFORE_TOOL_EXECUTION`, `AFTER_TOOL_EXECUTION`, `ON_TOOL_ERROR` (the tool span is already open when these fire)

No code change required to choose where to attach — `contextvars` gives us the right current span automatically.

---

## 8. Memory event emission

`memory_hooks.py:_inject_shared_memory` and `memory_hooks.py:_publish_to_memory` gain tracer-aware emission:

**Decision:** emit **one event per source** (one `memory.pull` for each predecessor that contributed data). Granular in the backend, easy to aggregate in UI. The tree chip `⇩N` shows total count across sources.

```python
# In _inject_shared_memory, after pulling:
for item in items:
    await self._tracer.add_event("memory.pull", {
        "from_agent": item.source_id,
        "policy": item.policy.value,
        "bytes": len(item.content or ""),
    })

# In _publish_to_memory:
if content:
    await self._tracer.add_event("memory.publish", {
        "kind": kind,                    # "final" or "tool_result"
        "bytes": len(content),
    })
```

---

## 9. `deferred_wait` span semantics

**Opening:** inside the A2A executor, when `event.deferred_tool_calls` is detected (`executor.py:234`). Before the final `TaskStatusUpdateEvent(state=input_required, final=True)` is emitted.

**Closing:** inside `inject_deferred_response()` (`deferred.py`), at the beginning of the resume path, before the executor restarts the loop. `duration_ms = resume_time - suspend_time` (wall clock of the pause).

**Attributes:** `tool_name`, `tool_call_id`, `suspend_reason` (always `"deferred_tool"` for now).

**Trace continuity:** the saved `trace_session` + `trace_context_span` in `ContextEntry` (today this already saves `trace_session` and `trace_span` for resume) must now also cover the context at time of suspend so that the `deferred_wait` span closes under the right parent on resume.

---

## 10. HTTP exporter protocol

Unchanged API between SDK and backend at `:8100`:

- `POST /ingest/trace` — header sent at `start_trace`
- `POST /ingest/span` — sent both at `start_span` (end_time=None, no events yet) and at `end_span` (completed, with events array)
- `PATCH /ingest/trace/{trace_id}` — sent at `end_trace` with final status

**New in payload:** the `events: [...]` field on span records. Backend must persist events and return them in span queries.

**New values on status:** `rejected`, `canceled` for `a2a_task` spans and trace-level status.

**New span types:** backend must accept `a2a_task`, `skill`, `deferred_wait` in `span_type`. Stripped values: `llm`, `memory`, `hook`.

### 10.1 Incremental vs full updates

Today the exporter sends the same span twice (start with `end_time=None`, end with full payload). It is allowed to send additional updates mid-flight if future use cases require it (e.g. progress on a long-running tool). Not in scope now.

---

## 11. Frontend implications

`C:\Users\GLoverde\PycharmProjects\obelix-tracer\frontend\src`:

- `types/trace.ts` — `SpanType` union updated. Drop `llm`, `memory`, `hook`. Add `a2a_task`, `skill`, `deferred_wait`. Keep `human`, `assistant`, `agent`, `sub_agent`, `tool`.
- `types/spanConfig.ts` — `SPAN_CONFIG` updated with new letters (`TK`, `SK`, `DW`) and color bindings. Palette proposal in `docs/tracer_mockup.html`.
- `components/TreeNode.tsx` — add chip rendering for `⚡/⇩/⇧/⚠` counters. Read counters from `span.events` filtered by name.
- `components/TreePanel.tsx` — add "show events" toggle in parent view, render event lines inline under the emitting span.
- `components/DetailPanel.tsx`:
  - Remove the `span.span_type === "llm"` branch; LLM stats now live on `agent`/`sub_agent` metadata (already reads from `llm_usage`).
  - Add an "Events" section listing `span.events[]` chronologically, with per-event card showing timestamp offset, name, attributes.
  - Special rendering for `a2a_task`: highlight final state, show `context_id` and `task_id`.
  - Special rendering for `deferred_wait`: show as a full-width panel with timer and the deferred tool call.
  - Keep `HumanDetail` and `AssistantDetail` as they are; they become the rendering for selected `human`/`assistant` spans.

---

## 12. Files changed in `src/obelix/`

Scope of the refactor (list, not diff):

- `core/tracer/models.py` — new `SpanType` values, new `SpanEvent` model, `events` field on `Span`, new `SpanStatus` values.
- `core/tracer/tracer.py` — add `add_event(name, attributes)` method; ensure `on_start` export carries initial metadata; ensure `on_end` export includes `events`.
- `core/tracer/exporters.py`:
  - `ConsoleExporter` — rewritten span formatters for new taxonomy; event inline rendering at verbosity ≥ 2; chip counters at verbosity ≥ 1.
  - `HTTPExporter` — include `events` in span payload; handle new statuses.
- `core/agent/agent_tracing.py` — drop `start_llm_span` / `end_llm_span` (replaced by metadata aggregation in BaseAgent); `start_tool_span` gains the 3-way branch; add helpers for skill spans and the metadata accumulator.
- `core/agent/base_agent.py`:
  - Remove per-call LLM spans.
  - Aggregate LLM usage on the agent span as iterations run.
  - Emit `hook.fired` events from `_run_hooks()` when decision ≠ CONTINUE or effects non-empty.
  - Move human/assistant span emission to happen only when `agent` is the root trace.
- `core/agent/memory_hooks.py` — emit `memory.pull` / `memory.publish` events.
- `core/agent/subagent_wrapper.py` — no change; BaseAgent decides the span type at dispatch. The inner agent's span naturally nests.
- `plugins/builtin/skill_tool.py` — no tracer awareness required. Fork/inline branching stays internal; inner agent's span nests under the `skill` span opened by BaseAgent.
- `adapters/inbound/a2a/server/executor.py`:
  - Replace the current implicit tracing (whatever exists) with explicit `a2a_task` span opened at task start, closed at task end.
  - Emit `a2a.state_change` events on every transition.
  - Open `deferred_wait` span on `input_required` emit, close on resume.
  - Emit `cancellation.requested` event on client cancel.
  - Emit `human` / `assistant` spans for the input / final output.

---

## 13. Console rendering (algorithmic sketch)

For each span `on_start`:
- Compute indent from parent chain depth.
- If span is `a2a_task` or `agent`: print header line with name and (if running) spinner placeholder.
- Else: buffer until `on_end` (so we print with final duration).

For each event `add_event`:
- If verbosity ≥ 2: print inline under the current span's indent with `· event.name attrs`.
- Else: increment the per-span event counter for the final chip rendering.

For each span `on_end`:
- Print the full line with badge, name, stat chips (calls, tokens), event chips (`⚡N ⇩N ⇧N ⚠`), status dot, duration.

Footer at `end_trace`:
- Aggregate across all spans: total duration, llm calls, tool calls, total tokens, final status.
- For traces with `deferred_wait` spans: compute `active = total - sum(deferred_wait.duration_ms)`, show both.

---

## 14. Test plan (high level)

- Unit: `Span.add_event` appends correctly; `SpanEvent` serializes.
- Unit: `ConsoleExporter` renders each new span type with correct badge/chip/color at each verbosity level.
- Unit: `HTTPExporter` includes `events` field in span payload.
- Integration: `dev_workflow_server.py` scenario — trace has one `a2a_task` root, one `agent` for CoordinatorAgent, three `sub_agent` spans, inner `agent` spans, `skill` spans with `mode=fork/inline`, `memory.pull` events on CommitAgent/SummaryAgent.
- Integration: deferred tool scenario — `deferred_wait` span opens on suspend, closes on resume, active vs suspended durations correct.
- Integration: rejection scenario — single `hook.fired` with `decision=REJECT` on agent, `a2a_task` status = `rejected`, reason propagated.
- Integration: cancellation scenario — `cancellation.requested` event, `a2a_task` status = `canceled`, in-flight tool span closed with `canceled` status.

---

## 15. Out of scope (future)

- Stream chunk events (`llm.chunk`, `llm.reasoning_delta`) in debug mode — possible future addition, will not pollute default trace.
- Distributed tracing A2A client → A2A server via W3C TraceContext headers — requires frontend/backend work and protocol decisions; separate spec.
- Trace linking across multi-turn conversations with the same `context_id` — UI concern, frontend can query by context_id.
- Per-MCP-server span aggregation or provider retry event — possible follow-ups.
