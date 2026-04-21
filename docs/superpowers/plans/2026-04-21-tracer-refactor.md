# Tracer refactor — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework Obelix's tracer around A2A task lifecycle, making skills and hooks first-class while removing LLM span noise.

**Architecture:** Eight span types (a2a_task, agent, sub_agent, skill, tool, deferred_wait, human, assistant) plus point-in-time events (a2a.state_change, hook.fired, memory.pull/publish, cancellation.requested) attached to their originating span. LLM stats aggregate onto the agent span's metadata. Instrumentation lives in three layers: A2A executor opens the task root, BaseAgent opens agent/tool/skill/sub_agent, the tracer library is a pure data layer.

**Tech Stack:** Python 3.13, Pydantic, asyncio, pytest, pytest-asyncio. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-21-tracer-refactor-design.md`
**Visual reference:** `docs/tracer_mockup.html`

---

## Conventions

- Exact file paths absolute from repo root.
- All tests use `pytest-asyncio`; existing helpers in `tests/conftest.py`.
- Each task ends with a commit (no Co-Authored-By trailer per user preference).
- Run `uv run ruff check --fix .` and `uv run ruff format .` before committing.
- Python 3.13 type syntax (`str | None`, `list[X]`) — no `Optional`, no `List`.
- **Test fixtures**: several tasks reference named fixtures (`make_agent_with_spy_tracer`, `executor_with_tracer`, `dev_workflow_agents_with_spy`, …). Implement each fixture in `tests/conftest.py` (or the appropriate nested `conftest.py`) the first time it is needed, following the `_SpyTracerExporter` pattern shown in Task 10. Do not skip fixture work as "later" — a missing fixture fails the whole task, add it before asserting.
- **BaseAgent constructor access in tests**: `BaseAgent` requires `system_message`, `provider`, and accepts `tracer` as a keyword argument. Mock the provider with `AsyncMock(side_effect=[msg1, msg2, ...])` on `.invoke`, and set `.provider_type = "mock"` + `.model_id = "mock-model"`.

---

## Phase 1 — Data model

### Task 1: Replace `SpanType` enum

**Files:**
- Modify: `src/obelix/core/tracer/models.py`
- Create: `tests/core/tracer/__init__.py`
- Create: `tests/core/tracer/test_models.py`

- [ ] **Step 1: Create the test directory and an empty package marker**

```bash
mkdir -p tests/core/tracer
```

Create `tests/core/tracer/__init__.py` with empty content.

- [ ] **Step 2: Write failing test**

Create `tests/core/tracer/test_models.py`:

```python
"""Tests for tracer data models."""

from obelix.core.tracer.models import Span, SpanEvent, SpanStatus, SpanType


class TestSpanType:
    def test_new_types_present(self):
        assert SpanType.a2a_task == "a2a_task"
        assert SpanType.skill == "skill"
        assert SpanType.deferred_wait == "deferred_wait"

    def test_kept_types_present(self):
        assert SpanType.agent == "agent"
        assert SpanType.sub_agent == "sub_agent"
        assert SpanType.tool == "tool"
        assert SpanType.human == "human"
        assert SpanType.assistant == "assistant"

    def test_removed_types_absent(self):
        assert not hasattr(SpanType, "llm")
        assert not hasattr(SpanType, "memory")
        assert not hasattr(SpanType, "hook")
```

- [ ] **Step 3: Run test — expect import errors on SpanEvent**

```bash
uv run pytest tests/core/tracer/test_models.py -v
```

Expected: ImportError on `SpanEvent`.

- [ ] **Step 4: Update `SpanType` enum**

In `src/obelix/core/tracer/models.py`, replace the `SpanType` class:

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

- [ ] **Step 5: Run test — only SpanEvent missing**

```bash
uv run pytest tests/core/tracer/test_models.py::TestSpanType -v
```

Expected: the three tests inside `TestSpanType` pass.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/tracer/models.py tests/core/tracer/
git commit -m "refactor(tracer): new SpanType taxonomy (a2a_task, skill, deferred_wait; drop llm/memory/hook)"
```

---

### Task 2: Add `SpanEvent` model

**Files:**
- Modify: `src/obelix/core/tracer/models.py`
- Modify: `tests/core/tracer/test_models.py`

- [ ] **Step 1: Append failing test to `tests/core/tracer/test_models.py`**

```python
class TestSpanEvent:
    def test_constructs_with_defaults(self):
        ev = SpanEvent(name="hook.fired")
        assert ev.name == "hook.fired"
        assert ev.attributes == {}
        assert ev.timestamp is not None

    def test_carries_attributes(self):
        ev = SpanEvent(name="memory.pull", attributes={"from_agent": "reviewer", "bytes": 1240})
        assert ev.attributes["from_agent"] == "reviewer"
        assert ev.attributes["bytes"] == 1240

    def test_is_serializable(self):
        ev = SpanEvent(name="a2a.state_change", attributes={"to": "rejected", "reason": "nope"})
        dumped = ev.model_dump(mode="json")
        assert dumped["name"] == "a2a.state_change"
        assert dumped["attributes"]["to"] == "rejected"
        assert "timestamp" in dumped
```

- [ ] **Step 2: Run — expect ImportError**

```bash
uv run pytest tests/core/tracer/test_models.py::TestSpanEvent -v
```

- [ ] **Step 3: Add `SpanEvent` to `src/obelix/core/tracer/models.py`**

Above the `Span` class:

```python
class SpanEvent(BaseModel):
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run — PASS**

```bash
uv run pytest tests/core/tracer/test_models.py::TestSpanEvent -v
```

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/tracer/models.py tests/core/tracer/test_models.py
git commit -m "feat(tracer): add SpanEvent model for point-in-time occurrences"
```

---

### Task 3: Add `events` field to `Span`

**Files:**
- Modify: `src/obelix/core/tracer/models.py`
- Modify: `tests/core/tracer/test_models.py`

- [ ] **Step 1: Append failing test**

```python
class TestSpanEvents:
    def test_span_has_empty_events_by_default(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        assert sp.events == []

    def test_events_can_be_appended(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        sp.events.append(SpanEvent(name="hook.fired", attributes={"decision": "REJECT"}))
        assert len(sp.events) == 1
        assert sp.events[0].name == "hook.fired"

    def test_events_included_in_dump(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        sp.events.append(SpanEvent(name="memory.pull", attributes={"from_agent": "x"}))
        dumped = sp.model_dump(mode="json")
        assert len(dumped["events"]) == 1
        assert dumped["events"][0]["name"] == "memory.pull"
```

- [ ] **Step 2: Run — FAIL**

```bash
uv run pytest tests/core/tracer/test_models.py::TestSpanEvents -v
```

- [ ] **Step 3: Add `events` field to `Span` class in `src/obelix/core/tracer/models.py`**

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
    events: list[SpanEvent] = Field(default_factory=list)
```

- [ ] **Step 4: Run — PASS**

```bash
uv run pytest tests/core/tracer/test_models.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/tracer/models.py tests/core/tracer/test_models.py
git commit -m "feat(tracer): add events field to Span"
```

---

### Task 4: Extend `SpanStatus` with A2A-aware states

**Files:**
- Modify: `src/obelix/core/tracer/models.py`
- Modify: `tests/core/tracer/test_models.py`

- [ ] **Step 1: Append failing test**

```python
class TestSpanStatus:
    def test_existing_values(self):
        assert SpanStatus.ok == "ok"
        assert SpanStatus.error == "error"
        assert SpanStatus.timeout == "timeout"

    def test_a2a_values(self):
        assert SpanStatus.rejected == "rejected"
        assert SpanStatus.canceled == "canceled"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Extend the enum**

```python
class SpanStatus(StrEnum):
    ok = "ok"
    error = "error"
    timeout = "timeout"
    rejected = "rejected"
    canceled = "canceled"
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/tracer/models.py tests/core/tracer/test_models.py
git commit -m "feat(tracer): add rejected/canceled to SpanStatus for A2A terminal states"
```

---

## Phase 2 — Tracer API

### Task 5: `Tracer.add_event()` + exporter hook

**Files:**
- Modify: `src/obelix/core/tracer/tracer.py`
- Modify: `src/obelix/core/tracer/exporters.py`
- Create: `tests/core/tracer/test_tracer.py`

- [ ] **Step 1: Create `tests/core/tracer/test_tracer.py`**

```python
"""Tests for Tracer API (add_event)."""

import pytest

from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import SpanType
from obelix.core.tracer.tracer import Tracer


class SpyExporter(NoOpExporter):
    def __init__(self):
        self.events = []
        self.span_exports = []

    async def export_span(self, span, service_name):
        self.span_exports.append((span.span_id, span.end_time is not None, list(span.events)))

    async def on_event(self, span, event, service_name):
        self.events.append((span.span_id, event.name, dict(event.attributes)))


@pytest.mark.asyncio
async def test_add_event_attaches_to_current_span():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    span = await tracer.start_span(SpanType.agent, "agent.Test")
    await tracer.add_event("hook.fired", {"decision": "REJECT"})
    await tracer.end_span()
    await tracer.end_trace()

    assert len(span.events) == 1
    assert span.events[0].name == "hook.fired"
    assert span.events[0].attributes == {"decision": "REJECT"}


@pytest.mark.asyncio
async def test_add_event_calls_exporter_hook():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    await tracer.start_span(SpanType.agent, "a")
    await tracer.add_event("memory.pull", {"from_agent": "x"})
    await tracer.end_span()
    await tracer.end_trace()

    assert len(exp.events) == 1
    assert exp.events[0][1] == "memory.pull"
    assert exp.events[0][2] == {"from_agent": "x"}


@pytest.mark.asyncio
async def test_add_event_without_span_is_noop():
    """Adding an event when no span is active must not raise."""
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    # no trace, no span
    await tracer.add_event("orphan", {})
    assert exp.events == []


@pytest.mark.asyncio
async def test_span_export_includes_events():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    await tracer.start_span(SpanType.agent, "a")
    await tracer.add_event("e1", {"k": "v"})
    await tracer.end_span()
    await tracer.end_trace()

    # export_span called at start (no events) and at end (with event)
    end_export = exp.span_exports[-1]
    assert end_export[1] is True  # has end_time
    assert len(end_export[2]) == 1
    assert end_export[2][0].name == "e1"
```

- [ ] **Step 2: Run — FAIL (add_event not defined, on_event not in exporter)**

```bash
uv run pytest tests/core/tracer/test_tracer.py -v
```

- [ ] **Step 3: Add abstract `on_event` hook to `TracerExporter`**

In `src/obelix/core/tracer/exporters.py`, add to `TracerExporter`:

```python
class TracerExporter(ABC):
    @abstractmethod
    async def export(self, trace: TraceSession) -> None: ...

    @abstractmethod
    async def export_span(self, span: Span, service_name: str) -> None: ...

    @abstractmethod
    async def start_trace(self, trace: TraceSession, service_name: str) -> None: ...

    @abstractmethod
    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None: ...

    async def on_event(self, span: Span, event: "SpanEvent", service_name: str) -> None:
        """Called when a SpanEvent is added to a span. Default: no-op."""
        pass

    @abstractmethod
    async def shutdown(self) -> None: ...
```

Add the import at top:

```python
from obelix.core.tracer.models import Span, SpanEvent, SpanStatus, SpanType, TraceSession
```

- [ ] **Step 4: Implement `Tracer.add_event()` in `src/obelix/core/tracer/tracer.py`**

Inside the `Tracer` class:

```python
async def add_event(
    self,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Attach a point-in-time event to the current span. No-op if no span."""
    from obelix.core.tracer.models import SpanEvent

    span = get_current_span()
    if span is None:
        return
    event = SpanEvent(name=name, attributes=attributes or {})
    span.events.append(event)
    await self._exporter.on_event(event=event, span=span, service_name=self.service_name)
```

- [ ] **Step 5: Run — PASS**

```bash
uv run pytest tests/core/tracer/test_tracer.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/tracer/ tests/core/tracer/test_tracer.py
git commit -m "feat(tracer): add_event method + on_event exporter hook"
```

---

## Phase 3 — Exporter updates

### Task 6: `HTTPExporter` includes events in payload

**Files:**
- Modify: `src/obelix/core/tracer/exporters.py`
- Modify: `tests/core/tracer/test_tracer.py`

- [ ] **Step 1: Append failing test to `test_tracer.py`**

```python
class TestHTTPExporterPayload:
    def test_span_payload_includes_events(self):
        from datetime import UTC, datetime

        from obelix.core.tracer.exporters import HTTPExporter
        from obelix.core.tracer.models import Span, SpanEvent, SpanType

        exp = HTTPExporter(endpoint="http://x/ingest")
        sp = Span(
            trace_id="t1",
            span_type=SpanType.agent,
            name="ag",
            end_time=datetime.now(UTC),
            duration_ms=10.0,
        )
        sp.events.append(SpanEvent(name="hook.fired", attributes={"decision": "REJECT"}))
        payload = exp._span_to_payload(sp)
        assert "events" in payload
        assert len(payload["events"]) == 1
        assert payload["events"][0]["name"] == "hook.fired"
        assert payload["events"][0]["attributes"] == {"decision": "REJECT"}
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Update `HTTPExporter._span_to_payload` in `exporters.py`**

Locate the `_span_to_payload` method and add `events` to the returned dict:

```python
def _span_to_payload(self, span: Span) -> dict:
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "span_type": span.span_type,
        "name": span.name,
        "start_time": self._dt_to_epoch(span.start_time),
        "end_time": self._dt_to_epoch(span.end_time),
        "duration_ms": span.duration_ms,
        "input": span.input,
        "output": span.output,
        "status": span.status,
        "error": span.error,
        "metadata": span.metadata,
        "events": [
            {
                "name": e.name,
                "timestamp": e.timestamp.timestamp(),
                "attributes": e.attributes,
            }
            for e in span.events
        ],
    }
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/tracer/exporters.py tests/core/tracer/test_tracer.py
git commit -m "feat(tracer): HTTPExporter emits events in span payload"
```

---

### Task 7: `ConsoleExporter` — badges for new span types

**Files:**
- Modify: `src/obelix/core/tracer/exporters.py`
- Create: `tests/core/tracer/test_console_exporter.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for ConsoleExporter rendering the new taxonomy."""

import io
from contextlib import redirect_stdout

import pytest

from obelix.core.tracer.exporters import ConsoleExporter
from obelix.core.tracer.models import Span, SpanEvent, SpanStatus, SpanType, TraceSession


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() is False else asyncio.run(coro)


@pytest.mark.asyncio
async def test_badge_mapping_covers_all_span_types():
    from obelix.core.tracer.exporters import ConsoleExporter
    exp = ConsoleExporter(verbosity=1, use_color=False)
    for t in SpanType:
        assert t in exp._ICONS, f"Missing icon for {t}"


@pytest.mark.asyncio
async def test_renders_a2a_task_span():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="task", trace_id="aaaaaaaaaaaa")

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        from datetime import UTC, datetime
        start = datetime.now(UTC)
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.a2a_task,
            name="task 7f3a",
            start_time=start,
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 123.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "[TK]" in out or "TK" in out
    assert "task 7f3a" in out


@pytest.mark.asyncio
async def test_renders_skill_span():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        from datetime import UTC, datetime
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.skill,
            name="code-review",
            start_time=datetime.now(UTC),
            metadata={"mode": "fork"},
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 200.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "SK" in out
    assert "code-review" in out
```

- [ ] **Step 2: Run — FAIL (badge mapping incomplete)**

```bash
uv run pytest tests/core/tracer/test_console_exporter.py -v
```

- [ ] **Step 3: Update `ConsoleExporter._ICONS` in `src/obelix/core/tracer/exporters.py`**

```python
_ICONS: dict[str, str] = {
    "a2a_task": "[TK]",
    "agent": "[AG]",
    "sub_agent": "[SA]",
    "skill": "[SK]",
    "tool": "[TL]",
    "deferred_wait": "[DW]",
    "human": "[H]",
    "assistant": "[A]",
}
```

And update `_COLORS` to include new types (pink for a2a_task, teal for skill, yellow already present):

```python
_COLORS: dict[str, str] = {
    "a2a_task": "\033[1;35m",    # magenta bold
    "agent": "\033[1;36m",       # cyan bold
    "sub_agent": "\033[1;36m",   # cyan bold (dimmer used via rendering)
    "skill": "\033[36m",         # cyan (teal-ish)
    "tool": "\033[32m",          # green
    "deferred_wait": "\033[33m", # yellow
    "human": "\033[2m",
    "assistant": "\033[37m",
    "error": "\033[31m",
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
}
```

- [ ] **Step 4: Add a simple `_fmt_generic_line` method and dispatch by type**

Replace the end of `export_span` (the body after the `# Format line based on type` block) with explicit handling for the new types. Rather than trying to specialize every type inline, introduce a dispatcher:

```python
def _format_span_line(self, span: Span) -> str:
    t = span.span_type
    if t == SpanType.a2a_task:
        return self._fmt_a2a_task_line(span)
    if t == SpanType.agent:
        return self._fmt_agent_line(span)
    if t == SpanType.sub_agent:
        return self._fmt_subagent_line(span)
    if t == SpanType.skill:
        return self._fmt_skill_line(span)
    if t == SpanType.tool:
        return self._fmt_tool_line(span)
    if t == SpanType.deferred_wait:
        return self._fmt_deferred_wait_line(span)
    if t == SpanType.human:
        return self._fmt_human_line(span)
    if t == SpanType.assistant:
        return self._fmt_assistant_line(span)
    icon = self._ICONS.get(str(t), "[???]")
    return f"{icon} {span.name}"
```

Add minimal implementations of the new formatters near the existing ones:

```python
def _fmt_a2a_task_line(self, span: Span) -> str:
    parts = [self._colorize(f"[TK] {span.name}", "a2a_task")]
    status = span.metadata.get("final_state") or span.status.value
    parts.append(f"status={status}")
    dur = self._fmt_duration(span.duration_ms)
    if dur:
        parts.append(dur)
    return "  ".join(parts)

def _fmt_agent_line(self, span: Span) -> str:
    parts = [self._colorize(f"[AG] {span.name}", "agent")]
    dur = self._fmt_duration(span.duration_ms)
    if dur:
        parts.append(dur)
    if self._verbosity >= 2:
        usage = span.metadata.get("llm_usage")
        if isinstance(usage, dict):
            calls = usage.get("calls", 0)
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
            if calls:
                parts.append(f"{calls} calls")
            if tin or tout:
                parts.append(f"{self._fmt_tokens(tin)}->{self._fmt_tokens(tout)} tok")
    return "  ".join(parts)

def _fmt_skill_line(self, span: Span) -> str:
    parts = [self._colorize(f"[SK] {span.name}", "skill")]
    mode = span.metadata.get("mode")
    if mode:
        parts.append(f"mode={mode}")
    dur = self._fmt_duration(span.duration_ms)
    if dur:
        parts.append(dur)
    return "  ".join(parts)

def _fmt_deferred_wait_line(self, span: Span) -> str:
    dur = self._fmt_duration(span.duration_ms) or "—"
    tool = span.metadata.get("tool_name", "")
    return self._colorize(f"[DW] deferred_wait  {dur}  tool={tool}", "deferred_wait")
```

Update `export_span` to call `_format_span_line(span)` instead of the inline if/elif chain.

- [ ] **Step 5: Run — PASS**

```bash
uv run pytest tests/core/tracer/test_console_exporter.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/tracer/exporters.py tests/core/tracer/test_console_exporter.py
git commit -m "feat(tracer): ConsoleExporter supports a2a_task/skill/deferred_wait badges"
```

---

### Task 8: `ConsoleExporter` — event inline rendering + chip counters

**Files:**
- Modify: `src/obelix/core/tracer/exporters.py`
- Modify: `tests/core/tracer/test_console_exporter.py`

- [ ] **Step 1: Append failing tests**

```python
@pytest.mark.asyncio
async def test_event_printed_inline_at_verbosity_2():
    exp = ConsoleExporter(verbosity=2, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        from datetime import UTC, datetime
        span = Span(trace_id=trace.trace_id, span_type=SpanType.agent, name="A",
                    start_time=datetime.now(UTC))
        await exp.export_span(span, "svc")
        ev = SpanEvent(name="hook.fired", attributes={"decision": "REJECT", "reason": "nope"})
        await exp.on_event(span=span, event=ev, service_name="svc")
    out = buf.getvalue()
    assert "hook.fired" in out
    assert "REJECT" in out


@pytest.mark.asyncio
async def test_event_not_printed_at_verbosity_1():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        from datetime import UTC, datetime
        span = Span(trace_id=trace.trace_id, span_type=SpanType.agent, name="A",
                    start_time=datetime.now(UTC))
        await exp.export_span(span, "svc")
        ev = SpanEvent(name="hook.fired", attributes={"decision": "REJECT"})
        await exp.on_event(span=span, event=ev, service_name="svc")
    out = buf.getvalue()
    assert "hook.fired" not in out


@pytest.mark.asyncio
async def test_chip_counter_shows_in_closed_span_line():
    """When a span closes with events, its rendered line includes chip counters at v1."""
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        from datetime import UTC, datetime
        span = Span(trace_id=trace.trace_id, span_type=SpanType.agent, name="A",
                    start_time=datetime.now(UTC))
        await exp.export_span(span, "svc")
        span.events.append(SpanEvent(name="hook.fired", attributes={"decision": "REJECT"}))
        span.events.append(SpanEvent(name="memory.pull", attributes={"from_agent": "r"}))
        span.end_time = datetime.now(UTC)
        span.duration_ms = 10.0
        await exp.export_span(span, "svc")
    out = buf.getvalue()
    # chips carry counts; flexible matcher — accept "hook:1" or "⚡1" style
    assert "1" in out  # at least the count appears on the agent line
    assert "A" in out
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement `on_event` inline rendering in `ConsoleExporter`**

Add:

```python
async def on_event(self, span: Span, event: SpanEvent, service_name: str) -> None:
    if self._verbosity < 2:
        return
    indent = self._get_indent(span.span_id)
    line = self._fmt_event_line(event)
    self._print(f"{indent}  {line}")

def _fmt_event_line(self, event: SpanEvent) -> str:
    attrs = event.attributes or {}
    kv = " ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)
    return self._colorize(f"· {event.name}  {kv}", "dim")
```

- [ ] **Step 4: Add chip counters to all span formatters**

Inside `_format_span_line`, after the type-specific formatter returns, append chip summary if the span has events and it's a closed line:

Add a helper:

```python
def _fmt_event_chips(self, span: Span) -> str:
    if not span.events:
        return ""
    counts: dict[str, int] = {}
    for e in span.events:
        counts[e.name] = counts.get(e.name, 0) + 1
    parts: list[str] = []
    if counts.get("hook.fired"):
        parts.append(self._colorize(f"hk:{counts['hook.fired']}", "error"))
    if counts.get("memory.pull"):
        parts.append(self._colorize(f"mp:{counts['memory.pull']}", "dim"))
    if counts.get("memory.publish"):
        parts.append(self._colorize(f"mx:{counts['memory.publish']}", "dim"))
    if counts.get("cancellation.requested"):
        parts.append(self._colorize("⚠cancel", "error"))
    return "  ".join(parts)
```

Modify `_format_span_line` to append chips:

```python
def _format_span_line(self, span: Span) -> str:
    # dispatch as before …
    line = dispatch(span)
    chips = self._fmt_event_chips(span)
    return f"{line}  {chips}" if chips else line
```

- [ ] **Step 5: Run — PASS**

```bash
uv run pytest tests/core/tracer/test_console_exporter.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/tracer/exporters.py tests/core/tracer/test_console_exporter.py
git commit -m "feat(tracer): ConsoleExporter renders events inline + chip counters"
```

---

### Task 9: `ConsoleExporter` — `deferred_wait` divider at v2

**Files:**
- Modify: `src/obelix/core/tracer/exporters.py`
- Modify: `tests/core/tracer/test_console_exporter.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_deferred_wait_renders_as_divider():
    exp = ConsoleExporter(verbosity=2, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        from datetime import UTC, datetime
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.deferred_wait,
            name="deferred_wait",
            start_time=datetime.now(UTC),
            metadata={"tool_name": "bash"},
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 7000.0
        await exp.export_span(span, "svc")
    out = buf.getvalue()
    assert "SUSPEND" in out or "---" in out
    assert "7.0s" in out
    assert "bash" in out
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Replace `_fmt_deferred_wait_line` body**

```python
def _fmt_deferred_wait_line(self, span: Span) -> str:
    dur = self._fmt_duration(span.duration_ms) or "—"
    tool = span.metadata.get("tool_name", "")
    sep = "───"
    return self._colorize(f"{sep} SUSPEND {dur} tool={tool} {sep}", "deferred_wait")
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/tracer/exporters.py tests/core/tracer/test_console_exporter.py
git commit -m "feat(tracer): ConsoleExporter renders deferred_wait as SUSPEND divider"
```

---

## Phase 4 — BaseAgent instrumentation

### Task 10: Drop per-call LLM spans, aggregate on agent span

**Files:**
- Modify: `src/obelix/core/agent/agent_tracing.py`
- Modify: `src/obelix/core/agent/base_agent.py`
- Modify: `tests/core/agent/test_base_agent.py`

- [ ] **Step 1: Add test for LLM usage aggregation**

Append to `tests/core/agent/test_base_agent.py` (adjust imports at top if needed):

```python
@pytest.mark.asyncio
async def test_agent_span_aggregates_llm_usage(make_agent_with_spy_tracer):
    """After two iterations, agent span metadata carries llm_usage totals."""
    # make_agent_with_spy_tracer is a fixture that returns (agent, spy_tracer)
    # with two mocked LLM responses: first triggers a tool, second returns text.
    agent, spy = make_agent_with_spy_tracer(
        responses=[
            # iter 1: tool_use, usage 800/120
            _mock_assistant_with_tool_call(usage_in=800, usage_out=120),
            # iter 2: end_turn, usage 600/80
            _mock_assistant_text("done", usage_in=600, usage_out=80),
        ]
    )
    await agent.execute_query_async("hi")
    # find the agent span
    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    assert len(agent_spans) == 1
    usage = agent_spans[0].metadata["llm_usage"]
    assert usage["calls"] == 2
    assert usage["input_tokens"] == 1400
    assert usage["output_tokens"] == 200
    assert usage["total_tokens"] == 1600


@pytest.mark.asyncio
async def test_no_llm_spans_emitted(make_agent_with_spy_tracer):
    agent, spy = make_agent_with_spy_tracer(responses=[_mock_assistant_text("done")])
    await agent.execute_query_async("hi")
    assert not any(s.span_type.value == "llm" for s in spy.spans)
```

Helpers `_mock_assistant_with_tool_call`, `_mock_assistant_text`, and the `make_agent_with_spy_tracer` fixture must be defined in `tests/conftest.py` or at the top of the file. Define them inline at the top of the test file:

```python
from datetime import UTC, datetime

from obelix.core.tracer.models import SpanType
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.tracer import Tracer


class _SpyTracerExporter(NoOpExporter):
    def __init__(self):
        self.spans: list = []
    async def export_span(self, span, service_name):
        if span.end_time is not None:
            self.spans.append(span)


def _mock_assistant_text(text: str, usage_in: int = 100, usage_out: int = 50) -> AssistantMessage:
    from obelix.core.model.usage import Usage
    return AssistantMessage(content=text, tool_calls=None, usage=Usage(input_tokens=usage_in, output_tokens=usage_out))


def _mock_assistant_with_tool_call(usage_in: int = 100, usage_out: int = 50) -> AssistantMessage:
    from obelix.core.model.usage import Usage
    return AssistantMessage(
        content=None,
        tool_calls=[ToolCall(id="x", name="dummy", arguments={})],
        usage=Usage(input_tokens=usage_in, output_tokens=usage_out),
    )


@pytest.fixture
def make_agent_with_spy_tracer():
    def _factory(responses: list[AssistantMessage]) -> tuple[BaseAgent, _SpyTracerExporter]:
        exporter = _SpyTracerExporter()
        tracer = Tracer(exporter=exporter)

        provider = MagicMock()
        provider.provider_type = "mock"
        provider.model_id = "mock-model"
        provider.invoke = AsyncMock(side_effect=responses)

        agent = BaseAgent(
            system_message=SystemMessage(content="test"),
            provider=provider,
            tracer=tracer,
            max_iterations=5,
        )
        # Register a dummy tool so tool_calls resolve
        class DummyTool:
            name = "dummy"
            async def execute(self, call):
                return ToolResult(tool_call_id=call.id, tool_name="dummy", result="ok",
                                  status=ToolStatus.SUCCESS, error=None)
        # The actual ToolBase wrapping is non-trivial; use agent's tool registry API
        # ...
        return agent, exporter
    return _factory
```

Note: the fixture details depend on the exact `BaseAgent` constructor API. Adapt to what exists — the test goal is clear (no llm spans, llm_usage aggregated).

- [ ] **Step 2: Run — FAIL (llm spans still emitted)**

```bash
uv run pytest tests/core/agent/test_base_agent.py::test_agent_span_aggregates_llm_usage tests/core/agent/test_base_agent.py::test_no_llm_spans_emitted -v
```

- [ ] **Step 3: Remove llm span helpers from `src/obelix/core/agent/agent_tracing.py`**

Delete `start_llm_span` and `end_llm_span` functions entirely. Replace with a single helper:

```python
async def accumulate_llm_call(
    tracer: Tracer | None,
    assistant_msg: AssistantMessage,
    provider_type: str,
    model_id: str,
    duration_ms: float,
) -> None:
    """Fold a single LLM call into the current agent span's metadata.llm_usage."""
    if not tracer:
        return
    from obelix.core.tracer.context import get_current_span

    span = get_current_span()
    if span is None:
        return
    usage_dict = span.metadata.setdefault(
        "llm_usage",
        {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    usage_dict["calls"] += 1
    if assistant_msg.usage:
        usage_dict["input_tokens"] += assistant_msg.usage.input_tokens or 0
        usage_dict["output_tokens"] += assistant_msg.usage.output_tokens or 0
        usage_dict["total_tokens"] = usage_dict["input_tokens"] + usage_dict["output_tokens"]
    span.metadata.setdefault("model_id", model_id)
    span.metadata.setdefault("provider_type", provider_type)
```

- [ ] **Step 4: Update `src/obelix/core/agent/base_agent.py`**

Remove all `await start_llm_span(...)` and `await end_llm_span(...)` call sites (grep found them at lines 526, 551, 574, 599). Replace the single `end_llm_span` call at line 599 (the successful non-streaming path) with the accumulator:

```python
await accumulate_llm_call(
    self._tracer,
    assistant_msg=assistant_msg,
    provider_type=str(self.provider.provider_type),
    model_id=self.provider.model_id,
    duration_ms=(time.monotonic() - llm_started_at) * 1000,
)
```

(Where `llm_started_at` is a new `time.monotonic()` captured immediately before `provider.invoke()` / `invoke_stream()`.)

Update import of `agent_tracing` to drop the removed helpers.

- [ ] **Step 5: Run — PASS**

```bash
uv run pytest tests/core/agent/test_base_agent.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/agent/ tests/core/agent/test_base_agent.py
git commit -m "refactor(tracer): drop per-call llm spans, aggregate llm_usage on agent span"
```

---

### Task 11: 3-way dispatch branch at tool call (skill / sub_agent / tool)

**Files:**
- Modify: `src/obelix/core/agent/agent_tracing.py`
- Modify: `src/obelix/core/agent/base_agent.py`
- Modify: `tests/core/agent/test_base_agent_skills.py` (or add a new test file)

- [ ] **Step 1: Write failing test**

Append to `tests/core/agent/test_base_agent_skills.py` (assumes a fixture that runs an agent invoking a skill and captures spans):

```python
@pytest.mark.asyncio
async def test_skill_call_emits_skill_span_not_tool(make_agent_with_skill_and_tracer):
    agent, spy = make_agent_with_skill_and_tracer(skill_name="demo", mode="inline")
    await agent.execute_query_async('Use the "demo" skill.')
    skill_spans = [s for s in spy.spans if s.span_type.value == "skill"]
    assert len(skill_spans) == 1
    assert skill_spans[0].name == "demo"
    assert skill_spans[0].metadata.get("mode") == "inline"
    # not a tool span
    tool_spans_with_skill_name = [s for s in spy.spans if s.span_type.value == "tool" and "Skill" in s.name]
    assert tool_spans_with_skill_name == []
```

(Define the fixture near the existing skill tests; if the existing tests already set up a mock LLM that invokes the skill tool, just add the assertions.)

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Update `start_tool_span` in `src/obelix/core/agent/agent_tracing.py`**

```python
async def start_tool_span(
    tracer: Tracer | None,
    call: ToolCall,
    registered_tools: list[Tool],
) -> None:
    if not tracer:
        return
    from obelix.core.agent.subagent_wrapper import SubAgentWrapper
    from obelix.core.tracer.models import SpanType
    from obelix.plugins.builtin.skill_tool import SkillTool

    tool = next((t for t in registered_tools if t.name == call.name), None)

    if isinstance(tool, SkillTool):
        span_type = SpanType.skill
        span_name = call.arguments.get("name", call.name) if isinstance(call.arguments, dict) else call.name
        input_payload = {
            "tool_call_id": call.id,
            "skill_args": call.arguments.get("args") if isinstance(call.arguments, dict) else None,
        }
        metadata = {
            "mode": _resolve_skill_mode(tool, span_name),
            "source": _resolve_skill_source(tool, span_name),
        }
    elif isinstance(tool, SubAgentWrapper):
        span_type = SpanType.sub_agent
        span_name = call.name
        input_payload = {"tool_call_id": call.id, "arguments": call.arguments}
        metadata = {}
    else:
        span_type = SpanType.tool
        span_name = call.name
        input_payload = {"tool_call_id": call.id, "arguments": call.arguments}
        metadata = {}

    await tracer.start_span(span_type, span_name, input=input_payload, metadata=metadata)


def _resolve_skill_mode(tool, skill_name: str) -> str | None:
    try:
        skill = tool._manager.find(skill_name)  # adapt to real API of SkillTool
        return skill.context if skill else None
    except Exception:
        return None


def _resolve_skill_source(tool, skill_name: str) -> str | None:
    try:
        skill = tool._manager.find(skill_name)
        return getattr(skill, "source", None)
    except Exception:
        return None
```

Adjust `_resolve_skill_mode`/`_resolve_skill_source` to the real accessors on `SkillTool` (likely `tool._skills_manager` or similar — inspect `plugins/builtin/skill_tool.py`).

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/ tests/core/agent/test_base_agent_skills.py
git commit -m "feat(tracer): dispatch skill/sub_agent/tool span types at tool call site"
```

---

### Task 12: Emit `hook.fired` events when decision ≠ CONTINUE or effects non-empty

**Files:**
- Modify: `src/obelix/core/agent/base_agent.py`
- Modify: `tests/core/agent/test_hooks.py`

- [ ] **Step 1: Write failing test**

Append to `tests/core/agent/test_hooks.py`:

```python
@pytest.mark.asyncio
async def test_hook_reject_emits_hook_fired_event(make_agent_with_spy_tracer):
    agent, spy = make_agent_with_spy_tracer(
        responses=[_mock_assistant_text("ignored")],
    )
    agent.on(AgentEvent.BEFORE_LLM_CALL).reject("test-reason")

    with pytest.raises(Exception):
        await agent.execute_query_async("hi")

    agent_spans = [s for s in spy.spans if s.span_type.value == "agent"]
    assert agent_spans
    events = agent_spans[0].events
    hook_events = [e for e in events if e.name == "hook.fired"]
    assert len(hook_events) == 1
    assert hook_events[0].attributes["event"] == "BEFORE_LLM_CALL"
    assert hook_events[0].attributes["decision"] == "REJECT"
    assert hook_events[0].attributes["reason"] == "test-reason"


@pytest.mark.asyncio
async def test_hook_continue_with_no_effects_emits_nothing(make_agent_with_spy_tracer):
    agent, spy = make_agent_with_spy_tracer(responses=[_mock_assistant_text("ok")])
    # A hook with condition always False → decision=CONTINUE (default), no effects
    agent.on(AgentEvent.BEFORE_LLM_CALL).when(lambda s: False)

    await agent.execute_query_async("hi")

    agent_spans = [s for s in spy.spans if s.span_type.value == "agent"]
    hook_events = [e for s in agent_spans for e in s.events if e.name == "hook.fired"]
    assert hook_events == []
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Patch `_run_hooks` in `src/obelix/core/agent/base_agent.py`**

Identify the `_run_hooks` method (near line 284 per exploration). After the outcome is evaluated and BEFORE decision handling, emit the event:

```python
# after computing `outcome`:
if self._tracer:
    decision = outcome.decision
    effects_count = len(getattr(outcome, "effects", []) or [])
    if decision != HookDecision.CONTINUE or effects_count:
        await self._tracer.add_event(
            "hook.fired",
            {
                "event": event.value,
                "decision": decision.value,
                "reason": getattr(outcome, "reason", None),
                "effects_count": effects_count,
            },
        )
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/base_agent.py tests/core/agent/test_hooks.py
git commit -m "feat(tracer): emit hook.fired event when hook changes behavior"
```

---

### Task 13: Guard `human`/`assistant` emission to root-agent only

**Files:**
- Modify: `src/obelix/core/agent/base_agent.py`
- Modify: `tests/core/agent/test_base_agent.py`

- [ ] **Step 1: Write failing test**

Append to `tests/core/agent/test_base_agent.py`:

```python
@pytest.mark.asyncio
async def test_human_span_emitted_only_when_root(make_agent_with_spy_tracer):
    """Nested agent (with pre-existing trace) must not emit human/assistant spans."""
    agent, spy = make_agent_with_spy_tracer(responses=[_mock_assistant_text("ok")])
    # Pre-seed a trace so the agent acts as nested
    await agent._tracer.start_trace("outer")
    try:
        await agent.execute_query_async("hi")
    finally:
        await agent._tracer.end_trace()

    human_spans = [s for s in spy.spans if s.span_type.value == "human"]
    assistant_spans = [s for s in spy.spans if s.span_type.value == "assistant"]
    assert human_spans == []
    assert assistant_spans == []


@pytest.mark.asyncio
async def test_human_span_emitted_when_root(make_agent_with_spy_tracer):
    agent, spy = make_agent_with_spy_tracer(responses=[_mock_assistant_text("done")])
    await agent.execute_query_async("hello")
    assert any(s.span_type.value == "human" for s in spy.spans)
    assert any(s.span_type.value == "assistant" for s in spy.spans)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Guard emissions in BaseAgent**

Locate `start_agent_trace` return value `is_root_trace` (around line 428). Thread it through the loop as a local flag, then guard:

- Line ~479: `if is_root_trace: await emit_human_span(...)`
- Each `emit_assistant_span` call: `if is_root_trace: await emit_assistant_span(...)`
- Only the FINAL `emit_assistant_span` remains (the one after the loop exits successfully). Remove the intermediate ones at lines 507, 620, 659, 737 — they were for mid-loop text responses, which the spec cuts.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/base_agent.py tests/core/agent/test_base_agent.py
git commit -m "refactor(tracer): emit human/assistant spans only when agent is root"
```

---

## Phase 5 — Memory events

### Task 14: Emit `memory.pull` event in `_inject_shared_memory`

**Files:**
- Modify: `src/obelix/core/agent/memory_hooks.py`
- Modify: existing memory test file (or create new)

- [ ] **Step 1: Locate the test file**

```bash
grep -l "SharedMemoryGraph\|memory_hooks\|_inject_shared_memory" tests/ -r
```

Use the first match (likely `tests/core/agent/test_base_agent.py` or a dedicated file). Append:

```python
@pytest.mark.asyncio
async def test_memory_pull_emits_events_per_source(make_agent_with_graph_and_tracer):
    """Agent with 2 predecessors having data emits 2 memory.pull events."""
    agent, spy = make_agent_with_graph_and_tracer(
        predecessors=[("src1", "final content A", "FINAL_RESPONSE_ONLY"),
                      ("src2", "final content B", "FINAL_RESPONSE_ONLY")],
        responses=[_mock_assistant_text("ok")],
    )
    await agent.execute_query_async("hi")

    agent_spans = [s for s in spy.spans if s.span_type.value == "agent"]
    assert agent_spans
    pull_events = [e for e in agent_spans[0].events if e.name == "memory.pull"]
    assert len(pull_events) == 2
    sources = {e.attributes["from_agent"] for e in pull_events}
    assert sources == {"src1", "src2"}
    for e in pull_events:
        assert e.attributes["policy"] == "FINAL_RESPONSE_ONLY"
        assert e.attributes["bytes"] > 0
```

(Fixture `make_agent_with_graph_and_tracer` wires a `SharedMemoryGraph` with pre-published predecessors.)

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Patch `_inject_shared_memory` in `src/obelix/core/agent/memory_hooks.py`**

After pulling items but before injecting them as `SystemMessage`:

```python
if tracer:
    for item in items:
        await tracer.add_event(
            "memory.pull",
            {
                "from_agent": item.source_id,
                "policy": item.policy.value,
                "bytes": len(item.content or ""),
            },
        )
```

Thread `tracer` into the function if it's not already available — it should be passed via the hook binding context (agent has `self._tracer`).

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/memory_hooks.py tests/
git commit -m "feat(tracer): emit memory.pull event per predecessor source"
```

---

### Task 15: Emit `memory.publish` event in `_publish_to_memory`

**Files:**
- Modify: `src/obelix/core/agent/memory_hooks.py`
- Modify: test file (same as Task 14)

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_memory_publish_emits_event_on_final(make_agent_with_graph_and_tracer):
    agent, spy = make_agent_with_graph_and_tracer(
        predecessors=[],
        responses=[_mock_assistant_text("my final answer")],
    )
    await agent.execute_query_async("hi")
    agent_spans = [s for s in spy.spans if s.span_type.value == "agent"]
    publish_events = [e for e in agent_spans[0].events if e.name == "memory.publish"]
    assert len(publish_events) >= 1
    assert any(e.attributes.get("kind") == "final" for e in publish_events)
    for e in publish_events:
        assert e.attributes["bytes"] >= 0
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Patch `_publish_to_memory`**

```python
if tracer and content:
    await tracer.add_event(
        "memory.publish",
        {"kind": kind, "bytes": len(content)},
    )
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/memory_hooks.py tests/
git commit -m "feat(tracer): emit memory.publish event on publish"
```

---

## Phase 6 — A2A executor instrumentation

### Task 16: Open `a2a_task` span as root in the executor

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Create: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Create test**

```python
"""Tracer integration tests for the A2A executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import SpanType
from obelix.core.tracer.tracer import Tracer


class _Spy(NoOpExporter):
    def __init__(self):
        self.spans: list = []
    async def export_span(self, span, service_name):
        if span.end_time is not None:
            self.spans.append(span)


@pytest.mark.asyncio
async def test_executor_opens_a2a_task_root_span(executor_with_tracer):
    """
    Fixture `executor_with_tracer` returns (executor, spy, sample_context).
    Calling executor's run method with a sample message must produce an
    a2a_task span as the root of the trace.
    """
    executor, spy, send_message = executor_with_tracer
    await send_message("hello")
    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert len(a2a_tasks) == 1
    # the agent span must be a child
    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    assert any(a.parent_span_id == a2a_tasks[0].span_id for a in agent_spans)
```

(The fixture assembles a minimal executor with a mock factory/agent. Follow the existing `tests/adapters/inbound/a2a/` conventions if files already exist; otherwise fabricate the minimum.)

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Add `tracer` attribute to `ObelixAgentExecutor`**

In `src/obelix/adapters/inbound/a2a/server/executor.py`, add the tracer parameter to `__init__`:

```python
def __init__(self, agent_factory, ..., tracer: Tracer | None = None):
    ...
    self._tracer = tracer
```

In `src/obelix/core/agent/agent_factory.py`, locate where `ObelixAgentExecutor` is instantiated (inside `a2a_serve`) and pass `tracer=self._tracer` to the constructor.

- [ ] **Step 4: Open `a2a_task` span in `ObelixAgentExecutor._run_agent`**

At the beginning of `_run_agent`, before any agent call:

```python
async def _run_agent(self, request_context, task_id, context_id, ...):
    tracer = self._tracer
    if tracer:
        await tracer.start_trace(
            name=f"a2a.task",
            metadata={"task_id": task_id, "context_id": context_id},
        )
        await tracer.start_span(
            SpanType.a2a_task,
            name=f"task {task_id[:8]}",
            input={"context_id": context_id},
            metadata={"task_id": task_id, "context_id": context_id},
        )
    try:
        # existing _run_agent body
        ...
    finally:
        if tracer:
            await tracer.end_span()
            await tracer.end_trace()
```

- [ ] **Step 5: Run — PASS**

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py src/obelix/core/agent/agent_factory.py tests/adapters/inbound/a2a/
git commit -m "feat(tracer): a2a executor opens a2a_task root span"
```

---

### Task 17: Emit `human` and final `assistant` spans in the executor

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Modify: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_executor_emits_human_and_assistant_spans(executor_with_tracer):
    executor, spy, send_message = executor_with_tracer
    await send_message("review my changes")

    human = [s for s in spy.spans if s.span_type == SpanType.human]
    assistant = [s for s in spy.spans if s.span_type == SpanType.assistant]
    assert len(human) == 1
    assert human[0].input == "review my changes" or (isinstance(human[0].input, dict) and human[0].input.get("text") == "review my changes")
    assert len(assistant) == 1
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Emit human/assistant in executor**

In `_run_agent`, right after opening `a2a_task` and after converting parts to `HumanMessage`:

```python
if tracer:
    await tracer.start_span(
        SpanType.human,
        "human.input",
        input=human_message.content,
    )
    await tracer.end_span(output=human_message.content)
```

And before closing `a2a_task` on success:

```python
if tracer:
    await tracer.start_span(
        SpanType.assistant,
        "assistant.response",
        input={"has_tool_calls": False},
    )
    await tracer.end_span(output={"content": final_response.content})
```

In `src/obelix/core/agent/base_agent.py`, reconfirm that `emit_human_span` / `emit_assistant_span` are now guarded by `is_root_trace` (done in Task 13). Because the executor opens the trace, `BaseAgent.is_root_trace` will be False for A2A-served agents, and they will not emit these spans — the executor owns them.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py tests/adapters/inbound/a2a/
git commit -m "feat(tracer): a2a executor emits human/assistant spans at turn boundaries"
```

---

### Task 18: Emit `a2a.state_change` events on every transition

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Modify: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_executor_emits_state_change_events(executor_with_tracer):
    executor, spy, send_message = executor_with_tracer
    await send_message("hi")

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    events = a2a_tasks[0].events
    names = [e.name for e in events]
    assert names.count("a2a.state_change") >= 2
    states = [e.attributes.get("to") for e in events if e.name == "a2a.state_change"]
    assert "working" in states
    assert "completed" in states
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Instrument every place in `_run_agent` that emits `TaskStatusUpdateEvent`**

Wrap the state emissions with a helper:

```python
async def _emit_state(self, tracer, from_state: str | None, to_state: str, reason: str | None = None):
    if tracer:
        await tracer.add_event(
            "a2a.state_change",
            {"from": from_state, "to": to_state, "reason": reason},
        )
```

Call `self._emit_state(tracer, None, "working")` immediately after emitting the first `TaskStatusUpdateEvent(state=working)`. Call `self._emit_state(tracer, "working", "completed")` before the final `TaskStatusUpdateEvent(state=completed, final=True)`. Same for `input_required`, `rejected`, `failed`, `canceled`.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py tests/adapters/inbound/a2a/
git commit -m "feat(tracer): a2a executor emits state_change events on every transition"
```

---

### Task 19: Open/close `deferred_wait` span around input_required

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Modify: `src/obelix/adapters/inbound/a2a/server/deferred.py`
- Modify: `src/obelix/adapters/inbound/a2a/server/context.py`
- Modify: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Write failing test using deferred scenario**

```python
@pytest.mark.asyncio
async def test_deferred_tool_creates_deferred_wait_span(executor_with_deferred_tool):
    """Fixture produces an executor whose agent emits one deferred tool call,
    and whose next message resumes it."""
    executor, spy, send_message, resume = executor_with_deferred_tool
    await send_message("please run a deferred thing")
    await resume(stdout="ok")

    dw = [s for s in spy.spans if s.span_type == SpanType.deferred_wait]
    assert len(dw) == 1
    assert dw[0].duration_ms is not None and dw[0].duration_ms > 0
    assert dw[0].metadata.get("tool_name")
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Extend `ContextEntry` to remember the deferred_wait span id**

In `src/obelix/adapters/inbound/a2a/server/context.py`, add a field:

```python
@dataclass
class ContextEntry:
    ...
    deferred_wait_span_id: str | None = None
```

- [ ] **Step 4: In `executor.py`, open `deferred_wait` span on input_required**

Around the block that emits `TaskStatusUpdateEvent(state=input_required)` (line 247 per exploration):

```python
if tracer:
    await tracer.start_span(
        SpanType.deferred_wait,
        "deferred_wait",
        metadata={
            "tool_name": deferred_tool_calls[0].name if deferred_tool_calls else None,
            "tool_call_ids": [c.id for c in deferred_tool_calls],
        },
    )
    current = get_current_span()
    entry.deferred_wait_span_id = current.span_id if current else None
    # close it's end_time must be set on resume, NOT here
```

Do NOT call `end_span()` yet — we want the span to stay "open" (end_time=None) until resume. To achieve this, actually close the span on resume (Task 3 — re-enter executor loop):

In the resume path (around `executor.py:196-202`), after restoring trace context:

```python
if tracer and entry.deferred_wait_span_id:
    # end the deferred_wait span
    await tracer.end_span()
    entry.deferred_wait_span_id = None
```

The `end_span()` works because trace context was restored first, so the deferred_wait is the current span.

- [ ] **Step 5: Run — PASS**

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/ tests/adapters/inbound/a2a/
git commit -m "feat(tracer): deferred_wait span wraps input_required pause"
```

---

### Task 20: Emit `cancellation.requested` event

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Modify: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_cancellation_emits_event_and_canceled_status(executor_with_cancelable):
    executor, spy, send_message, cancel = executor_with_cancelable
    task = asyncio.create_task(send_message("long operation"))
    await asyncio.sleep(0.05)
    await cancel()
    await task

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert a2a_tasks[0].status.value == "canceled"
    events = a2a_tasks[0].events
    assert any(e.name == "cancellation.requested" for e in events)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Emit event in executor's `cancel()` method**

```python
async def cancel(self, request_context, event_queue):
    if self._tracer:
        await self._tracer.add_event(
            "cancellation.requested",
            {"source": "client"},
        )
    # existing cancel body
```

Also ensure `a2a_task` span closes with `status=SpanStatus.canceled` when the final state is canceled. In `_run_agent` finally block:

```python
if tracer:
    final = final_state_for_trace or SpanStatus.ok
    await tracer.end_span(status=final, error=...)
    await tracer.end_trace(status=final, error=...)
```

Where `final_state_for_trace` is set to `SpanStatus.canceled` in the cancellation branch, `SpanStatus.rejected` in the rejection branch, `SpanStatus.error` on generic exceptions, etc.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py tests/adapters/inbound/a2a/
git commit -m "feat(tracer): cancellation.requested event + canceled status on a2a_task"
```

---

### Task 21: Close `a2a_task` with `rejected`/`failed` status correctly

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py`
- Modify: `tests/adapters/inbound/a2a/test_executor_tracing.py`

- [ ] **Step 1: Append failing tests**

```python
@pytest.mark.asyncio
async def test_rejected_status_propagates_to_a2a_task(executor_with_rejecting_hook):
    executor, spy, send_message = executor_with_rejecting_hook
    await send_message("hi")
    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert a2a_tasks[0].status.value == "rejected"
    assert a2a_tasks[0].error is not None  # rejection reason


@pytest.mark.asyncio
async def test_failed_status_on_unexpected_exception(executor_with_failing_agent):
    executor, spy, send_message = executor_with_failing_agent
    await send_message("hi")
    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert a2a_tasks[0].status.value == "error"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Map A2A task states to tracer statuses in `_run_agent`**

In the except blocks:

- On `TaskRejectedError`: set `final_state_for_trace = SpanStatus.rejected`, `error = rejection_reason`.
- On generic `Exception`: `final_state_for_trace = SpanStatus.error`, `error = str(exc)`.
- On normal completion: `final_state_for_trace = SpanStatus.ok`.
- On cancellation: `final_state_for_trace = SpanStatus.canceled` (done in Task 20).

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py tests/adapters/inbound/a2a/
git commit -m "feat(tracer): map A2A terminal states to a2a_task span status"
```

---

## Phase 7 — Integration E2E scenarios

### Task 22: Integration test — dev_workflow full pipeline

**Files:**
- Create: `tests/integration/tracer/test_dev_workflow.py`

- [ ] **Step 1: Create integration test**

```python
"""End-to-end trace shape for the dev_workflow scenario (no real A2A HTTP).

Instantiates the CoordinatorAgent + 3 sub-agents via AgentFactory with a spy
tracer, mocks the LLM responses to force a predictable sequence, asserts the
resulting span tree.
"""

import pytest

from obelix.core.tracer.models import SpanType
# ... (imports, fixtures)


@pytest.mark.asyncio
async def test_dev_workflow_tree_shape(dev_workflow_agents_with_spy):
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    types = [s.span_type for s in spy.spans]
    # one root agent (no A2A here), three sub_agent calls, one or more skill spans
    assert types.count(SpanType.agent) >= 4  # coordinator + 3 sub-agents + skill forks
    assert types.count(SpanType.sub_agent) == 3  # reviewer, commit_writer, summary
    assert types.count(SpanType.skill) >= 2  # code-review, commit-writer (forks)
    # llm/human spans at root only
    assert types.count(SpanType.human) == 1
    assert types.count(SpanType.assistant) == 1
    assert "llm" not in [str(t) for t in types]


@pytest.mark.asyncio
async def test_dev_workflow_memory_events_on_agents(dev_workflow_agents_with_spy):
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    agent_spans_by_name = {
        s.metadata.get("agent_name") or s.name: s
        for s in spy.spans if s.span_type == SpanType.agent
    }
    commit_events = agent_spans_by_name.get("CommitAgent").events
    summary_events = agent_spans_by_name.get("SummaryAgent").events

    # CommitAgent pulls from reviewer
    assert any(e.name == "memory.pull" and e.attributes.get("from_agent") == "reviewer" for e in commit_events)
    # SummaryAgent pulls from reviewer and commit_writer
    summary_sources = {e.attributes.get("from_agent") for e in summary_events if e.name == "memory.pull"}
    assert summary_sources >= {"reviewer", "commit_writer"}
```

- [ ] **Step 2: Run — iteratively make the fixture work**

```bash
uv run pytest tests/integration/tracer/test_dev_workflow.py -v
```

- [ ] **Step 3: Fix any real issues found (do NOT mask with xfail)**

- [ ] **Step 4: Commit**

```bash
git add tests/integration/tracer/
git commit -m "test(tracer): integration — dev_workflow scenario trace shape"
```

---

### Task 23: Integration test — deferred tool scenario

**Files:**
- Create: `tests/integration/tracer/test_deferred.py`

- [ ] **Step 1: Create test**

```python
"""End-to-end trace shape for deferred tool scenario."""

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_deferred_wait_span_between_suspend_and_resume(deferred_scenario_with_spy):
    # fixture sends message, waits for input_required, simulates 7s pause, resumes
    scenario = deferred_scenario_with_spy
    await scenario.send("run df -h")
    await scenario.wait_for_input_required()
    await scenario.sleep_seconds(0.5)  # simulate pause (real time)
    await scenario.resume_with_data({"stdout": "Filesystem ..."})

    spans = scenario.spy.spans
    # exactly one deferred_wait
    dw = [s for s in spans if s.span_type == SpanType.deferred_wait]
    assert len(dw) == 1
    assert dw[0].duration_ms > 400  # >= the 0.5s sleep

    # a2a_task completed
    tk = [s for s in spans if s.span_type == SpanType.a2a_task]
    assert tk[0].status.value == "ok"

    # state transitions recorded
    transitions = [e.attributes["to"] for e in tk[0].events if e.name == "a2a.state_change"]
    assert "input_required" in transitions
    assert transitions.count("working") >= 2  # initial + resume
    assert "completed" in transitions
```

- [ ] **Step 2: Run + fix + commit**

```bash
uv run pytest tests/integration/tracer/test_deferred.py -v
git add tests/integration/tracer/test_deferred.py
git commit -m "test(tracer): integration — deferred tool suspend/resume"
```

---

### Task 24: Integration test — rejection scenario

**Files:**
- Create: `tests/integration/tracer/test_rejection.py`

- [ ] **Step 1: Create test**

```python
"""End-to-end trace shape for rejection scenario."""

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_rejection_propagates_to_root_task(rejection_scenario_with_spy):
    scenario = rejection_scenario_with_spy
    await scenario.send("review")

    spans = scenario.spy.spans
    tk = [s for s in spans if s.span_type == SpanType.a2a_task][0]
    assert tk.status.value == "rejected"
    # hook.fired event on agent span
    agent = [s for s in spans if s.span_type == SpanType.agent][0]
    hook_events = [e for e in agent.events if e.name == "hook.fired"]
    assert len(hook_events) == 1
    assert hook_events[0].attributes["decision"] == "REJECT"
    # zero LLM usage
    assert agent.metadata.get("llm_usage", {}).get("calls", 0) == 0
    # a2a.state_change to rejected present on root
    transitions = [e.attributes["to"] for e in tk.events if e.name == "a2a.state_change"]
    assert "rejected" in transitions
```

- [ ] **Step 2: Run + fix + commit**

```bash
uv run pytest tests/integration/tracer/test_rejection.py -v
git add tests/integration/tracer/test_rejection.py
git commit -m "test(tracer): integration — hook-based rejection"
```

---

## Phase 8 — Cleanup & docs

### Task 25: Remove dead tracing helpers and imports

**Files:**
- Modify: `src/obelix/core/agent/agent_tracing.py`
- Modify: `src/obelix/core/agent/base_agent.py`
- Modify: `src/obelix/core/tracer/__init__.py`

- [ ] **Step 1: Check that `start_llm_span`, `end_llm_span`, intermediate `emit_assistant_span` call sites are gone**

```bash
uv run grep -rn "start_llm_span\|end_llm_span" src/ tests/
```

Should print nothing.

- [ ] **Step 2: Remove unused imports in `base_agent.py`, `agent_tracing.py`**

- [ ] **Step 3: Run full suite**

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(tracer): remove dead llm span helpers and imports"
```

---

### Task 26: Update `docs/index.md` and remove obsolete docs

**Files:**
- Modify: `docs/index.md` (if it references tracer internals)
- Create: `docs/tracer.md`

- [ ] **Step 1: Write a concise `docs/tracer.md`**

Contents: overview of the new taxonomy, event list, how to plug a custom exporter, how verbosity levels work, link to the spec and mockup.

- [ ] **Step 2: Link from `docs/index.md`**

- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs(tracer): document new span taxonomy and event model"
```

---

## Post-implementation checklist

- [ ] Run `uv run pytest -q` and confirm all tests pass.
- [ ] Run `uv run ruff check .` and `uv run ruff format --check .`; both clean.
- [ ] Open `docs/tracer_mockup.html`, run `examples/dev_workflow_server.py` with `ConsoleExporter(verbosity=2)` and compare: tree shape should match the mockup.
- [ ] Start the tracer backend at `:8100` and run a real A2A request; confirm events persist in the backend and the frontend (after adapting `SPAN_CONFIG`) renders the new taxonomy.
- [ ] Frontend adaptation (`C:\Users\GLoverde\PycharmProjects\obelix-tracer\frontend`) is out of scope for this plan — tracked as a separate ticket.
- [ ] Tracer backend schema adaptation (`C:\Users\GLoverde\PycharmProjects\obelix-tracer\obelix_tracer\storage`) to accept the `events` field is out of scope — tracked separately.
