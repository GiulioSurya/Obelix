# A2A CLI polling-only Implementation Plan (spec 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the CLI→server webhook (TEMP-PATCH-SPEC-1) by switching the CLI to pure polling and surfacing drain-spawn / dispatched-peer state through `Task.metadata`, while preserving the user-visible behavior of spec 1 end-to-end.

**Architecture:** The server-side drainer writes `T1.metadata.spawned_task_ids` (drain-spawn child task discovery) and `T1.metadata.dispatched_peers` (peer status visibility) via in-place `TaskStore` patch — verified as the only working pattern on a2a-sdk 0.3.25 (research finding `Task.metadata mutability`). The CLI runs `tasks/get(T1)` every 0.5s, recursively spawns polling workers for each new `spawned_task_id`, and renders peers in the status bar. ESC cancels only the focused task; `cancel_task` errors `-32001` / `-32002` are silenced as no-ops.

**Tech Stack:**
- a2a-sdk 0.3.25 (`a2a.types.Task.metadata`, `a2a.server.tasks.task_store.TaskStore`, `a2a.client.ClientConfig`, JSON-RPC transport).
- httpx 0.28.1 with `ASGITransport` for in-process FakeA2AServer (already in tests).
- Textual 8.1.1 (CLI UI surface — unchanged).
- pytest + pytest-asyncio (existing test framework).

**Reference docs (read-only inputs):**
- Spec: `docs/superpowers/specs/2026-05-08-a2a-cli-polling-spec2-design.md`
- Research: `docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md`
- Spike artefacts (gitignored, local): `.claude/spikes/a2a-metadata-mutability.py`, `.claude/spikes/a2a-cancel-no-cascade.py`, `.claude/spikes/a2a-context-id-clientconfig.py`

**Iron rule (non-negotiable):**
- Integration tests only. **No `unittest.mock`, `pytest-mock`, `monkeypatch`, `MagicMock` against any `a2a.*` symbol.**
- Use the FakeA2AServer infrastructure built in Task 1.
- Hand-written Fake classes that implement real Protocol contracts. See `CLAUDE.md` (project root) "Niente mock di dipendenze esterne nei test".

---

## File structure

### Created
- `tests/_fakes/__init__.py` — empty package marker.
- `tests/_fakes/fake_a2a_server.py` — in-process A2A JSON-RPC server (Starlette + InMemoryTaskStore) reusable by every spec-2 integration test. Single responsibility: stand up an A2A endpoint behind `httpx.ASGITransport`.
- `src/obelix/adapters/inbound/a2a/client/task_tracker.py` — `TaskInfo` + `TaskTracker` migrated from `webhook_server.py`. Adds `update_peers()` / `get_active_peers()` for status-bar rendering. Drops `context_resolver` callback.
- `src/obelix/adapters/inbound/a2a/server/metadata_patch.py` — small helper module exposing `async update_task_metadata(task_store, task_id, patch_fn)`. Single responsibility: read-modify-write a Task in the SDK's TaskStore using the verified pattern from research.
- `tests/adapters/inbound/a2a/server/test_metadata_patch.py` — exercises the helper end-to-end against a real `InMemoryTaskStore`.
- `tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py` — verifies drainer writes `spawned_task_ids` to `T1.metadata` after registering T3.
- `tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py` — verifies `dispatch_agent` adds an entry to `T1.metadata.dispatched_peers` and that webhook/polling/handler updates the `state` field.
- `tests/adapters/inbound/a2a/client/test_polling_discovery.py` — verifies the CLI polling worker discovers a drain-spawn child via `spawned_task_ids` and pulls its artifact.
- `tests/adapters/inbound/a2a/client/test_polling_termination.py` — verifies the polling loop exits only when T1 + all children + all peers are terminal.
- `tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py` — verifies the CLI surfaces dispatched-peer state via `TaskTracker.get_active_peers()`.
- `tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py` — verifies `-32001` and `-32002` from `cancel_task` are treated as no-ops.

### Modified
- `src/obelix/adapters/inbound/a2a/server/executor.py` — removes `httpx_client` slot, `_apply_webhook_metadata_patch`, `_DrainSpawnEventQueue`, `_NullEventQueue`. Accepts `task_store` injection. `_run_drain_task` becomes a thin wrapper over the SDK's normal flow.
- `src/obelix/adapters/inbound/a2a/server/context.py` — removes `client_webhook_url` / `client_webhook_token` slots and their init.
- `src/obelix/adapters/outbound/a2a/tools/dispatch.py` — accepts a `task_store` setter; on dispatch, appends to `T1.metadata.dispatched_peers` via the helper.
- `src/obelix/adapters/outbound/a2a/webhook.py` — on push update, also patches `T1.metadata.dispatched_peers[*].state`.
- `src/obelix/adapters/outbound/a2a/polling.py` — on poll detection of state change, same patch as webhook.py.
- `src/obelix/adapters/outbound/a2a/handler.py` — local handler path (e.g., respond-to-remote completion), same patch.
- `src/obelix/core/agent/agent_factory.py` — removes `_resolve_webhook_host`, `_WILDCARD_BIND_HOSTS`, the wildcard-host rewrite for the (now-removed) CLI webhook URL, and removes `httpx_client` from the `ObelixAgentExecutor` wiring. Injects the SDK `task_store`.
- `src/obelix/adapters/inbound/a2a/client/cli_client.py` — switches `ClientConfig` to polling-only, replaces `_polling_fallback` / `_poll_warned` with `_start_polling` / `_poll_task` (recursive discovery), wires `update_peers` into status bar, silences `-32001` / `-32002` from cancel, drops webhook server boot.
- Existing tests `tests/adapters/inbound/a2a/server/test_drain_spawn_*.py` — adapted to assert the new metadata-based delivery path (no webhook POSTs anymore).

### Deleted
- `src/obelix/adapters/inbound/a2a/client/webhook_server.py`
- `tests/adapters/inbound/a2a/client/test_webhook_server_token.py`
- `tests/core/agent/test_a2a_serve_webhook_url.py`
- `tests/adapters/inbound/a2a/client/test_cli_webhook_metadata.py`
- `tests/adapters/inbound/a2a/client/test_task_tracker_unknown_agent.py`
- `tests/adapters/inbound/a2a/server/test_temp_patch_marker.py` (deleted in the **last** task once `EXPECTED_COUNT=0` is reached)

---

## Task 1: Build the FakeA2AServer test infrastructure

**Why this task exists:** Every spec-2 integration test needs an in-process A2A JSON-RPC endpoint backed by a real `InMemoryTaskStore`. The iron rule forbids mocking `a2a.*` symbols. The existing test `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py` already uses the `httpx.ASGITransport` pattern; we extract a reusable fixture so subsequent tasks don't reinvent it.

**Files:**
- Create: `tests/_fakes/__init__.py`
- Create: `tests/_fakes/fake_a2a_server.py`

- [ ] **Step 1: Create the empty package marker**

```python
# tests/_fakes/__init__.py
```

- [ ] **Step 2: Verify how the existing e2e test wires the SDK**

Read `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py` to learn the existing pattern: `httpx.ASGITransport` against a Starlette app built from `A2AStarletteApplication(agent_card, http_handler=DefaultRequestHandler(...))`. The fake is just that wiring, parameterized.

- [ ] **Step 3: Write the fake**

```python
# tests/_fakes/fake_a2a_server.py
"""In-process A2A JSON-RPC server backed by InMemoryTaskStore.

Instantiate one per test; the .client attribute is a ready-to-use
a2a.client.Client wired through httpx.ASGITransport — no network.

Iron rule: we never mock a2a.* symbols; the SDK runs against itself.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


class _NoopExecutor(AgentExecutor):
    """Default executor used when a test only needs the request handlers
    (tasks/get, tasks/cancel) — never produces events, never runs an agent."""

    async def execute(self, context, event_queue: EventQueue) -> None:  # noqa: D401
        return

    async def cancel(self, context, event_queue: EventQueue) -> None:  # noqa: D401
        return


def _default_card(url: str = "http://fake-a2a") -> AgentCard:
    return AgentCard(
        name="fake-agent",
        description="In-process fake A2A agent for tests.",
        url=url,
        version="0.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=True,
            supports_authenticated_extended_card=False,
        ),
        skills=[
            AgentSkill(
                id="echo",
                name="echo",
                description="Echo what you send.",
                tags=[],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


class FakeA2AServer:
    """Test fixture: a Starlette A2A app + an SDK Client over ASGITransport.

    Public attributes:
        task_store: the SDK's InMemoryTaskStore (mutate directly to seed
            tasks or to assert on stored Task.metadata after the code under
            test ran).
        client: an a2a.client.Client wired to this app via ASGITransport.
        executor: the AgentExecutor handed to DefaultRequestHandler (override
            via constructor to inject custom behavior).

    Polling-only by default (matches the spec-2 CLI config). Pass
    ``streaming=True`` to test code that may opt into SSE.
    """

    def __init__(
        self,
        executor: AgentExecutor | None = None,
        *,
        card: AgentCard | None = None,
        streaming: bool = False,
    ) -> None:
        self.task_store = InMemoryTaskStore()
        self.executor = executor or _NoopExecutor()
        self._card = card or _default_card()
        self._handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
        )
        self._app = A2AStarletteApplication(
            agent_card=self._card,
            http_handler=self._handler,
        ).build()
        self._transport = httpx.ASGITransport(app=self._app)
        self._httpx = httpx.AsyncClient(
            transport=self._transport,
            base_url=self._card.url,
            timeout=30.0,
        )
        self._streaming = streaming
        self.client = None  # populated in __aenter__

    async def __aenter__(self) -> "FakeA2AServer":
        resolver = A2ACardResolver(httpx_client=self._httpx, base_url=self._card.url)
        # The card resolver fetches /.well-known/agent-card; we already have
        # the card object, but a fresh fetch through the ASGI transport
        # exercises the full path the way the CLI does.
        card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=self._httpx,
            streaming=self._streaming,
            polling=not self._streaming,
            push_notification_configs=[],
        )
        self.client = ClientFactory(config).create(card)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._httpx.aclose()

    @asynccontextmanager
    async def lifespan(self):
        """Convenience: ``async with server.lifespan(): ...`` for tests
        that don't want to use the class itself as a context manager."""
        async with self as s:
            yield s


def make_server(
    executor: AgentExecutor | None = None,
    **kwargs: Any,
) -> FakeA2AServer:
    """Module-level helper for symmetry with pytest fixture style."""
    return FakeA2AServer(executor, **kwargs)
```

- [ ] **Step 4: Sanity-check the fake compiles and resolves a card**

Write a throwaway one-liner test to verify wiring:

```python
# tests/_fakes/test_smoke.py  (delete after step 5)
import pytest
from tests._fakes.fake_a2a_server import FakeA2AServer


@pytest.mark.asyncio
async def test_fake_server_resolves_card():
    async with FakeA2AServer() as server:
        assert server.client is not None
        assert server.task_store is not None
```

Run: `uv run pytest tests/_fakes/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Delete the smoke test**

```bash
rm tests/_fakes/test_smoke.py
```

The fake is now consumed only by real tests in subsequent tasks. Smoke is for the green-light only.

- [ ] **Step 6: Commit**

```bash
git add tests/_fakes/__init__.py tests/_fakes/fake_a2a_server.py
git commit -m "test(spec2): FakeA2AServer infrastructure (ASGITransport, real SDK)"
```

---

## Task 2: Inject `task_store` into `ObelixAgentExecutor` (replaces `httpx_client`)

**Why this task exists:** Spec 2 abandons the webhook (`httpx_client` was only used by `_DrainSpawnEventQueue`). In its place, the executor needs a reference to the SDK's `TaskStore` to perform in-place metadata patches (research finding "Task.metadata mutability"). This task does the wiring and nothing else; metadata writes come in later tasks.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py:96-117` (constructor signature + slots)
- Modify: `src/obelix/core/agent/agent_factory.py` (the call site that constructs `ObelixAgentExecutor`)
- Test: `tests/adapters/inbound/a2a/server/test_executor_construction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/inbound/a2a/server/test_executor_construction.py
"""Verify ObelixAgentExecutor accepts a task_store kwarg and exposes it
internally for the metadata-patch helpers (introduced in later tasks).

Iron rule: real SDK types only — no MagicMock substitution for TaskStore.
"""

from __future__ import annotations

from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model import SystemMessage


def _agent_factory() -> BaseAgent:
    # A trivial agent — no provider needed for construction-only checks.
    return BaseAgent(system_message=SystemMessage(content="x"), provider=None)


def test_executor_accepts_task_store():
    store = InMemoryTaskStore()
    executor = ObelixAgentExecutor(
        agent_factory=_agent_factory,
        task_store=store,
    )
    assert executor._task_store is store


def test_executor_no_longer_accepts_httpx_client():
    """Regression: spec 2 removes the httpx_client slot. Passing it must
    raise (TypeError on unexpected kwarg)."""
    import httpx
    import pytest

    store = InMemoryTaskStore()
    with pytest.raises(TypeError, match="httpx_client"):
        ObelixAgentExecutor(
            agent_factory=_agent_factory,
            task_store=store,
            httpx_client=httpx.AsyncClient(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_executor_construction.py -v`
Expected: FAIL — `task_store` parameter does not exist; the second test passes accidentally if `httpx_client` is still there (we'll see one fail, one pass).

- [ ] **Step 3: Modify `executor.py:96-117` (constructor)**

Replace the existing `__init__` block (the bit shown in spec context, around lines 96-117):

```python
# src/obelix/adapters/inbound/a2a/server/executor.py
# (replacing the constructor — keep the rest of the file unchanged)

from a2a.server.tasks.task_store import TaskStore

class ObelixAgentExecutor(AgentExecutor):
    # ...docstring unchanged...

    def __init__(
        self,
        agent_factory: Callable[[], BaseAgent],
        *,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
        tracer: Tracer | None = None,
        registry: RemoteAgentRegistry | None = None,
        context_store: ContextStore | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._store = (
            context_store if context_store is not None else ContextStore(max_contexts)
        )
        self._store_lock = asyncio.Lock()
        self._tracer = tracer
        self._registry = registry
        # SDK's TaskStore — used by metadata-patch helpers to expose
        # spawned_task_ids and dispatched_peers to polling clients.
        self._task_store = task_store
```

Also remove the `if TYPE_CHECKING: import httpx` block — `httpx` is no longer referenced at runtime in this file. (Leave other TYPE_CHECKING imports.)

- [ ] **Step 4: Update `agent_factory.py` wiring**

In `src/obelix/core/agent/agent_factory.py`, locate the `ObelixAgentExecutor(...)` constructor call inside the `a2a_serve` flow. Replace `httpx_client=httpx_client` with `task_store=task_store`. The TaskStore is the same one passed to `DefaultRequestHandler`; obtain it from there. If the current code instantiates `DefaultRequestHandler` and `ObelixAgentExecutor` separately, share the store explicitly:

```python
# inside agent_factory.a2a_serve (replace the executor construction)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

task_store = InMemoryTaskStore()
executor = ObelixAgentExecutor(
    agent_factory=agent_factory,
    max_contexts=max_contexts,
    tracer=tracer,
    registry=registry,
    context_store=context_store,
    task_store=task_store,
)
http_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=task_store,
)
```

(If a `task_store` is already constructed in this region, just reuse it instead of creating a new one — there must only be ONE TaskStore per server, otherwise the executor patches a different store than the one the request handler reads from.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_executor_construction.py -v`
Expected: PASS for both tests.

- [ ] **Step 6: Run the full test suite to surface fallout**

Run: `uv run pytest -q`
Expected: a few failures referencing `httpx_client` in `test_drain_spawn_*.py`. Note them but do not fix yet — those tests are deleted/adapted in Task 17 once the new flow is fully in place. Add `@pytest.mark.skip(reason="adapted in spec2 Task 17")` to those tests for now.

- [ ] **Step 7: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py \
        src/obelix/core/agent/agent_factory.py \
        tests/adapters/inbound/a2a/server/test_executor_construction.py
git commit -m "refactor(a2a): swap executor's httpx_client for task_store injection"
```

---

## Task 3: Add the `update_task_metadata` helper

**Why this task exists:** The drainer, dispatch tool, and webhook handler all need to do the same read-modify-write pattern on a `Task.metadata`. Centralize it once, with its own test, and stop duplicating.

**Files:**
- Create: `src/obelix/adapters/inbound/a2a/server/metadata_patch.py`
- Create: `tests/adapters/inbound/a2a/server/test_metadata_patch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/inbound/a2a/server/test_metadata_patch.py
"""End-to-end test of update_task_metadata against a real InMemoryTaskStore.

Iron rule: no mocks. The SDK runs against itself.
"""

from __future__ import annotations

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata


def _seed_task(store: InMemoryTaskStore, task_id: str, metadata=None) -> Task:
    task = Task(
        id=task_id,
        context_id="ctx-test",
        status=TaskStatus(state=TaskState.completed),
        metadata=metadata,
    )
    return task


@pytest.mark.asyncio
async def test_appends_to_list_field_when_metadata_is_none():
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1")
    await store.save(task)

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = list(meta.get("spawned_task_ids", [])) + ["t3"]
        return meta

    await update_task_metadata(store, "t1", add_child)

    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata == {"spawned_task_ids": ["t3"]}


@pytest.mark.asyncio
async def test_preserves_other_metadata_fields():
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1", metadata={"existing": "value"})
    await store.save(task)

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = ["t3"]
        return meta

    await update_task_metadata(store, "t1", add_child)

    refreshed = await store.get("t1")
    assert refreshed.metadata == {"existing": "value", "spawned_task_ids": ["t3"]}


@pytest.mark.asyncio
async def test_noop_when_task_evicted():
    """If the task is gone from the store, the helper must NOT raise."""
    store = InMemoryTaskStore()

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = ["t3"]
        return meta

    # Should silently no-op — task never existed.
    await update_task_metadata(store, "missing", add_child)


@pytest.mark.asyncio
async def test_concurrent_appends_no_loss_under_serial_calls():
    """Two sequential calls (the realistic case for spec 2's drainer)
    must not lose data."""
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1")
    await store.save(task)

    async def add(child_id: str):
        async def patch(meta: dict) -> dict:
            meta["spawned_task_ids"] = list(meta.get("spawned_task_ids", [])) + [child_id]
            return meta
        await update_task_metadata(store, "t1", patch)

    await add("t3")
    await add("t4")

    refreshed = await store.get("t1")
    assert refreshed.metadata["spawned_task_ids"] == ["t3", "t4"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_metadata_patch.py -v`
Expected: FAIL — `update_task_metadata` does not exist.

- [ ] **Step 3: Implement the helper**

```python
# src/obelix/adapters/inbound/a2a/server/metadata_patch.py
"""In-place mutation of Task.metadata via the SDK's TaskStore.

Verified pattern from pre-impl research (spike a2a-metadata-mutability):
- TaskStatusUpdateEvent post-terminal is dropped silently (queue closed).
- TaskStore.save() is the only working path.
- DefaultRequestHandler.on_get_task always re-reads from the store, so
  the next polling client sees the patched metadata immediately.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from a2a.server.tasks.task_store import TaskStore


PatchFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def update_task_metadata(
    store: TaskStore,
    task_id: str,
    patch_fn: PatchFn,
) -> None:
    """Read the Task, run patch_fn over a mutable copy of its metadata, save.

    No-ops silently if the task is not in the store (evicted, never existed).

    The patch_fn receives a dict that is safe to mutate in-place; whatever
    dict it returns becomes the new ``Task.metadata``. Returning ``{}``
    clears the metadata; returning ``None`` is treated as ``{}``.
    """
    task = await store.get(task_id)
    if task is None:
        return
    current = dict(task.metadata) if task.metadata else {}
    new_meta = await patch_fn(current)
    task.metadata = new_meta or {}
    await store.save(task)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_metadata_patch.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/metadata_patch.py \
        tests/adapters/inbound/a2a/server/test_metadata_patch.py
git commit -m "feat(a2a): update_task_metadata helper for in-place TaskStore patch"
```

---

## Task 4: Drainer writes `spawned_task_ids` to T1.metadata (with race-safe ordering)

**Why this task exists:** Section 2 of the spec mandates that drain-spawn task IDs surface to polling clients via `T1.metadata.spawned_task_ids`. The research finding "race ordering vincolante" requires saving T3 to the TaskStore BEFORE writing its id to T1.metadata, so a polling client never sees a child id it can't yet `tasks/get`.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py` — `maybe_spawn_drain_task` and `_run_drain_task`.
- Create: `tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py`

- [ ] **Step 1: Locate the spawn site**

Read `src/obelix/adapters/inbound/a2a/server/executor.py` around `maybe_spawn_drain_task` (uses `asyncio.create_task` and is named like the function appearing in spec 1 implementation). Identify the exact line where the drain-spawn task gets its `task_id` (a fresh UUID).

- [ ] **Step 2: Write the failing test**

```python
# tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py
"""Verify that when the drainer spawns T3 it:
  1. saves T3 to the SDK TaskStore FIRST (so tasks/get(T3) returns 200).
  2. patches T1.metadata.spawned_task_ids to include T3.id.

Order matters: a polling client that learns T3.id but cannot yet retrieve
the Task gets a confusing -32001. Race verified empirically in spec 1
smoke testing (Bug 6).

Iron rule: no SDK mocks. We feed a real InMemoryTaskStore and observe.
"""

from __future__ import annotations

import asyncio

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


# Helper: build the executor with the minimum scaffolding needed to
# exercise the drainer entry-point.  We deliberately avoid the
# higher-level a2a_serve / agent factory machinery — we want a pure
# observation of side effects on the TaskStore.
def _build_executor(task_store):
    def _agent_factory():
        raise AssertionError("agent_factory should not run in this test")

    return ObelixAgentExecutor(
        agent_factory=_agent_factory,
        task_store=task_store,
    )


@pytest.mark.asyncio
async def test_drainer_appends_spawned_task_id_after_saving_child():
    store = InMemoryTaskStore()
    parent = Task(
        id="t1",
        context_id="ctx-share",
        status=TaskStatus(state=TaskState.completed),
        metadata=None,
    )
    await store.save(parent)

    executor = _build_executor(store)
    entry = ContextEntry()
    entry.context_id = "ctx-share"
    # Replicate the post-completion situation: agent done, parent terminal.
    entry.history = []

    # Trigger the drainer entry point. The exact API surface is
    # ``maybe_spawn_drain_task`` (spec 1).  We pass a synthetic remote
    # update so the function takes the spawn branch.
    new_id = await executor.maybe_spawn_drain_task(
        parent_task_id="t1",
        context_id="ctx-share",
        entry=entry,
    )

    # Wait briefly for the spawned background task to register T3 + patch
    # T1.metadata. The drainer is fire-and-forget; we only need to observe
    # both side-effects landed.
    await asyncio.sleep(0.05)

    # 1. Child task is in the store.
    child = await store.get(new_id)
    assert child is not None, "T3 must be saved to TaskStore before T1.metadata is patched"

    # 2. T1.metadata.spawned_task_ids contains T3.id.
    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata is not None
    assert new_id in refreshed.metadata.get("spawned_task_ids", []), (
        f"expected T3.id={new_id!r} in T1.metadata.spawned_task_ids, "
        f"got metadata={refreshed.metadata!r}"
    )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py -v`
Expected: FAIL — either `maybe_spawn_drain_task` is not awaitable, doesn't return a task_id, or doesn't touch the store.

- [ ] **Step 4: Modify `maybe_spawn_drain_task`**

Update the function (in `executor.py`) to (a) generate the new task_id, (b) save a minimal Task to the SDK TaskStore, (c) patch T1.metadata via the helper, (d) only THEN call `asyncio.create_task` for the actual drain run. Pseudocode in the function:

```python
# inside ObelixAgentExecutor (file: src/obelix/adapters/inbound/a2a/server/executor.py)

from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata
from a2a.types import Task as A2ATask
from a2a.types import TaskState, TaskStatus

async def maybe_spawn_drain_task(
    self,
    *,
    parent_task_id: str,
    context_id: str,
    entry: ContextEntry,
) -> str:
    """Spawn a drain task. Returns the new task_id once it has been
    registered in the TaskStore AND parent T1.metadata has been patched.
    The actual agent execution starts in a fire-and-forget background
    task immediately after."""
    new_task_id = str(uuid.uuid4())

    if self._task_store is not None:
        # 1. Register T3 in the SDK store FIRST so tasks/get(T3) succeeds
        #    the moment the polling client learns the id.
        await self._task_store.save(
            A2ATask(
                id=new_task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.submitted),
                metadata=None,
            )
        )
        # 2. Patch T1.metadata.spawned_task_ids — append-only.
        async def _append_child(meta: dict) -> dict:
            existing = list(meta.get("spawned_task_ids", []))
            existing.append(new_task_id)
            meta["spawned_task_ids"] = existing
            return meta
        await update_task_metadata(self._task_store, parent_task_id, _append_child)

    # 3. Now fire-and-forget the actual run.
    synthetic_message = self._build_synthetic_drain_message(entry)
    asyncio.create_task(
        self._run_drain_task(
            task_id=new_task_id,
            context_id=context_id,
            entry=entry,
            message=synthetic_message,
        ),
        name=f"drain-spawn-{new_task_id[:8]}",
    )
    return new_task_id
```

If the existing function shape differs (different signature, different inline message construction), preserve the existing behavior and only add the three numbered side effects in the order shown.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py \
        tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py
git commit -m "feat(a2a): drainer writes spawned_task_ids to T1.metadata (race-safe order)"
```

---

## Task 5: Drainer uses the SDK's normal EventQueue (delete `_DrainSpawnEventQueue` / `_NullEventQueue`)

**Why this task exists:** Spec 2 §5 deletes both classes and §3 mandates the drain-spawn task is a regular A2A task in the SDK store. The previous `_run_drain_task` selected between two custom queues based on `entry.client_webhook_url`; with the webhook gone, it must use the same EventQueue path the SDK uses for any task spawned by `message/send` (so that `tasks/get(T3)` reflects the task progressing through working/completed naturally, with artifacts visible to the polling client).

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py` — `_run_drain_task`, delete `_DrainSpawnEventQueue` and `_NullEventQueue` classes.
- Existing test fallout: spec-1 `test_drain_spawn_*.py` will need updates (deferred to Task 17).

- [ ] **Step 1: Verify the SDK contract for spawning a task on an existing TaskStore**

Read `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:199-260` to understand `_setup_message_execution`. The pattern: it builds a per-task `EventQueue`, registers it in `_queue_manager`, runs the executor, and `EventConsumer` merges events into the TaskStore via the server-side `TaskManager`. We replicate that for the drain spawn.

- [ ] **Step 2: Refactor `_run_drain_task` to use the SDK queue manager**

```python
# src/obelix/adapters/inbound/a2a/server/executor.py
# (replacing the existing _run_drain_task body — keep the surrounding code)

from a2a.server.events.event_queue import EventQueue
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager

async def _run_drain_task(
    self,
    *,
    task_id: str,
    context_id: str,
    entry: ContextEntry,
    message: Message,
) -> None:
    """Run a drain-spawned task using a fresh SDK EventQueue.

    The task is already registered in self._task_store by maybe_spawn_drain_task
    (see Task 4). We open a per-task EventQueue, drive the agent to completion,
    and EventConsumer merges the events back into the TaskStore via the SDK's
    TaskManager. The polling client picks the result up via tasks/get(T3).
    """
    if self._task_store is None:
        logger.warning(
            f"[A2A drain] no task_store available, skipping drain spawn task_id={task_id}"
        )
        return

    # Drain pending notifications into the agent history before running.
    if entry.pending_notifications:
        drained = entry.pending_notifications
        entry.pending_notifications = []
        entry.history.extend(drained)

    queue = EventQueue()
    # Build a TaskManager / consumer pair the same way the SDK does for
    # message/send. The simplest path is to use the SDK's helper if exposed;
    # otherwise we construct manually.
    from a2a.server.tasks.task_manager import TaskManager
    from a2a.server.events.event_consumer import EventConsumer

    task_manager = TaskManager(
        task_id=task_id,
        context_id=context_id,
        task_store=self._task_store,
        initial_message=None,
    )
    consumer = EventConsumer(queue=queue, task_manager=task_manager)
    consumer_task = asyncio.create_task(
        consumer.consume_all(),
        name=f"drain-consume-{task_id[:8]}",
    )

    try:
        await self._run_agent(
            task_id=task_id,
            context_id=context_id,
            user_text="",
            attachments=[],
            entry=entry,
            event_queue=queue,
            is_resume=False,
            is_drain_spawn=True,
        )
    except Exception as e:
        logger.exception(
            f"[A2A drain] spawned task failed | task_id={task_id} error={e}"
        )
    finally:
        # Wait for the consumer to drain the final event(s) into the store.
        try:
            await asyncio.wait_for(consumer_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"[A2A drain] consumer did not finish within 5s task_id={task_id}"
            )
            consumer_task.cancel()
```

NOTE: if the SDK already exposes a single helper (e.g. `DefaultRequestHandler` has a private method that does this whole dance), prefer that instead. Read the source under `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py` and locate it. If found, the body of `_run_drain_task` collapses to a call to that helper. Document the call site verbatim in the commit message.

- [ ] **Step 3: Delete `_DrainSpawnEventQueue` and `_NullEventQueue`**

Find both classes in `executor.py` (currently around lines 1301-1394 from the prior listing). Delete them entirely along with any helper methods (`_post`, `_merge_artifact_update`) that only existed for `_DrainSpawnEventQueue`. Remove the now-unused imports (`httpx`, `Artifact` if no longer referenced elsewhere — re-check).

- [ ] **Step 4: Update or skip existing drain-spawn tests**

The tests in `tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py`, `test_drain_spawn_e2e.py`, and `test_drain_spawn_webhook_post.py` were written against `_DrainSpawnEventQueue`. They will fail to import after the deletion. Add an `@pytest.mark.skip(reason="rewritten for spec 2 polling, see Task 17")` decorator at module level to keep them parseable, OR delete the import lines that reference removed classes (preferred — fewer surprises). Do NOT rewrite the tests yet — that's Task 17.

- [ ] **Step 5: Run the new drainer test from Task 4**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py -v`
Expected: PASS — the metadata side-effect from Task 4 must still hold; this task only changed how T3 is *executed*, not how it's announced.

- [ ] **Step 6: Run the smoke suite to surface fallout outside drain-spawn**

Run: `uv run pytest tests/ -q -x --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: PASS (the rest of the suite must keep working).

- [ ] **Step 7: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py \
        tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py \
        tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py \
        tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py
git commit -m "refactor(a2a): drainer uses SDK EventQueue, drop _DrainSpawnEventQueue/_NullEventQueue"
```

---

## Task 6: `dispatch_agent` writes `dispatched_peers` on dispatch

**Why this task exists:** Spec §2 calls `dispatched_peers` "the visibility channel for peer status". The CLI reads this list off `T1.metadata` to render the status bar. The dispatch tool is the right place to append a new entry, since it is the only code path that decides "this remote agent has been delegated to right now".

**Files:**
- Modify: `src/obelix/adapters/outbound/a2a/tools/dispatch.py`
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py:1005-1044` (the `_inject_context_entry` helper that wires tools)
- Create: `tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py`

- [ ] **Step 1: Decide the parent_task_id source**

The dispatch tool runs inside an agent loop driven by the executor. The parent task id (T1.id) is set on `RequestContext.task_id` when the executor starts the run; we forward it to the tool via the existing `set_context_entry(entry, *, context_id)` setter pattern. Extend the setter signature: add `parent_task_id`. The executor passes it from `RequestContext.task_id`.

- [ ] **Step 2: Write the failing test**

```python
# tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py
"""Verify dispatch_agent appends an entry to T1.metadata.dispatched_peers
the moment it submits the message to the remote agent.

Iron rule: no SDK mocks. We use the FakeA2AServer for the remote and
exercise the tool's execute() against a real registry / store.
"""

from __future__ import annotations

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.tools.dispatch import DispatchAgentTool

from tests._fakes.fake_a2a_server import FakeA2AServer


@pytest.mark.asyncio
async def test_dispatch_appends_peer_to_t1_metadata():
    async with FakeA2AServer() as remote:
        # Seed T1 in the LOCAL store (not the remote's). Dispatch tool reads
        # local store via the helper.
        local_store = InMemoryTaskStore()
        await local_store.save(
            Task(
                id="t1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.working),
                metadata=None,
            )
        )

        # Build registry with one remote whose card name is "fake-agent".
        registry = RemoteAgentRegistry()
        registry.register(name="fake-agent", url=remote._card.url, client=remote.client, card=remote._card)

        # Construct the tool and inject its context.
        entry = ContextEntry()
        tool = DispatchAgentTool(registry=registry)
        tool.set_context_entry(entry, context_id="ctx-1", parent_task_id="t1")
        tool.set_webhook_url("http://placeholder/webhook")
        tool.set_task_store(local_store)
        tool.agent_name = "fake-agent"
        tool.query = "do something"

        result = await tool.execute()
        assert result["status"] == "submitted"

        # T1.metadata.dispatched_peers must now contain the new entry.
        refreshed = await local_store.get("t1")
        assert refreshed is not None
        peers = (refreshed.metadata or {}).get("dispatched_peers", [])
        assert len(peers) == 1
        assert peers[0]["name"] == "fake-agent"
        assert peers[0]["task_id"] == result["task_id"]
        assert peers[0]["state"] == "working"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py -v`
Expected: FAIL — `set_task_store` and `parent_task_id` don't exist on the tool yet.

- [ ] **Step 4: Modify `dispatch.py`**

Add the two new setters and the metadata write at the end of `execute()`:

```python
# src/obelix/adapters/outbound/a2a/tools/dispatch.py
# (add to the existing class)

from a2a.server.tasks.task_store import TaskStore
from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata


class DispatchAgentTool:
    # ...existing attrs...
    def __init__(self, registry: RemoteAgentRegistry) -> None:
        self._registry = registry
        self._ctx_entry: ContextEntry | None = None
        self._context_id: str | None = None
        self._webhook_url: str | None = None
        self._parent_task_id: str | None = None
        self._task_store: TaskStore | None = None

    def set_context_entry(
        self,
        entry: ContextEntry,
        *,
        context_id: str,
        parent_task_id: str | None = None,
    ) -> None:
        self._ctx_entry = entry
        self._context_id = context_id
        self._parent_task_id = parent_task_id

    def set_task_store(self, store: TaskStore) -> None:
        """Inject the SDK TaskStore so the tool can patch T1.metadata."""
        self._task_store = store

    # (existing set_webhook_url, system_prompt_fragment, ...)

    async def execute(self) -> dict:
        # ...existing precondition + send_message logic — unchanged...
        # task is the resolved Task from the remote.

        # NEW: surface the dispatched peer on T1.metadata.dispatched_peers.
        if self._task_store is not None and self._parent_task_id is not None:
            async def _append(meta: dict) -> dict:
                peers = list(meta.get("dispatched_peers", []))
                peers.append({
                    "name": self.agent_name,
                    "task_id": task.id,
                    "state": "working",
                })
                meta["dispatched_peers"] = peers
                return meta
            await update_task_metadata(
                self._task_store, self._parent_task_id, _append
            )

        return {
            "status": "submitted",
            "task_id": task.id,
            "agent": self.agent_name,
        }
```

- [ ] **Step 5: Update the executor's `_inject_context_entry` to pass `parent_task_id`**

In `executor.py` around lines 1005-1044, change the call site to forward the current task_id (it is already in scope as `task_id` inside `_run_agent`):

```python
# inside _inject_context_entry — already passes context_id when supported.
# Add parent_task_id forwarding similarly.

if "context_id" in sig.parameters:
    if "parent_task_id" in sig.parameters:
        setter(entry, context_id=context_id, parent_task_id=parent_task_id)
    else:
        setter(entry, context_id=context_id)
else:
    setter(entry)
```

The `parent_task_id` argument is added to `_inject_context_entry`'s signature and forwarded from the caller (`_run_agent`) which knows the current task_id. Also: in the same helper, after the existing setter loop, attach the task_store to any tool that exposes `set_task_store`:

```python
for tool in agent.registered_tools:
    setter = getattr(tool, "set_task_store", None)
    if callable(setter) and self._task_store is not None:
        setter(self._task_store)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/obelix/adapters/outbound/a2a/tools/dispatch.py \
        src/obelix/adapters/inbound/a2a/server/executor.py \
        tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py
git commit -m "feat(a2a): dispatch_agent surfaces dispatched_peers on T1.metadata"
```

---

## Task 7: webhook / polling / handler update `dispatched_peers[*].state`

**Why this task exists:** State transitions on the peer side (`working → completed/failed/canceled`) reach the parent agent through three code paths: the inbound webhook from a remote (push A↔B), the outbound polling worker, and the local respond_to_remote handler. All three must mirror the new state into `T1.metadata.dispatched_peers[*].state`. Without this update, the CLI status bar would freeze on "working".

**Files:**
- Modify: `src/obelix/adapters/outbound/a2a/webhook.py:120-145` (the line that mutates `entry.remote_tasks[task_id].status`).
- Modify: `src/obelix/adapters/outbound/a2a/polling.py:115-135`.
- Modify: `src/obelix/adapters/outbound/a2a/handler.py:175-205`.
- Modify: `src/obelix/core/agent/agent_factory.py` to inject the `task_store` into webhook/polling constructors.
- Create: extend `tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py` with the state-transition cases.

- [ ] **Step 1: Add the helper for the three call sites**

Centralize: extend `metadata_patch.py` (created in Task 3) with a peer-specific helper so callers don't reimplement the dict surgery.

```python
# append to src/obelix/adapters/inbound/a2a/server/metadata_patch.py

async def update_dispatched_peer_state(
    store: TaskStore,
    parent_task_id: str,
    peer_task_id: str,
    new_state: str,
) -> None:
    """Patch T_parent.metadata.dispatched_peers[*].state where the entry's
    task_id == peer_task_id. No-op if parent or peer entry is gone."""

    async def _patch(meta: dict) -> dict:
        peers = list(meta.get("dispatched_peers", []))
        for peer in peers:
            if peer.get("task_id") == peer_task_id:
                peer["state"] = new_state
        meta["dispatched_peers"] = peers
        return meta

    await update_task_metadata(store, parent_task_id, _patch)
```

- [ ] **Step 2: Extend the test file with state-transition cases**

```python
# append to tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py

import pytest

from obelix.adapters.inbound.a2a.server.metadata_patch import (
    update_dispatched_peer_state,
)


@pytest.mark.asyncio
async def test_update_dispatched_peer_state_changes_only_matching_entry():
    store = InMemoryTaskStore()
    await store.save(
        Task(
            id="t1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.working),
            metadata={
                "dispatched_peers": [
                    {"name": "A", "task_id": "tA", "state": "working"},
                    {"name": "B", "task_id": "tB", "state": "working"},
                ]
            },
        )
    )

    await update_dispatched_peer_state(store, "t1", "tA", "completed")

    refreshed = await store.get("t1")
    peers = refreshed.metadata["dispatched_peers"]
    assert peers[0]["state"] == "completed"
    assert peers[1]["state"] == "working"


@pytest.mark.asyncio
async def test_update_dispatched_peer_state_noop_when_peer_unknown():
    store = InMemoryTaskStore()
    await store.save(
        Task(
            id="t1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.working),
            metadata={"dispatched_peers": [{"name": "A", "task_id": "tA", "state": "working"}]},
        )
    )

    await update_dispatched_peer_state(store, "t1", "tUnknown", "completed")

    refreshed = await store.get("t1")
    peers = refreshed.metadata["dispatched_peers"]
    assert peers == [{"name": "A", "task_id": "tA", "state": "working"}]
```

- [ ] **Step 3: Run, verify failing**

Run: `uv run pytest tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py::test_update_dispatched_peer_state_changes_only_matching_entry -v`
Expected: FAIL — `update_dispatched_peer_state` does not exist.

- [ ] **Step 4: Implement the helper (already drafted in Step 1) and call it from the three sites**

In `webhook.py:120-145`, immediately after `entry.remote_tasks[task_id].status = ...`, add:

```python
# webhook.py — after updating entry.remote_tasks[task_id].status
if self._task_store is not None and entry.context_id is not None:
    parent_task_id = entry.current_task_id  # set by executor on each turn
    if parent_task_id:
        await update_dispatched_peer_state(
            self._task_store, parent_task_id, task_id, new_state
        )
```

`new_state` is the state already computed for `entry.remote_tasks`. The webhook handler must accept a `task_store` constructor kwarg; thread it through `make_webhook_handler` (the factory invoked from `agent_factory.a2a_serve`).

Apply the same edit at `polling.py:115-135` and `handler.py:175-205`.

NOTE: `entry.current_task_id` is the task_id of the request currently in flight on this context. If `ContextEntry` doesn't already expose this, add it: a single string slot updated by the executor at the top of `_run_agent`. (Sketch: `entry.current_task_id = task_id` at the start, `entry.current_task_id = None` in the finally.)

- [ ] **Step 5: Wire `task_store` through `agent_factory.a2a_serve`**

The `make_webhook_handler` and `PollingWorker` constructors must accept `task_store` and forward it to the helper call sites. In `agent_factory.a2a_serve` pass the same `task_store` instance used by the executor (Task 2).

- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py -v`
Expected: 4/4 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/metadata_patch.py \
        src/obelix/adapters/inbound/a2a/server/context.py \
        src/obelix/adapters/inbound/a2a/server/executor.py \
        src/obelix/adapters/outbound/a2a/webhook.py \
        src/obelix/adapters/outbound/a2a/polling.py \
        src/obelix/adapters/outbound/a2a/handler.py \
        src/obelix/core/agent/agent_factory.py \
        tests/adapters/outbound/a2a/test_dispatched_peers_metadata.py
git commit -m "feat(a2a): mirror peer state to T1.metadata.dispatched_peers across webhook/polling/handler"
```

---

## Task 8: Strip TEMP-PATCH-SPEC-1 metadata reading from executor

**Why this task exists:** `_apply_webhook_metadata_patch` in `executor.py` exists only to extract `client_webhook_url` / `client_webhook_token` from the first message metadata. The CLI no longer sends those fields. The slot on `ContextEntry` and the helper itself become dead code.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/server/executor.py` — remove `_apply_webhook_metadata_patch` and the call site at line 316.
- Modify: `src/obelix/adapters/inbound/a2a/server/context.py:46-47, 95-99` — remove the two slots and their init.

- [ ] **Step 1: Remove the call site and the method from `executor.py`**

Locate the line `# TEMP-PATCH-SPEC-1` followed by `self._apply_webhook_metadata_patch(entry=entry, metadata=message.metadata)` (around line 316) and delete both lines. Then locate the method definition (lines 967-983) and delete the whole method.

- [ ] **Step 2: Remove the slots from `ContextEntry`**

In `src/obelix/adapters/inbound/a2a/server/context.py`, delete the two `__slots__` entries marked `TEMP-PATCH-SPEC-1` and the matching init block (`self.client_webhook_url = None`, `self.client_webhook_token = None`).

- [ ] **Step 3: Sanity check — full test suite excluding the still-skipped drain-spawn legacy tests**

Run: `uv run pytest tests/ -q --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: PASS.

- [ ] **Step 4: Verify TEMP-PATCH-SPEC-1 counter dropped**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_temp_patch_marker.py -v`
Expected: FAIL — count went down (was 13, now ~8). Update `EXPECTED_COUNT` to the actual current count and rerun. PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/server/executor.py \
        src/obelix/adapters/inbound/a2a/server/context.py \
        tests/adapters/inbound/a2a/server/test_temp_patch_marker.py
git commit -m "cleanup(a2a): drop _apply_webhook_metadata_patch + ContextEntry webhook slots"
```

---

## Task 9: Strip CLI-webhook-host rewrite from `agent_factory.py`

**Why this task exists:** `_resolve_webhook_host` and `_WILDCARD_BIND_HOSTS` were added in spec 1 specifically to fix the wildcard-bind URL handed to the CLI's webhook (Bug 3). With the CLI webhook gone, neither helper is referenced.

**Files:**
- Modify: `src/obelix/core/agent/agent_factory.py`
- Delete: `tests/core/agent/test_a2a_serve_webhook_url.py`

- [ ] **Step 1: Find the helpers and their call sites**

Search the file for `_WILDCARD_BIND_HOSTS`, `_resolve_webhook_host`, and `webhook_url`. Identify any remaining references — typically the helpers and the place where the URL was constructed for the executor wiring.

- [ ] **Step 2: Remove**

Delete the constant `_WILDCARD_BIND_HOSTS = {...}`, delete the function `_resolve_webhook_host`, and delete the line that constructed `webhook_url = f"http://{_resolve_webhook_host(host)}:{port}/webhook"` (or wherever it lives). Anything that consumed `webhook_url` (typically forwarded to the executor) is now gone.

- [ ] **Step 3: Delete the regression test**

```bash
git rm tests/core/agent/test_a2a_serve_webhook_url.py
```

- [ ] **Step 4: Run the suite**

Run: `uv run pytest tests/ -q --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/agent_factory.py
git commit -m "cleanup(a2a): remove _resolve_webhook_host (CLI webhook gone)"
```

---

## Task 10: Migrate `TaskTracker` to a dedicated module (drop `context_resolver`, add `update_peers`)

**Why this task exists:** The current `TaskTracker` lives in `webhook_server.py` because it was conceived as the consumer of webhook POSTs. With the webhook gone, the tracker survives — its job is to be the in-memory store the CLI status bar reads from. Moving it to its own module clarifies the responsibility and removes the misleading filename.

**Files:**
- Create: `src/obelix/adapters/inbound/a2a/client/task_tracker.py`
- Update imports: `src/obelix/adapters/inbound/a2a/client/cli_client.py`

- [ ] **Step 1: Create the new module**

```python
# src/obelix/adapters/inbound/a2a/client/task_tracker.py
"""In-memory tracker for A2A tasks observed by the CLI (polling-only).

Tracks task state per task_id and the live dispatched_peers seen on each
parent task. Read by the CLI status bar; written by the polling worker.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInfo:
    """A single tracked task."""

    task_id: str
    agent_name: str
    state: str = "submitted"
    timestamp: float = field(default_factory=time.time)
    task_data: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in ("completed", "failed", "canceled", "rejected")


@dataclass
class PeerInfo:
    """A peer dispatched from a tracked task. Read off ``T1.metadata.dispatched_peers``."""

    name: str
    task_id: str
    state: str  # "working" / "completed" / "failed" / "canceled" / "rejected"

    @property
    def is_terminal(self) -> bool:
        return self.state in ("completed", "failed", "canceled", "rejected")


class TaskTracker:
    """Thread-safe tracker of A2A task state for the CLI."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        # Per parent task_id, the latest snapshot of dispatched_peers
        # seen on T1.metadata.
        self._peers_by_parent: dict[str, list[PeerInfo]] = {}
        self._lock = asyncio.Lock()
        self._on_update: asyncio.Event = asyncio.Event()

    async def register(self, task_id: str, agent_name: str) -> None:
        async with self._lock:
            self._tasks[task_id] = TaskInfo(task_id=task_id, agent_name=agent_name)

    async def update(self, task_data: dict[str, Any]) -> None:
        task_id = task_data.get("id", "")
        status = task_data.get("status", {})
        state = status.get("state", "unknown")
        async with self._lock:
            if task_id in self._tasks:
                info = self._tasks[task_id]
                info.state = state
                info.timestamp = time.time()
                info.task_data = task_data
            else:
                # Unknown task_id (e.g. drain-spawn discovered via metadata
                # before the polling worker registered it). Record under
                # placeholder agent name; the worker that discovered it
                # will overwrite agent_name on its first poll.
                self._tasks[task_id] = TaskInfo(
                    task_id=task_id,
                    agent_name="(spawned)",
                    state=state,
                    task_data=task_data,
                )
        self._on_update.set()
        self._on_update = asyncio.Event()

    async def update_peers(self, parent_task_id: str, peers: list[dict[str, Any]]) -> None:
        async with self._lock:
            self._peers_by_parent[parent_task_id] = [
                PeerInfo(
                    name=p.get("name", ""),
                    task_id=p.get("task_id", ""),
                    state=p.get("state", "working"),
                )
                for p in peers
            ]

    def get_all(self) -> list[TaskInfo]:
        return list(self._tasks.values())

    def get_active(self) -> list[TaskInfo]:
        return [t for t in self._tasks.values() if not t.is_terminal]

    def get_by_agent(self, agent_name: str) -> list[TaskInfo]:
        return [t for t in self._tasks.values() if t.agent_name == agent_name]

    def get_pending_input(self, agent_name: str) -> TaskInfo | None:
        for t in self._tasks.values():
            if t.agent_name == agent_name and t.state == "input-required":
                return t
        return None

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def force_terminal(self, task_id: str, state: str) -> None:
        info = self._tasks.get(task_id)
        if info:
            info.state = state
            info.timestamp = time.time()

    def get_active_peers(self, parent_task_id: str) -> list[PeerInfo]:
        """Return the peers whose state is non-terminal for the given
        parent task. Used by the status bar."""
        return [
            p
            for p in self._peers_by_parent.get(parent_task_id, [])
            if not p.is_terminal
        ]

    def get_all_peers(self, parent_task_id: str) -> list[PeerInfo]:
        return list(self._peers_by_parent.get(parent_task_id, []))

    async def wait_for_terminal(self, task_id: str, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        while True:
            info = self._tasks.get(task_id)
            if info and (info.is_terminal or info.state == "input-required"):
                return info
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return info
            try:
                await asyncio.wait_for(self._on_update.wait(), timeout=min(remaining, 1.0))
            except TimeoutError:
                pass
```

- [ ] **Step 2: Update `cli_client.py` imports**

Find the line `from obelix.adapters.inbound.a2a.client.webhook_server import (TaskTracker, WebhookServer)` (around lines 54-58) and change it to:

```python
from obelix.adapters.inbound.a2a.client.task_tracker import TaskTracker
```

(Drop `WebhookServer` — its uses are removed in Task 11.)

- [ ] **Step 3: Run the suite**

Run: `uv run pytest tests/ -q --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: many existing CLI tests will still pass since `TaskTracker` API stayed compatible. Anything still importing from `webhook_server` will fail — fix those imports too (likely `test_cli_webhook_metadata.py` and `test_task_tracker_unknown_agent.py`, both deleted in Task 17 anyway, so just add `pytest.skip` at the top for now).

- [ ] **Step 4: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/client/task_tracker.py \
        src/obelix/adapters/inbound/a2a/client/cli_client.py
git commit -m "refactor(a2a): TaskTracker moves to task_tracker.py with peer support"
```

---

## Task 11: CLI ClientConfig switches to polling-only; webhook setup deleted

**Why this task exists:** This is the keystone change of spec 2. The CLI stops booting a webhook server, stops generating a token, stops attaching `PushNotificationConfig`, and from now on only polls.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/client/cli_client.py`

- [ ] **Step 1: Remove the webhook block from `_connect_agents`**

Inside `_connect_agents`, delete:
- The `WebhookServer` instantiation and `await self._webhook_server.start()`.
- The `chat.write(... f"Webhook: {self._webhook_url}" ...)` line.
- The `push_configs = [PushNotificationConfig(...)]` block.
- Set `push_notification_configs=[]` in `ClientConfig`.

- [ ] **Step 2: Delete the webhook-related instance attrs and CLI argparse arg**

In `__init__`: delete `_webhook_server`, `_webhook_url`, `_webhook_host`, `_webhook_token`. Drop the `secrets` and `PushNotificationConfig` imports. The `TaskTracker` no longer needs `context_resolver` — keep `tracker = TaskTracker()` only.

In `from_cli`: remove the `--webhook-host` argparse argument and the `webhook_host=args.webhook_host` constructor kwarg.

- [ ] **Step 3: Delete the metadata-injection block in `_send_message`**

Lines 1031-1044 of the original `cli_client.py` (the `if agent.context_id is None:` block that adds `client_webhook_url` and `client_webhook_token`) — keep `client_info` but drop the two webhook keys. After cleanup the block becomes:

```python
metadata = None
if agent.context_id is None and self._shell_info:
    metadata = {"client_info": self._shell_info}
```

- [ ] **Step 4: Delete `_resolve_agent_by_context`**

Method existed only as a callback for `TaskTracker.context_resolver`. Drop the method body.

- [ ] **Step 5: Delete the unmount cleanup line**

In `on_unmount`, drop `if self._webhook_server: await self._webhook_server.stop()`.

- [ ] **Step 6: Run the existing suite to ensure no regressions in code paths that don't touch polling yet**

Run: `uv run pytest tests/ -q --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: PASS. The polling worker's recursive discovery isn't yet in (next task), but the existing `_polling_fallback` should still drive the basic happy path.

- [ ] **Step 7: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/client/cli_client.py
git commit -m "refactor(cli): drop webhook server, switch to polling-only ClientConfig"
```

---

## Task 12: Replace `_polling_fallback` with `_start_polling` (recursive discovery)

**Why this task exists:** The previous polling code was a *fallback* triggered when push notifications failed. In spec 2 polling is the canonical channel. The new worker walks `T1.metadata.spawned_task_ids` and starts a sibling polling loop for every newly-discovered child. Cadence drops from 1.0s to 0.5s. Polling continues past terminal state on the parent until all spawned children are also terminal.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/client/cli_client.py`
- Create: `tests/adapters/inbound/a2a/client/test_polling_discovery.py`
- Create: `tests/adapters/inbound/a2a/client/test_polling_termination.py`

- [ ] **Step 1: Sketch the new worker shape — read it through before writing code**

The worker:
1. Tracks a per-instance `self._tracked_tasks: set[str]` to dedupe recursive spawns.
2. Per `task_id`, runs an asyncio `Task` that polls `client.get_task` every 0.5s.
3. On each poll: `tracker.update(task_data)`, walk `task_data.metadata.spawned_task_ids`, for any child not in `_tracked_tasks` start a new worker.
4. Walk `task_data.metadata.dispatched_peers`, call `tracker.update_peers(task_id, peers)`.
5. Exit when: parent state is terminal AND all spawned children's workers have exited AND all peers in `dispatched_peers` are terminal.

- [ ] **Step 2: Write the discovery test**

```python
# tests/adapters/inbound/a2a/client/test_polling_discovery.py
"""When the polling worker sees T1.metadata.spawned_task_ids grow, it
starts polling each new task_id and pulls its artifact.

Iron rule: real SDK + FakeA2AServer.
"""

from __future__ import annotations

import asyncio

import pytest
from a2a.types import (
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from tests._fakes.fake_a2a_server import FakeA2AServer

from obelix.adapters.inbound.a2a.client.task_tracker import TaskTracker


# We test the worker as a free-standing function rather than through the
# Textual app — the worker logic must be importable.
from obelix.adapters.inbound.a2a.client.cli_client import poll_task_loop  # NEW symbol


@pytest.mark.asyncio
async def test_worker_discovers_spawned_child_and_pulls_artifact():
    async with FakeA2AServer() as server:
        tracker = TaskTracker()
        await tracker.register("t1", "orchestrator")

        # Seed T1 (terminal) + T3 (will be the spawn).
        await server.task_store.save(
            Task(
                id="t1",
                context_id="ctx",
                status=TaskStatus(state=TaskState.completed),
                metadata=None,
            )
        )
        await server.task_store.save(
            Task(
                id="t3",
                context_id="ctx",
                status=TaskStatus(state=TaskState.completed),
                artifacts=[
                    Artifact(
                        artifact_id="a1",
                        parts=[Part(root=TextPart(text="async response"))],
                    )
                ],
                metadata=None,
            )
        )

        # Start the worker on T1, then 50 ms later patch T1.metadata to expose
        # T3 — simulating the drainer's append.
        async def _patch_after_delay():
            await asyncio.sleep(0.05)
            t = await server.task_store.get("t1")
            t.metadata = {"spawned_task_ids": ["t3"]}
            await server.task_store.save(t)

        patcher = asyncio.create_task(_patch_after_delay())

        await asyncio.wait_for(
            poll_task_loop(
                client=server.client,
                tracker=tracker,
                task_id="t1",
                agent_name="orchestrator",
                tracked=set(),
                interval=0.05,
            ),
            timeout=3.0,
        )
        await patcher

        # The discovered child must now be in the tracker, terminal.
        assert tracker.get("t3") is not None
        assert tracker.get("t3").is_terminal
        assert tracker.get("t3").task_data is not None
```

- [ ] **Step 3: Write the termination test**

```python
# tests/adapters/inbound/a2a/client/test_polling_termination.py
"""The polling worker must NOT exit while a spawned child or a dispatched
peer is still non-terminal."""

from __future__ import annotations

import asyncio

import pytest
from a2a.types import Task, TaskState, TaskStatus

from tests._fakes.fake_a2a_server import FakeA2AServer
from obelix.adapters.inbound.a2a.client.task_tracker import TaskTracker
from obelix.adapters.inbound.a2a.client.cli_client import poll_task_loop


@pytest.mark.asyncio
async def test_worker_keeps_polling_until_peer_terminal():
    async with FakeA2AServer() as server:
        tracker = TaskTracker()
        await tracker.register("t1", "orchestrator")

        await server.task_store.save(
            Task(
                id="t1",
                context_id="ctx",
                status=TaskStatus(state=TaskState.completed),
                metadata={"dispatched_peers": [{"name": "C", "task_id": "tC", "state": "working"}]},
            )
        )

        completed_event = asyncio.Event()

        async def _flip_peer_after_delay():
            await asyncio.sleep(0.15)
            t = await server.task_store.get("t1")
            t.metadata["dispatched_peers"][0]["state"] = "completed"
            await server.task_store.save(t)
            completed_event.set()

        flipper = asyncio.create_task(_flip_peer_after_delay())

        await asyncio.wait_for(
            poll_task_loop(
                client=server.client,
                tracker=tracker,
                task_id="t1",
                agent_name="orchestrator",
                tracked=set(),
                interval=0.05,
            ),
            timeout=3.0,
        )
        assert completed_event.is_set(), "loop must not exit before peer flips terminal"
        await flipper
```

- [ ] **Step 4: Run, expect failures**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_polling_discovery.py tests/adapters/inbound/a2a/client/test_polling_termination.py -v`
Expected: FAIL — `poll_task_loop` does not exist.

- [ ] **Step 5: Implement `poll_task_loop` as a free function in `cli_client.py`**

Place it near the top of the file, just under the existing helpers and **before** the `CLIClient` class. The free-function form makes it directly testable; the Textual class delegates to it.

```python
# src/obelix/adapters/inbound/a2a/client/cli_client.py
# (add near the top, after the existing helpers)

import asyncio
from a2a.client import A2AClientHTTPError, A2AClientJSONError, A2AClientTimeoutError
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import TaskQueryParams

from obelix.adapters.inbound.a2a.client.task_tracker import TaskTracker

_TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}


async def poll_task_loop(
    *,
    client,
    tracker: TaskTracker,
    task_id: str,
    agent_name: str,
    tracked: set[str],
    interval: float = 0.5,
) -> None:
    """Poll a single task and recursively spawn polling for any spawned
    children discovered via T_x.metadata.spawned_task_ids. Returns when
    this task plus all of its descendants and dispatched peers are
    terminal."""
    tracked.add(task_id)
    child_workers: dict[str, asyncio.Task] = {}
    try:
        while True:
            try:
                task = await client.get_task(TaskQueryParams(id=task_id))
            except (A2AClientTimeoutError, A2AClientHTTPError):
                # Transient — try again next tick.
                await asyncio.sleep(interval)
                continue
            except A2AClientJSONRPCError as exc:
                if getattr(exc.error, "code", None) in (-32001, -32002):
                    # Task evicted or terminal-from-our-perspective —
                    # nothing more to observe.
                    return
                raise
            except A2AClientJSONError:
                await asyncio.sleep(interval)
                continue

            data = task.model_dump(mode="json", exclude_none=True)
            await tracker.update(data)
            metadata = data.get("metadata") or {}

            # Discover & spawn workers for new children.
            for child_id in metadata.get("spawned_task_ids", []):
                if child_id not in tracked and child_id not in child_workers:
                    child_workers[child_id] = asyncio.create_task(
                        poll_task_loop(
                            client=client,
                            tracker=tracker,
                            task_id=child_id,
                            agent_name=agent_name,
                            tracked=tracked,
                            interval=interval,
                        ),
                        name=f"poll-{child_id[:8]}",
                    )

            # Surface peer state on the tracker for status-bar rendering.
            await tracker.update_peers(task_id, metadata.get("dispatched_peers", []))

            # Exit conditions.
            state = data.get("status", {}).get("state", "unknown")
            children_done = all(t.done() for t in child_workers.values())
            peers = metadata.get("dispatched_peers", [])
            peers_done = all(p.get("state") in _TERMINAL_STATES for p in peers)
            if state in _TERMINAL_STATES and children_done and peers_done:
                break

            await asyncio.sleep(interval)
    finally:
        # Wait for any in-flight child workers to finish before returning.
        if child_workers:
            await asyncio.gather(*child_workers.values(), return_exceptions=True)
```

- [ ] **Step 6: Replace the old `_poll_tasks` plumbing in `CLIClient`**

In the `CLIClient` class:
1. Remove `self._last_poll`, `self._poll_warned`, the periodic `set_interval(0.15, self._poll_tasks)` (the polling worker now drives itself).
2. After `tracker.register(task.id, agent.name)` in `_send_message`, kick off a worker:

```python
asyncio.create_task(
    poll_task_loop(
        client=agent.client,
        tracker=self.tracker,
        task_id=task.id,
        agent_name=agent.name,
        tracked=self._tracked_tasks,
        interval=0.5,
    ),
    name=f"poll-{task.id[:8]}",
)
```

3. Add `self._tracked_tasks: set[str] = set()` to `__init__`.
4. Keep a small `set_interval(0.15, self._tick_status_bar)` method that only renders the status bar (reads from `tracker.get_all` + `tracker.get_active_peers`). All side-effect logic (polling, recovery messaging, push-warning) is gone.

- [ ] **Step 7: Run the new tests**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_polling_discovery.py tests/adapters/inbound/a2a/client/test_polling_termination.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/client/cli_client.py \
        tests/adapters/inbound/a2a/client/test_polling_discovery.py \
        tests/adapters/inbound/a2a/client/test_polling_termination.py
git commit -m "feat(cli): polling-only worker with recursive spawned-task discovery"
```

---

## Task 13: Status bar surfaces dispatched peers

**Why this task exists:** Spec §2 promises the user sees `O is thinking | C: working` while the orchestrator is delegating. The status bar previously read only from `tracker.get_all()`. Now it must additionally read `tracker.get_active_peers(parent_task_id)` and render one segment per peer.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/client/cli_client.py` — `_update_status_bar`.
- Create: `tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py
"""When the tracker has active peers for the focused task, the status
bar must contain one segment per peer."""

from __future__ import annotations

import pytest

from obelix.adapters.inbound.a2a.client.task_tracker import TaskTracker
from obelix.adapters.inbound.a2a.client.cli_client import render_status_segments


@pytest.mark.asyncio
async def test_status_segments_include_active_peers():
    tracker = TaskTracker()
    await tracker.register("t1", "orchestrator")
    await tracker.update({"id": "t1", "status": {"state": "working"}})
    await tracker.update_peers("t1", [{"name": "C", "task_id": "tc", "state": "working"}])

    segments = render_status_segments(
        tracker=tracker,
        agents=[("orchestrator", "t1")],
        unseen_by_agent={},
        spinner_frame=".",
    )
    flat = " | ".join(segments)
    assert "orchestrator" in flat
    assert "C: working" in flat


@pytest.mark.asyncio
async def test_status_segments_exclude_terminal_peers():
    tracker = TaskTracker()
    await tracker.register("t1", "orchestrator")
    await tracker.update({"id": "t1", "status": {"state": "completed"}})
    await tracker.update_peers("t1", [
        {"name": "C", "task_id": "tc", "state": "completed"},
        {"name": "D", "task_id": "td", "state": "working"},
    ])

    segments = render_status_segments(
        tracker=tracker,
        agents=[("orchestrator", "t1")],
        unseen_by_agent={},
        spinner_frame=".",
    )
    flat = " | ".join(segments)
    assert "C:" not in flat
    assert "D: working" in flat
```

- [ ] **Step 2: Run the test, expect failure**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py -v`
Expected: FAIL — `render_status_segments` does not exist.

- [ ] **Step 3: Extract a free function `render_status_segments`**

In `cli_client.py`, factor out the segment-building loop currently inside `_update_status_bar` into a free function (just like `poll_task_loop`):

```python
# in src/obelix/adapters/inbound/a2a/client/cli_client.py
def render_status_segments(
    *,
    tracker,
    agents,  # list[tuple[agent_name: str, last_task_id: str | None]]
    unseen_by_agent,
    spinner_frame,
):
    segments = []
    for agent_name, last_task_id in agents:
        latest = max(
            (t for t in tracker.get_by_agent(agent_name)),
            key=lambda t: t.timestamp,
            default=None,
        )
        if latest is None:
            continue
        unseen = len(unseen_by_agent.get(agent_name, []))
        if latest.state == "input-required":
            segments.append(f"[bold red]⚠ {agent_name}: INPUT[/]")
        elif latest.state in ("working", "submitted"):
            segments.append(f"[yellow]{spinner_frame} {agent_name} is thinking...[/]")
        elif latest.state == "completed":
            if unseen > 0:
                segments.append(f"[bold green]{agent_name}: {unseen} new[/]")
            else:
                segments.append(f"[green]✓ {agent_name}[/]")
        elif latest.state == "failed":
            if unseen > 0:
                segments.append(f"[bold red]{agent_name}: {unseen} new[/]")
            else:
                segments.append(f"[red]✗ {agent_name}[/]")
        else:
            segments.append(f"{agent_name}: {latest.state}")

        # NEW: append peer segments for this parent task.
        if last_task_id:
            for peer in tracker.get_active_peers(last_task_id):
                segments.append(f"[dim]{peer.name}: {peer.state}[/]")
    return segments
```

- [ ] **Step 4: Make `_update_status_bar` call the new helper**

Adapt `_update_status_bar` to a thin wrapper that builds the `agents` list from `self.agents` and calls `render_status_segments`.

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/client/cli_client.py \
        tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py
git commit -m "feat(cli): status bar renders dispatched peer state"
```

---

## Task 14: Silence `-32001` / `-32002` from `cancel_task`

**Why this task exists:** Spec §4 "Errori su `cancel_task`" + research finding "cancel-no-cascade" Q4: cancelling a task that's already terminal returns `A2AClientJSONRPCError(-32002)`; cancelling a task evicted from the store returns `-32001`. These are no-ops — the user pressed ESC and the task is already done. They must NOT surface as red error messages.

**Files:**
- Modify: `src/obelix/adapters/inbound/a2a/client/cli_client.py` — `_cancel_current_task`.
- Create: `tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py
"""When ESC fires after the task has already gone terminal, the cancel
RPC raises -32002. The CLI must NOT print an error message."""

from __future__ import annotations

import pytest
from a2a.types import Task, TaskState, TaskStatus

from tests._fakes.fake_a2a_server import FakeA2AServer
from obelix.adapters.inbound.a2a.client.cli_client import safe_cancel_task


@pytest.mark.asyncio
async def test_cancel_terminal_task_is_silent():
    async with FakeA2AServer() as server:
        await server.task_store.save(
            Task(
                id="t1",
                context_id="ctx",
                status=TaskStatus(state=TaskState.completed),
                metadata=None,
            )
        )
        # Should NOT raise, returns False to indicate "no cancel performed".
        ok = await safe_cancel_task(client=server.client, task_id="t1")
        assert ok is False


@pytest.mark.asyncio
async def test_cancel_unknown_task_is_silent():
    async with FakeA2AServer() as server:
        ok = await safe_cancel_task(client=server.client, task_id="unknown")
        assert ok is False
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py -v`
Expected: FAIL — `safe_cancel_task` not defined.

- [ ] **Step 3: Implement `safe_cancel_task` and rewire `_cancel_current_task`**

```python
# src/obelix/adapters/inbound/a2a/client/cli_client.py
# (free helper near poll_task_loop)

from a2a.types import TaskIdParams

async def safe_cancel_task(*, client, task_id: str) -> bool:
    """Cancel `task_id`. Returns True if the cancel actually happened,
    False if the SDK reported the task is gone or already terminal
    (silent no-op)."""
    try:
        await client.cancel_task(TaskIdParams(id=task_id))
        return True
    except A2AClientJSONRPCError as exc:
        if getattr(exc.error, "code", None) in (-32001, -32002):
            return False
        raise
```

In `_cancel_current_task`, replace the existing `await agent.client.cancel_task(...)` block with `await safe_cancel_task(client=agent.client, task_id=task_id)` and use the boolean to decide whether to print the "Task canceled." line. If False, write `Text("  Task already done.", style="dim")` instead of an error.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/inbound/a2a/client/cli_client.py \
        tests/adapters/inbound/a2a/client/test_cancel_silent_terminal.py
git commit -m "feat(cli): silence -32001/-32002 from cancel_task as no-op"
```

---

## Task 15: Delete `webhook_server.py` and the obsolete test files

**Why this task exists:** All consumers of `webhook_server.py` and its companion regression tests are gone. Removing dead code closes the loop on the cleanup.

**Files:**
- Delete: `src/obelix/adapters/inbound/a2a/client/webhook_server.py`
- Delete: `tests/adapters/inbound/a2a/client/test_webhook_server_token.py`
- Delete: `tests/adapters/inbound/a2a/client/test_cli_webhook_metadata.py`
- Delete: `tests/adapters/inbound/a2a/client/test_task_tracker_unknown_agent.py`

- [ ] **Step 1: Verify no remaining imports**

Run: `uv run rg -l "from obelix.adapters.inbound.a2a.client.webhook_server" src/ tests/`
Expected: empty.

- [ ] **Step 2: Delete**

```bash
git rm src/obelix/adapters/inbound/a2a/client/webhook_server.py
git rm tests/adapters/inbound/a2a/client/test_webhook_server_token.py
git rm tests/adapters/inbound/a2a/client/test_cli_webhook_metadata.py
git rm tests/adapters/inbound/a2a/client/test_task_tracker_unknown_agent.py
```

- [ ] **Step 3: Run the suite**

Run: `uv run pytest tests/ -q --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py --ignore=tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`
Expected: PASS.

- [ ] **Step 4: Verify the marker counter dropped further**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_temp_patch_marker.py -v`
Expected: FAIL — count keeps shrinking. Update `EXPECTED_COUNT` to the actual current count and rerun. PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/adapters/inbound/a2a/server/test_temp_patch_marker.py
git commit -m "cleanup(a2a): delete webhook_server + obsolete tests"
```

---

## Task 16: Adapt the legacy `test_drain_spawn_*.py` to the new flow

**Why this task exists:** Three skipped tests from Task 5 still claim spec-1 behavior (POST to webhook, `_DrainSpawnEventQueue`). Spec 2's drain-spawn flow delivers via `T1.metadata.spawned_task_ids` + a regular SDK Task. Either rewrite each test to assert the new flow, or delete it if the new equivalent already exists.

**Files:**
- Modify: `tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py`
- Modify: `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py`
- Modify: `tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py`

- [ ] **Step 1: Decide per-file**

For each file, audit:
- `test_drain_spawn_artifacts_bug.py` (spec-1 Bug 5): asserted that `_DrainSpawnEventQueue` accumulates artifacts via `_merge_artifact_update`. Replacement: an integration test that verifies `tasks/get(T3)` returns artifacts after the drain run finishes. Rewrite using `FakeA2AServer`.
- `test_drain_spawn_e2e.py`: full drain-spawn end-to-end. Replacement: rewrite to use `FakeA2AServer`, assert that after the drain run the parent's `T1.metadata.spawned_task_ids` includes T3 AND `tasks/get(T3)` returns the synthesized artifact.
- `test_drain_spawn_webhook_post.py`: asserted webhook POST behavior. Spec 2 has no webhook → **delete**.

- [ ] **Step 2: Rewrite `test_drain_spawn_e2e.py`**

```python
# tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py
"""End-to-end drain-spawn test for spec 2: the drainer registers T3 in
the SDK TaskStore, patches T1.metadata.spawned_task_ids, and the polling
client retrieves the synthesized artifact via tasks/get(T3)."""

from __future__ import annotations

import asyncio

import pytest
from a2a.types import (
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from tests._fakes.fake_a2a_server import FakeA2AServer

# (rest of the test exercises maybe_spawn_drain_task end-to-end against
#  FakeA2AServer's task_store, asserting the spec-2 contract:
#  - T3 saved before T1.metadata patched
#  - tasks/get(T3) returns the artifact
#  - polling client sees the new child via metadata)
# Keep the body small but focused on those three asserts.
```

(Implementer: complete the body following the same shape as Task 4's test, with the addition that T3's artifact is observable via `await server.client.get_task(TaskQueryParams(id=t3))`.)

- [ ] **Step 3: Rewrite `test_drain_spawn_artifacts_bug.py`**

Same pattern. After the drain run, assert `tasks/get(T3).artifacts` contains the expected text. Drop all references to `_DrainSpawnEventQueue`, `_merge_artifact_update`, etc.

- [ ] **Step 4: Delete `test_drain_spawn_webhook_post.py`**

```bash
git rm tests/adapters/inbound/a2a/server/test_drain_spawn_webhook_post.py
```

- [ ] **Step 5: Run the rewritten tests**

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py -v`
Expected: PASS.

- [ ] **Step 6: Run the entire suite — no exclusions this time**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/adapters/inbound/a2a/server/test_drain_spawn_artifacts_bug.py \
        tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py
git commit -m "test(spec2): rewrite drain-spawn tests for metadata-based delivery"
```

---

## Task 17: Drop the `TEMP-PATCH-SPEC-1` marker counter

**Why this task exists:** Once every patch site is gone, the counter fires zero. The counter test exists only to detect drift while the marker survives. With the marker gone the test becomes noise.

**Files:**
- Modify: `tests/adapters/inbound/a2a/server/test_temp_patch_marker.py` (set EXPECTED_COUNT=0 and verify) then delete.

- [ ] **Step 1: Verify zero markers in src/**

Run (PowerShell): `uv run pytest tests/adapters/inbound/a2a/server/test_temp_patch_marker.py -v`
Expected: count is non-zero (we may still have some sentinel comments in places we missed). If non-zero, locate and remove every remaining `TEMP-PATCH-SPEC-1` mention via Grep over `src/`. Re-run until count is 0.

- [ ] **Step 2: Set EXPECTED_COUNT to 0, run, expect PASS**

Edit `tests/adapters/inbound/a2a/server/test_temp_patch_marker.py` line `EXPECTED_COUNT = ...` to `0`.

Run: `uv run pytest tests/adapters/inbound/a2a/server/test_temp_patch_marker.py -v`
Expected: PASS.

- [ ] **Step 3: Delete the test file**

```bash
git rm tests/adapters/inbound/a2a/server/test_temp_patch_marker.py
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "cleanup(a2a): drop TEMP-PATCH-SPEC-1 marker counter (count=0, all sites gone)"
```

---

## Task 18: Update roadmap status, project memory, and run the smoke

**Why this task exists:** The roadmap doc and memory entries advertise spec 2 as `pending`. The implementation is now complete; downstream readers (and future agent sessions) need to see it as `implemented`.

**Files:**
- Modify: `docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md`
- Create: `C:\Users\GLoverde\.claude\projects\C--Users-GLoverde-PycharmProjects-Obelix\memory\project_a2a_spec2_polling.md`
- Modify: `C:\Users\GLoverde\.claude\projects\C--Users-GLoverde-PycharmProjects-Obelix\memory\MEMORY.md`
- Modify: `C:\Users\GLoverde\.claude\projects\C--Users-GLoverde-PycharmProjects-Obelix\memory\project_a2a_temporary_webhook_patch.md` — mark resolved.

- [ ] **Step 1: Edit the roadmap**

Open `docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md`. Under § Spec 2 — C, change `Status: pending` to `Status: implemented (2026-05-08)`. Add a one-liner under the existing scope:

> Implemented 2026-05-08. CLI is polling-only; webhook eliminated; `T1.metadata.{spawned_task_ids,dispatched_peers}` carries discovery + peer status.

- [ ] **Step 2: Create the new memory entry**

```markdown
# Spec 2 — A2A CLI polling-only (IMPLEMENTED 2026-05-08)

Branch: `a2a-spec2-cli-streaming` (despite the name, ended up polling-only — SSE was scoped out).

## What changed
- CLI uses `ClientConfig(streaming=False, polling=True, push_notification_configs=[])`.
- Polling worker `poll_task_loop` (free function in `cli_client.py`) walks `T1.metadata.spawned_task_ids` recursively.
- Status bar reads `tracker.get_active_peers(parent_task_id)` for peer visibility.
- Drainer writes `T1.metadata.spawned_task_ids` via `update_task_metadata` helper (in-place TaskStore patch).
- `dispatch_agent` writes `T1.metadata.dispatched_peers`; webhook/polling/handler mirror state changes.
- Cancel of terminal/unknown task is silent (-32001 / -32002).
- `webhook_server.py` deleted, `_DrainSpawnEventQueue` / `_NullEventQueue` deleted, slot `client_webhook_url`/`client_webhook_token` deleted.
- TEMP-PATCH-SPEC-1 marker count → 0; counter test deleted.

## Why polling, not SSE
SSE only buys token-streaming live, which the CLI didn't have anyway (`_collect_task` consumed the iterator). Polling is simpler, NAT-friendly, no reconnect logic. Pattern: Claude Code's `RemoteAgentTask.tsx:564`.

## Key SDK research findings
- `TaskStatusUpdateEvent` post-terminal does NOT propagate (queue closed). Only `TaskStore.save` works.
- `JsonRpcTransport.get_task` has no client-side caching. Fresh metadata visible on the very next poll.
- Cancel is task-scoped (no cascade across `context_id`).
- Multi-Task per `context_id` supported; SDK does not provide context-level lookup.

## Files of interest (after spec 2)
- `src/obelix/adapters/inbound/a2a/server/metadata_patch.py` — the in-place patch helper.
- `src/obelix/adapters/inbound/a2a/client/task_tracker.py` — moved out of `webhook_server.py`.
- `src/obelix/adapters/inbound/a2a/client/cli_client.py` — `poll_task_loop`, `safe_cancel_task`, `render_status_segments` are testable free functions.
- `tests/_fakes/fake_a2a_server.py` — reusable in-process A2A endpoint.

## Spec & research
- `docs/superpowers/specs/2026-05-08-a2a-cli-polling-spec2-design.md`
- `docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md`
```

- [ ] **Step 3: Update `MEMORY.md` index**

Append (under the A2A section):

```
- [Spec 2 polling (IMPLEMENTED 2026-05-08)](project_a2a_spec2_polling.md) — CLI polling-only, webhook eliminated, metadata-based discovery via T1.metadata.{spawned_task_ids,dispatched_peers}.
```

- [ ] **Step 4: Mark `project_a2a_temporary_webhook_patch.md` resolved**

Open the file, change the marker count to 0 and add status `RESOLVED 2026-05-08 by spec 2 implementation`.

- [ ] **Step 5: Run the full smoke suite once more**

Run: `uv run pytest tests/ -q`
Expected: PASS.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: PASS.

- [ ] **Step 6: Final smoke (manual e2e)**

Run the orchestrator example to verify the CLI shows the dispatched peer and receives the async response without webhook involvement:

```
# terminal 1 — start orchestrator + the 4 example agents
uv run python examples/orchestrator_server.py  # this runs all 5 servers per spec 1 setup
# terminal 2 — CLI client
uv run python -m obelix.adapters.inbound.a2a.client.cli_client http://localhost:8005
# in CLI:
> hi orchestrator, ask the coordinator what 12 + 30 is
```

Expected behavior:
- Status bar: `O is thinking | coordinator: working` → `O ✓ | coordinator: working` → `O ✓ | coordinator ✓`.
- Final synthesized response from O appears in chat panel within ~1s of coordinator completing.
- No `Webhook: http://...` line on connect.
- No `unknown: 1 new` segment.
- Pressing ESC during O's first turn cancels T1 silently if T1 already completed.

- [ ] **Step 7: Final commit + push**

```bash
git add docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md
# memory files are outside the repo, no need to git-add — just save them.
git commit -m "docs(spec2): roadmap status -> implemented, smoke pass"
git push origin a2a-spec2-cli-streaming
```

---

## Self-review

**Spec coverage check (each spec section maps to a task):**
- § 1 Problema → Tasks 11-15 (CLI cleanup) and Tasks 8-9 (server cleanup). Both axes covered.
- § 2 Architettura → Tasks 4 (spawned_task_ids), 6 (dispatched_peers), 11 (ClientConfig), 12 (poll_task_loop), 14 (cancel silent).
- § 3 Data flow & metadata schema → Tasks 3 (helper), 4 (drainer), 6 (dispatch tool), 7 (state mirroring).
- § 4 Lifecycle → Task 12 covers polling exit conditions; Task 14 covers cancel error semantics; Task 4 covers race ordering.
- § 5 Cleanup → Tasks 8, 9, 10, 15, 16, 17.
- § 6 Testing → All tasks include integration tests via FakeA2AServer (Task 1).
- § 7 Out of scope → no implementation tasks needed.
- § 8 Resolved open assumptions → tasks already incorporate the resolutions; the section is informational.
- § 9 Success criteria → Task 18 step 6 manual smoke covers each criterion (TEMP-PATCH count 0, no webhook, status bar peers, ESC no cascade, NAT-able).
- § 10 File list → matches the "File structure" section above.

**Placeholder scan:**
- No "TBD"/"TODO"/"add appropriate error handling" anywhere in the body.
- One open invariant in Task 5 step 2 (the implementer may find a single SDK helper that replaces the manual `EventConsumer` plumbing). The instruction is explicit: read the SDK source, prefer the helper if found, document in commit message. Not a placeholder, just a tactical choice the implementer makes with the source in hand.

**Type/symbol consistency:**
- `update_task_metadata(store, task_id, patch_fn)`: defined Task 3, used Tasks 4, 6, 7. ✓
- `update_dispatched_peer_state(store, parent_task_id, peer_task_id, new_state)`: defined Task 7, used Task 7. ✓
- `poll_task_loop(*, client, tracker, task_id, agent_name, tracked, interval)`: defined Task 12, used Task 12. ✓
- `safe_cancel_task(*, client, task_id)`: defined Task 14, used Task 14. ✓
- `render_status_segments(*, tracker, agents, unseen_by_agent, spinner_frame)`: defined Task 13, used Task 13. ✓
- `ObelixAgentExecutor.__init__` adds `task_store`, drops `httpx_client`: enforced Task 2, used in Tasks 4-7. ✓
- `DispatchAgentTool.set_context_entry(entry, *, context_id, parent_task_id)` + `set_task_store`: defined Task 6, called by `_inject_context_entry` Task 6 step 5. ✓
- `TaskTracker.update_peers(parent_task_id, peers)` / `get_active_peers(parent_task_id)`: defined Task 10, used Task 12 (write) + Task 13 (read). ✓
- `ContextEntry.current_task_id`: introduced in Task 7 step 4 NOTE for the executor to set. The executor change must land alongside Task 7 (mentioned explicitly in step 4).

The plan is internally consistent. Implementer moves task-by-task with TDD discipline, frequent commits, and integration tests gated on the FakeA2AServer foundation.
