# A2A Agent Client — Async Inter-Agent Dispatch

**Date**: 2026-04-30
**Status**: Draft (pending review)
**Branch**: `a2a_agent_client`
**Related**: `docs/superpowers/specs/2026-04-21-tracer-refactor-design.md`, A2A spec §13.2 (Push Notification Security)

## 1. Context

Today an Obelix `BaseAgent` exposed via `AgentFactory.a2a_serve()` can be **contacted** as an A2A server, but cannot **call other A2A agents**. The only existing A2A client code lives inside the human-facing CLI (`adapters/inbound/a2a/client/cli_client.py`). There is no `adapters/outbound/a2a/`.

We want a server agent A to be able to dispatch tasks to other server agents B, C, …, in a **fire-and-forget asynchronous fashion**:

- A's reasoning loop is not blocked while B works.
- A continues its turn (or ends it) immediately after dispatch.
- B reports state changes (`working`, `completed`, `failed`, `canceled`, `rejected`, `input_required`) via A2A push notifications to A.
- A surfaces those updates to its LLM at the next turn, allowing it to react.
- A can dispatch to multiple remotes in parallel and handle responses as they arrive.

This is the same pattern Claude Code implements with `AgentTool` + `RemoteAgentTask` + a notification queue, adapted to the A2A protocol semantics.

## 2. Goals

- A `BaseAgent` configured with `remote_agents=[urls]` discovers and can call those agents over A2A.
- Dispatch is non-blocking: the dispatch tool returns a `task_id` immediately, A continues.
- Remote state changes propagate back to A via A2A push notifications, authenticated via per-task tokens.
- Updates of interest (terminal states, `input_required`) are injected into A's conversation as user-role messages wrapped in `<remote_task_update>` XML at the start of A's next loop iteration.
- The LLM in A has tools to dispatch, list, inspect, respond to, and stop tracking remote tasks.
- Multi-tenant isolation: when A serves multiple conversations (`context_id`s), notifications for one context do not leak into another.
- Polling fallback when push delivery fails (no NAT/firewall complications).

## 3. Non-Goals

- Cancellation cascade across the agent network. `task_stop` only stops local tracking; the remote keeps running until it terminates on its own (matches Claude Code's model).
- Persistence of in-flight tasks across A's process restart. In v1, a crash drops the token map; B's subsequent webhooks return 401.
- Distributed tracing across A2A boundaries. Each agent opens an independent trace; cross-agent linkage via shared task_id metadata is deferred.
- Continued (multi-turn) conversations with the same remote agent. v1 is stateless: every `dispatch_agent` call uses a fresh `context_id` on the remote.
- Replacing the CLI's `WebhookServer`. The CLI runs without an A2A server backbone and must keep its standalone webhook. Shared concerns (polling fallback, task state struct) may be factored later but are not in scope here.
- Authentication beyond `PushNotificationConfig.token`. The richer `AuthenticationInfo` scheme (Bearer/OAuth/etc.) is reserved for a future iteration.

## 4. Architecture

```
┌──────────────── A's process (BaseAgent + a2a_serve) ────────────────┐
│                                                                      │
│  uvicorn :8000                                                       │
│   ├── /a2a/...           inbound A2A server (existing)              │
│   └── /webhook           NEW — receives push notifications          │
│                                                                      │
│  ContextStore                                                        │
│   └── ContextEntry["ctx-AAA"]                                        │
│        ├── (existing fields: history, idle, deferred_*)             │
│        ├── remote_tasks: dict[task_id → RemoteTaskState]   NEW      │
│        └── pending_notifications: list[HumanMessage]        NEW      │
│                                                                      │
│  RemoteAgentRegistry  (process-wide singleton)              NEW      │
│   ├── cards: dict[name → AgentCard]                                  │
│   ├── clients: dict[name → A2AClient]                                │
│   └── token_map: dict[token → TokenRoute]                            │
│                                                                      │
│  BaseAgent (extended with remote_agents= and 5 new tools)            │
│   └── dispatch_agent · respond_to_remote · task_list ·               │
│        task_get · task_stop                                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 Component summary

| Component | New / Existing | Location |
|---|---|---|
| `RemoteAgentRegistry` | New | `adapters/outbound/a2a/registry.py` |
| `RemoteTaskState` (dataclass) | New | `adapters/outbound/a2a/state.py` |
| `TokenRoute` (dataclass) | New | `adapters/outbound/a2a/state.py` |
| `/webhook` Starlette route | New | `adapters/outbound/a2a/webhook.py` |
| `DispatchAgentTool` | New | `adapters/outbound/a2a/tools/dispatch.py` |
| `RespondToRemoteTool` | New | `adapters/outbound/a2a/tools/respond.py` |
| `TaskListTool` / `TaskGetTool` / `TaskStopTool` | New | `adapters/outbound/a2a/tools/task_ops.py` |
| `ContextEntry` extension | Existing (extended) | `adapters/inbound/a2a/server/context.py` |
| `BaseAgent.__init__(remote_agents=...)` | Existing (extended) | `core/agent/base_agent.py` |
| Executor injection of context entry | Existing (extended) | `adapters/inbound/a2a/server/executor.py` |
| Pending-notification drain | New | `adapters/inbound/a2a/server/executor.py` |
| Polling fallback worker | New | `adapters/outbound/a2a/polling.py` |

### 4.2 Reused infrastructure

- `a2a.client.A2ACardResolver` — agent card discovery (well-known path).
- `a2a.client.ClientFactory.create()` — A2A client construction.
- `Client.send_message()` — outbound dispatch with `pushNotificationConfig`.
- `Client.get_task()` — polling fallback.
- `Client.cancel_task()` — exposed but **not** used in v1 (`task_stop` is local-only).
- `SmartPushNotificationSender` (server-side) — already attaches `X-A2A-Notification-Token` to outgoing webhooks.
- `AgentFactory.a2a_serve()` — existing server entry point, mounts the new `/webhook` route on the same Starlette app.

## 5. Data Flow

### 5.1 Outbound dispatch

User Mario is in `ctx-AAA` on agent A. The LLM calls `dispatch_agent("B", "do X")`.

1. `DispatchAgentTool.execute()` reads its injected `ctx_entry` (per-request, see §6.4).
2. Looks up the AgentCard and A2A client for "B" via `registry.client_for("B")`.
3. Generates `token = secrets.token_urlsafe(32)`.
4. `registry.register_token(token, context_id="ctx-AAA", agent_name="B")` — stores route with `task_id=None`, fresh `asyncio.Event`.
5. Builds `PushNotificationConfig(url="http://a:8000/webhook", token=token)`.
6. `task = await client_for_B.send_message(message, push_config=cfg)` — synchronous on the wire but fast (B returns the new task_id immediately, before doing any work).
7. `registry.claim_task_id(token, task.id)` — fills in the missing field, sets the `Event`.
8. `ctx_entry.remote_tasks[task.id] = RemoteTaskState(agent_name="B", status="submitted", token=token, created_at=now, last_update=now, last_artifact=None, deferred_calls=None)`.
9. Tool returns `{"task_id": task.id, "status": "submitted", "agent": "B"}`.
10. The agent loop continues. The LLM may dispatch more or end the turn.

### 5.2 Inbound webhook

B emits a state change. `SmartPushNotificationSender` (B's side) POSTs to `http://a:8000/webhook` with header `X-A2A-Notification-Token: <token>` and the JSON-serialized `Task` in the body.

1. `/webhook` handler reads the token from the header.
2. `route = registry.lookup(token)` — if `None`, respond 401, log warning, exit.
3. If `route.task_id is None` (race: webhook arrived before `send_message` returned), use `body.id` as task_id.
4. Locate `ctx_entry = context_store.get(route.context_id)`. If evicted (see §7.2), respond 200, log, exit.
5. Idempotency check: if `body.status.state == ctx_entry.remote_tasks[task_id].status`, respond 200, exit.
6. Update `ctx_entry.remote_tasks[task_id]` with new status, last_update, last_artifact, deferred_calls (if `input_required`).
7. If state is **terminal** (`completed`/`failed`/`canceled`/`rejected`) or **`input_required`**:
   - Build an XML-wrapped `HumanMessage`:
     ```xml
     <remote_task_update>
       <task_id>...</task_id>
       <agent>B</agent>
       <status>completed</status>
       <result>... text from artifacts ...</result>
       <!-- on input_required, instead of <result>: -->
       <deferred_tool_calls>[{tool_name, arguments}, ...]</deferred_tool_calls>
       <!-- on failed/rejected, instead of <result>: -->
       <error>... reason ...</error>
     </remote_task_update>
     ```
   - Append to `ctx_entry.pending_notifications`.
   - On terminal states only: `registry.revoke(token)` to free the slot.
8. Respond 200.

Intermediate states (`working`, `submitted`) update `remote_tasks` but do **not** generate notifications. The LLM can poll proactively via `task_get(task_id)` if it wants progress visibility.

### 5.3 Notification drain

When the next request arrives on `ctx-AAA` (Mario speaks again, or any other event scheduling a turn), the executor's `_run_agent` enters. Before delegating to `BaseAgent.execute_query_stream()` it drains the queue:

```python
if entry.pending_notifications:
    for note in entry.pending_notifications:
        agent.conversation_history.append(note)
    entry.pending_notifications.clear()
```

The LLM at the next iteration sees the user's new message *and* the `<remote_task_update>` XML in the conversation. It reacts naturally: report to Mario, dispatch follow-up to another agent, ignore, etc.

### 5.4 input_required cycle

When B's `<remote_task_update>` carries `<status>input_required</status>`, the LLM reads the embedded `<deferred_tool_calls>` payload and decides what to do. To answer, it calls:

```
respond_to_remote(task_id="t-001", data={...payload matching B's tool OutputSchema...})
```

The `RespondToRemoteTool.execute()`:

1. Looks up `ctx_entry.remote_tasks[task_id]` (must exist and be in `input_required` state).
2. Resolves `agent_name → A2AClient`.
3. Sends a follow-up `Message` to B with a `DataPart(data=...)` and the same `task_id`/`context_id` B used.
4. Updates local state to `submitted` (B will resume and emit further updates via webhook).
5. Returns `{"task_id": ..., "status": "submitted"}` to the LLM.

If A's LLM has no information sufficient to answer, it can:
- Ask its own user (a normal conversational turn).
- If A is itself a remote called by another parent P, propagate by emitting its **own** `request_user_input` deferred tool — that becomes A's `input_required` to P, and the chain continues. (This composes naturally; no special code needed.)

### 5.5 Polling fallback

A worker per `ctx_entry` (started lazily on first dispatch) inspects `remote_tasks` periodically:

- Every 30s, scan non-terminal tasks where `now - last_update > 30s`.
- For each, call `client.get_task(task_id)`.
- Feed the response into the **same code path** the webhook handler uses (idempotency check applies, notifications generated identically).
- On terminal state, no further polling.

This means: if pushes arrive, polling is silent. If pushes fail, polling delivers the same outcome with up-to-30s latency. There is no observable behavior difference for the LLM.

## 6. Components in Detail

### 6.1 `RemoteAgentRegistry`

```python
# adapters/outbound/a2a/registry.py
class RemoteAgentRegistry:
    """Process-wide registry of known remote agents and their in-flight tokens.

    Singleton lifecycle: created in AgentFactory.a2a_serve() before uvicorn
    starts. Owns the long-lived httpx client used for all outbound A2A calls
    and AgentCard fetches.
    """

    def __init__(self, urls: list[str], httpx_client: httpx.AsyncClient): ...

    async def resolve_all(self) -> None:
        """Fetch /.well-known/agent-card.json for each URL.
        Logs and skips on individual failures; raises on duplicate names."""

    def card_for(self, name: str) -> AgentCard: ...
    def client_for(self, name: str) -> A2AClient: ...
    def names(self) -> list[str]: ...
    def descriptions(self) -> dict[str, dict]:
        """Returns {name: {description, skills}} for system_prompt_fragment."""

    def register_token(self, token, context_id, agent_name) -> None: ...
    def claim_task_id(self, token, task_id) -> None: ...
    def lookup(self, token: str) -> TokenRoute | None: ...
    def revoke(self, token: str) -> None: ...
    async def gc_expired(self, ttl_seconds: int = 86400) -> int:
        """Remove tokens older than TTL. Called periodically (every 1h)."""
```

### 6.2 State dataclasses

```python
# adapters/outbound/a2a/state.py
@dataclass
class RemoteTaskState:
    task_id: str
    agent_name: str
    status: str  # submitted | working | input_required | completed | failed | canceled | rejected | killed
    created_at: datetime
    last_update: datetime
    last_artifact: dict | None
    deferred_calls: list[dict] | None  # populated when status == "input_required"
    token: str

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "canceled", "rejected", "killed")


@dataclass
class TokenRoute:
    context_id: str
    agent_name: str
    task_id: str | None
    awaiting_task_id: asyncio.Event
    registered_at: datetime
```

### 6.3 `ContextEntry` extension

Two new slots added to the existing `ContextEntry` in `adapters/inbound/a2a/server/context.py`:

```python
self.remote_tasks: dict[str, RemoteTaskState] = {}
self.pending_notifications: list[HumanMessage] = []
```

Added method to support eviction protection:

```python
def is_evictable(self) -> bool:
    """LRU should not evict this context if there are non-terminal remote
    tasks in flight. Otherwise their webhook returns will hit 401 and the
    user-visible behavior is broken."""
    return all(t.is_terminal for t in self.remote_tasks.values())
```

`ContextStore.get_or_create()` is updated to skip non-evictable entries when picking eviction victims.

### 6.4 Per-request context injection

The executor (`adapters/inbound/a2a/server/executor.py`) already has the precedent of `_inject_client_info(agent, entry.client_info)`. We add the symmetric:

```python
def _inject_context_entry(self, agent: BaseAgent, entry: ContextEntry) -> None:
    for tool in agent.registered_tools:
        if hasattr(tool, "set_context_entry"):
            tool.set_context_entry(entry)
```

Called once after `_inject_client_info`. Each remote-task tool implements `set_context_entry(self, entry: ContextEntry) -> None` storing the reference for the duration of its `execute()`.

Thread safety follows from: each request creates a fresh `BaseAgent` and fresh tool instances via `agent_factory()`; no shared mutable state.

### 6.5 The five tools

#### `dispatch_agent`

```python
@tool(name="dispatch_agent", description=...)
class DispatchAgentTool:
    agent_name: str = Field(..., description="Name of remote agent to call")
    query: str = Field(..., description="Task description for the remote agent")

    def system_prompt_fragment(self) -> str: ...   # see §6.6
    def set_context_entry(self, entry: ContextEntry): ...
    def __init__(self, registry: RemoteAgentRegistry): ...

    async def execute(self) -> dict:
        # §5.1 flow
        return {"task_id": ..., "status": "submitted", "agent": ...}
```

Returns immediately. `is_deferred=False`.

#### `respond_to_remote`

```python
@tool(name="respond_to_remote", description="...")
class RespondToRemoteTool:
    task_id: str = Field(...)
    data: dict = Field(..., description="Response payload (matches remote tool's OutputSchema)")
    # see §5.4
```

Returns `{"task_id": ..., "status": "submitted"}`. Errors if task is not in `input_required` state.

#### `task_list`

```python
@tool(name="task_list", description="...", read_only=True)
class TaskListTool:
    async def execute(self) -> dict:
        return {
            "tasks": [
                {"task_id": ..., "agent": ..., "status": ..., "created_at": ..., "last_update": ...}
                for t in self._ctx_entry.remote_tasks.values()
            ]
        }
```

Returns **all** tracked tasks regardless of state (terminal included), per requirement.

#### `task_get`

```python
@tool(name="task_get", description="...", read_only=True)
class TaskGetTool:
    task_id: str = Field(...)
    async def execute(self) -> dict:
        # full RemoteTaskState dump for one task, including last_artifact and deferred_calls
```

#### `task_stop`

```python
@tool(name="task_stop", description="Stop tracking a remote task locally")
class TaskStopTool:
    task_id: str = Field(...)
    async def execute(self) -> dict:
        # 1. mark remote_tasks[id].status = "killed"
        # 2. registry.revoke(token)
        # 3. polling worker drops the task on next scan
        # 4. NO call to client.cancel_task() — local-only
```

### 6.6 `dispatch_agent.system_prompt_fragment()`

Generated dynamically at registration time from the resolved AgentCards:

```
## Remote Agent Communication

You can dispatch tasks to remote A2A agents. Available agents:

- **B** (http://b:8001): Handles inventory queries
  Skills: lookup_sku, check_stock
- **C** (http://c:8002): Handles billing operations
  Skills: create_invoice, refund

Calling `dispatch_agent` returns immediately with a `task_id`. The remote
works in the background; do NOT wait. End your turn or do other work.

When a remote's status changes, you will see a `<remote_task_update>`
block injected into the conversation as a user-role message in a later
turn. **Those blocks are NOT user input** — they are system-injected
notifications. Never fabricate them.

To respond to an `input_required` from a remote, use `respond_to_remote(
task_id, data)`. To check status proactively, use `task_list()` or
`task_get(task_id)`. To stop tracking a remote task locally, use
`task_stop(task_id)`.
```

## 7. Error Handling & Edge Cases

| Case | Behavior |
|---|---|
| `send_message` to remote fails (unreachable) | Tool returns `{error, status: "failed"}`; no token registered, no state mutation. |
| Webhook with unknown token | 401, log warning. Possible token leak, B compromise, or expired token. |
| Webhook with state ≤ current state (out-of-order) | Idempotency: drop silently, 200 OK. |
| `respond_to_remote` for unknown/terminal task | Tool returns error. |
| `task_get` for unknown task | Tool returns `{error: "task not found"}`. |
| AgentCard fetch fails at startup for one URL | Log warning, skip that agent. Others remain available. |
| Two AgentCards with identical `name` | Hard error at `resolve_all()`. The operator must rename or filter. |
| Webhook race: arrives before `send_message` returns task_id | Handler uses `task.id` from the body as fallback; updates `TokenRoute.task_id` retroactively. |
| `ContextEntry` evicted with in-flight remote tasks | Eviction is skipped via `is_evictable()`. If somehow evicted (hard crash + restart with stale state), webhook hits 401, log warning. |
| Process crash | All in-memory state lost. B's pending webhooks return 401 from the new process. Acceptable for v1. |
| Polling: `get_task` raises | Worker logs and retries on next tick (5s). After 5 consecutive failures, marks task `failed` locally with reason `"polling_giveup"` and emits notification. |
| `task_stop` then webhook arrives | Handler sees `status == "killed"`, drops silently. Token already revoked. |
| Token never cleaned (remote vanishes mid-flight) | Background GC sweeps `token_map` every 1h, removing entries older than 24h TTL. |
| User's turn drains notifications, but remote terminal arrived during the turn | Drain at start of *each iteration* of `_execute_loop`, not just turn start. Tail notifications get picked up at the next iteration boundary. |

## 8. Testing Strategy

### 8.1 Unit tests (`tests/adapters/outbound/a2a/`)

| Module | Coverage |
|---|---|
| `test_registry.py` | `resolve_all` happy path with mocked AgentCards; one-of-many failure → skip; duplicate names → hard error; `register_token` / `claim_task_id` race handling; `lookup` after revoke returns None; `gc_expired` removes only old entries |
| `test_state.py` | `RemoteTaskState.is_terminal` matrix; `TokenRoute.awaiting_task_id` event semantics |
| `test_dispatch_tool.py` | Schema validation; happy path with mocked client; remote unreachable → error; `set_context_entry` injection works; correct token registered before send_message; correct state added on success |
| `test_respond_tool.py` | Reject if task unknown; reject if task not in `input_required`; happy path mocks `send_message`; state transitions to `submitted` |
| `test_task_ops.py` | `task_list` returns all tracked states (terminal included); `task_get` with unknown id; `task_stop` flips state, revokes token, does NOT call `cancel_task` |
| `test_webhook_handler.py` | Valid token → state update; unknown token → 401; idempotent retransmit → no duplicate notification; race "task_id None" → fallback to body; evicted ctx → 200, log; terminal state → notification accodata + revoke; input_required → notification with deferred_calls |
| `test_polling.py` | Worker scans non-terminal tasks; calls `get_task`; feeds response into common handler; stops on terminal; gives up after 5 failures |
| `test_context_entry.py` | `is_evictable` true when empty/all-terminal, false with non-terminal in flight; `pending_notifications` queue semantics |
| `test_executor_drain.py` | Drain happens BEFORE LLM call on each iteration; cleared after drain; messages appended in order received |

### 8.2 Integration tests (`tests/integration/a2a_outbound/`)

Pattern: real `a2a_serve` for agent B (and C) on ephemeral ports, real `BaseAgent` A configured with `remote_agents=[B_url, C_url]`. Real httpx; real Starlette; mocked LLM provider returning canned tool calls. Reuse `tests/integration/tracer/conftest.py` patterns for fixture setup.

| Scenario | Verification |
|---|---|
| **Happy path single agent** | A.execute_query → dispatch_agent → B completes → webhook → next turn shows `<remote_task_update>` with `<status>completed</status>` and the result text |
| **Two parallel dispatches** | A dispatches to B and C in same turn; both complete; both notifications drained at next turn in arrival order |
| **input_required round-trip** | A dispatches to B; B emits `request_user_input` (deferred); webhook delivers `input_required` with deferred_calls; A's next turn sees the update and calls `respond_to_remote`; B receives, resumes, completes; final notification arrives |
| **Polling fallback when push fails** | A dispatches with a webhook URL pointed at black hole; B's pushes drop; after 30s polling kicks in and discovers `completed`; same `<remote_task_update>` flow |
| **Multi-context isolation** | A receives two parallel requests on `ctx-MARIO` and `ctx-LUCIA`; both dispatch to B with different tasks; responses route correctly to their respective `pending_notifications` queues |
| **Token spoofing** | External POST to /webhook with a randomly generated token (not in token_map) → 401, no state mutation, no notification |
| **Eviction protection** | Fill `ContextStore` with `max_contexts` empty entries; one entry has a non-terminal remote_task; trigger eviction; that entry survives, an empty one is evicted |
| **`task_stop` then late webhook** | A dispatches; before completion calls `task_stop(task_id)`; B later POSTs the completion webhook; handler drops silently (200 OK); no notification accodata |
| **Polling giveup** | get_task always raises; after 5 failures local state flips to `failed`; notification with `<error>polling_giveup</error>` accodata |
| **Race: webhook before send_message returns** | Mocked B that POSTs `working` *before* its send_message responds; handler uses body.id as fallback; `claim_task_id` later resolves cleanly |
| **AgentCard collision at startup** | Two URLs returning cards with same `name` → `resolve_all` raises clear error |
| **Tracer integration** | Each `dispatch_agent` execute opens a tool span; webhook arrival emits `remote_task.update` event on the agent span; `task_stop` emits `remote_task.stopped` event |

### 8.3 Test infrastructure to add

- `tests/integration/a2a_outbound/conftest.py`:
  - `remote_agent_factory(name, behavior)` → spawns a real A2A server with a stub agent that follows a scripted behavior (e.g. complete after N seconds, emit input_required, fail).
  - `mock_llm_with_dispatches(scripts)` → LLM provider mock that returns predetermined tool calls including `dispatch_agent`.
  - `webhook_blackhole(percent)` → fixture that drops a percent of outgoing pushes for fallback testing.

- `tests/adapters/outbound/a2a/conftest.py`:
  - `mocked_a2a_client()` → replaces real `A2AClient` with a controllable double.
  - `fake_card(name, skills)` → AgentCard factory.
  - `fresh_registry(urls, *, ...)` → registry with mocked card resolution.

### 8.4 Test quality bar

- Each integration test asserts not just the final state but the **notification XML structure** (so prompt design stays stable).
- Each "race" test uses explicit `asyncio.sleep(0)` and `Event` orchestration, not real time waits.
- Polling-fallback tests use a synthetic clock (monkey-patched `time.monotonic`) to avoid 30s real waits.
- Mock LLM provider returns deterministic sequences of `tool_calls` so the same scenario is reproducible.

Target: **all 9 unit modules + 12 integration scenarios green** before merging. ~30 unit tests, ~12 integration tests.

## 9. Decision Log

| # | Decision | Rationale |
|---|---|---|
| 1 | Discovery via static URL list in `BaseAgent(remote_agents=[...])` | Simplest; matches ADK's `RemoteA2aAgent` pattern; dynamic registry deferred. |
| 2 | Single dispatcher tool `dispatch_agent(agent_name, query)`, not one tool per agent | Matches Claude Code's `AgentTool`; scales without bloating the LLM tool catalog. |
| 3 | Webhook mounted on existing A2A server uvicorn | Avoids a second HTTP listener; consistent with the constraint that the client agent is always also a server. |
| 4 | Token-based routing via `PushNotificationConfig.token` and `X-A2A-Notification-Token` header | A2A-native (spec-compliant per §13.2); already supported by `SmartPushNotificationSender`; provides routing AND auth in one mechanism. |
| 5 | Polling fallback (30s start, 5s interval) | Same robustness pattern as the CLI client today; resilient to NAT/firewall issues. |
| 6 | Stateless dispatch (fresh remote context per call) | Simpler v1; multi-turn continuation can be added as opt-in `continue=True` parameter later. |
| 7 | No cancellation cascade; `task_stop` is local-only | Matches Claude Code's `RemoteAgentTask.kill`; remote owns its own lifecycle. |
| 8 | `input_required` propagated as a notification with `<deferred_tool_calls>` payload | Cleanest semantic; A2A protocol-aligned (the sender of the task is the consumer of input_required). |
| 9 | Notifications are user-role messages with `<remote_task_update>` XML wrapper | Matches Claude Code's `<TASK_NOTIFICATION>` pattern; user-role is the only injectable mid-conversation in mainstream LLM provider APIs. |
| 10 | Per-context queues on `ContextEntry`, not global | Required for multi-tenant isolation; aligns with existing `ContextStore` boundaries. |
| 11 | Per-request context injection via `_inject_context_entry` in the executor | Mirrors existing `_inject_client_info` precedent; thread-safe by construction; no `contextvars` needed. |
| 12 | Token TTL 24h with hourly GC | Backstop against orphan tokens; long enough for legitimate slow tasks. |
| 13 | In-memory only; no persistence in v1 | Crash recovery is out of scope; matches Claude Code (their persistence is for separate goals). |
| 14 | `task_list` returns all tracked tasks regardless of status | Per-user requirement; lets the LLM see history of completed/failed too. |
| 15 | No remote `cancel_task` from `task_stop` | Honors remote autonomy; the remote's TTL or own `cancel` paths handle cleanup. |
| 16 | Token only for v1; richer `AuthenticationInfo` deferred | Sufficient for v1 spec compliance; OAuth/Bearer/etc. is a future iteration. |

## 10. Future Work (out of scope)

- Persistence of `token_map` and `pending_notifications` across restart (SQLite or similar).
- Cross-A2A trace propagation via custom message metadata.
- `AuthenticationInfo`-based mutual auth (Bearer tokens, OAuth).
- Multi-turn continuation with explicit `continue=True` flag on dispatch.
- A `discover_agent(url)` runtime tool for dynamic registry expansion.
- Backpressure / rate limiting of dispatches per remote.
- Graceful shutdown that drains in-flight tasks and notifies remotes.
- Automatic cleanup on completion: option to optionally call `cancel_task` on remotes when a parent terminates abnormally (opt-in flag, default off).
