# A2A Agent Client — Async Inter-Agent Dispatch

**Date**: 2026-04-30
**Status**: Draft v2 (revised after architecture review)
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
3. Generates `token = secrets.token_urlsafe(32)` (cryptographically random, 256 bits).
4. `registry.register_token(token, context_id="ctx-AAA", agent_name="B")` — stores `TokenRoute(context_id, agent_name, task_id=None, registered_at=now)`.
5. Builds the `MessageSendConfiguration`:
   ```python
   cfg = MessageSendConfiguration(
       blocking=False,
       push_notification_config=PushNotificationConfig(
           url="http://a:8000/webhook",
           token=token,
       ),
   )
   ```
6. Calls `client.send_message(message, configuration=cfg)` — this returns an `AsyncIterator[ClientEvent | Message]`, **not** a Task directly. The dispatch tool consumes the iterator only until it has the task (single tuple yielded by the SDK in non-streaming mode), then breaks. Pattern reused from `cli_client.py:257-265`:
   ```python
   task = None
   async for event in client.send_message(message, configuration=cfg):
       if isinstance(event, tuple):
           task, _update = event
           break
       elif isinstance(event, Message):
           # Remote returned a final Message instead of a Task
           # (simple-interaction agents). Synthesize a "completed" notification
           # locally and skip token tracking.
           registry.revoke(token)
           return {"status": "completed", "result": ...}
   ```
   The A2A `Client` for outbound use is constructed with `streaming=False, polling=False` so the iterator yields just the initial task tuple and exits.
7. `registry.claim_task_id(token, task.id)` — fills in the previously-`None` `task_id` on the `TokenRoute`. (No synchronization primitive needed: see "race fallback" in §5.2 step 3.)
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

#### 5.2.1 Tracer integration across the HTTP boundary

The webhook handler runs in a separate Starlette request task with empty `contextvars` — `get_current_trace()` returns `None`. To attach `remote_task.update` and `remote_task.stopped` events to the agent's trace, the handler reuses the same pattern `_emit_cancellation_event` already uses (executor.py:135-176):

```python
# Inside webhook handler, after looking up route → ctx_entry
saved_trace = ctx_entry.trace_session  # set by the executor when the
                                       # original turn opened a2a_task span
if tracer and saved_trace is not None:
    a2a_span = next(
        (s for s in saved_trace.spans if s.span_type == SpanType.a2a_task),
        None,
    )
    if a2a_span is not None:
        prior_trace = get_current_trace()
        prior_span = get_current_span()
        set_current_trace(saved_trace)
        set_current_span(a2a_span)
        try:
            await tracer.add_event(
                "remote_task.update",
                {"task_id": ..., "agent": ..., "from": ..., "to": ...},
            )
        finally:
            set_current_trace(prior_trace)
            set_current_span(prior_span)
```

`task_stop` emits `remote_task.stopped` via the **same** dance (it runs in the LLM's tool dispatch path, so contextvars *are* set, but we use the same helper for symmetry).

**New event names** to register in `docs/superpowers/specs/2026-04-21-tracer-refactor-design.md` event taxonomy:
- `remote_task.update` — attributes: `task_id`, `agent`, `from` (previous status), `to` (new status), `iteration` (optional).
- `remote_task.stopped` — attributes: `task_id`, `agent`, `reason` (e.g. `"llm_stop"`, `"polling_giveup"`).

### 5.3 Notification drain

When the next request arrives on `ctx-AAA` (Mario speaks again, or any other event scheduling a turn), the executor's `_run_agent_impl` enters. The drain happens **after** any deferred-resume injection (`inject_deferred_response` at executor.py:303-306) and **before** the agent's own loop starts — i.e. drain goes immediately after the resume-path block, before `_run_agent_impl` calls `agent.execute_query_stream()` / `agent.resume_after_deferred()`:

```python
# executor.py — _run_agent_impl, after the resume-path block
if is_resume:
    inject_deferred_response(entry, message)

# NEW: drain pending remote-task notifications BEFORE starting the agent.
# Order matters: deferred ToolMessage (above) must stay adjacent to its
# AssistantMessage. Notifications go after, as fresh user-role messages.
if entry.pending_notifications:
    entry.history.extend(entry.pending_notifications)
    entry.pending_notifications.clear()

await self._run_agent(...)  # entry.history is then injected into agent.conversation_history
```

The LLM at the next iteration sees the user's new message *and* the `<remote_task_update>` XML in the conversation. It reacts naturally: report to Mario, dispatch follow-up to another agent, ignore, etc.

**Drain happens at request boundary only.** Notifications arriving during a single turn (between iterations of `_execute_loop`) are queued in `pending_notifications` and surface at the start of the **next** request. This keeps `BaseAgent` (`core/`) free of dependencies on `adapters/inbound/a2a/` (hexagonal layering). A future enhancement could expose this via a `BEFORE_LLM_CALL` hook if intra-turn delivery becomes desirable, but it is **not** in v1 scope.

### 5.4 input_required cycle

When B's `<remote_task_update>` carries `<status>input_required</status>`, the LLM reads the embedded `<deferred_tool_calls>` payload and decides what to do. To answer, it calls:

```
respond_to_remote(task_id="t-001", data={...payload matching B's tool OutputSchema...})
```

The `RespondToRemoteTool.execute()`:

1. Looks up `ctx_entry.remote_tasks[task_id]`.
   - **Idempotency / replay protection** — if `state.status != "input_required"` or `state.deferred_calls is None`, return `{"error": "task is not awaiting input (status=...)"}`. This blocks duplicate `respond_to_remote` calls within a single input_required cycle. If B later emits a *new* `input_required` (different deferred call ids), `status` returns to `"input_required"` with fresh `deferred_calls`, and respond_to_remote becomes valid again — naturally idempotent per cycle.
2. Resolves `agent_name → A2AClient`.
3. Sends a follow-up `Message` to B with a `DataPart(data=...)` and the same `task_id`/`context_id` B used. Same `MessageSendConfiguration` shape as initial dispatch (push_notification_config with the **same token** so the resume notifications still route).
4. Updates local state: `status = "submitted"`, `deferred_calls = None`. (B will resume and emit further updates via webhook.)
5. Returns `{"task_id": ..., "status": "submitted"}` to the LLM.

If A's LLM has no information sufficient to answer, it can:
- Ask its own user (a normal conversational turn).
- If A is itself a remote called by another parent P, propagate by emitting its **own** `request_user_input` deferred tool — that becomes A's `input_required` to P, and the chain continues. (This composes naturally; no special code needed.)

### 5.5 Polling fallback

A **single global worker** (one per process, owned by `RemoteAgentRegistry`, started at server boot) iterates all known contexts every **5 seconds**:

```python
async def poll_loop(self):
    while not self._stop.is_set():
        await asyncio.sleep(5.0)
        now = time.monotonic()
        for ctx_entry in context_store.iter_entries():
            for task in list(ctx_entry.remote_tasks.values()):
                if task.is_terminal:
                    continue
                if now - task.last_update_monotonic < 30.0:
                    continue
                try:
                    fresh = await client_for(task.agent_name).get_task(
                        TaskQueryParams(id=task.task_id)
                    )
                except Exception as e:
                    task.poll_failures += 1
                    if task.poll_failures >= 5:
                        # Locally mark failed and emit notification.
                        # Token revoked, no further polling.
                        ...
                    continue
                # Feed response through the same handler path used by /webhook.
                await handle_remote_update(ctx_entry, task.task_id, fresh)
```

Cadence rules:
- **Worker tick: 5 seconds** (the loop's `sleep`).
- **Per-task delay before polling: 30 seconds of silence** (`now - last_update_monotonic > 30`).
- **Giveup: 5 consecutive `get_task` failures** for the same task → mark `failed` locally with reason `"polling_giveup"` and emit a notification.

`last_update_monotonic` uses `time.monotonic()` (immune to wall-clock jumps). The wall-clock `last_update: datetime` field stays for human-readable logs and `task_get` output.

Behaviorally: if pushes arrive, the worker sees `last_update_monotonic` always fresh and never polls. If pushes fail, polling delivers the same outcome with up-to-35s latency (30s delay + ≤5s scan tick). No observable difference for the LLM.

**One global worker, not per-context**: avoids leaks (no per-context lifecycle), simplifies tests (one task to monkey-patch), trivially handles `ContextEntry` LRU eviction (the worker just iterates whatever's currently in the store).

### 5.6 Cancel of A's own task with in-flight remotes

When A's task is canceled (user ESC, async cancel, hook-driven cancel — handled in `executor.py:718, 887`), A has potentially several non-terminal `remote_tasks` for that context. **No wire call to the remotes** — they keep running autonomously (per Decision 7).

What A does locally:

```python
# In executor.py cancel paths, after flipping was_canceled=True:
for tid, state in list(entry.remote_tasks.items()):
    if state.is_terminal:
        continue
    registry.revoke(state.token)        # silence late webhooks
    state.status = "killed"             # local marker
    state.last_update = now()
    # No emit of <remote_task_update>: the user requested abort, surfacing
    # post-hoc completions of the same task would be confusing.
```

This means:
- Future webhooks from the canceled remotes hit 401 silently (token revoked).
- Polling worker skips them (terminal state).
- The remotes keep computing wastefully — accepted cost; alternative (cascade cancel) was explicitly rejected by the design.

The LLM's `task_stop(task_id)` tool follows the **same** local-only logic but is scoped to one task instead of the whole context. The two paths share the same helper:

```python
def stop_remote_locally(entry, task_id, *, reason: str):
    state = entry.remote_tasks.get(task_id)
    if state is None or state.is_terminal:
        return
    registry.revoke(state.token)
    state.status = "killed"
    state.last_update = now()
    # tracer event remote_task.stopped with reason
```

`reason="llm_stop"` for `task_stop`, `reason="context_canceled"` for the cancel-of-A path.

**Late webhook after cancel + new dispatch to same agent (the "filo conduttore" scenario)**: A cancels, t-001 token revoked. Mario asks again, A dispatches t-002 with a **fresh token**. Independent. When agent_1 eventually finishes t-001 → 401. When agent_1 finishes t-002 → token valid → notification accodata. No interference; tokens are per-dispatch.

## 6. Components in Detail

### 6.1 `RemoteAgentRegistry`

```python
# adapters/outbound/a2a/registry.py
class RemoteAgentRegistry:
    """Process-wide registry of known remote agents and their in-flight tokens.

    Singleton lifecycle: created in AgentFactory.a2a_serve() before uvicorn
    starts. Reuses the httpx.AsyncClient already created by a2a_serve for the
    SmartPushNotificationSender (agent_factory.py:581) — connection pooling
    matters and double clients waste fds.
    """

    def __init__(
        self,
        urls: list[str],
        httpx_client: httpx.AsyncClient,
        *,
        client_config: ClientConfig | None = None,
    ):
        self._urls = urls
        self._httpx = httpx_client
        # Outbound clients run with streaming=False, polling=False — we want
        # send_message to yield exactly the initial task tuple and stop.
        self._client_config = client_config or ClientConfig(
            httpx_client=httpx_client,
            streaming=False,
            polling=False,
        )

    async def resolve_all(self) -> None:
        """Fetch /.well-known/agent-card.json for each URL.

        - On individual fetch failure: log warning, skip that agent.
        - On duplicate name: log warning + append discriminator (`B (1)`,
          `B (2)`) so legitimate replicated topologies (multi-AZ) work.
          Operators can pass `strict_names=True` later to upgrade to hard
          error if/when they need uniqueness invariants.
        """

    def card_for(self, name: str) -> AgentCard: ...
    def client_for(self, name: str) -> A2AClient: ...
    def names(self) -> list[str]: ...
    def descriptions(self) -> dict[str, dict]:
        """Returns {name: {description, skills}} for system_prompt_fragment."""

    def register_token(self, token, context_id, agent_name) -> None: ...
    def claim_task_id(self, token, task_id) -> None: ...
    def lookup(self, token: str) -> TokenRoute | None:
        """Constant-time lookup by token. Implementation MUST use a dict
        (hash equality, no linear comparison) to avoid timing-attack
        token enumeration."""
    def revoke(self, token: str) -> None: ...
    async def gc_expired(self, ttl_seconds: int = 86400) -> int:
        """Remove tokens older than TTL. Called periodically (every 1h)."""
```

#### Lifecycle wiring in `a2a_serve()`

`AgentFactory.a2a_serve()` is **synchronous** (calls `uvicorn.run`, agent_factory.py:523). The registry's `resolve_all()` is async. We bridge with `asyncio.run` before `uvicorn.run` is invoked:

```python
# agent_factory.py — a2a_serve() body, before uvicorn.run
registry = RemoteAgentRegistry(
    urls=remote_agent_urls,
    httpx_client=push_httpx_client,  # the existing one
)
asyncio.run(registry.resolve_all())   # blocking, runs to completion

# Mount /webhook on the FastAPI app, sharing the registry instance:
fastapi_app.add_api_route("/webhook", make_webhook_handler(registry, context_store), methods=["POST"])

# Patch the agent_factory closure to inject the dispatch tools:
def agent_factory_with_remotes():
    instance = original_agent_factory()
    instance.register_tool(DispatchAgentTool(registry))
    instance.register_tool(RespondToRemoteTool(registry))
    instance.register_tool(TaskListTool())
    instance.register_tool(TaskGetTool())
    instance.register_tool(TaskStopTool(registry))
    return instance

executor = ObelixAgentExecutor(agent_factory_with_remotes, ...)

# Start the global polling worker as part of the FastAPI lifespan
fastapi_app.add_event_handler("startup", registry.start_polling)
fastapi_app.add_event_handler("shutdown", registry.stop_polling)

uvicorn.run(fastapi_app, ...)
```

If `remote_agent_urls` is empty, the registry construction and tool injection are skipped entirely — `a2a_serve()` for legacy agents stays bit-for-bit identical.

### 6.2 State dataclasses

```python
# adapters/outbound/a2a/state.py
@dataclass
class RemoteTaskState:
    task_id: str
    agent_name: str
    status: str  # submitted | working | input_required | completed | failed | canceled | rejected | killed
    token: str
    created_at: datetime           # wall clock, for human-readable logs
    last_update: datetime          # wall clock, for task_get output
    last_update_monotonic: float   # time.monotonic(), for polling cadence math
    last_artifact: dict | None
    deferred_calls: list[dict] | None  # populated when status == "input_required"
    poll_failures: int = 0         # incremented on consecutive get_task errors

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "canceled", "rejected", "killed")


@dataclass
class TokenRoute:
    context_id: str
    agent_name: str
    task_id: str | None     # None until claim_task_id; race fallback uses body.id
    registered_at: datetime  # for TTL-based GC
```

**No `asyncio.Event`** in `TokenRoute`. The earlier draft had `awaiting_task_id: asyncio.Event` to handle the "webhook arrives before send_message returns" race, but no consumer awaits the event. The webhook handler's body-fallback (§5.2 step 3) handles the race directly.

### 6.3 `ContextEntry` extension

`ContextEntry` (`adapters/inbound/a2a/server/context.py`) uses `__slots__`. Both the `__slots__` tuple **and** `__init__` must be updated together, otherwise `AttributeError` at runtime:

```python
class ContextEntry:
    __slots__ = (
        ... existing slots ...,
        "remote_tasks", "pending_notifications",
    )

    def __init__(self):
        ...
        self.remote_tasks: dict[str, RemoteTaskState] = {}
        self.pending_notifications: list[HumanMessage] = []
```

Method to support eviction protection:

```python
def is_evictable(self) -> bool:
    """LRU should not evict this context if there are non-terminal remote
    tasks in flight. Otherwise their webhook returns hit 401 silently and
    the user loses the result without warning."""
    return all(t.is_terminal for t in self.remote_tasks.values())
```

#### `ContextStore.get_or_create()` eviction policy with non-evictable entries

The existing eviction at `context.py:97-99` walks the OrderedDict from oldest:

```python
while len(self._contexts) >= self._max_contexts:
    evicted_id, _evicted = self._contexts.popitem(last=False)
```

Naive change "skip non-evictable" creates an **unbounded growth** vector: under bursty multi-tenant load with many parallel dispatches, every entry could become non-evictable, and the store grows without limit. Mitigation:

```python
def _evict_one(self) -> None:
    # First pass: find oldest evictable entry.
    for cid, entry in self._contexts.items():
        if entry.is_evictable():
            self._contexts.pop(cid)
            return
    # Hard cap: if everyone is non-evictable AND we exceed 2x max_contexts,
    # force-evict the oldest to prevent OOM. Log a clear warning so operators
    # see the saturation; this is a last-resort safety valve, not normal flow.
    if len(self._contexts) >= self._max_contexts * 2:
        oldest_id, _ = self._contexts.popitem(last=False)
        logger.warning(
            f"[A2A] Forced eviction of context {oldest_id} with in-flight "
            f"remote tasks: hard cap (2x max_contexts) reached. Their "
            f"webhooks will return 401."
        )
```

So normal operation: protect non-evictable. Saturation: hard cap kicks in with a loud warning.

### 6.4 Per-request context injection

The executor (`adapters/inbound/a2a/server/executor.py`) already has the precedent of `_inject_client_info` — a `@staticmethod` at executor.py:826-844 called from `_run_agent_impl` at line 462-463. We add the symmetric, also as `@staticmethod` for consistency:

```python
@staticmethod
def _inject_context_entry(agent: BaseAgent, entry: ContextEntry) -> None:
    """Mirror of _inject_client_info but for context-aware tools.

    Each remote-task tool implements set_context_entry(entry) and stores the
    reference for the duration of its execute(). Tools are fresh per-request
    (created in agent_factory()), so this is naturally thread-safe."""
    for tool in agent.registered_tools:
        if hasattr(tool, "set_context_entry"):
            tool.set_context_entry(entry)
```

Called once in `_run_agent_impl` immediately after `_inject_client_info`.

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
    limit: int = Field(default=50, ge=1, le=500, description="Max entries to return")

    async def execute(self) -> dict:
        all_tasks = sorted(
            self._ctx_entry.remote_tasks.values(),
            key=lambda t: t.last_update,
            reverse=True,
        )
        shown = all_tasks[: self.limit]
        return {
            "tasks": [
                {"task_id": t.task_id, "agent": t.agent_name, "status": t.status,
                 "created_at": t.created_at.isoformat(),
                 "last_update": t.last_update.isoformat()}
                for t in shown
            ],
            "shown": len(shown),
            "total": len(all_tasks),
        }
```

Returns **all** tracked tasks regardless of state (terminal included), most recently updated first. Bounded to `limit` entries (default 50, max 500) to prevent context-window blowup on long-running agents. The response includes `shown` and `total` so the LLM knows when more exist and can re-call with a higher `limit` if it actually needs them.

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
| Two AgentCards with identical `name` | Log warning and append `(1)`, `(2)`, … discriminator. Legitimate replicated topologies (multi-AZ behind separate URLs) keep working. A `strict_names=True` registry flag is reserved for future use. |
| Webhook race: arrives before `send_message` returns task_id | Handler uses `task.id` from the body as fallback; updates `TokenRoute.task_id` retroactively. No `asyncio.Event` needed. |
| `ContextEntry` evicted with in-flight remote tasks | Eviction is skipped via `is_evictable()` until 2x `max_contexts` is reached, then forced with warning (see §6.3). |
| Cancel of A's own task with in-flight remotes | All non-terminal `remote_tasks` for that context have their tokens revoked locally; status flipped to `"killed"`. **No wire call to the remotes** — they keep running. Late webhooks from those tasks return 401 silently. See §5.6. |
| Process crash | All in-memory state lost. B's pending webhooks return 401 from the new process. Acceptable for v1. |
| Polling: `get_task` raises | Worker logs and retries on next tick (5s). After 5 consecutive failures **for the same task**, marks task `failed` locally with reason `"polling_giveup"` and emits notification. |
| `task_stop` then webhook arrives | Handler sees `status == "killed"`, drops silently. Token already revoked. |
| `respond_to_remote` called twice for same input_required cycle | Idempotency check at tool entry: if `state.status != "input_required"` or `state.deferred_calls is None`, returns error. The first call flips status to `"submitted"` and clears `deferred_calls`, blocking duplicates within the cycle. New cycles are not blocked. |
| Token never cleaned (remote vanishes mid-flight) | Background GC sweeps `token_map` every 1h, removing entries older than 24h TTL. |
| Notification arrives during a single turn (after iteration start) | Queued in `pending_notifications`, drained at the **next** request boundary (§5.3). Intra-turn delivery is intentionally not supported in v1 to keep `BaseAgent` (`core/`) free of `adapters/inbound` dependencies. |

## 8. Testing Strategy

### 8.1 Unit tests (`tests/adapters/outbound/a2a/`)

| Module | Coverage |
|---|---|
| `test_registry.py` | `resolve_all` happy path with mocked AgentCards; one-of-many failure → skip; duplicate names → warning + discriminator (`B`, `B (1)`); `register_token` / `claim_task_id` semantics; `lookup` constant-time path uses dict; `revoke` removes; `gc_expired` removes only entries older than TTL |
| `test_state.py` | `RemoteTaskState.is_terminal` matrix; `TokenRoute` shape (no `asyncio.Event`); `last_update_monotonic` populated on every state change |
| `test_dispatch_tool.py` | Schema validation; happy path with mocked client; remote unreachable → error tool result (never returns `None`, never trips deferred); `set_context_entry` injection works; token registered **before** `send_message`; `claim_task_id` after `send_message` returns; state added on success; `MessageSendConfiguration.push_notification_config.token` matches registered token |
| `test_respond_tool.py` | Reject if task unknown; reject if task not in `input_required`; reject on second call within same cycle (idempotency); happy path mocks `send_message`; state transitions to `submitted` and clears `deferred_calls`; new `input_required` cycle later allows respond again |
| `test_task_ops.py` | `task_list` bounded by `limit` param (default 50); `task_list` returns terminal entries too; `task_list` shown/total accuracy; `task_get` with unknown id; `task_stop` flips state, revokes token, does NOT call `cancel_task`; `task_stop` emits `remote_task.stopped` event with `reason="llm_stop"` |
| `test_webhook_handler.py` | Valid token → state update; unknown token → 401; idempotent retransmit (same state) → no duplicate notification; race "task_id None" → fallback to `body.id`; evicted ctx → 200, log; terminal state → notification accodata + revoke; input_required → notification with deferred_calls; tracer integration uses `entry.trace_session` correctly |
| `test_polling.py` | Single global worker tick at 5s (monkey-patched `time.monotonic`); per-task delay 30s; task with `last_update_monotonic` < 30s ago skipped; `get_task` raises 5 consecutive times → `failed` + notification; recovery after transient failure resets `poll_failures` |
| `test_context_entry.py` | `__slots__` includes `remote_tasks` and `pending_notifications`; `is_evictable` true when empty/all-terminal, false with non-terminal in flight; eviction skips non-evictable until 2x cap then forces with warning |
| `test_executor_drain.py` | Drain happens AFTER `inject_deferred_response` and BEFORE agent loop start; cleared after drain; messages appended in arrival order; intra-turn arrivals deferred to next request |
| `test_cancel_handling.py` | A's task canceled with N non-terminal remote_tasks → all tokens revoked, statuses flipped to `"killed"`, no `cancel_task` wire call to remotes; subsequent late webhooks for those tasks return 401 |

### 8.2 Integration tests (`tests/integration/a2a_outbound/`)

Pattern: real `a2a_serve` for agent B (and C) on ephemeral ports, real `BaseAgent` A configured with `remote_agents=[B_url, C_url]`. Real httpx; real Starlette; mocked LLM provider returning canned tool calls. Reuse `tests/integration/tracer/conftest.py` patterns for fixture setup.

| Scenario | Verification |
|---|---|
| **Happy path single agent** | A.execute_query → dispatch_agent → B completes → webhook → next turn shows `<remote_task_update>` with `<status>completed</status>` and the result text |
| **Two parallel dispatches** | A dispatches to B and C in same turn; both complete; both notifications drained at next turn in arrival order |
| **input_required round-trip** | A dispatches to B; B emits `request_user_input` (deferred); webhook delivers `input_required` with deferred_calls; A's next turn sees the update and calls `respond_to_remote`; B receives, resumes, completes; final notification arrives |
| **Polling fallback when push fails** | A dispatches with a webhook URL pointed at black hole; B's pushes drop; after 30s+5s polling kicks in (monkey-patched clock) and discovers `completed`; same `<remote_task_update>` flow |
| **Multi-context isolation** | A receives two parallel requests on `ctx-MARIO` and `ctx-LUCIA`; both dispatch to B with different tasks; responses route correctly to their respective `pending_notifications` queues |
| **Token spoofing** | External POST to /webhook with a randomly generated token (not in token_map) → 401, no state mutation, no notification |
| **Eviction protection** | Fill `ContextStore` with `max_contexts` empty entries; one entry has a non-terminal remote_task; trigger eviction; that entry survives, an empty one is evicted. **Saturation**: fill 2x with all non-evictable → forced eviction with warning logged. |
| **`task_stop` then late webhook** | A dispatches; before completion calls `task_stop(task_id)`; B later POSTs the completion webhook; handler drops silently (200 OK); no notification accodata |
| **Polling giveup** | get_task always raises; after 5 failures local state flips to `failed`; notification with `<error>polling_giveup</error>` accodata |
| **Race: webhook before send_message returns** | Mocked B that POSTs `working` *before* its send_message responds; handler uses body.id as fallback; `claim_task_id` later resolves cleanly |
| **AgentCard duplicate name** | Two URLs returning cards with same `name` → both registered, second gets `(1)` suffix; both callable independently |
| **Cancel of A's turn with in-flight remotes** | A has 3 non-terminal remote tasks; A is canceled (via cancel_task on A's executor); all 3 tokens revoked locally; later webhooks for those tasks → 401; verify NO outbound `cancel_task` was sent to the remotes |
| **Cancel + new dispatch ("filo conduttore")** | A cancels with t-001 in flight; same context dispatches t-002 to same agent; t-001 webhook arrives → 401; t-002 webhook arrives → notification accodata cleanly |
| **respond_to_remote idempotency** | Single input_required cycle: first respond → submitted; second respond → error; verify only one DataPart was sent on the wire |
| **Tracer integration** | Each `dispatch_agent` execute opens a tool span; webhook arrival emits `remote_task.update` event on the saved `a2a_task` span (across HTTP boundary); `task_stop` emits `remote_task.stopped` event with reason |

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

Target: **all 10 unit modules + 13 integration scenarios green** before merging. ~35 unit tests, ~13 integration tests.

## 9. Decision Log

| # | Decision | Rationale |
|---|---|---|
| 1 | Discovery via static URL list in `BaseAgent(remote_agents=[...])` | Simplest; matches Google ADK's `RemoteA2aAgent` pattern; dynamic registry deferred. |
| 2 | Single dispatcher tool `dispatch_agent(agent_name, query)`, not one tool per agent | Matches Claude Code's `AgentTool`; scales without bloating the LLM tool catalog. |
| 3 | Webhook mounted on existing A2A server uvicorn (FastAPI `add_api_route`) | Avoids a second HTTP listener; consistent with the constraint that the client agent is always also a server. |
| 4 | Token-based routing via `PushNotificationConfig.token` and `X-A2A-Notification-Token` header | A2A-native (spec-compliant per §13.2); already supported by `SmartPushNotificationSender`; provides routing AND auth in one mechanism. |
| 5 | Polling fallback: single global worker, 5s tick, 30s per-task delay, 5 consecutive failures → giveup | Resilient to NAT/firewall; one worker scales better than per-context workers; uses `time.monotonic()` for cadence. |
| 6 | Stateless dispatch (fresh remote context per call) | Simpler v1; multi-turn continuation can be added as opt-in `continue=True` parameter later. |
| 7 | No cancellation cascade; `task_stop` and context-cancel are both local-only (no wire `cancel_task` to remotes) | Honors remote autonomy; matches Claude Code's `RemoteAgentTask.kill`; tradeoff: remote may waste compute, accepted. Token revocation silences late webhooks. |
| 8 | `input_required` propagated as a notification with `<deferred_tool_calls>` payload | Cleanest semantic; A2A protocol-aligned (the sender of the task is the consumer of input_required). |
| 9 | Notifications are user-role messages with `<remote_task_update>` XML wrapper | Matches Claude Code's `<TASK_NOTIFICATION>` pattern; user-role is the only injectable mid-conversation in mainstream LLM provider APIs. |
| 10 | Per-context queues on `ContextEntry`, not global | Required for multi-tenant isolation; aligns with existing `ContextStore` boundaries. |
| 11 | Per-request context injection via `_inject_context_entry` (`@staticmethod`) in the executor | Mirrors existing `_inject_client_info` precedent; thread-safe by construction; no `contextvars` needed. |
| 12 | Drain `pending_notifications` at request boundary only (not per-iteration) | Keeps `BaseAgent` (`core/`) free of `adapters/inbound` dependencies (hexagonal layering). Intra-turn delivery is future work. |
| 13 | Drain happens AFTER `inject_deferred_response` and BEFORE agent loop start | Deferred `ToolMessage` must stay adjacent to its `AssistantMessage`; notifications are fresh user-role messages that go after. |
| 14 | Eviction protection with hard cap at 2x `max_contexts` | Prevents unbounded growth under saturation while protecting in-flight tasks under normal load. |
| 15 | `respond_to_remote` idempotent within an `input_required` cycle via status check | Reuses existing `RemoteTaskState.status` and `deferred_calls` fields; no extra "responded" flag needed. |
| 16 | Webhook handler emits tracer events using `entry.trace_session` saved by the executor | Mirrors `_emit_cancellation_event` pattern; events `remote_task.update` and `remote_task.stopped` registered in tracer taxonomy. |
| 17 | Token TTL 24h with hourly GC | Backstop against orphan tokens; long enough for legitimate slow tasks. |
| 18 | In-memory only; no persistence in v1 | Crash recovery is out of scope. |
| 19 | `task_list` returns all tracked tasks regardless of status, bounded by `limit` (default 50, max 500) | Per requirement (terminal entries visible) but bounded to prevent context-window blowup. |
| 20 | Soft-handle duplicate AgentCard names: warn + append `(N)` discriminator | Allows legitimate replicated topologies (multi-AZ); strict mode reserved for future flag. |
| 21 | Reuse existing `httpx.AsyncClient` from `a2a_serve` for outbound calls and registry | Connection pool sharing; avoid fd duplication. |
| 22 | Token only for v1; richer `AuthenticationInfo` deferred | Sufficient for v1 spec compliance; OAuth/Bearer/etc. is a future iteration. |
| 23 | `lookup(token)` uses dict (constant-time hash equality), not linear comparison | Avoid timing-attack token enumeration. |

## 10. Future Work (out of scope)

- Persistence of `token_map` and `pending_notifications` across restart (SQLite or similar).
- Cross-A2A trace propagation via custom message metadata.
- `AuthenticationInfo`-based mutual auth (Bearer tokens, OAuth).
- Multi-turn continuation with explicit `continue=True` flag on dispatch.
- A `discover_agent(url)` runtime tool for dynamic registry expansion.
- Backpressure / rate limiting of dispatches per remote.
- Graceful shutdown that drains in-flight tasks and notifies remotes.
- Automatic cleanup on completion: option to optionally call `cancel_task` on remotes when a parent terminates abnormally (opt-in flag, default off).
