# Pre-implementation research: A2A CLI polling-only (spec 2)

**Spec**: `docs/superpowers/specs/2026-05-08-a2a-cli-polling-spec2-design.md`
**Date**: 2026-05-08
**Status**: complete (3 T2 spike runs, all passing)

---

## Tech inventory & classification

| # | Technology | Version | Tier | Justification |
|---|---|---|---|---|
| 1 | a2a-sdk: `Task.metadata` mutability post-creation | 0.3.25 | T2 | User request (a2a all T2). Core of open assumption #1. |
| 2 | a2a-sdk: `TaskStatusUpdateEvent(metadata={...}, final=False)` on terminal task | 0.3.25 | T2 | Open assumption #1 option A — SDK-canonical event-based path. |
| 3 | a2a-sdk: `EventQueue` lifecycle post-terminal | 0.3.25 | T2 | Determines whether option A fails silently or propagates. |
| 4 | a2a-sdk: `TaskStore` API for in-place Task update | 0.3.25 | T2 | Open assumption #1 option B fallback. |
| 5 | a2a-sdk: `client.get_task(TaskQueryParams)` freshness/caching | 0.3.25 | T2 | Open assumption #4 — every poll must reveal current Task metadata. |
| 6 | a2a-sdk: `client.cancel_task(TaskIdParams)` server-side propagation | 0.3.25 | T2 | Open assumption #5 — confirm no cascade on context_id siblings. |
| 7 | a2a-sdk: Multiple Tasks with shared `context_id` | 0.3.25 | T2 | Open assumption #6 — drain-spawn T3 reuses T1.context_id. |
| 8 | a2a-sdk: `ClientConfig(streaming=False, polling=True, push_notification_configs=[])` | 0.3.25 | T2 | Confirm `send_message` returns Task immediately (non-blocking). |
| 9 | httpx: `ASGITransport` for in-memory FakeA2AServer | 0.28.1 | T0 | Already used in `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py` (verified 2026-05-08). |

Subagent dispatch: 3 spike-runner in parallel covering questions #1–#8 grouped by spike scope.

---

## Per-tech findings

### a2a-sdk@0.3.25 — Task.metadata mutability (questions #1, #2, #3, #4, #5)

**Verified facts** (from spike execution + cross-checked with source)
- `Task.metadata` is `dict[str, Any] | None` with default `None`; once assigned to a dict it's freely mutable, and Pydantic `model_dump(mode="json")` round-trips faithfully. [source: spike step 5 + `.venv/Lib/site-packages/a2a/types.py:1880`]
- The server-side `TaskStore` ABC exposes only `async save(task, context=None)`, `async get(task_id, context=None)`, `async delete(task_id, context=None)`. `InMemoryTaskStore.save` is an upsert keyed by `task.id`. [source: `.venv/Lib/site-packages/a2a/server/tasks/task_store.py:7-29` + `.venv/Lib/site-packages/a2a/server/tasks/inmemory_task_store.py:25-31`]
- **Option B works.** Mutating the stored `Task.metadata` (`store.get(id)` → assign `metadata` → `store.save(task)`) makes the next `client.get_task(TaskQueryParams(id))` return the fresh metadata. [source: spike step 2 — `polled metadata after Option B: {'option_b_marker': 'wrote-this', 'spawned_task_ids': ['c1']}` + `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:111-122` (`on_get_task` reads directly from `task_store.get` with no in-memory cache)]
- **Option A is dead on the canonical path.** After the agent executor returns and `_cleanup_producer` runs, `_queue_manager.close(task_id)` removes the EventQueue entry entirely; `queue_manager.get(task_id)` returns `None`, so there is no live queue whose `enqueue_event` would route a post-terminal `TaskStatusUpdateEvent` into the server's `TaskManager`. [source: spike step 3 — `queue_manager.get(task_id) → None` + `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:438-452` (`_cleanup_producer` → `_queue_manager.close(task_id)`) + `.venv/Lib/site-packages/a2a/server/events/in_memory_queue_manager.py:62-72`]
- Even if a queue were still around, `EventQueue.enqueue_event` early-returns with a `WARNING` log on a closed queue (no exception, no propagation) — silently dropped. [source: `.venv/Lib/site-packages/a2a/server/events/event_queue.py:46-62`]
- The merge logic referenced in the spec (`if event.metadata: task.metadata.update(event.metadata)`) lives in `ClientTaskManager.save_task_event` and on the server-side `TaskManager.save_task_event` — both driven by `EventConsumer` over a live queue; neither runs on a polling `tasks/get` call. [source: `.venv/Lib/site-packages/a2a/client/client_task_manager.py:125-128` + `.venv/Lib/site-packages/a2a/server/tasks/task_manager.py:148-152`]
- **No client-side caching in `JsonRpcTransport.get_task`.** Each call builds a fresh `GetTaskRequest`, posts JSON-RPC, validates `GetTaskResponse`, returns `response.root.result`. Two consecutive polls observed two distinct freshly-saved metadata values. [source: spike step 4 — first poll `{'q4_freshness': 1}`, second poll `{'q4_freshness': 2}` + `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:224-247`]

**Spike file:** `.claude/spikes/a2a-metadata-mutability.py`

**Spike execution output** (key excerpts only)
```
[1] send_message → task_id=bd806d8e-...
    initial state=completed metadata=None
[2] Option B — mutate Task in TaskStore directly, then poll.
    polled metadata after Option B: {'option_b_marker': 'wrote-this', 'spawned_task_ids': ['c1']}
    OPTION B WORKS = True
[3] Option A — emit TaskStatusUpdateEvent(metadata=..., final=False) to the original task's queue.
    queue_manager.get(task_id) → None
    OPTION A WORKS = False
[4] Q4 — JsonRpcTransport.get_task client-side caching check.
    first poll metadata = {'q4_freshness': 1}
    second poll metadata = {'q4_freshness': 2}
    NO CLIENT-SIDE CACHING = True
```

**Constraints discovered**
- The producer queue is gone after the task terminates (`DefaultRequestHandler._cleanup_producer` calls `_queue_manager.close(task_id)`). Even creating a new EventQueue ad-hoc would not be picked up — there is no `TaskManager` listening for that task anymore. **The drainer MUST write to the TaskStore.**
- `EventQueue.enqueue_event` on a closed queue is a silent no-op (warns, returns). Code paths trying option A on a closed queue will not raise but also not propagate.
- `TaskUpdater.update_status` enforces an internal `_terminal_state_reached` guard and raises `RuntimeError` if you try to drive the same `TaskUpdater` after `final=True`. [source: `.venv/Lib/site-packages/a2a/server/tasks/task_updater.py:82-89`]
- `TaskStore.save` does NOT trigger any client invalidation hook — invalidation is unnecessary because `on_get_task` always re-reads from the store.

**Open assumptions** (this spike could not verify)
- `DatabaseTaskStore.save` under concurrent writes — only `InMemoryTaskStore` exercised. Production may need read-modify-write or row-level locking.
- Behavior when an SSE subscription is open against the same task at the moment metadata is mutated. Polling path verified, streaming path not.

**Sources**
- `.venv/Lib/site-packages/a2a/types.py:1855-1888`
- `.venv/Lib/site-packages/a2a/server/tasks/task_store.py:7-29`
- `.venv/Lib/site-packages/a2a/server/tasks/inmemory_task_store.py:25-58`
- `.venv/Lib/site-packages/a2a/server/tasks/task_manager.py:90-158`
- `.venv/Lib/site-packages/a2a/server/tasks/task_updater.py:65-108`
- `.venv/Lib/site-packages/a2a/server/events/event_queue.py:46-62, 135-187`
- `.venv/Lib/site-packages/a2a/server/events/in_memory_queue_manager.py:62-72`
- `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:111-122, 438-452`
- `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:224-247`
- `.venv/Lib/site-packages/a2a/client/client_task_manager.py:114-128`

**Recommendation:** Option A is unusable on a2a-sdk 0.3.25 — the queue is gone before the drainer fires. **Use Option B**: in the drainer, do `task = await task_store.get(task_id); task.metadata = {**(task.metadata or {}), "spawned_task_ids": [...]}; await task_store.save(task)`. Polling clients see the update on the next `tasks/get` (no caching).

---

### a2a-sdk@0.3.25 — cancel-no-cascade (question #6)

**Verified facts**
- **Cancel does NOT cascade across `context_id`.** Two pre-populated tasks T1 and T3 share `context_id=ctx-shared`, both `state=working`. `tasks/cancel(T1)` returns T1 in `canceled` state; `tasks/get(T3)` afterwards still shows `state=working`, untouched timestamp. Custom `SpikeAgentExecutor.cancel_calls` records exactly `["task-T1"]`. [source: spike + `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:124-185` — `on_cancel_task` reads exactly one task via `task_store.get(params.id)`. No context_id lookup, no sibling sweep.]
- **Response shape**: returns the canceled `Task` (camelCase JSON over wire). Client `cancel_task` returns `Task`; raises `A2AClientJSONRPCError` if `JSONRPCErrorResponse`. [source: `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:249-272`]
- **Server-side hook on cancel**: invokes `agent_executor.cancel(RequestContext(task_id=T1, context_id=ctx-shared, task=T1), queue)` for the requested task only. Confirmed empirically. [source: `default_request_handler.py:143-185`]
- **Cancel on already-terminal task**: returns JSON-RPC error, NOT no-op.
  - Code: `-32002` (`TaskNotCancelableError`)
  - Message: `"Task cannot be canceled - current state: TaskState.canceled"`
- **Cancel on unknown id**: `-32001` (`TaskNotFoundError`), message `"Task not found"`.

**Spike file:** `.claude/spikes/a2a-cancel-no-cascade.py`

**Constraints discovered**
- **Executor `cancel()` MUST NOT call `event_queue.close()` itself.** `EventQueue.close()` awaits `queue.join()` until events are `task_done()`'d, but `default_request_handler.on_cancel_task` builds the `EventConsumer` *after* `agent_executor.cancel(...)` — closing inside the executor deadlocks. The handler closes the queue once `EventConsumer.consume_all` sees a `final=True` event. [source: `event_consumer.py:128`, `event_queue.py:135-187`]
- **Cancel on a terminal task is a hard error, not a no-op.** CLI code calling `cancel_task` after ESC must catch `A2AClientJSONRPCError` with `error.code == -32002` and treat as "already done". Same for `-32001`.
- **Cascade behavior is impossible at SDK layer** — no API surface or hook takes a `context_id` and cancels all tasks under it. If Obelix ever needs cascade, it must loop in application code.

**Divergence from spec expectations**: none. The plan's "no cascade" is the SDK default and only behavior.

**Open assumptions**
- Spike pre-populates the task store with idle tasks rather than going through `message/send`. Sufficient to verify cancel-handler scoping. Cancel propagation through an in-flight `agent_executor.cancel()` (with `CancelledError` injected into a running loop) was not exercised — out of scope for the no-cascade question.

**Sources**
- `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:249-272`
- `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:53-58, 124-185`
- `.venv/Lib/site-packages/a2a/server/agent_execution/agent_executor.py:31-44`
- `.venv/Lib/site-packages/a2a/server/events/event_consumer.py:71-149`
- `.venv/Lib/site-packages/a2a/server/events/event_queue.py:135-187`

---

### a2a-sdk@0.3.25 — context_id siblings & ClientConfig polling-only (questions #7, #8)

**Verified facts**
- **Q1 — Sibling Tasks under same `context_id`: ALLOWED, no SDK warning/error.** Server received two `message/send` calls with same `context_id` and no `task_id`. SDK assigned two fresh `task_id`s and persisted both. [source: spike + `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:199-242` — `_setup_message_execution` only does `task_manager.get_task()` keyed on `params.message.task_id`. When no `task_id` supplied, creates a new Task; inbound `context_id` propagated unchanged.]
- **Q2 — `client.get_task` works for both siblings, returns each one's distinct state.** Distinct task identities, same context, distinct lifecycle states.
- **Q3 — NO context-level Task lookup in the SDK.**
  - `dir(client)` filtered by `*context*task*` returned `[]`.
  - `TaskStore` ABC: only `save(task)`, `get(task_id)`, `delete(task_id)`. No `list_by_context`.
  - `InMemoryTaskStore`: single `dict[str, Task]` keyed on `task.id`.
  - Implication for spec 2: any "give me all tasks in this context" lookup must be implemented in Obelix's own `ContextStore`, never delegated to the SDK.
- **Q4 — `ClientConfig(streaming=False, polling=True, push_notification_configs=[])` non-blocking confirmed.** `client.send_message(msg)` returns the iterator immediately yielding `(Task, None)`. First-yield delay 8–28ms, identical to total iterator time. Yielded `Task.status.state` was `submitted`/`working`, NOT terminal — proving no blocking. [source: `.venv/Lib/site-packages/a2a/client/base_client.py:62-118` — L89: `blocking=not self._config.polling`; L109-118: when `streaming=False`, single `transport.send_message`, wraps as `(response, None)`, yields once, returns.]
- **Q5 — Iterator yield shape**: one `(Task, None)` tuple, then iterator ends. The consumer must poll via `client.get_task` (which is exactly the CLI plan).

**Spike file:** `.claude/spikes/a2a-context-id-clientconfig.py`

**Constraints discovered**
- **Same `task_id` reuse with same `context_id` is NOT a sibling — it's task continuation.** If caller sets `params.message.task_id` to existing task id, `_setup_message_execution` calls `task_manager.update_with_message`; if task is already terminal, raises `InvalidParamsError`. CLI plan only creates siblings by sending `task_id=None` (or fresh id), which is what spec 1's drain-spawn already does.
- **`AgentCard.capabilities.streaming` matters.** `base_client.py:109` uses `not self._config.streaming or not self._card.capabilities.streaming` — even if `ClientConfig.streaming=True`, a card with `capabilities.streaming=False` forces non-streaming path. Spec 2 sets `streaming=False` in ClientConfig for safety.
- **`push_notification_configs=[]`**: confirmed harmless. `base_client.py:90-94` only attaches FIRST entry; empty list yields `push_notification_config=None`. Server does not require a push config when `blocking=False`.

**Open assumptions**
- Obelix server-side context propagation (Obelix's own `ContextStore` for trace_id reuse across siblings) is verified by the existing test `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py`, not here.
- Behavior under long-running task with actual polling cadence + cancellation — separate concern, covered by integration tests in spec 2 plan.

**Sources**
- `.venv/Lib/site-packages/a2a/client/base_client.py:62-118`
- `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:114-137`
- `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:199-260`
- `.venv/Lib/site-packages/a2a/server/tasks/task_store.py`
- `.venv/Lib/site-packages/a2a/server/tasks/inmemory_task_store.py:22`
- `.venv/Lib/site-packages/a2a/client/client.py:30-72`

**Verdict**: Spec 2's two assumptions — (a) multi-Task-per-context_id supported and SDK-friendly, with SDK refusing to do context-level lookups itself; (b) `ClientConfig(streaming=False, polling=True, push_notification_configs=[])` returns iterator non-blocking with one `(Task, None)` yield — are both empirically and source-level confirmed.

---

### httpx@0.28.1 — ASGITransport (T0)

Verified at `tests/adapters/inbound/a2a/server/test_drain_spawn_e2e.py` (existing pattern, in use 2026-05-08). No subagent dispatch needed.

---

## Open assumptions

The following remain unverified after this research and must be carried into implementation as known fragilities:

1. `DatabaseTaskStore.save` under concurrent writes is not characterized. Production deployments using non-`InMemoryTaskStore` may need read-modify-write or row-level locking. Mitigation: spec 2 development uses `InMemoryTaskStore`; switching to `DatabaseTaskStore` would warrant a follow-up spike.
2. Behavior when an SSE subscription is open against the same task at the moment metadata is mutated via TaskStore. Spec 2's polling path is fully verified; if external clients use SSE this should be explicitly tested.
3. Cancel propagation through an in-flight `agent_executor.cancel()` — the spike used pre-populated idle tasks for scoping verification. Real cancel-during-running-loop is exercised by integration tests in the implementation plan, not here.

---

## Out of scope

Deliberately not researched:

- **Push notifications API** (`tasks/pushNotificationConfig/set`, `get`). Spec 2 eliminates these — no need to characterize their behavior.
- **`message/stream` / `tasks/resubscribe` SSE behavior**. Spec 2 doesn't use SSE — already covered by SDK source review during brainstorming.
- **`a2a-sdk` versions other than 0.3.25**. Future upgrades require a fresh research run.
- **Non-Python A2A clients**. Spec 2 only ships a Python CLI client; cross-language interop unaffected.

---

## Required spec amendments

### Amendment 1 — § 3 "Vincolo SDK aperto (open assumption #1)" — Option A is dead on a2a-sdk 0.3.25, mandate Option B

**Spec section:** § 3 — Data flow e metadata schema → Vincolo SDK aperto (open assumption #1)

**Current text (in spec):**
> A2A SDK non documenta come modificare `Task.metadata` post-creation. Due
> candidati di implementazione, da verificare in pre-impl research:
>
> 1. **In-place TaskStore patch**: il drainer recupera il Task dallo store
>    SDK, modifica `metadata`, lo riscrive. Il prossimo `tasks/get(T1)` legge
>    il Task aggiornato. Pro: semplice, immediato. Contro: bypassa il
>    meccanismo eventi (potenziale race con altri reader).
>
> 2. **TaskStatusUpdateEvent metadata-only**: il drainer emette un evento
>    `TaskStatusUpdateEvent(task_id=T1, status=<unchanged>, metadata={...},
>    final=False)` sulla EventQueue di T1. `ClientTaskManager.save_task_event`
>    merge-a `event.metadata` su `task.metadata` (vedi
>    `client_task_manager.py:125-128`). Pro: SDK-canonical. Contro: **se T1 è
>    già terminal la EventQueue è chiusa**, l'evento non si propaga. Da
>    verificare se l'SDK lo accetta o solleva.
>
> Decisione: si parte preferendo l'opzione 2 (SDK-canonical). Se la
> pre-impl research dimostra che opzione 2 fallisce dopo task terminale, si
> ripiega su opzione 1.

**Issue:** Pre-impl research (spike `a2a-metadata-mutability.py`) ha verificato che opzione 2 (`TaskStatusUpdateEvent` post-terminal) **NON funziona** sull'a2a-sdk 0.3.25: dopo che il task raggiunge stato terminal, `DefaultRequestHandler._cleanup_producer` chiama `_queue_manager.close(task_id)` che rimuove la EventQueue dal manager (`queue_manager.get(task_id) → None`). Anche se la queue esistesse ancora, `EventQueue.enqueue_event` su queue chiusa è un silent no-op (warning, no exception, no propagation). Inoltre `ClientTaskManager.save_task_event` non viene mai eseguito su un polling `tasks/get` — è driven solo da `EventConsumer` su una queue live. Vedi finding section "a2a-sdk@0.3.25 — Task.metadata mutability".

**Replacement text:**
> Verificato in pre-impl research (`docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md` finding "Task.metadata mutability"): l'a2a-sdk 0.3.25 supporta solo l'**in-place TaskStore patch**.
>
> **Pattern obbligatorio**:
>
> ```python
> task = await task_store.get(task_id)
> if task is None:
>     return  # task evicted, niente da fare
> task.metadata = {**(task.metadata or {}), "spawned_task_ids": [...]}
> await task_store.save(task)
> ```
>
> `task_store.save` è un upsert keyed by `task.id` (`InMemoryTaskStore.save` at `.venv/Lib/site-packages/a2a/server/tasks/inmemory_task_store.py:25-31`). Il prossimo `client.get_task(T1)` ritorna il Task con metadata aggiornato senza alcuna logica di invalidation, perché `DefaultRequestHandler.on_get_task` rilegge sempre dal `task_store` (no in-memory cache, verificato a `.venv/Lib/site-packages/a2a/server/request_handlers/default_request_handler.py:111-122`). Niente caching client-side in `JsonRpcTransport.get_task` (verificato a `.venv/Lib/site-packages/a2a/client/transports/jsonrpc.py:224-247`).
>
> **Path eliminato**: emettere `TaskStatusUpdateEvent(metadata={...}, final=False)` sulla EventQueue del task terminale è impossibile — la queue viene chiusa da `_cleanup_producer` prima che il drainer possa fire. L'evento sarebbe silently dropped (`EventQueue.enqueue_event` su queue chiusa = warning + early return).

**Status:** Applied at 2026-05-08 17:30 — spec patch in commit (next)

---

### Amendment 2 — § 4 "Errori transienti" — aggiungere edge case cancel su task terminal

**Spec section:** § 4 — Lifecycle e edge cases → Errori transienti

**Current text (in spec):**
> **Errori transienti** durante `tasks/get`:
>
> - 5xx, timeout, network error: continua il loop al prossimo ciclo.
> - 4xx: stop di quel task, log dell'errore in chat (non più con prefisso
>   `[poll]` — è il canale primario, non un fallback).
> - JSON-RPC `-32001 Task not found`: non dovrebbe più accadere (i drain-spawn
>   task in spec 2 sono task A2A regolari nello SDK store). Se accade è un
>   bug genuino — log e stop di quel task. **Open assumption #2**: la
>   pre-impl research deve verificare che il drainer registri T3 nel
>   TaskStore SDK PRIMA di scrivere `T3.id` in `T1.metadata.spawned_task_ids`,
>   per evitare race fra annuncio del task e disponibilità via `tasks/get`.

**Issue:** Manca il caso `cancel_task` su task in stato terminale, identificato dallo spike `a2a-cancel-no-cascade.py`. Quando l'utente preme ESC dopo che T1 è già transitato in stato terminale (es. completed nel breve istante prima che il keypress arrivi), l'SDK solleva `A2AClientJSONRPCError(code=-32002, "TaskNotCancelableError")`. Questo non è un errore reale — il task è già done. La CLI deve ignorarlo silenziosamente. Stessa cosa per `-32001` (task evicted from store fra il momento in cui la CLI tracciava `last_task_id` e il momento del cancel).

**Replacement text:**
> **Errori transienti** durante `tasks/get`:
>
> - 5xx, timeout, network error: continua il loop al prossimo ciclo.
> - 4xx: stop di quel task, log dell'errore in chat (non più con prefisso
>   `[poll]` — è il canale primario, non un fallback).
> - JSON-RPC `-32001 Task not found` su `tasks/get`: non dovrebbe più accadere
>   (i drain-spawn task in spec 2 sono task A2A regolari nello SDK store). Se
>   accade è un bug genuino — log e stop di quel task.
>
> **Errori su `cancel_task`** (ESC dell'utente):
>
> - JSON-RPC `-32002 TaskNotCancelableError`: il task era già in stato terminale
>   al momento del cancel (race fra completion naturale e keypress utente).
>   Trattare come no-op silenzioso (log debug, niente messaggio in chat).
>   Verificato in pre-impl research finding "cancel-no-cascade".
> - JSON-RPC `-32001 TaskNotFoundError`: il task era stato evict dal TaskStore
>   fra il tracking di `last_task_id` e il cancel. Trattare come no-op silenzioso.
> - Altri errori: log e messaggio in chat come oggi.
>
> **Race ordering vincolante** (drain-spawn append vs TaskStore registration):
> il drainer DEVE registrare T3 nel `TaskStore` SDK (via `task_store.save(T3)`)
> PRIMA di scrivere `T3.id` in `T1.metadata.spawned_task_ids` (via il pattern
> in-place patch di Amendment 1). Se invertito, la CLI può fare poll del
> nuovo `T3.id` prima che il task sia disponibile e ricevere `-32001`.

**Status:** Applied at 2026-05-08 17:30 — spec patch in commit (next)

---

### Amendment 3 — § 8 "Open assumptions" — chiudere le assumption verificate dalla research

**Spec section:** § 8 — Open assumptions (per pre-implementation research)

**Current text (in spec):**
> Da risolvere in `docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md`
> prima del piano:
>
> 1. **Mutabilità di `Task.metadata` post-creation nell'a2a-sdk**.
>    - Verificare se `TaskStatusUpdateEvent(metadata={...}, final=False)`
>      emesso DOPO che il Task è in stato terminal (state=completed) è
>      accettato dal server SDK e propagato correttamente al
>      `ClientTaskManager` lato client.
>    - Se sì → opzione 2 (SDK-canonical).
>    - Se no → opzione 1 (in-place TaskStore patch). Verificare API SDK per
>      update di Task nello store.
> 2. **Race fra append a `spawned_task_ids` e registrazione del child task
>    nel TaskStore**. Il drainer deve garantire ordine: prima registra T3 nel
>    TaskStore SDK (così `tasks/get(T3)` ritorna 200), POI append in
>    `T1.metadata.spawned_task_ids`. Verificare API SDK per task
>    pre-registration.
> 3. **Come `dispatched_peers` viene aggiornato in metadata**: stessa
>    questione di #1 ma per update di un campo esistente (non solo append).
>    Probabile stessa soluzione.
> 4. **Comportamento di `client.get_task` su task in stato terminal**.
>    Risponde sempre con il Task corrente (compreso `metadata` aggiornato),
>    o l'SDK lo cache? Se cache, eviction policy?
> 5. **Cancellazione di un task con `dispatched_peers` attivi**. Quando
>    `cancel_task(T1)` arriva al server O, cosa succede ai T2 (peer) e ai T3
>    (drain-spawn) attivi server-side? Probabile: nulla automatico (è il caso
>    "no cascade" che vogliamo). Verificare per essere sicuri.
> 6. **Conformità al ciclo di vita del Task SDK**. Il drain-spawn task T3 è
>    un task "child" con un suo lifecycle indipendente, ma condivide
>    `context_id` con T1 (per riuso trace_id, già da spec 1). Verificare che
>    l'SDK accetti più Task con stesso `context_id` senza errori.
>
> Tier subagent dispatch: **T2** (spike-runner) per #1, #2, #4 — comportamento
> SDK richiede esecuzione contro istanza reale. **T1** (doc-verifier) per #5
> e #6 (semantica documentata, non comportamento runtime). **T0** se
> applicabile (dipende da quanto è già verificato nel codebase corrente).

**Issue:** Pre-impl research è stata eseguita (3 spike T2). Tutte le 6 open assumption sono ora verificate o esplicitamente lasciate come fragilità note. La sezione deve riflettere la chiusura, non lo stato pre-research.

**Replacement text:**
> Pre-implementation research completata 2026-05-08, vedi
> `docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md`.
> Esiti delle 6 open assumption iniziali:
>
> 1. **`Task.metadata` post-creation** → risolto. Vedi Amendment 1: solo
>    in-place TaskStore patch funziona; option SDK-canonical via
>    `TaskStatusUpdateEvent` è impossibile (queue chiusa post-terminal).
> 2. **Race fra append a `spawned_task_ids` e TaskStore registration** →
>    risolto. Vedi Amendment 2: il drainer DEVE chiamare `task_store.save(T3)`
>    PRIMA di scrivere `T3.id` in `T1.metadata.spawned_task_ids`.
> 3. **Update di `dispatched_peers` in metadata** → risolto, stesso pattern di
>    #1 (in-place TaskStore patch).
> 4. **`client.get_task` freshness** → risolto. Nessun caching client-side in
>    `JsonRpcTransport.get_task`; nessun caching server-side in
>    `DefaultRequestHandler.on_get_task` (rilegge sempre dal `task_store`).
> 5. **Cancel-no-cascade** → risolto. Vedi Amendment 2: cancel è strettamente
>    task-scoped, no API per cascade su `context_id`. Edge case su cancel di
>    task terminale aggiunto (`-32002`).
> 6. **Multi-Task con stesso `context_id`** → risolto. Supportato senza
>    errori; `TaskStore` indicizza solo per `task.id`, non per `context_id`.
>    Lookup context-level deve essere fatto in Obelix (ContextStore), non
>    delegato all'SDK.
>
> **Fragilità note residue** (da carry come Open assumptions nel piano):
>
> - `DatabaseTaskStore.save` sotto write concorrenti non caratterizzato.
>   Spec 2 sviluppa contro `InMemoryTaskStore`; switch futuri richiederanno
>   un follow-up spike.
> - Comportamento di SSE subscription aperta durante mutation di
>   `Task.metadata` via TaskStore non testato. Spec 2 non usa SSE → non
>   blocking, ma se client esterni la usano serve test esplicito.
> - Cancel-during-in-flight `agent_executor.cancel()` non esercitato dallo
>   spike (test contro pre-populated idle tasks). Coperto dai test
>   integrazione del piano spec 2.

**Status:** Applied at 2026-05-08 17:30 — spec patch in commit (next)

---

## Subagent dispatch log

- 2026-05-08 17:00 a2a-sdk@0.3.25 (T2) — Task.metadata mutability spike → success, all 5 questions answered, ~234s, agentId=a71b4d75b52ffaacb
- 2026-05-08 17:00 a2a-sdk@0.3.25 (T2) — cancel-no-cascade spike → success, all 4 questions answered, ~501s, agentId=af3a8d6ab850d9c94
- 2026-05-08 17:00 a2a-sdk@0.3.25 (T2) — context_id-and-clientconfig spike → success, all 5 questions answered, ~201s, agentId=ab610b7db9d653e76
