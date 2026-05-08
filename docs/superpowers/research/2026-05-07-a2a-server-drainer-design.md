# Pre-implementation research: A2A server drainer + tracer trace_id reuse

**Spec**: `docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md`
**Date**: 2026-05-07
**Owner**: f.crino@ads.it

## Tech inventory & classification

| # | Technology | Version | Tier | Justification |
|---|---|---|---|---|
| 1 | `a2a.types.Task.model_dump(mode="json", exclude_none=True)` | a2a-sdk 0.3.25 | **T2** | Spike-runner verified: payload **camelCase**, contratti per 7 stati documentati. Vedi spike `.claude/spikes/a2a-types-spike.py` |
| 2 | `a2a.types.Message(role=Role.user, parts=[], context_id=...)` | a2a-sdk 0.3.25 | **T2** | Spike-runner verified: `parts=[]` accettato, shape camelCase documentata |
| 3 | `a2a.types.Role` enum | a2a-sdk 0.3.25 | **T2** | Spike-runner verified: solo `Role.user`, `Role.agent` (nessun `system`/`tool`) |
| 4 | `uuid.uuid4()` | stdlib (Python 3.13) | **T0** | stdlib; usato già `dispatch.py:139, 154` |
| 5 | `asyncio.create_task(coro, name=...)` | stdlib (Python 3.13) | **T0** | stdlib; usato `webhook_server.py:212`, `polling.py:55` |
| 6 | `httpx.AsyncClient.post(url, json=, headers=, timeout=)` | httpx ~0.28 | **T0** | Usato `push_sender.py:53-62` per pattern identico (POST Task JSON + token header) |
| 7 | `Tracer.start_trace(name, metadata) -> TraceSession` | Obelix interno | **T0** | Definita `src/obelix/core/tracer/tracer.py:57`, usata `executor.py:373` |
| 8 | `Tracer.start_span(SpanType, name, input, metadata)` | Obelix interno | **T0** | Definita `tracer.py:99`, usata `executor.py:377` |
| 9 | `set_current_trace(trace)` / `get_current_trace()` | Obelix interno | **T0** | Pattern di trace cross-context riusato da `webhook.py:113-115, 138` |
| 10 | `SpanType.a2a_task` | Obelix interno | **T0** | Usato `executor.py:378` |
| 11 | `ContextEntry.idle: asyncio.Event` | Obelix interno | **T0** | Definita `context.py:49-50`, usata `executor.py:309-343` per gating turni — **uso questo invece di un nuovo `has_active_task()`** |
| 12 | `ContextEntry.pending_notifications: list[HumanMessage]` | Obelix interno | **T0** | Definita `context.py:87`, drenata `executor.py:323-330` (swap-pattern atomico) |
| 13 | `ContextEntry.trace_session` | Obelix interno | **T0** | Definita `context.py:33,53`, salvata `executor.py:389`, riusata `webhook.py:115` |
| 14 | `ContextStore.peek(context_id)` / `iter_entries()` | Obelix interno | **T0** | Definite `context.py:127-141` |
| 15 | `HumanMessage` (Pydantic) | Obelix interno | **T0** | Definito `obelix.core.model.human_message`, costruito in `notification.py:85` |
| 16 | `starlette.responses.JSONResponse` | Starlette ~0.41 | **T0** | Usato `webhook.py`, `webhook_server.py` |

## Per-tech findings

### T2 — `a2a.types.Task` (spike-runner verified)

**File definizione**: `.venv/Lib/site-packages/a2a/types.py:1855`

**Campi**: `id` (req), `context_id` (req), `kind: Literal['task']='task'`, `status: TaskStatus` (req), `artifacts: list[Artifact]|None`, `history: list[Message]|None`, `metadata: dict|None`.

**Serialization base**: `A2ABaseModel` (`.venv/Lib/site-packages/a2a/_base.py:21-38`) ha `serialize_by_alias=True` + `alias_generator=to_camel_custom` → l'output JSON è **camelCase** (`contextId`, `taskId`, `messageId`, `artifactId`), NON snake_case.

**Shape `state=working`** (esempio runtime):
```json
{"contextId":"ctx-spike-001","id":"task-1","kind":"task","status":{"state":"working","timestamp":"2026-05-07T12:00:00+00:00"}}
```

**Shape `state=completed` con artifacts**:
```json
{"artifacts":[{"artifactId":"art-1","name":"reply","parts":[{"kind":"text","text":"hello world"}]}],"contextId":"...","id":"...","kind":"task","status":{"state":"completed","timestamp":"..."}}
```

**Shape `state=input-required` con DataPart deferred**:
```json
{"contextId":"...","id":"...","kind":"task","status":{"state":"input-required","message":{"kind":"message","messageId":"...","parts":[{"data":{"deferred_tool_calls":[...]},"kind":"data"}],"role":"agent"},"timestamp":"..."}}
```
Nota: `data` dentro DataPart resta snake_case perché è contenuto utente, non un nome di campo Pydantic.

**Shape `state=failed` con error message**: `status.message.parts[0]` = `{"kind":"text","text":"boom: provider 500"}`.

**`exclude_none=True` behavior**: rimuove `artifacts`, `history`, `metadata`, `status.message` quando None. Mantiene sempre `id`, `contextId`, `kind`, `status.state`. `status.timestamp` mantenuto se settato.

**Timestamp serializzazione**: `status.timestamp` è già `str | None` (NON datetime) — `.venv/Lib/site-packages/a2a/types.py:1652`. Nessuna conversione datetime→ISO automatica: il chiamante deve passare ISO string.

**Confronto con codice esistente**: `push_sender.py:53` usa già `task.model_dump(mode="json", exclude_none=True)`. Il drainer può usare lo stesso pattern senza modifiche.

### T2 — `a2a.types.Message` (spike-runner verified)

**File definizione**: `.venv/Lib/site-packages/a2a/types.py:1436`

**Campi richiesti**: `message_id: str`, `parts: list[Part]`, `role: Role`. **Opzionali**: `context_id`, `task_id`, `metadata`, `extensions`, `reference_task_ids`, `kind: Literal['message']='message'`.

**Costruzione `Message(message_id=..., role=Role.user, parts=[], context_id="ctx-test")`**: **ACCETTATA** — `parts: list[Part]` non ha `min_length`, lista vuota OK.

**Shape `model_dump(mode="json", exclude_none=True)`**:
```json
{"contextId":"ctx-spike-001","kind":"message","messageId":"<uuid>","parts":[],"role":"user"}
```

### T2 — `a2a.types.Role` (spike-runner verified)

**File definizione**: `.venv/Lib/site-packages/a2a/types.py:713`

`class Role(str, Enum)`: due soli valori — `Role.agent = 'agent'`, `Role.user = 'user'`. Niente `system`, `tool`, `assistant`.

### Bonus — `TaskState`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`

- `TaskState` (`.venv/Lib/site-packages/a2a/types.py:989`): valori serializzati `submitted`, `working`, `input-required`, `completed`, `canceled`, `failed`, `rejected`, `auth-required`, `unknown`. Hyphenated forma sul wire; gli attr Python usano underscore (`input_required`, `auth_required`). `_state_str` in `handler.py:39` già normalizza.
- `TaskStatusUpdateEvent` (`:1660`): `kind='status-update'`, campi camelCase `taskId`, `contextId`, `final: bool`, `status`, `metadata?`.
- `TaskArtifactUpdateEvent` (`:1603`): `kind='artifact-update'`, camelCase `taskId`, `contextId`, `artifact`, `append?`, `lastChunk?`, `metadata?`.

### T0 — Tutti gli altri

Verificati via `git grep` in repo, già usati con citation file:linea nella tabella sopra. Nessun rischio di drift sui contratti perché le call site esistenti definiscono il pattern atteso.

**Punto critico T0**: `ContextEntry.idle` è il meccanismo già esistente per "context busy detection". La spec inizialmente proponeva un nuovo metodo `entry.has_active_task()`; la verifica nel repo mostra che `entry.idle.is_set()` (Event interno) è la primitiva esistente per lo stesso concetto. **Vedi amendment 1 sotto.**

## Open assumptions

_no amendments — tutti i punti del design verificati contro contratti reali_.

Una sola nota: i Fake class nei test devono produrre payload **camelCase** (`contextId`, `taskId`, ...). Sbagliarli passerebbe il test ma fallerebbe il parsing lato client. Va annotato esplicitamente nelle istruzioni dei sub-agent implementatori (vedi spec § 8.7).

## Out of scope

Coerentemente con lo spec § 7, non sono stati ricercati:
- Contratti di `a2a.client.Client` (streaming, resubscribe) — competenza spec 2
- `MessageSendConfiguration` / `PushNotificationConfig` SDK — il drainer bypassa il sender SDK e fa POST diretta
- Comportamento `ClientFactory` / `ClientConfig` — competenza spec 2

## Required spec amendments

### Amendment 1 — `entry.has_active_task()` non serve, riusare `entry.idle`

**Spec section**: `§ 4.3 Logica` e `§ 4.4 Dettagli sui check`, e `Appendice A — File toccati` riga "EDIT: src/obelix/adapters/inbound/a2a/server/context.py".

**Current text (in spec)**:

> **Check 2**: serve un metodo `entry.has_active_task() -> bool` che ritorni True se esiste un task A2A non terminale per quel context. La logica si appoggia al `task_store` dell'executor (o equivalente). Il metodo ritorna False quando tutti i task sono in stato `completed`/`failed`/`canceled`/`rejected`. Implementazione iniziale: itera su `task_store.list_tasks_for_context(context_id)` e verifica gli stati. Se il `task_store` non espone questa API, va aggiunta.

**Issue**: la verifica nel repo (`context.py:49-50` + `executor.py:309-343`) mostra che il meccanismo "context ha un turno A2A in corso?" esiste già: `entry.idle: asyncio.Event` viene clearato all'inizio del turno (`executor.py:310`) e settato alla fine (`:343`). Non serve aggiungere un nuovo metodo né accedere a un `task_store` che — verificato via `grep -n "task_store"` su `src/obelix/adapters/inbound/a2a/server/` — **non esiste**.

**Replacement text**:

> **Check 2**: usa `entry.idle: asyncio.Event` esistente (`context.py:49-50`). L'executor già clear/set questo event all'inizio/fine di ogni turno A2A (`executor.py:310, 343`). Quindi `entry.idle.is_set() == True` significa "nessun turno A2A in corso per il context". La condizione del drainer è:
>
> ```python
> if not entry.idle.is_set():
>     return  # turno in corso, le notifiche verranno drenate da lui
> ```
>
> Niente nuovo metodo da aggiungere a `ContextEntry`. Niente nuovo store da introdurre.

**Status**: Applied at 2026-05-07 — spec patch in commit (next).

### Amendment 2 — chiarire che il payload webhook è camelCase

**Spec section**: `§ 6.2 POST diretta dal drainer`.

**Current text (in spec)**:

> ```python
> # executor.py — TEMP-PATCH-SPEC-1
> if is_drain_spawn and entry.client_webhook_url:
>     try:
>         await self._httpx_client.post(
>             entry.client_webhook_url,
>             json=task.model_dump(mode="json", exclude_none=True),
>             headers={"X-A2A-Notification-Token": entry.client_webhook_token or ""},
>             timeout=5.0,
>         )
>     except Exception:
>         logger.warning("[A2A drain] webhook POST failed (best-effort)")
> ```

**Issue**: il `model_dump(mode="json", exclude_none=True)` produce JSON con campi **camelCase** (`contextId`, `taskId`, `messageId`, `artifactId`) per via dell'`A2ABaseModel.alias_generator` (`.venv/Lib/site-packages/a2a/_base.py:21-38`). I Fake class nei test devono rispettare camelCase, e qualunque consumer client-side ne deve essere aware. Da annotare per evitare confusione sui Fake (che altrimenti userebbero snake_case e farebbero passare i test ma fallirebbero in prod).

**Replacement text**:

> ```python
> # executor.py — TEMP-PATCH-SPEC-1
> if is_drain_spawn and entry.client_webhook_url:
>     try:
>         await self._httpx_client.post(
>             entry.client_webhook_url,
>             json=task.model_dump(mode="json", exclude_none=True),
>             headers={"X-A2A-Notification-Token": entry.client_webhook_token or ""},
>             timeout=5.0,
>         )
>     except Exception:
>         logger.warning("[A2A drain] webhook POST failed (best-effort)")
> ```
>
> **Nota sul payload**: `Task.model_dump(mode="json", exclude_none=True)` produce JSON **camelCase** (`contextId`, `taskId`, `messageId`, `artifactId`) per via dell'`A2ABaseModel.alias_generator`. I Fake class nei test devono rispettare camelCase per essere allineati alla realtà del wire format. Esempio della shape per `state=working`:
> ```json
> {"contextId":"...","id":"...","kind":"task","status":{"state":"working","timestamp":"..."}}
> ```

**Status**: Applied at 2026-05-07 — spec patch in commit (next).

### Amendment 3 — Aggiornare Appendice A per riflettere amendment 1

**Spec section**: `Appendice A — File toccati`.

**Current text (in spec)**:

> EDIT: src/obelix/adapters/inbound/a2a/server/context.py
>   - aggiungere campi client_webhook_url, client_webhook_token su ContextEntry
>   - aggiungere metodo has_active_task() (o equivalente)

**Issue**: `has_active_task()` non va aggiunto (vedi amendment 1).

**Replacement text**:

> EDIT: src/obelix/adapters/inbound/a2a/server/context.py
>   - aggiungere campi client_webhook_url, client_webhook_token su ContextEntry (aggiungerli a __slots__ e __init__)

**Status**: Applied at 2026-05-07 — spec patch in commit (next).

## Subagent dispatch log

- 2026-05-07 — `a2a-sdk Task/Message/Role`@0.3.25 (T2) → spike-runner ok in 104s, file `.claude/spikes/a2a-types-spike.py` creato + report verbatim incluso in "Per-tech findings"
