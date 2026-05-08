# A2A Server-Side Drainer + Tracer trace_id Reuse — Design Spec

**Status**: design (in review)
**Created**: 2026-05-07
**Owner**: f.crino@ads.it
**Roadmap**: [docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md](../2026-05-07-a2a-async-agents-roadmap.md) — questa è la **spec 1** di una serie di 5.

## 1. Problema

Quando un agent A (server A2A in Obelix) dispatcha un task asincrono verso un agent B tramite `DispatchAgentTool` (`src/obelix/adapters/outbound/a2a/tools/dispatch.py`), il flusso oggi è:

1. A esegue `dispatch_agent("B", "...")` → ritorna immediatamente con `task_id=X`.
2. A finisce il proprio turno → il suo task A2A passa a stato `completed`.
3. B esegue il proprio lavoro in background.
4. B termina e notifica A (via push se il sender è attivo, o via polling fallback in `src/obelix/adapters/outbound/a2a/polling.py`).
5. `handle_remote_update` (`src/obelix/adapters/outbound/a2a/handler.py:93`) accoda un `HumanMessage` con XML `<remote_task_update>` in `entry.pending_notifications`.
6. **Nessuno fa partire un nuovo turno di A**. La notifica resta in coda finché l'utente non manda un nuovo messaggio.

Diagnosi confermata via tracer (sessione 2026-05-07): l'orchestrator dispatcha il coordinator, il coordinator completa, l'utente alle 14:31:39 chiede "ancora niente?" e solo a quel punto l'agent risponde con conoscenza già drenata. La pipeline interna funziona; manca il **trigger automatico** che fa partire un nuovo turno di A quando arriva la notifica di B.

In più, anche se il fix server-side fosse risolto, il client CLI (`src/obelix/adapters/inbound/a2a/client/cli_client.py`) non vedrebbe il task spawnato dal trigger automatico, perché la sua connessione (stream/polling/webhook) è legata al task originale di A che è già `completed`.

Infine, il tracer Obelix oggi crea un nuovo `trace_id` per ogni task A2A (`executor.py:373` chiama `tracer.start_trace`). Una conversazione utente che innesca multipli turni di A (turno user-triggered + turno auto-triggered da notifica di B) produce N trace separati, frammentando la storia visibile nel frontend.

## 2. Obiettivi e non-obiettivi

### Obiettivi

- L'agent A risponde all'utente non appena arriva la risposta del remote B, senza richiedere input utente.
- Il client CLI vede questa risposta apparire spontaneamente (per la durata di questa spec, tramite patch temporanea).
- I turni multipli di A nello stesso context (user-triggered + auto-triggered) restano sotto lo stesso `trace_id`, formando un solo tree visivo nel frontend tracer.
- Niente regressione su task user-triggered esistenti.

### Non-obiettivi (esclusi esplicitamente — vedi § 6)

Si rimanda al roadmap doc per le altre spec. Out-of-scope qui:

- CLI streaming SSE migration (spec 2)
- Peer discovery A→B nella CLI (spec 3)
- Peer UI nella status bar (spec 4)
- Revisione fire-and-forget di `dispatch_agent` (spec 5)
- Modifiche al frontend tracer
- JWT/signing/retry sul push del drainer
- Re-enable del `BasePushNotificationSender` SDK lato server
- Modifiche a `DispatchAgentTool`

## 3. Architettura

Lo spec introduce **3 componenti server-side** strettamente coordinati:

```
                                      ┌─────────────────────────────┐
                                      │ entry.pending_notifications │
                                      │  list[HumanMessage]         │
                                      └──────┬──────────────────────┘
                                             │ append
   handle_remote_update ──────────────────┐  │
   (webhook outbound o polling)           │  │
                                          ▼  ▼
                                      ┌─────────────────────────────┐
                                      │  drainer.maybe_spawn(...)   │  (1)
                                      │  - notif vuota? return      │
                                      │  - task attivo? return      │
                                      │  - else: executor.spawn()   │
                                      └──────┬──────────────────────┘
                                             │
                                             ▼
                                      ┌─────────────────────────────┐
                                      │  executor.spawn_drain_task  │
                                      │  - NUOVO task A2A interno   │
                                      │  - Message synthetic vuoto  │
                                      │  - is_drain_spawn=True      │
                                      └──────┬──────────────────────┘
                                             │
                                             ▼
                                      ┌─────────────────────────────┐
                                      │  _run_agent (esistente)     │
                                      │  - skip start_trace          │  (2)
                                      │  - set_current_trace(saved) │
                                      │  - drain pending_notif       │
                                      │  - LLM, tools, response      │
                                      └──────┬──────────────────────┘
                                             │ task state changes
                                             ▼
                                      ┌─────────────────────────────┐
                                      │  POST direct → CLI webhook  │  (3)
                                      │  # TEMP-PATCH-SPEC-1         │
                                      └─────────────────────────────┘
```

I 3 componenti numerati:

1. **Drainer** — funzione async che decide se spawnare o no un nuovo task A2A.
2. **Tracer trace_id reuse** — l'executor, per task spawnati dal drainer, riusa il trace esistente invece di crearne uno nuovo.
3. **Patch temporanea webhook CLI** — POST diretta al webhook locale del client per i task spawnati, in modo che la CLI li veda. Da rimuovere in spec 2.

## 4. Componente 1 — Drainer

### 4.1 Posizione nel codice

Nuovo modulo `src/obelix/adapters/inbound/a2a/server/drainer.py`. Funzione pura, niente classe, niente loop:

```python
async def maybe_spawn_drain_task(
    *,
    entry: ContextEntry,
    context_id: str,
    executor: ObelixAgentExecutor,
) -> None:
    ...
```

### 4.2 Quando viene chiamata

`handle_remote_update` è una funzione **sync** (`handler.py:93-99`). Il drainer è **async** (deve creare task asyncio). Quindi non può essere chiamato direttamente dal corpo di `handle_remote_update`. Va invece chiamato dai due call site async, **dopo** la chiamata sync a `handle_remote_update`:

- `src/obelix/adapters/outbound/a2a/webhook.py` — nella `webhook_handler` async closure, dopo `handle_remote_update(...)`.
- `src/obelix/adapters/outbound/a2a/polling.py` — dentro `_poll_one` async, dopo `handle_remote_update(...)`.

Convenzione: la chiamata al drainer è sempre `await maybe_spawn_drain_task(entry=ctx_entry, context_id=ctx_entry.context_id, executor=self._executor)` immediatamente dopo. Il drainer è idempotente quindi due chiamate consecutive (caso teorico in cui webhook e polling arrivano in finestra brevissima sullo stesso task) sono safe.

<!-- amended per docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md amendment 1 on 2026-05-07 -->
### 4.3 Logica

Pseudo-codice:

```
async def maybe_spawn_drain_task(entry, context_id, executor):
    # Check 1: c'è una notifica da drenare?
    if not entry.pending_notifications:
        return  # niente da fare, idempotente

    # Check 2: c'è già un turno A2A attivo per questo context?
    # (entry.idle è asyncio.Event esistente, gestito da executor.py:310, 343)
    if not entry.idle.is_set():
        return  # turno in corso, le notifiche verranno drenate da lui

    # Spawn
    await executor.spawn_drain_task(entry=entry, context_id=context_id)
```

<!-- amended per docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md amendment 1 on 2026-05-07 -->
### 4.4 Dettagli sui check

**Check 1**: `entry.pending_notifications` è già una `list[HumanMessage]` esistente. Verifica `if not lst` è sufficiente (lock asyncio non serve perché Python list operations sono atomiche e l'append è la sola operazione concorrente).

**Check 2**: usa `entry.idle: asyncio.Event` esistente (`context.py:49-50`). L'executor già clear/set questo event all'inizio/fine di ogni turno A2A (`executor.py:310, 343`). Quindi `entry.idle.is_set() == True` significa "nessun turno A2A in corso per il context". La condizione del drainer è:

```python
if not entry.idle.is_set():
    return  # turno in corso, le notifiche verranno drenate da lui
```

Niente nuovo metodo da aggiungere a `ContextEntry`. Niente nuovo store da introdurre.

### 4.5 Spawn del task

Il metodo `executor.spawn_drain_task(entry, context_id)` deve essere **fire-and-forget**: ritorna immediatamente dopo aver schedulato il task in background, senza attendere il completamento del nuovo task A2A. Altrimenti il drainer (chiamato dal webhook handler) bloccherebbe la response al webhook per tutto il tempo del nuovo turno (potenzialmente decine di secondi).

Implementazione:

```python
async def spawn_drain_task(self, *, entry, context_id) -> None:
    task_id = str(uuid.uuid4())
    synthetic_message = Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[],
        context_id=context_id,
    )
    # Fire-and-forget: schedule il task in background, ritorna subito.
    asyncio.create_task(
        self._run_drain_task(task_id, context_id, entry, synthetic_message),
        name=f"drain-spawn-{task_id[:8]}",
    )
```

`_run_drain_task` è una closure interna che riusa la stessa pipeline di `_run_agent` ma con il flag `is_drain_spawn=True`.

Il flag `is_drain_spawn=True` (argomento esplicito propagato attraverso `_run_agent` → `_run_agent_impl`) ha questi effetti:
- Indirizza il branch tracer (vedi § 5)
- Indirizza la patch webhook (vedi § 6)
- Diagnostico (log structured `[A2A drain] spawned task_id=X for context=Y`)
- **Riusa il primitivo "resume"**: `_run_agent_impl` chiama `agent.resume_after_deferred()` invece di `agent.execute_query_stream(user_text)`. Il drain-spawn condivide la semantica del deferred-tool-resume: "ripartì il loop sulla history corrente SENZA appendere un nuovo HumanMessage". Questo evita il bug dell'HumanMessage(content="") sintetizzato dal branch `isinstance(query, str)` di `BaseAgent._execute_loop` (base_agent.py:493-494).
- **Skip del tracer span `human.input`**: per coerenza con `is_resume`, niente nuovo span "human.input" (non c'è input utente nuovo).

**Nota sul Message synthetic**: `parts=[]` è la scelta principale. Il Message non viene mai trasformato in `user_text`/`HumanMessage` lato BaseAgent — è solo un veicolo per le metadata del task A2A. Il segnale "questo è un drain-spawn" viaggia esclusivamente come argomento esplicito `is_drain_spawn`.

**Drain delle pending_notifications**: `_run_drain_task` deve drenare `entry.pending_notifications` in `entry.history` PRIMA di chiamare `_run_agent`. Il drain non può essere demandato a `execute()` (che il drain-spawn bypassa per design) né a `_run_agent_impl` (che gira il loop dopo il drain). La logica replicata è la stessa di `execute():334-341` (swap-pattern atomico per evitare race con webhook concorrenti).

### 4.6 Race condition: due notifiche in rapida successione

Scenario: B termina e notifica → drainer spawn task per A → task di A inizia → arriva una seconda notifica da C (un altro remote dispatchato in precedenza).

- Drainer scatta, vede check 1 OK (la nuova notifica è in coda)
- Check 2: c'è un task A2A attivo (quello di A appena partito) → return
- La notifica resta in `pending_notifications`

Risoluzione: quando il task di A inizia il proprio turno, l'executor drena **tutto** ciò che è in `pending_notifications` (codice esistente in `executor.py:323-330`, swap pattern). Quindi entrambe le notifiche vengono mostrate ad A nello stesso turno. Niente notifica persa, niente duplicazione.

### 4.7 Race condition: utente vs drainer

FIFO temporale puro tramite il lock per context_id già esistente (`executor.py` cita "executor's per-context idle gate" alla linea 491). Decisione di brainstorming 2026-05-07: **niente meccanismo di priorità, vince chi arriva prima**.

- Notifica arriva, drainer prende lock, parte spawn → utente manda messaggio → utente aspetta lock → spawn finisce → utente parte
- Utente prende lock prima → drainer aspetta → utente termina turno (drena pending_notifications come già succede oggi) → drainer ricontrolla check 1: vuoto → return

Idempotenza per costruzione.

## 5. Componente 2 — Tracer trace_id reuse

### 5.1 Modifica all'executor

In `src/obelix/adapters/inbound/a2a/server/executor.py`, il blocco `if tracer and not is_resume:` alle linee 372-389 va esteso:

```python
if tracer:
    if is_resume:
        # path esistente: deferred tool resume, trace già attivo
        pass
    elif is_drain_spawn and entry.trace_session is not None:
        # NUOVO path: task spawnato dal drainer
        set_current_trace(entry.trace_session)
        a2a_task_span = await tracer.start_span(
            SpanType.a2a_task,
            name=f"task {task_id[:8]} (drain-spawn)",
            input={"context_id": context_id, "drain_spawn": True},
            metadata={"task_id": task_id, "context_id": context_id, "drain_spawn": True},
        )
        # NON aggiorna entry.trace_session (è già giusto)
    else:
        # path esistente: nuovo task user-triggered
        await tracer.start_trace(...)
        a2a_task_span = await tracer.start_span(SpanType.a2a_task, ...)
        entry.trace_session = get_current_trace()
```

`set_current_trace` è già usato in `src/obelix/adapters/outbound/a2a/webhook.py` (linee 30-31) per cross-context attribution.

### 5.2 Edge case: trace_session vuoto

Se `entry.trace_session is None` (es. tracer disabilitato, o primo turno avvenuto senza tracer attivo), il drainer cade nel branch standard `start_trace`. Niente errore, perdi solo il raggruppamento per quella conversazione. Comportamento accettabile.

### 5.3 Frontend

Nessuna modifica frontend richiesta. `obelix-tracer/frontend/src/utils/spanTree.ts` già supporta più nodi root (`parent_span_id == null`) sotto lo stesso trace_id, restituendo un albero con root multipli. Il rendering li impagina come fratelli sotto il header del trace.

## 6. Componente 3 — Patch temporanea webhook CLI

**Marker**: tutto il codice nuovo della patch va annotato con `# TEMP-PATCH-SPEC-1`. Rimozione futura: `grep -rn TEMP-PATCH-SPEC-1 src/` + cancellazione.

### 6.1 Trasporto del webhook URL dal client al server

Il client CLI già attacca `metadata.client_info` al primo `Message` di una connessione (`cli_client.py:986-988`). Estendiamo lo stesso punto:

```python
# cli_client.py — TEMP-PATCH-SPEC-1
if agent.context_id is None and self._shell_info:
    metadata = {
        "client_info": self._shell_info,
        "client_webhook_url": self._webhook_url,        # NEW
        "client_webhook_token": self._webhook_token,    # NEW (random alla connessione)
    }
```

Server-side, `executor.py` al primo task del context legge `metadata` e salva i due campi sulla `entry`:

```python
# executor.py — TEMP-PATCH-SPEC-1
if metadata := message.metadata:
    if entry.client_webhook_url is None:
        entry.client_webhook_url = metadata.get("client_webhook_url")
        entry.client_webhook_token = metadata.get("client_webhook_token")
```

I due campi vivono su `ContextEntry` (`src/obelix/adapters/inbound/a2a/server/context.py`).

<!-- amended per docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md amendment 2 on 2026-05-07 -->
### 6.2 POST diretta dal drainer

In `executor.spawn_drain_task` (o subito dopo, nel loop di event emission), quando il task spawnato cambia stato, fa:

```python
# executor.py — TEMP-PATCH-SPEC-1
if is_drain_spawn and entry.client_webhook_url:
    try:
        await self._httpx_client.post(
            entry.client_webhook_url,
            json=task.model_dump(mode="json", exclude_none=True),
            headers={"X-A2A-Notification-Token": entry.client_webhook_token or ""},
            timeout=5.0,
        )
    except Exception:
        logger.warning("[A2A drain] webhook POST failed (best-effort)")
```

**Nota sul payload (verificata via spike runtime)**: `Task.model_dump(mode="json", exclude_none=True)` produce JSON **camelCase** (`contextId`, `taskId`, `messageId`, `artifactId`) per via dell'`A2ABaseModel.alias_generator` (`.venv/Lib/site-packages/a2a/_base.py:21-38`). I Fake class nei test DEVONO rispettare camelCase per allinearsi alla realtà del wire format — sbagliarli passa i test ma fallisce il parsing client-side. Esempio della shape per `state=working`:

```json
{"contextId":"...","id":"...","kind":"task","status":{"state":"working","timestamp":"..."}}
```

Per la lista completa delle shape per ogni stato (`completed`, `input-required`, `failed`, ecc.), vedi il research artifact `docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md` § "Per-tech findings → T2 — `a2a.types.Task`".

Best-effort: niente retry, niente queue, niente JWT. Se la CLI è offline, l'utente non vede la risposta — ma può rilanciare un messaggio e l'agent risponde dal context aggiornato.

### 6.3 Lato CLI

`WebhookServer` esistente (`src/obelix/adapters/inbound/a2a/client/webhook_server.py`) riceve i POST e aggiorna il `TaskTracker` come fa già oggi. Nessuna modifica strutturale al webhook server. Solo verifica del token in arrivo: confronto stretto con `self._webhook_token`. Se non combacia, 401.

### 6.4 Cleanup futuro (spec 2)

Quando la spec 2 (CLI streaming SSE) viene implementata, vanno rimossi:
- Tutto il codice marcato `# TEMP-PATCH-SPEC-1`
- I due campi `entry.client_webhook_url`/`token` su `ContextEntry`
- I metadati `client_webhook_url`/`token` sul Message del client
- Il `WebhookServer` (intera classe, file `webhook_server.py`)
- I flag/env `webhook_host`, `OBELIX_WEBHOOK_HOST`, `OBELIX_WEBHOOK_PORT`

## 7. Out of scope esplicito

Vedi anche memoria `project_a2a_spec1_exclusions.md`.

| Tema | Stato | Dove sarà affrontato |
|---|---|---|
| CLI streaming SSE migration | Out | Spec 2 |
| Peer discovery A→B (CLI vede B) | Out | Spec 3 |
| Peer UI nella status bar | Out | Spec 4 |
| Revisione fire-and-forget `dispatch_agent` | Out | Spec 5 |
| Modifiche frontend tracer | Out | Non necessario |
| JWT/signing/retry sul push del drainer | Out | Patch è temporanea |
| Re-enable `BasePushNotificationSender` SDK lato server | Out | Bypassiamo direttamente |
| Modifiche a `DispatchAgentTool` | Out | Tool resta com'è |

Durante l'implementazione e il code review, qualunque tentativo di "sistemare anche X" che cade in queste righe va respinto con riferimento a questo documento.

## 8. Testing strategy

**IRON RULE non derogabile** (memoria `feedback_test_iron_rule.md`):

1. **Integration test**, non unit con tutto mockato.
2. **Niente mock di SDK esterni** (a2a-sdk, anthropic, openai, ecc.) — mai `unittest.mock.MagicMock`, mai `pytest-mock`, mai `monkeypatch` su client.
3. **Se serve sostituire una dipendenza, è una FAKE CLASS scritta a mano** che implementa il `Protocol` esatto e restituisce gli stessi output reali della dipendenza.
4. **Pre-implementation research è OBBLIGATORIA** per documentare i contratti delle dipendenze esterne **prima** di scrivere i test.

### 8.1 Pre-implementation research (vincolante)

Prima di scrivere qualunque test, va prodotto un report che documenta:

- Per `a2a-sdk` `Client`: tipi di ritorno reali di `send_message` (AsyncIterator[ClientEvent | Message]), `get_task` (Task), `cancel_task` (Task). File `:linea` di riferimento nel `.venv/Lib/site-packages/a2a/`.
- Per `Task` model: shape esatta del `model_dump(mode="json", exclude_none=True)` quando state=working, completed, failed, input-required.
- Per `EventQueue` server-side: tipo degli eventi che l'executor enqueua (`TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`).
- Per il tracer Obelix: API esatta di `start_trace`, `start_span`, `set_current_trace`, `get_current_trace`, e cosa restituiscono.
- Per `httpx.AsyncClient.post`: forma dell'eccezione `HTTPStatusError`, comportamento di `timeout=5.0`.

Il report va in `docs/superpowers/research/2026-05-07-a2a-server-drainer-research.md` o equivalente, e va citato file:linea nei test.

### 8.2 Test del drainer (`tests/.../test_drainer.py`)

Scenari:
- queue vuota + nessun task attivo → no-op
- queue vuota + task attivo → no-op
- queue piena + task attivo → no-op
- queue piena + nessun task attivo → spawn (1 chiamata a `executor.spawn_drain_task`)
- chiamato due volte di fila in stessa condizione (queue piena + nessun task) → spawn UNA volta perché il primo ha cambiato lo state

Fake class:
- `FakeContextEntry`: implementa `pending_notifications: list[HumanMessage]` e `has_active_task() -> bool`
- `FakeExecutor`: implementa `spawn_drain_task` che incrementa un contatore

Niente mock SDK, niente HTTP.

### 8.3 Integration test executor + drainer (`tests/.../test_executor_drain_spawn.py`)

Scenario completo end-to-end con server reali:
- FastAPI test client che ospita il server A2A reale (con executor reale)
- Provider LLM Fake che restituisce risposte fisse (Fake già esistente nei test del progetto, da riusare se conforme)
- Webhook receiver fixture (Starlette TestClient) per ricevere la POST della patch

Verifiche:
- Dopo handle_remote_update, viene spawnato un nuovo task A2A interno entro N millisecondi
- Il nuovo task ha `trace_id` identico al primo
- L'agent vede `<remote_task_update>` come ultimo messaggio nella history
- Il webhook fixture riceve la POST con il task serializzato e header `X-A2A-Notification-Token` corretto
- Token errato → 401 lato webhook fixture (non aggiornamento del task)

### 8.4 Integration test tracer trace_id reuse

Scenario:
- Primo task user-triggered → `trace_session_1`
- Drainer spawna secondo task → deve avere stesso `trace_id`
- Span del secondo task: `parent_span_id == null`, `metadata.drain_spawn == True`
- `tracer.list_traces()` ritorna 1 trace, non 2

### 8.5 Integration test patch webhook (con marker)

Test grep-based: `tests/.../test_temp_patch_marker.py` esegue `grep -rn TEMP-PATCH-SPEC-1 src/` e verifica che il numero di occorrenze sia uguale a un valore atteso (es. 6). Test che fallirà se qualcuno aggiunge/rimuove patch senza aggiornare il counter — utile per non perdere il tracciamento durante la rimozione futura.

### 8.6 E2E manuale

Riproduzione del bug originale:
- Avvia `examples/orchestrator_server.py` su :8005 e `examples/dev_workflow_server.py` su :8001
- Avvia CLI connessa a entrambi
- Manda "ho bisogno di un check sugli ultimi commit"
- Attendi senza fare nulla
- Verifica: l'orchestrator risponde spontaneamente con il risultato del coordinator entro N secondi
- Apri il tracer frontend: un solo tree per la conversazione, due `a2a_task` come root

### 8.7 Briefing per sub-agent implementatori

Quando il plan invocherà sub-agent per scrivere i test, ogni prompt deve includere esplicitamente:

- Riferimento alla iron rule `feedback_test_iron_rule.md`
- Riferimento al pre-implementation research report con file:linea
- "Se durante l'implementazione scopri che il contratto è diverso da quanto documentato, ferma e segnala — NON inventare"
- Per ogni dipendenza esterna usata: il path della Fake class da scrivere e gli output reali documentati

## 9. Removal plan (per spec 2)

Quando la spec 2 (CLI streaming SSE) arriva, il cleanup è meccanico:

1. `grep -rn TEMP-PATCH-SPEC-1 src/` → cancella tutte le occorrenze (codice + commenti)
2. Rimuovi i campi `client_webhook_url`/`client_webhook_token` da `ContextEntry`
3. Rimuovi i metadati `client_webhook_url`/`client_webhook_token` dal lato CLI
4. Rimuovi `WebhookServer` (file `webhook_server.py`)
5. Rimuovi env vars e flag `webhook_host`, `OBELIX_WEBHOOK_HOST`, `OBELIX_WEBHOOK_PORT`
6. Aggiorna `test_temp_patch_marker.py` per atteso=0 e poi cancellalo

I componenti **drainer** e **tracer trace_id reuse** restano: non sono temporanei, sono il cuore della spec 1.

## 10. Success criteria

La spec è considerata completa quando:

- Test integration di `test_executor_drain_spawn.py` passano e coprono i 5 scenari del § 8.2 + lo scenario del § 8.3
- Test grep `test_temp_patch_marker.py` passa con counter atteso correto
- Test manuale § 8.6 mostra l'orchestrator rispondere spontaneamente senza input utente, con un solo tree nel tracer
- Niente regressione su test esistenti in `tests/.../a2a/`
- Pre-implementation research report è committato e citato dai test

## Appendice A — File toccati

```
NEW: src/obelix/adapters/inbound/a2a/server/drainer.py
EDIT: src/obelix/adapters/inbound/a2a/server/executor.py
  - aggiungere is_drain_spawn parameter, branch tracer trace_id reuse
  - aggiungere spawn_drain_task method
  - aggiungere read di metadata.client_webhook_url/token + save su entry
  - aggiungere POST patch al webhook CLI per task spawn
<!-- amended per docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md amendment 3 on 2026-05-07 -->
EDIT: src/obelix/adapters/inbound/a2a/server/context.py
  - aggiungere campi client_webhook_url, client_webhook_token su ContextEntry (aggiungere a __slots__ e __init__)
  - NESSUN nuovo metodo: il check "context busy" usa entry.idle.is_set() già esistente
EDIT: src/obelix/adapters/outbound/a2a/webhook.py
  - chiamata a `await maybe_spawn_drain_task(...)` nel webhook_handler async, dopo handle_remote_update sync
EDIT: src/obelix/adapters/outbound/a2a/polling.py
  - chiamata a `await maybe_spawn_drain_task(...)` in _poll_one async, dopo handle_remote_update sync
NOTE: src/obelix/adapters/outbound/a2a/handler.py NON viene modificato — handle_remote_update resta sync
EDIT: src/obelix/adapters/inbound/a2a/client/cli_client.py
  - aggiunta di client_webhook_url/token in metadata del primo Message
  - generazione random del token al boot
EDIT: src/obelix/adapters/inbound/a2a/client/webhook_server.py
  - verifica del token su POST in entrata, 401 se mismatch

NEW: tests/inbound/a2a/server/test_drainer.py
NEW: tests/inbound/a2a/server/test_executor_drain_spawn.py
NEW: tests/inbound/a2a/server/test_tracer_trace_reuse.py
NEW: tests/inbound/a2a/server/test_temp_patch_marker.py
NEW: docs/superpowers/research/2026-05-07-a2a-server-drainer-research.md
```

Stima: ~600-800 righe nuove, ~200 righe modificate, ~400 righe di test.

## Appendice B — Note implementative

- Il `Httpx.AsyncClient` per la POST della patch va riusato da quello già istanziato dall'executor (probabilmente in `AgentFactory.a2a_serve` o nel webhook outbound). Niente nuovo client.
- Il flag `is_drain_spawn` si propaga come argomento esplicito attraverso `_run_agent` → `_run_agent_impl`. Niente var globali, niente contextvars.
- Il logger structured va con prefisso `[A2A drain]` (in linea con `[A2A polling]`, `[A2A dispatch]` esistenti).
