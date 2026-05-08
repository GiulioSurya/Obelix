# A2A CLI polling-only — Spec 2 design

**Created**: 2026-05-08
**Owner**: f.crino@ads.it
**Status**: design (pending pre-implementation research)
**Roadmap**: `docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md` § Spec 2 — C
**Predecessor**: `docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md` (spec 1)

---

## 1. Problema

La CLI A2A (`src/obelix/adapters/inbound/a2a/client/cli_client.py`) oggi
combina tre canali per ricevere updates dal server:

1. **Push notifications** (server → webhook locale CLI). Reverse-connection.
   Spec 1 ha aggiunto auth token e bug fix; resta la patch
   `TEMP-PATCH-SPEC-1` (13 occorrenze in `src/`).
2. **Polling fallback** (`tasks/get` ogni 1s su task non-terminali).
3. **No streaming SSE** (la CLI usa `streaming=False` in `ClientConfig`).

Il modello reverse-connection (server → client webhook) impedisce deployment
con server in container/NAT/cloud remoti dove il client non è raggiungibile
dall'esterno (limite #4 della roadmap). Inoltre la patch spec 1 ha
introdotto codice transitorio (`_DrainSpawnEventQueue`, slot
`client_webhook_url/token` su `ContextEntry`, validazione token su
`WebhookServer`, `OBELIX_WEBHOOK_HOST/PORT` env vars) marcato
`TEMP-PATCH-SPEC-1` con un test di counting (`test_temp_patch_marker.py`,
`EXPECTED_COUNT=13`).

Spec 2 elimina entrambi: niente push, niente patch. Il polling — già
robusto e implementato come fallback — diventa l'**unico canale primario**.

### Perché non SSE

Considerata e scartata. SSE darebbe latenza ~0 e supporto a
`TaskArtifactUpdateEvent` con `append=True` (artifact che crescono in
tempo reale, p.es. token streaming live). Ma:

- La CLI **già oggi non visualizza token streaming**: `_collect_task` consuma
  l'iteratore e mostra solo l'output finale. SSE aggiungerebbe complessità
  (long-lived connection, reconnect via `resubscribe`, gestione disconnect)
  per una feature non utilizzata.
- Polling è strettamente più semplice: stateless short request, niente
  reconnect, NAT-friendly identico (outbound-only).
- A2A spec 3.5.1 elenca polling come canale ufficiale, non degraded.
- Pattern Claude Code (`RemoteAgentTask.tsx:564`): polling con
  `pollRemoteSessionEvents(sessionId, lastEventId)` ogni 1s + queue locale di
  notifiche è il canale primary per remote agent task. WebSocket esiste solo
  per la sessione UI primaria, non per i delegati.

Se in futuro vorremo token streaming live, aggiungere SSE sarà estensione
non-breaking.

---

## 2. Architettura

**Cambio configurazione client A2A**:

```python
# Prima (spec 1)
ClientConfig(streaming=False, polling=True,
             push_notification_configs=[PushNotificationConfig(url=..., token=...)])

# Dopo (spec 2)
ClientConfig(streaming=False, polling=True,
             push_notification_configs=[])
```

**Discovery di task figli via metadata**:

Il server arricchisce il `Task` con due metadata distinti:

```python
T1.metadata = {
    "spawned_task_ids": ["task-uuid-3"],            # drain-spawn (delivery)
    "dispatched_peers": [
        {"name": "coordinator", "task_id": "task-uuid-2", "state": "working"},
    ],
}
```

- **`spawned_task_ids`**: task figli **drain-spawn**. Sono il path di delivery
  della CLI per le risposte async. Discovery: la CLI polla `T1`, legge
  `spawned_task_ids`, apre un loop polling per ogni nuovo `task_id` non già
  tracciato.
- **`dispatched_peers`**: peer dispatcciati dal tool `dispatch_agent`.
  Servono solo a UX (status bar mostra "coordinator: working" quando il
  peer sta lavorando). La CLI **non polla** i task dei peer — il loro stato
  arriva via push interno A↔B che aggiorna `state` sul metadata di T1.

**Cadenza polling**: 0.5s fisso. Niente backoff, niente adaptive — KISS.

**Niente cancel cascade**: ESC cancella solo `agent.last_task_id`. I T_n
drain-spawn e i dispatched_peers proseguono finché non finiscono
naturalmente. Pattern preso da Claude Code (`useCancelRequest.ts`):
ESC è scope-locale, kill-all è gesto deliberato separato (in spec 2 non
implementiamo kill-all — KISS).

---

## 3. Data flow e metadata schema

### Schema metadata su Task

| Campo | Tipo | Scope | Mutabilità |
|---|---|---|---|
| `spawned_task_ids` | `list[str]` | Solo su Task root del context (T1). Append-only durante la vita del context. | Append da drainer alla creazione di ogni drain-spawn task. |
| `dispatched_peers[*].name` | `str` | T1 (e ogni T_n che a sua volta dispatcha). | Set una volta al dispatch, immutabile. |
| `dispatched_peers[*].task_id` | `str` | idem | idem |
| `dispatched_peers[*].state` | `str` (`working`/`completed`/`failed`/`canceled`/`rejected`) | idem | Update da push handler A↔B. |

### Chi scrive cosa, quando

| Operazione | Componente server | Trigger |
|---|---|---|
| Append a `spawned_task_ids` | `executor.maybe_spawn_drain_task` (drainer) | Subito prima di `asyncio.create_task(spawn_drain_task)`. |
| Append a `dispatched_peers` | Tool `dispatch_agent` | Subito dopo `client.send_message` verso il peer. State iniziale `working`. |
| Update di `dispatched_peers[*].state` | Push handler A↔B (esistente) | Ogni notifica push da peer: cambio stato. |

### Vincolo SDK aperto (open assumption #1)

A2A SDK non documenta come modificare `Task.metadata` post-creation. Due
candidati di implementazione, da verificare in pre-impl research:

1. **In-place TaskStore patch**: il drainer recupera il Task dallo store
   SDK, modifica `metadata`, lo riscrive. Il prossimo `tasks/get(T1)` legge
   il Task aggiornato. Pro: semplice, immediato. Contro: bypassa il
   meccanismo eventi (potenziale race con altri reader).

2. **TaskStatusUpdateEvent metadata-only**: il drainer emette un evento
   `TaskStatusUpdateEvent(task_id=T1, status=<unchanged>, metadata={...},
   final=False)` sulla EventQueue di T1. `ClientTaskManager.save_task_event`
   merge-a `event.metadata` su `task.metadata` (vedi
   `client_task_manager.py:125-128`). Pro: SDK-canonical. Contro: **se T1 è
   già terminal la EventQueue è chiusa**, l'evento non si propaga. Da
   verificare se l'SDK lo accetta o solleva.

Decisione: si parte preferendo l'opzione 2 (SDK-canonical). Se la
pre-impl research dimostra che opzione 2 fallisce dopo task terminale, si
ripiega su opzione 1.

### Flow CLI

```
1. CLIClient._send_message(text)
   ├── client.send_message(msg) → ritorna T1
   ├── tracker.register(T1.id, agent.name)
   └── _start_polling(T1.id, agent.name)

2. _start_polling(task_id, agent_name) [worker background]
   while not stopped:
       task = await client.get_task(TaskQueryParams(id=task_id))
       tracker.update(task.model_dump(mode="json", exclude_none=True))
       # Discover spawned children
       for child_id in task.metadata.get("spawned_task_ids", []):
           if child_id not in self._tracked_tasks:
               self._tracked_tasks.add(child_id)
               self._start_polling(child_id, agent_name)
       # Update peers (status bar)
       tracker.update_peers(task_id, task.metadata.get("dispatched_peers", []))
       # Exit condition
       if (task.status.state in TERMINAL
           and all_children_terminal(task_id)
           and all_peers_terminal(task_id)):
           break
       await asyncio.sleep(0.5)

3. Status bar render
   - Per ogni agent connesso, segmento del suo last_task_id (come oggi).
   - + segmento "<peer.name>: <peer.state>" per ogni peer non-terminale.
   - Esempio: "O ✓ | coordinator: working" → "O ✓ | coordinator ✓".
```

### Cancellazione

ESC → `cancel_task(agent.last_task_id)` solo. I poll loop dei T_n figli
continuano (e termineranno naturalmente quando i task figli arrivano a
terminal state, eventualmente cancellandosi a loro volta in cascata se il
server propaga la cancellazione di T1 internamente — fuori scope di spec 2,
sta al server scegliere). Niente two-press kill-all in spec 2.

---

## 4. Lifecycle e edge cases

**Quando un poll loop esce**:

- `task.status.state in {completed, failed, canceled, rejected}` E
- nessun child in `self._tracked_tasks` con loop ancora attivo per ramo
  discendente E
- nessun `dispatched_peer.state` non-terminale.

L'ultimo punto evita di perdere drain-spawn tardivi: T1 può essere
`completed` da molti secondi quando finalmente arriva la risposta async.

**Errori transienti** durante `tasks/get`:

- 5xx, timeout, network error: continua il loop al prossimo ciclo.
- 4xx: stop di quel task, log dell'errore in chat (non più con prefisso
  `[poll]` — è il canale primario, non un fallback).
- JSON-RPC `-32001 Task not found`: non dovrebbe più accadere (i drain-spawn
  task in spec 2 sono task A2A regolari nello SDK store). Se accade è un
  bug genuino — log e stop di quel task. **Open assumption #2**: la
  pre-impl research deve verificare che il drainer registri T3 nel
  TaskStore SDK PRIMA di scrivere `T3.id` in `T1.metadata.spawned_task_ids`,
  per evitare race fra annuncio del task e disponibilità via `tasks/get`.

**Side-effect UI**:

- `_handle_pending_input` (deferred tool) resta attivo: quando il poll vede
  `state=input-required` su un task currently-focused, mostra il pannello
  come oggi.
- `_show_result` resta attivo: quando il poll vede `state=completed` con
  artifact, mostra il panel.
- Nessun cambiamento all'esperienza utente per deferred / display.

**Cleanup all'unmount**:

- Tutti i poll loop attivi vengono cancellati (`task.cancel()` su ognuno).
- Niente `WebhookServer.stop()` (eliminato).

---

## 5. Cleanup codice (rimozione spec 1 patch)

### File interi eliminati

- `src/obelix/adapters/inbound/a2a/client/webhook_server.py` (tutto il file).
- Test `tests/adapters/inbound/a2a/client/test_webhook_server_token.py`.
- Test `tests/core/agent/test_a2a_serve_webhook_url.py` (rewriting wildcard
  non più necessario per webhook URL CLI).
- Test `tests/adapters/inbound/a2a/client/test_cli_webhook_metadata.py`.
- Test `tests/adapters/inbound/a2a/client/test_task_tracker_unknown_agent.py`.

### File rinominato

- `webhook_server.py` → contenuto di `TaskTracker` migrato in
  `task_tracker.py` (nome onesto). `TaskInfo` resta nello stesso modulo.

### Modifiche per file

**`src/obelix/adapters/inbound/a2a/client/cli_client.py`**

Rimossi:
- Attributi: `_webhook_server`, `_webhook_url`, `_webhook_host`,
  `_webhook_token`.
- CLI arg `--webhook-host` + relativa logica `from_cli`.
- Import di `secrets`, `PushNotificationConfig`.
- `push_configs = [PushNotificationConfig(...)]` da `ClientConfig`.
- Block `metadata["client_webhook_url"]/[token]` in `_send_message`
  (linee ~1031-1044).
- `_resolve_agent_by_context` (callback per `context_resolver`).
- `_polling_fallback` rinominato in `_poll_task` (canonical, non fallback).
- `_poll_warned: set` (niente più "recovered via polling" warning).
- Prefisso `[poll]` da messaggi di errore.

Aggiunti:
- `_start_polling(task_id: str, agent_name: str)`: dispatcher che apre il
  worker `_poll_task` e tiene bookkeeping in `self._tracked_tasks: set[str]`.
- Loop ricorsivo che scopre `spawned_task_ids` e attiva `_start_polling`
  ricorsivo per ogni nuovo child.
- `tracker.update_peers(parent_task_id, peers)`: aggiorna struttura interna
  della status bar.

Modificati:
- `set_interval(0.15, self._poll_tasks)` resta come tick UI; ma le decisioni
  di poll si basano su 0.5s, non 1.0s.
- `_handle_pending_input`, `_show_result`: invariati.

**`src/obelix/adapters/inbound/a2a/client/task_tracker.py`** (nuovo, da
`webhook_server.py`)

Modifiche al `TaskTracker`:
- Rimosso `context_resolver` callback (era per attribuire push
  drain-spawn → ora drain-spawn discovery via metadata polling).
- Rimossa logica "unknown task_id from push" (linee 73-94 di
  `webhook_server.py`).
- Aggiunto `update_peers(parent_task_id: str, peers: list[dict])`:
  store interno `self._peers_by_parent: dict[str, list[PeerInfo]]`.
- API readers per status bar: `get_active_peers(agent_name) → list[PeerInfo]`.

**`src/obelix/adapters/inbound/a2a/server/context.py`**

Rimossi:
- Slot `client_webhook_url`, `client_webhook_token`.
- Init di entrambi nel `__init__`.

**`src/obelix/adapters/inbound/a2a/server/executor.py`**

Rimossi:
- Parametro `httpx_client` da `__init__` (era usato solo dal
  `_DrainSpawnEventQueue`).
- Slot `_httpx_client`.
- Metodo `_apply_webhook_metadata_patch` + chiamata in `execute`.
- Classi `_DrainSpawnEventQueue`, `_NullEventQueue`.
- Factory `_make_drain_spawn_event_queue` (selettore basato su
  `client_webhook_url`).

Modificati:
- `_run_drain_task`: usa la **EventQueue normale dell'SDK** (drain-spawn task
  è ora un task A2A regolare, registrato nel TaskStore SDK).
- `maybe_spawn_drain_task`: prima di `asyncio.create_task(spawn_drain_task)`,
  append `T3.id` a `T1.metadata.spawned_task_ids` (meccanismo da
  pre-impl research).
- Push handler A↔B (codice esistente che riceve push da peer): aggiunto
  update di `T_parent.metadata.dispatched_peers[*].state` quando arriva
  notifica.

Aggiunti:
- Logica nel tool `dispatch_agent` (codice esistente in
  `adapters/outbound/a2a/...`): subito dopo `send_message` verso peer,
  append a `T_parent.metadata.dispatched_peers`.

**`src/obelix/core/agent/agent_factory.py`**

Rimossi:
- `_resolve_webhook_host`, `_WILDCARD_BIND_HOSTS` (servivano solo per
  webhook URL CLI).
- Passaggio `httpx_client` a `ObelixAgentExecutor`.
- Eventuali param doc su webhook config.

### Conteggio TEMP-PATCH-SPEC-1

A fine spec 2: **0 occorrenze in `src/`**. Eliminato anche
`test_temp_patch_marker.py` (più nessun marker da contare).

---

## 6. Testing strategy (iron rule)

**Iron rule confermata**: integration test only. **No mock di SDK
esterni** (`unittest.mock`, `pytest-mock`, `monkeypatch` su SDK A2A sono
proibiti). Hand-written Fake class che implementano i Protocol contracts
reali del SDK.

### Scenarios coperti

1. **Polling discovery di drain-spawn** (regression del path async di
   spec 1)
   - `FakeA2AServer` finge un server O che: emette T1 completed, dopo 200ms
     appende `spawned_task_ids=["T3"]` al metadata di T1, registra T3 nel
     proprio TaskStore con un artifact "risposta async".
   - CLI polla T1 → vede metadata cresciuto → scopre T3 → polla T3 → vede
     artifact → mostra panel.
   - Asserzione outcome: `tracker.get(T3).is_terminal and panel mostrato`.

2. **`dispatched_peers` visualization in status bar**
   - Server scrive `dispatched_peers=[{"name":"C","task_id":"t2",
     "state":"working"}]` in T1.metadata.
   - Polling → `tracker.update_peers()` → status bar contiene segmento
     "C: working".
   - Server aggiorna `state=completed` → polling → status bar segmento
     "C ✓".

3. **Recursive drain-spawn discovery** (T3 spawna T4)
   - `T1.spawned_task_ids=[T3]`, `T3.spawned_task_ids=[T4]`. CLI deve
     scoprire T4 via polling ricorsivo del loop di T3.

4. **Polling termination correctness**
   - Loop di T1 esce **solo** quando T1 terminale + tutti spawned terminali +
     tutti peer terminali.
   - Test: T1 completed ma T3 ancora working → loop di T1 continua.
     T3 completed → loop di T1 esce.

5. **Cancellazione no-cascade**
   - Utente preme ESC. Solo `cancel_task(T1)` chiamato. T3 e T2 NON
     vengono cancellati. I rispettivi loop polling proseguono.

6. **TEMP-PATCH-SPEC-1 marker count**
   - Test esistente: aggiornato a `EXPECTED_COUNT=0`, poi eliminato.

7. **Server side: drainer scrive metadata correttamente**
   - Drain handler riceve push da peer, spawna T3 → verifica che
     `T1.metadata.spawned_task_ids` contenga `T3.id` E che `T3` sia
     recuperabile via `tasks/get(T3)` (registrato nello SDK store, no -32001).

### Fake da scrivere

- **`FakeA2AServer`**: ASGI mini-app che implementa `tasks/get`,
  `tasks/cancel`, `message/send` con uno store in-memory di Task.
  Configurabile con scenarios (sequenze pre-programmate di update).
- **`FakeAgent`**: BaseAgent subclass che simula dispatch verso peer +
  trigger drainer.
- **`FakeRichLog`**: output di Textual cattura di assertion outcome.

### Tools NON usabili nei test

- `unittest.mock`, `pytest-mock`, `monkeypatch`, `MagicMock` su:
  - Qualsiasi tipo importato da `a2a.client` o `a2a.server` o `a2a.types`.
  - `httpx.AsyncClient` reale (usare ASGI transport contro `FakeA2AServer`).
  - `asyncio.sleep` non va monkeypatchato per "fast-forward" del polling:
    nei test usare `asyncio.Event` o helper di sincronizzazione esplicita
    fra produzione delle update lato `FakeA2AServer` e consumo lato CLI.

---

## 7. Out of scope

- **Token streaming live** (SSE). Considerato e scartato. Estensione futura
  non-breaking se servirà.
- **Peer discovery completa** (CLI apre client A2A diretto verso un peer
  scoperto, p.es. per chat con esso). Coperta in spec 3 — `dispatched_peers`
  in spec 2 è solo visibility/status, non interaction.
- **Webhook A↔B** (push interno tra agent server). Resta com'è. Razionale:
  i server sono outbound-capable nel deployment standard, non hanno il NAT
  problem del CLI; sostituire con polling sarebbe quadratico (N task in
  flight × M peer). Spec 5 ("fire-and-forget revisited") può rivisitarlo.
- **Two-press ESC kill-all del context**. Pattern preso da Claude Code,
  rimandato — KISS.
- **Cadenza polling adaptive / backoff**. KISS, 0.5s fisso.
- **Modifiche al protocollo A2A o alla agent card capabilities**. Server
  continua a dichiarare `streaming=True, push_notifications=True` — la
  CLI semplicemente non li usa.

---

## 8. Open assumptions (per pre-implementation research)

Da risolvere in `docs/superpowers/research/2026-05-08-a2a-cli-polling-spec2-design.md`
prima del piano:

1. **Mutabilità di `Task.metadata` post-creation nell'a2a-sdk**.
   - Verificare se `TaskStatusUpdateEvent(metadata={...}, final=False)`
     emesso DOPO che il Task è in stato terminal (state=completed) è
     accettato dal server SDK e propagato correttamente al
     `ClientTaskManager` lato client.
   - Se sì → opzione 2 (SDK-canonical).
   - Se no → opzione 1 (in-place TaskStore patch). Verificare API SDK per
     update di Task nello store.
2. **Race fra append a `spawned_task_ids` e registrazione del child task
   nel TaskStore**. Il drainer deve garantire ordine: prima registra T3 nel
   TaskStore SDK (così `tasks/get(T3)` ritorna 200), POI append in
   `T1.metadata.spawned_task_ids`. Verificare API SDK per task
   pre-registration.
3. **Come `dispatched_peers` viene aggiornato in metadata**: stessa
   questione di #1 ma per update di un campo esistente (non solo append).
   Probabile stessa soluzione.
4. **Comportamento di `client.get_task` su task in stato terminal**.
   Risponde sempre con il Task corrente (compreso `metadata` aggiornato),
   o l'SDK lo cache? Se cache, eviction policy?
5. **Cancellazione di un task con `dispatched_peers` attivi**. Quando
   `cancel_task(T1)` arriva al server O, cosa succede ai T2 (peer) e ai T3
   (drain-spawn) attivi server-side? Probabile: nulla automatico (è il caso
   "no cascade" che vogliamo). Verificare per essere sicuri.
6. **Conformità al ciclo di vita del Task SDK**. Il drain-spawn task T3 è
   un task "child" con un suo lifecycle indipendente, ma condivide
   `context_id` con T1 (per riuso trace_id, già da spec 1). Verificare che
   l'SDK accetti più Task con stesso `context_id` senza errori.

Tier subagent dispatch: **T2** (spike-runner) per #1, #2, #4 — comportamento
SDK richiede esecuzione contro istanza reale. **T1** (doc-verifier) per #5
e #6 (semantica documentata, non comportamento runtime). **T0** se
applicabile (dipende da quanto è già verificato nel codebase corrente).

---

## 9. Success criteria

Spec 2 si considera implementata quando:

- ✅ TEMP-PATCH-SPEC-1 count in `src/`: **0**.
- ✅ `WebhookServer` rimosso (file eliminato).
- ✅ Test suite full passa con la nuova polling-only architecture.
- ✅ Smoke test e2e: orchestrator dispatcha coordinator, coordinator finisce
  in 5s, CLI mostra la risposta sintetizzata di orchestrator entro
  ~1s dal completamento (1s = 0.5s poll + roundtrip). Niente "unknown:
  1 new" nella status bar, niente "completed (no content)", niente JSON-RPC
  -32001 silenziato.
- ✅ Smoke test status bar: durante il dispatch, la status bar mostra
  "O is thinking | coordinator: working" e poi "O ✓ | coordinator ✓".
- ✅ Smoke test cancel no-cascade: ESC su T1 mentre coordinator sta lavorando
  → T1 canceled, coordinator continua a girare lato server fino a
  completion naturale.
- ✅ Deployment in container/NAT possibile: il client CLI funziona dietro
  NAT (verificabile testando contro server in Docker con port mapping
  asimmetrico).

---

## 10. File list

### Eliminati

- `src/obelix/adapters/inbound/a2a/client/webhook_server.py`
- `tests/adapters/inbound/a2a/client/test_webhook_server_token.py`
- `tests/core/agent/test_a2a_serve_webhook_url.py`
- `tests/adapters/inbound/a2a/client/test_cli_webhook_metadata.py`
- `tests/adapters/inbound/a2a/client/test_task_tracker_unknown_agent.py`
- `tests/adapters/inbound/a2a/server/test_temp_patch_marker.py` (a fine spec 2)

### Creati

- `src/obelix/adapters/inbound/a2a/client/task_tracker.py` (estratto da
  `webhook_server.py`, ripulito).
- `tests/adapters/inbound/a2a/client/test_polling_discovery.py`
- `tests/adapters/inbound/a2a/client/test_polling_termination.py`
- `tests/adapters/inbound/a2a/client/test_dispatched_peers_status_bar.py`
- `tests/adapters/inbound/a2a/server/test_drainer_metadata_append.py`
- `tests/_fakes/fake_a2a_server.py` (FakeA2AServer + FakeAgent + helpers).

### Modificati (cleanup + nuova logica)

- `src/obelix/adapters/inbound/a2a/client/cli_client.py`
- `src/obelix/adapters/inbound/a2a/server/executor.py`
- `src/obelix/adapters/inbound/a2a/server/context.py`
- `src/obelix/core/agent/agent_factory.py`
- `src/obelix/adapters/outbound/a2a/tools/dispatch.py` (tool
  `dispatch_agent`: scrive `dispatched_peers` su T1.metadata).
- Test esistenti `test_drain_spawn_*` rivisti per nuovo flow.

---

## 11. Roadmap update post-implementation

A merge avvenuto:

- Aggiornare `docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md` §
  Spec 2 → status `implemented`.
- Memory entry: nuovo `project_a2a_spec2_polling.md` con file map +
  invariants. Aggiornare `MEMORY.md` index.
- `project_a2a_temporary_webhook_patch.md`: marker count → 0, status
  → resolved.
