# Roadmap — A2A async agents (multi-spec)

**Created**: 2026-05-07
**Owner**: f.crino@ads.it
**Status**: spec 1 implemented (2026-05-08), 6 smoke bugs fixed end-to-end

## Background

Oggi la CLI A2A (`src/obelix/adapters/inbound/a2a/client/cli_client.py`) e il
sistema outbound A2A degli agenti hanno una serie di limiti strutturali
nel modo in cui gestiscono il propagamento degli **status update** tra
server e client, e tra agent peer (server A2A che si dispatcha task tra
loro tramite tool come `dispatch_agent`). I limiti sono:

1. Il server A2A ha `AgentCapabilities.push_notifications=False` (commit
   30c6415) → il sender push è no-op. La CLI compensa via `_polling_fallback`,
   ma il webhook locale è codice morto.
2. Quando un agent A dispatcha un task verso B (peer A2A), il client CLI
   non lo vede mai: lo stream/polling/push del client sono legati al
   task originale di A. B è invisibile.
3. Quando l'orchestrator dispatcha un remote_task e il proprio task
   termina con `completed`, **nessuno lo risveglia** quando il remote
   risponde. La notifica `<remote_task_update>` resta in
   `entry.pending_notifications` finché l'utente non manda manualmente
   un nuovo messaggio.
4. Il modello reverse-connection del webhook (server → client) impedisce
   deployment con server in container/NAT/cloud remoti, dove il client
   non è raggiungibile dall'esterno.
5. Ogni nuovo task A2A apre un nuovo trace nel tracer
   (`executor.py:373` chiama `start_trace`). Una catena
   user→A→B→A→risposta produce 3+ tree separati che frammentano la
   storia visibile nel frontend.

Questi problemi sono interconnessi ma non equivalenti. Vanno affrontati
in più fasi separate, ognuna con la propria spec → plan → implementazione.

## Decomposizione in 5 spec

### Spec 1 — A+B: Server-side drainer + tracer trace_id reuse

**Status**: implemented (2026-05-08), smoke-tested end-to-end
**Spec file**: `docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md`
**Plan file**: `docs/superpowers/plans/2026-05-07-a2a-server-drainer.md`
**Research artifact**: `docs/superpowers/research/2026-05-07-a2a-server-drainer-design.md`

**Smoke fixes applied 2026-05-08** (6 bugs found in real e2e run):
- Bug 1: CLI must include `token` in `PushNotificationConfig` (else server push 401)
- Bug 2: drain-spawn must drain `pending_notifications` + use `resume_after_deferred()` (else Anthropic 400 on empty HumanMessage)
- Bug 3: `agent_factory.a2a_serve` rewrites wildcard host (`0.0.0.0`/`::`) to `127.0.0.1` for the auto-built webhook URL (else `connection refused` for local peers)
- Bug 4: `TaskTracker(context_resolver=...)` attributes server-spawned tasks to the originating agent based on payload `contextId` (else `unknown: 1 new` in status bar)
- Bug 5: `_DrainSpawnEventQueue` accumulates artifacts via `_merge_artifact_update`, removes false-positive dedup-by-state (else CLI shows "completed (no content)")
- Bug 6: `_polling_fallback` suppresses JSON-RPC `-32001 Task not found` for drain-spawn task_ids (these are not registered in the SDK store; expected silent failure mode)
TEMP-PATCH-SPEC-1 marker count: 13 (was 12 before fix Bug 1).

**Cosa risolve**: limiti #3 e #5.

**Scope**:
- Drainer auto-trigger lato server: quando arriva una notifica remote
  terminale e il task del context è già `completed`, l'executor avvia
  spontaneamente un nuovo task A2A interno con un Message synthetic. Il
  nuovo task drena `pending_notifications` come primo turno.
- Tracer `trace_id` reuse: il task auto-spawned non chiama
  `tracer.start_trace`; riusa `entry.trace_session`. Il nuovo
  `a2a_task` span è fratello del precedente sotto lo stesso trace_id.
  Il frontend lo vede come un solo tree con N a2a_task sotto.

**Patch temporanea inclusa**:
- Per far sì che la CLI veda il task auto-spawned, lo spec abilita un
  push notification dal server verso il webhook locale della CLI **per
  questo specifico caso** (task spawned dal drainer). Il webhook
  esistente in `cli_client.py` resta in vita.
- Questa è una **patch temporanea**: la soluzione definitiva (CLI
  streaming SSE only, niente reverse-connection) è in spec 3.
- La patch va rimossa quando spec 3 arriva.

**Esclusioni esplicite**:
- Nessuna modifica al modello di dispatch tool (`dispatch_agent` resta
  fire-and-forget, push verso webhook server padre).
- Nessuna riprogettazione del webhook locale CLI (rimane Starlette su
  random port).
- Nessuna gestione P2P discovery (il client CLI vede solo i task spawned
  dell'agent direttamente connesso).

---

### Spec 2 — C: CLI streaming SSE only

**Status**: pending
**Spec file**: TBD

**Cosa risolve**: limite #4 (NAT/Docker/cloud).

**Scope**:
- CLI passa da `ClientConfig(streaming=False, polling=True,
  push_notification_configs=[...])` a `ClientConfig(streaming=True,
  polling=False, push_notification_configs=[])`.
- `_send_message` consuma direttamente l'`AsyncIterator[ClientEvent]`
  di `client.send_message()`. Ogni `(Task, UpdateEvent)` aggiorna il
  `TaskTracker` mentre lo stream è aperto.
- Su disconnect non terminale: fallback a `client.resubscribe(task_id)`.
- Polling residuo solo come safety net eccezionale.
- **Rimozione completa di** `WebhookServer`, `webhook_host`,
  `OBELIX_WEBHOOK_HOST`/`PORT` env vars, `_poll_warned` flag.
- **Rimozione della patch temporanea introdotta in spec 1** (push verso
  webhook CLI per task spawned).

**Dipendenze**: spec 1 deve essere implementata e mergata prima.

---

### Spec 3 — D: Discovery di agent peer A→B nella CLI

**Status**: pending
**Spec file**: TBD

**Cosa risolve**: limite #2 (B invisibile alla CLI).

**Scope**:
- `dispatch_agent` ritorna anche `url` del remote.
- Server A emette un `TaskStatusUpdateEvent` con `metadata.discovered_agent
  = {name, url, task_id}` ad ogni dispatch riuscito.
- CLI legge questa metadata dallo stream di A: aggiunge B alla propria
  lista di agent connessi (via `A2ACardResolver` su `url`), apre
  `client.resubscribe(task_id)` verso B.
- Il meccanismo è ricorsivo: se B dispatcha C, la stessa logica si
  applica.

**Dipendenze**: spec 2 (CLI deve essere già su streaming SSE).

**Esclusioni**: niente auth verso B (decisione presa dall'utente). Se
B richiede credenziali che A possiede e CLI no, la spec va estesa.

---

### Spec 4 — E: Visualizzazione di agent peer scoperti nella CLI

**Status**: pending
**Spec file**: TBD

**Cosa risolve**: visibilità UX.

**Scope**:
- B (e i peer scoperti dinamicamente) appaiono nella status bar della
  CLI come segmento separato (es. `coordinator is thinking |
  research_agent: working`).
- B può essere selezionato da `/switch` come gli agent configurati
  all'avvio.
- Cleanup: dopo N secondi senza task attivi su B, lo si toglie dalla
  lista (decisione di policy nello spec).

**Dipendenze**: spec 3.

---

### Spec 5 — fire-and-forget design rivisitato

**Status**: pending
**Spec file**: TBD

**Cosa risolve**: alternativa al pattern attuale dove A chiude il
proprio task subito dopo `dispatch_agent`. Spec valuta se per task
critici A debba restare in stato `awaiting_remote` finché B non termina.

**Dipendenze**: tutte le precedenti. Va valutato dopo aver visto in
produzione il comportamento del drainer auto-trigger.

---

## Ordine di esecuzione raccomandato

1. **Spec 1 (A+B)** — fix server-side immediato, sblocca il caso
   orchestrator→coordinator→risposta visibile alla CLI tramite patch
   temporanea webhook.
2. **Spec 2 (C)** — migrazione CLI a streaming. Rimuove patch di spec 1
   e risolve NAT/Docker.
3. **Spec 3 (D)** — discovery peer. Sblocca visibilità A→B nella CLI.
4. **Spec 4 (E)** — UX peer.
5. **Spec 5** — eventuale revisione fire-and-forget.

## Tracking

- Ogni spec finisce in `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md`
- Ogni piano in `docs/superpowers/plans/YYYY-MM-DD-<slug>.md`
- Progresso aggiornato qui (status di ogni spec) man mano che si chiudono.
