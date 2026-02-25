# A2A Serve — Piano di Implementazione

## Contesto

Vogliamo esporre gli agent Obelix come **A2A Server** conformi al protocollo [A2A](https://a2a-protocol.org/latest/).
Un client A2A (altro agent, orchestrator esterno, UI) potra' scoprire l'agent via Agent Card, inviargli messaggi via JSON-RPC 2.0, e tracciare i task.

L'entry point sara' `AgentFactory.serve()` che avvia un server FastAPI + Uvicorn con tutti gli endpoint richiesti dal protocollo.

---

## Architettura

```
Client A2A
    |
    |  HTTP
    v
FastAPI app (uvicorn)
    |
    |-- GET  /.well-known/agent-card.json   -> AgentCard JSON
    |-- POST /                               -> JSON-RPC 2.0 dispatcher
    |-- GET  /health                         -> health check
    |
    v
ObelixA2AController (APIRouter)
    |
    |-- SendMessage        -> BaseAgent.execute_query()
    |-- GetTask            -> TaskStore.get()
    |-- ListTasks          -> TaskStore.list()
    |-- CancelTask         -> TaskStore.cancel()
    |
    v
BaseAgent  <-->  Provider LLM
```

### Posizionamento nell'architettura hexagonal

```
src/obelix/
  adapters/
    inbound/                    # NUOVO — adapter in ingresso
      a2a/
        __init__.py
        controller.py           # ObelixA2AController (APIRouter)
        handlers.py             # handler per ogni metodo JSON-RPC
        task_store.py           # in-memory task store
        models.py               # dataclass/pydantic di supporto (se servono oltre a2a-sdk)
    outbound/                   # gia' esistente (provider LLM)
  core/
    agent/
      agent_factory.py          # aggiunge serve() e _create_a2a_app()
```

`adapters/inbound/` e' il duale di `adapters/outbound/`: se outbound sono le porte verso i provider LLM, inbound e' la porta d'ingresso HTTP per i client A2A.

---

## Agent Card — Struttura conforme

Campi **obbligatori** secondo la spec:

```json
{
  "id": "obelix-coordinator-001",
  "name": "Coordinator Agent",
  "version": "0.1.0",
  "description": "Orchestrates weather and planner sub-agents",
  "provider": {
    "organization": "Obelix",
    "url": "https://github.com/user/obelix"
  },
  "interfaces": [
    {
      "type": "json-rpc",
      "url": "http://0.0.0.0:8000",
      "version": "2.0"
    }
  ],
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "extendedAgentCard": false
  },
  "skills": [
    {
      "id": "weather_agent",
      "name": "weather_agent",
      "description": "Provides weather forecasts"
    }
  ],
  "securitySchemes": {},
  "security": []
}
```

### Generazione automatica della card

- `id` — generato da `agent_name` o passato esplicitamente
- `name` — dall'agent servito
- `skills` — derivate dai **sub-agent registrati** (nome + descrizione da `SubAgentWrapper`) e/o dai **tool registrati** (nome + descrizione da `ToolBase`)
- `provider` — configurabile via `serve()` kwargs
- `interfaces` — costruito da host/port passati a `serve()`

---

## JSON-RPC 2.0 — Method Names conformi

Tutti i metodi arrivano su `POST /`. Il dispatcher legge `request.method` e instrada:

| Metodo JSON-RPC | Operazione | MVP? |
|---|---|---|
| `SendMessage` | Invia messaggio all'agent, ritorna Task | **Si** |
| `GetTask` | Recupera stato di un task | **Si** |
| `ListTasks` | Lista task con filtri | **Si** |
| `CancelTask` | Cancella un task | **Si** |
| `SendStreamingMessage` | Streaming SSE | No (v2) |
| `SubscribeToTask` | Subscribe a task esistente | No (v2) |
| `CreateTaskPushNotificationConfig` | Webhook push | No (v3) |
| `GetTaskPushNotificationConfig` | — | No (v3) |
| `ListTaskPushNotificationConfigs` | — | No (v3) |
| `DeleteTaskPushNotificationConfig` | — | No (v3) |
| `GetExtendedAgentCard` | Card autenticata | No (v3) |

I metodi non implementati rispondono con `UnsupportedOperationError` (come richiede la spec quando `capabilities` dichiara `false`).

---

## Task Store (in-memory)

```python
@dataclass
class TaskRecord:
    task: a2a_types.Task
    created_at: datetime
    updated_at: datetime

class TaskStore:
    """Thread-safe in-memory task store."""

    def create(self, context_id: str | None = None) -> Task
    def get(self, task_id: str) -> Task | None
    def list(self, context_id: str | None, status: str | None, page_size: int, page_token: str | None) -> ListTasksResponse
    def update_status(self, task_id: str, state: TaskState, message: str | None = None) -> Task
    def add_artifact(self, task_id: str, artifact: Artifact) -> Task
    def cancel(self, task_id: str) -> Task
```

- Dict `{task_id: TaskRecord}` protetto da `asyncio.Lock`
- Nessuna persistenza nel MVP (in-memory only)
- `context_id` generato server-side se non fornito dal client

---

## Flusso SendMessage

```
1. Client invia POST / con:
   { "jsonrpc": "2.0", "method": "SendMessage", "id": 1,
     "params": { "message": { "role": "user", "parts": [{"text": "..."}] } } }

2. Controller:
   a. Crea Task in TaskStore (state: working)
   b. Estrae testo dal primo TextPart del messaggio
   c. Chiama agent.execute_query(testo) in modo asincrono
   d. Al completamento:
      - Crea Artifact con la risposta
      - Aggiorna Task (state: completed, artifacts: [...])
   e. Ritorna JSONRPCResponse con il Task completo

3. In caso di errore:
   - Aggiorna Task (state: failed, message: errore)
   - Ritorna JSONRPCResponse con il Task in stato failed
```

---

## API di `AgentFactory.serve()`

```python
def serve(
    self,
    agent: str,                    # nome dell'agent registrato da servire
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    agent_id: str | None = None,   # id per la card (default: auto)
    version: str = "0.1.0",       # versione della card
    provider_name: str = "Obelix",
    provider_url: str | None = None,
    log_level: str = "info",
) -> None:
    """Avvia un A2A server per l'agent specificato."""
```

Internamente:
1. Chiama `self.create(agent)` per ottenere l'istanza BaseAgent
2. Costruisce `ObelixA2AController` passando agent, metadata, TaskStore
3. Crea `FastAPI()`, include il controller come router
4. Chiama `uvicorn.run(app, ...)`

---

## Dipendenze

Nel `pyproject.toml`, nuovo gruppo **extras** `serve`:

```toml
[project.optional-dependencies]
serve = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "a2a-sdk>=0.3.10",
]
```

Installazione: `uv sync --extra serve`

Import lazy: `agent_factory.serve()` importa fastapi/uvicorn/a2a solo quando viene chiamato, cosi' il core resta leggero.

---

## File da creare/modificare

| File | Azione |
|---|---|
| `src/obelix/adapters/inbound/__init__.py` | Creare (vuoto) |
| `src/obelix/adapters/inbound/a2a/__init__.py` | Creare (re-export controller) |
| `src/obelix/adapters/inbound/a2a/controller.py` | Creare — ObelixA2AController |
| `src/obelix/adapters/inbound/a2a/handlers.py` | Creare — handler per SendMessage, GetTask, ListTasks, CancelTask |
| `src/obelix/adapters/inbound/a2a/task_store.py` | Creare — in-memory TaskStore |
| `src/obelix/core/agent/agent_factory.py` | Modificare — aggiungere `serve()` e `_create_a2a_app()` |
| `pyproject.toml` | Modificare — aggiungere extra `serve` |

---

## Fasi di implementazione

### Fase 1 — Infrastruttura (questo MVP)

1. Aggiungere dipendenze `serve` in pyproject.toml
2. Creare `adapters/inbound/a2a/task_store.py` — TaskStore in-memory
3. Creare `adapters/inbound/a2a/controller.py` — ObelixA2AController
4. Creare `adapters/inbound/a2a/handlers.py` — SendMessage, GetTask, ListTasks, CancelTask
5. Modificare `agent_factory.py` — aggiungere `serve()` + `_create_a2a_app()`
6. Test unitari per TaskStore + handler + controller
7. Test e2e: avviare server, GET agent card, POST SendMessage

### Fase 2 — Streaming

- Implementare streaming su BaseAgent (yield chunks)
- Aggiungere `SendStreamingMessage` + `SubscribeToTask`
- Aggiornare `capabilities.streaming = true` nella card

### Fase 3 — Funzionalita' avanzate

- Push notifications (webhook)
- Extended Agent Card (autenticazione)
- Persistenza task store (Redis/SQLite)
- Multi-agent serving (piu' agent sullo stesso server con routing per skill)

---

## Verifica

1. `uv sync --extra serve` — dipendenze installate
2. `uv run ruff check . && uv run ruff format --check .` — lint ok
3. `uv run pytest tests/` — test unitari passano
4. **Test manuale**:
   ```bash
   # Avvia server
   uv run python demo_a2a.py

   # Agent Card
   curl http://localhost:8000/.well-known/agent-card.json | jq .

   # SendMessage
   curl -X POST http://localhost:8000 \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"SendMessage","id":1,"params":{"message":{"role":"user","parts":[{"text":"What is the weather in Rome?"}]}}}'

   # GetTask
   curl -X POST http://localhost:8000 \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"GetTask","id":2,"params":{"id":"<task_id>"}}'
   ```