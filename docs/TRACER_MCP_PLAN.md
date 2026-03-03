# Piano: Obelix Tracer MCP Server

## Contesto e Obiettivo

### Il problema
Quando si usa Obelix per costruire sistemi multi-agent, dall'esterno si vede solo input del coordinator e output finale. Tutto cio' che sta nel mezzo (deleghe ai sub-agent, reasoning di ogni agent, tool call intermedi, errori interni, retry) e' una scatola nera.

### La soluzione
Un **server MCP** che espone il tracer di Obelix come tool leggibili da un agent esterno (tipicamente Claude Code). Questo permette a un **meta-agent** di:
1. Osservare il runtime del sistema multi-agent in tempo reale
2. Capire cosa succede internamente (reasoning, deleghe, errori)
3. Modificare il sistema (system prompt, hook, tool) basandosi su cio' che osserva
4. Iterare: osserva → modifica → rilancia → osserva

### Flusso d'uso tipico
```
1. L'utente istruisce Claude Code su come usare Obelix
2. Claude Code costruisce un sistema di agenti (coordinator + sub-agent)
3. L'utente lancia il sistema, il tracer invia dati al tracer backend
4. Claude Code, tramite il server MCP, osserva le trace in tempo reale
5. Claude Code identifica problemi (reasoning sbagliato, tool falliti, deleghe inutili)
6. Claude Code modifica direttamente i file (system prompt, hook, codice) — non servono tool MCP per questo
7. L'utente rilancia, Claude Code osserva di nuovo → ciclo iterativo
```

---

## Architettura

```
Claude Code (meta-agent)
    |
    |-- MCP Server (obelix-tracer-mcp)        <-- DA COSTRUIRE
    |     |-- Tool read-only → HTTP client verso tracer backend
    |     └── Layer sottile, nessuna logica propria
    |
    |-- Tracer Backend (obelix-tracer)          <-- GIA' ESISTENTE + ESTENSIONI
    |     |-- SQLite (traces, spans)            <-- esistente
    |     |-- API REST /api/v1/                 <-- esistente
    |     |-- WebSocket live                    <-- esistente
    |     |-- API REST /api/v1/meta/            <-- DA AGGIUNGERE (sessioni + metalog)
    |     └── Tabelle sessions, metalog         <-- DA AGGIUNGERE
    |
    └── File system
          └── Claude Code modifica direttamente system prompt, hook, codice Obelix
```

---

## Parte 1: Estensioni al Tracer Backend (obelix-tracer)

### Progetto: `C:\Users\GLoverde\PycharmProjects\obelix-tracer`

### Decisione architetturale
Il meta-layer (sessioni + metalog) vive nello **stesso progetto del tracer**, ma come **modulo separato**:
- Router separato: `/api/v1/meta/` — non tocca i router esistenti (ingest, traces, websocket)
- Modelli separati: `models/meta.py` — non tocca `models/trace.py` e `models/responses.py`
- Storage esteso: si aggiungono metodi all'ABC e all'implementazione SQLite

**Motivazione**: sessioni e metalog sono concetti diversi dal tracing puro ("perche' il sistema e' cambiato" vs "cosa e' successo nel runtime"), ma sono strettamente correlati (le foreign key tra metalog → trace sono naturali). Stesso DB, stesso frontend, un solo servizio da gestire.

### Schema DB (nuove tabelle)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    goal TEXT,                    -- obiettivo dell'esperimento/iterazione
    status TEXT NOT NULL DEFAULT 'active',  -- active | completed | abandoned
    start_time REAL NOT NULL,
    end_time REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_traces (
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    added_at REAL NOT NULL,
    notes TEXT,                   -- annotazione opzionale del meta-agent su questa trace
    PRIMARY KEY (session_id, trace_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);

CREATE TABLE IF NOT EXISTS metalog (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT,                -- a quale trace si riferisce (nullable)
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,         -- tipo di azione: observe, modify_prompt, add_hook, remove_tool, etc.
    target TEXT,                  -- su quale agent/componente
    observation TEXT,             -- cosa ha osservato il meta-agent
    change TEXT,                  -- cosa ha cambiato
    reasoning TEXT,               -- perche' ha fatto questa scelta
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_session_traces_session ON session_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_metalog_session ON metalog(session_id);
CREATE INDEX IF NOT EXISTS idx_metalog_trace ON metalog(trace_id);
```

### Nuovi file nel tracer backend

```
obelix_tracer/
  models/
    meta.py              # NUOVO: Session, SessionTrace, MetalogEntry, response models
  api/
    meta.py              # NUOVO: router /api/v1/meta/ con tutti gli endpoint
  storage/
    abc.py               # ESTESO: nuovi metodi astratti per sessions/metalog
    sqlite.py            # ESTESO: implementazione dei nuovi metodi
  app.py                 # ESTESO: montare il nuovo router
```

### Endpoint REST del meta-layer

#### Sessioni

| Metodo | Endpoint | Body/Params | Risposta | Descrizione |
|--------|----------|-------------|----------|-------------|
| POST | `/meta/sessions` | `{name, description?, goal?}` | `{session_id, ...}` | Crea una nuova sessione |
| GET | `/meta/sessions` | `?status=active&limit=20` | `[Session, ...]` | Lista sessioni |
| GET | `/meta/sessions/{id}` | - | `Session` con trace associate e metalog | Dettaglio sessione completo |
| PATCH | `/meta/sessions/{id}` | `{status?, description?, goal?}` | `{status: "ok"}` | Aggiorna sessione (es. chiudi) |

#### Associazione trace ↔ sessione

| Metodo | Endpoint | Body/Params | Risposta | Descrizione |
|--------|----------|-------------|----------|-------------|
| POST | `/meta/sessions/{id}/traces` | `{trace_id, notes?}` | `{status: "ok"}` | Associa una trace alla sessione |
| GET | `/meta/sessions/{id}/traces` | - | `[TraceListItem + notes, ...]` | Lista trace della sessione in ordine |

#### Metalog

| Metodo | Endpoint | Body/Params | Risposta | Descrizione |
|--------|----------|-------------|----------|-------------|
| POST | `/meta/sessions/{id}/log` | `{trace_id?, action, target?, observation?, change?, reasoning?}` | `{entry_id, ...}` | Aggiunge entry al metalog |
| GET | `/meta/sessions/{id}/log` | `?action=modify_prompt` | `[MetalogEntry, ...]` | Legge il metalog della sessione |

### Pydantic Models (`models/meta.py`)

```python
class SessionCreate(BaseModel):
    name: str
    description: str | None = None
    goal: str | None = None

class SessionUpdate(BaseModel):
    status: str | None = None  # active, completed, abandoned
    description: str | None = None
    goal: str | None = None

class SessionResponse(BaseModel):
    session_id: str
    name: str
    description: str | None
    goal: str | None
    status: str
    start_time: float
    end_time: float | None
    metadata: dict
    trace_count: int           # quante trace associate
    metalog_count: int         # quante entry nel metalog

class SessionDetailResponse(SessionResponse):
    traces: list[...]          # TraceListItem + notes
    metalog: list[...]         # MetalogEntry ordinati per timestamp

class MetalogCreate(BaseModel):
    trace_id: str | None = None
    action: str                # observe, modify_prompt, add_hook, remove_tool, add_tool, restructure, etc.
    target: str | None = None  # agent name o componente
    observation: str | None = None
    change: str | None = None
    reasoning: str | None = None

class MetalogEntryResponse(BaseModel):
    entry_id: str
    session_id: str
    trace_id: str | None
    timestamp: float
    action: str
    target: str | None
    observation: str | None
    change: str | None
    reasoning: str | None
    metadata: dict

class AddTraceToSession(BaseModel):
    trace_id: str
    notes: str | None = None
```

### Endpoint aggiuntivo per ricerca cross-trace (utile al MCP)

Aggiungere al router traces esistente:

| Metodo | Endpoint | Params | Descrizione |
|--------|----------|--------|-------------|
| GET | `/spans` | `?status=error&span_type=tool&from_time=...&to_time=...&limit=50` | Ricerca span cross-trace |

Questo serve al meta-agent per fare query tipo "ci sono errori nelle ultime trace?" senza scorrere trace per trace.

---

## Parte 2: Server MCP (obelix-tracer-mcp)

### Progetto: nuovo, separato — `C:\Users\GLoverde\PycharmProjects\obelix-tracer-mcp`

### Tech stack
- Python 3.13+
- `mcp` SDK (Model Context Protocol Python SDK)
- `httpx` (async HTTP client verso tracer backend)
- `uv` come package manager

### Struttura progetto

```
obelix-tracer-mcp/
  pyproject.toml
  src/
    obelix_tracer_mcp/
      __init__.py
      server.py            # MCP server principale, registra tool
      client.py            # HTTP client async verso tracer backend
      tools/
        __init__.py
        traces.py          # Tool: list_traces, get_trace, get_trace_summary, get_span, get_agent_spans
        services.py        # Tool: list_services
        sessions.py        # Tool: create_session, list_sessions, get_session, end_session, add_trace_to_session
        metalog.py         # Tool: add_metalog_entry, get_metalog
        search.py          # Tool: search_spans (cross-trace)
      config.py            # URL del tracer backend, timeout, etc.
```

### Lista completa dei tool MCP

#### Osservazione runtime (proxy → tracer GET endpoints)

| Tool MCP | Tracer API | Descrizione per il meta-agent |
|----------|-----------|-------------------------------|
| `list_services` | `GET /services` | Lista servizi Obelix attivi con conteggio trace e stato di salute |
| `list_traces` | `GET /traces` | Lista trace recenti. Filtri: service_name, agent_name, status, from_time, to_time, limit. Usa per trovare esecuzioni specifiche |
| `get_trace` | `GET /traces/{id}` | Albero completo di span di una trace. Mostra la struttura di orchestrazione: chi ha delegato cosa a chi, in che ordine, con che risultato |
| `get_trace_summary` | `GET /traces/{id}/summary` | Riassunto testuale rapido di una trace: agent coinvolti, tool eseguiti, errori, durate |
| `get_span` | `GET /spans/{id}` | Dettaglio di un singolo span. Su span LLM: reasoning completo del modello. Su span tool: argomenti e risultato. Su span hook: cosa ha intercettato |
| `get_agent_spans` | `GET /traces/{id}/agents/{name}` | Tutti gli span di un agent specifico dentro una trace. Utile per isolare il comportamento di un sub-agent |
| `search_spans` | `GET /spans` (nuovo) | Ricerca cross-trace. Trova span per tipo, status, timerange. Esempio: "tutti i tool error degli ultimi 5 minuti" |

#### Sessioni (proxy → tracer /meta/ endpoints)

| Tool MCP | Tracer API | Descrizione per il meta-agent |
|----------|-----------|-------------------------------|
| `create_session` | `POST /meta/sessions` | Crea una sessione di lavoro. Una sessione raggruppa trace successive dello stesso esperimento iterativo |
| `list_sessions` | `GET /meta/sessions` | Lista sessioni esistenti |
| `get_session` | `GET /meta/sessions/{id}` | Dettaglio sessione con trace associate e metalog completo. Mostra l'evoluzione dell'esperimento |
| `end_session` | `PATCH /meta/sessions/{id}` | Chiude una sessione (status: completed o abandoned) |
| `add_trace_to_session` | `POST /meta/sessions/{id}/traces` | Associa una trace a una sessione. Opzionalmente aggiungi note su questa specifica esecuzione |

#### Metalog (proxy → tracer /meta/ endpoints)

| Tool MCP | Tracer API | Descrizione per il meta-agent |
|----------|-----------|-------------------------------|
| `add_metalog_entry` | `POST /meta/sessions/{id}/log` | Registra una decisione: cosa hai osservato, cosa hai cambiato, perche'. Questo log e' visibile all'utente nel frontend del tracer |
| `get_metalog` | `GET /meta/sessions/{id}/log` | Legge il metalog di una sessione. Utile per ricordare cosa hai gia' provato e quali modifiche hai fatto |

### Configurazione MCP per Claude Code

File `claude_desktop_config.json` o `.claude/settings.json`:

```json
{
  "mcpServers": {
    "obelix-tracer": {
      "command": "uv",
      "args": ["--directory", "C:\\Users\\GLoverde\\PycharmProjects\\obelix-tracer-mcp", "run", "python", "-m", "obelix_tracer_mcp"],
      "env": {
        "TRACER_URL": "http://localhost:8100/api/v1"
      }
    }
  }
}
```

---

## Parte 3: Estensione Frontend Tracer (opzionale, fase successiva)

Il frontend del tracer (`obelix-tracer/frontend/`) potrebbe mostrare:
- Pagina sessioni: lista sessioni, click per vedere l'evoluzione
- Timeline sessione: trace in sequenza con metalog entry intercalate
- Dettaglio metalog entry: cosa ha osservato, cambiato, perche'

Questo e' opzionale e puo' venire dopo che il backend + MCP server funzionano.

---

## Ordine di implementazione

### Fase 1: Estensioni tracer backend
1. Creare `models/meta.py` con i Pydantic model
2. Estendere `storage/abc.py` con i nuovi metodi astratti
3. Implementare in `storage/sqlite.py` (nuove tabelle, CRUD sessions/metalog)
4. Creare `api/meta.py` con il router FastAPI
5. Montare il router in `app.py`
6. Aggiungere endpoint `GET /spans` cross-trace in `api/traces.py`
7. Testare manualmente con curl/httpie

### Fase 2: Server MCP
1. Scaffolding progetto con uv
2. Creare `client.py` (httpx async client)
3. Implementare tool uno per uno, partendo da quelli di osservazione
4. Aggiungere tool sessioni e metalog
5. Configurare in Claude Code
6. Test end-to-end: lanciare sistema Obelix → osservare via MCP → iterare

### Fase 3: Frontend (opzionale)
1. Pagina sessioni
2. Vista timeline sessione
3. Integrazione metalog nella UI

---

## Dipendenze e prerequisiti

### Tracer backend (obelix-tracer) — gia' funzionante
- Progetto: `C:\Users\GLoverde\PycharmProjects\obelix-tracer`
- Stack: FastAPI + aiosqlite + pydantic
- DB: `traces.db` (SQLite)
- Port: 8100
- API base: `/api/v1/`
- Endpoint esistenti:
  - Ingest: `POST /ingest`, `POST /ingest/trace`, `POST /ingest/span`, `PATCH /ingest/trace/{id}`
  - Query: `GET /services`, `GET /traces`, `GET /traces/{id}`, `GET /traces/{id}/flat`, `GET /traces/{id}/summary`, `GET /traces/{id}/agents/{name}`, `GET /spans/{id}`
  - WebSocket: `/ws/live/{trace_id}`, `/ws/live/service/{service_name}`
- Storage ABC: `obelix_tracer.storage.abc.TraceStorage`
- Storage impl: `obelix_tracer.storage.sqlite.SQLiteTraceStorage`

### Obelix framework — gia' funzionante
- Progetto: `C:\Users\GLoverde\PycharmProjects\Obelix`
- Tracer SDK: `src/obelix/core/tracer/` (Tracer, exporters, models, context)
- HTTPExporter: invia a tracer backend via POST endpoints
- BaseAgent: integra tracer opzionalmente (parametro `tracer` nel costruttore)

### MCP SDK
- Package: `mcp` (PyPI)
- Docs: https://modelcontextprotocol.io
- Transport: stdio (Claude Code lo lancia come subprocess)
