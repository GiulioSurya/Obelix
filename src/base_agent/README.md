# BaseAgent

BaseAgent e' la classe base per tutti gli agent del sistema. Fornisce:

- Gestione della conversation history
- Registrazione e esecuzione di tool
- Sistema middleware per intercettare eventi del ciclo di vita
- Integrazione con LLM provider

## Indice

1. [Creazione di un Agent](#creazione-di-un-agent)
2. [Registrazione Tool](#registrazione-tool)
3. [Esecuzione Query](#esecuzione-query)
4. [Sistema Middleware](#sistema-middleware)
5. [Eventi Disponibili](#eventi-disponibili)
6. [API Middleware](#api-middleware)
7. [Esempi Pratici](#esempi-pratici)

---

## Creazione di un Agent

Per creare un agent personalizzato, estendere `BaseAgent`:

```python
from src.base_agent.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, provider=None):
        super().__init__(
            system_message="Sei un assistente specializzato in...",
            provider=provider,
            agent_name="MyAgent",
            description="Descrizione delle capacita dell'agent"
        )
```

### Parametri del costruttore

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `system_message` | `str` | - | Prompt di sistema per l'agent |
| `provider` | `AbstractLLMProvider` | `None` | Provider LLM (usa GlobalConfig se None) |
| `agent_name` | `str` | `None` | Nome dell'agent (usa nome classe se None) |
| `description` | `str` | `None` | Descrizione delle capacita |
| `agent_comment` | `bool` | `True` | Se True, LLM commenta i risultati dei tool |

---

## Registrazione Tool

I tool permettono all'agent di eseguire azioni concrete (query SQL, creazione grafici, etc.).

```python
from src.tools.my_tool import MyTool

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        # Registra un tool
        my_tool = MyTool()
        self.register_tool(my_tool)
```

I tool devono estendere `ToolBase` e implementare:
- `execute(tool_call: ToolCall) -> ToolResult`
- `create_schema() -> MCPToolSchema`

---

## Esecuzione Query

### Metodo sincrono (CLI)

```python
agent = MyAgent()
response = agent.execute_query("Qual e' il totale delle vendite?")

print(response.content)       # Risposta testuale
print(response.tool_results)  # Risultati dei tool eseguiti
print(response.error)         # Eventuale errore
```

### Metodo asincrono (FastAPI)

```python
response = await agent.execute_query_async("Qual e' il totale delle vendite?")
```

### Input come lista di messaggi

E' possibile passare una lista di messaggi pre-costruiti:

```python
from src.messages import HumanMessage, AssistantMessage

messages = [
    HumanMessage(content="Query dell'utente"),
    AssistantMessage(content="Contesto aggiuntivo")
]
response = agent.execute_query(messages)
```

La lista deve contenere esattamente un `HumanMessage`.

---

## Sistema Middleware

Il sistema middleware permette di intercettare eventi durante il ciclo di vita dell'agent senza modificare il codice base.

### Concetti chiave

- **AgentEvent**: Enum che definisce i punti di intercettazione
- **MiddlewareContext**: Contesto passato ai middleware con informazioni sullo stato corrente
- **Middleware**: Oggetto con API fluente per definire condizioni e azioni

### Registrazione middleware

```python
from src.base_agent.middleware import AgentEvent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        # Registra un middleware
        self.on(AgentEvent.ON_TOOL_ERROR) \
            .when(self._condizione) \
            .inject(self._crea_messaggio)
```

---

## Eventi Disponibili

### Lifecycle

| Evento | Quando scatta | Uso tipico |
|--------|---------------|------------|
| `ON_QUERY_START` | Prima di iniziare l'esecuzione | Setup, iniezione contesto iniziale |
| `ON_QUERY_END` | Fine esecuzione (successo o errore) | Cleanup, logging |

### LLM

| Evento | Quando scatta | Uso tipico |
|--------|---------------|------------|
| `BEFORE_LLM_CALL` | Prima di chiamare `provider.invoke()` | Modificare history, logging |
| `AFTER_LLM_CALL` | Dopo la risposta LLM | Trasformare risposta, validazione |

### Tool Calls

| Evento | Quando scatta | Uso tipico |
|--------|---------------|------------|
| `BEFORE_TOOL_EXECUTION` | Prima di eseguire un tool | Modificare parametri, validazione |
| `AFTER_TOOL_EXECUTION` | Dopo l'esecuzione del tool | Trasformare risultato, arricchimento |
| `BEFORE_HISTORY_UPDATE` | Prima di aggiungere alla history | Logging, side effects |
| `ON_TOOL_ERROR` | Quando un tool ritorna errore | Recovery, iniezione contesto |

### Condizionali

| Evento | Quando scatta | Uso tipico |
|--------|---------------|------------|
| `ON_MAX_ITERATIONS` | Raggiunto limite iterazioni | Warning, fallback |

---

## API Middleware

### MiddlewareContext

Contesto passato a tutti i callback del middleware:

```python
@dataclass
class MiddlewareContext:
    event: AgentEvent              # Evento corrente
    agent: BaseAgent               # Riferimento all'agent
    iteration: int                 # Numero iterazione (1-based)
    tool_call: Optional[ToolCall]  # Tool call corrente (se applicabile)
    tool_result: Optional[ToolResult]  # Risultato tool (se applicabile)
    assistant_message: Optional[AssistantMessage]  # Risposta LLM (se applicabile)
    error: Optional[str]           # Messaggio di errore (se applicabile)

    @property
    def conversation_history(self) -> List[StandardMessage]:
        """Accesso diretto alla conversation history"""
```

#### Campi popolati per evento

Non tutti i campi sono sempre presenti. Dipende dall'evento:

| Evento | iteration | tool_call | tool_result | assistant_message | error |
|--------|-----------|-----------|-------------|-------------------|-------|
| `ON_QUERY_START` | 0 | - | - | - | - |
| `BEFORE_LLM_CALL` | 1+ | - | - | - | - |
| `AFTER_LLM_CALL` | 1+ | - | - | presente | - |
| `BEFORE_TOOL_EXECUTION` | 1+ | presente | - | - | - |
| `AFTER_TOOL_EXECUTION` | 1+ | - | presente | - | - |
| `ON_TOOL_ERROR` | 1+ | - | presente | - | presente |
| `BEFORE_HISTORY_UPDATE` | 1+ | - | presente | - | - |
| `ON_QUERY_END` | 1+ | - | - | - | - |
| `ON_MAX_ITERATIONS` | max | - | - | - | - |

### Metodi del Middleware

#### when(condition)

Imposta una condizione per attivare il middleware. Se la condizione ritorna `False`, il middleware non viene eseguito.

```python
def mia_condizione(ctx: MiddlewareContext) -> bool:
    return ctx.iteration > 2

agent.on(AgentEvent.BEFORE_LLM_CALL) \
    .when(mia_condizione) \
    .do(lambda ctx: print("Troppe iterazioni"))
```

La condizione puo essere sincrona o asincrona.

#### inject(message_factory)

Inietta un messaggio alla fine della conversation history (append).

```python
def crea_messaggio(ctx: MiddlewareContext) -> HumanMessage:
    return HumanMessage(content="Informazione aggiuntiva")

agent.on(AgentEvent.ON_TOOL_ERROR) \
    .inject(crea_messaggio)
```

#### inject_at(position, message_factory)

Inietta un messaggio a una posizione specifica nella conversation history.

```python
# Inietta alla posizione 2 (dopo system message e primo messaggio)
agent.on(AgentEvent.ON_QUERY_START) \
    .inject_at(2, lambda ctx: AssistantMessage(content="Piano di esecuzione"))

# Posizione calcolata dinamicamente
agent.on(AgentEvent.ON_QUERY_START) \
    .inject_at(
        lambda ctx: len(ctx.conversation_history),  # Alla fine
        lambda ctx: HumanMessage(content="...")
    )
```

#### transform(transformer)

Trasforma il valore corrente. Usare con eventi che hanno un valore trasformabile:
- `AFTER_TOOL_EXECUTION`: trasforma `ToolResult`
- `AFTER_LLM_CALL`: trasforma `AssistantMessage`
- `BEFORE_TOOL_EXECUTION`: trasforma `ToolCall`

```python
async def arricchisci_errore(result: ToolResult, ctx: MiddlewareContext) -> ToolResult:
    if result.error:
        return result.model_copy(update={"error": f"[ENRICHED] {result.error}"})
    return result

agent.on(AgentEvent.AFTER_TOOL_EXECUTION) \
    .when(lambda ctx: ctx.tool_result.status == ToolStatus.ERROR) \
    .transform(arricchisci_errore)
```

#### do(action)

Esegue un'azione generica senza modificare valori. Utile per logging, side effects.

```python
agent.on(AgentEvent.ON_QUERY_END) \
    .do(lambda ctx: print(f"Query completata in {ctx.iteration} iterazioni"))
```

---

## Esempi Pratici

### Esempio 1: Iniezione contesto iniziale

Iniettare un piano di esecuzione all'inizio della conversazione:

```python
plan = "1. Analizza la query\n2. Genera SQL\n3. Esegui"

sql_agent.on(AgentEvent.ON_QUERY_START) \
    .inject_at(2, lambda ctx: AssistantMessage(content=plan))

result = sql_agent.execute_query("Mostra le vendite totali")
```

### Esempio 2: Recovery su errore

Iniettare lo schema database quando si verifica un errore "invalid identifier":

```python
class SQLGeneratorAgent(BaseAgent):
    def __init__(self, database_schema):
        super().__init__(system_message="...")
        self.database_schema = database_schema
        self._schema_injected = False

        self.on(AgentEvent.ON_TOOL_ERROR) \
            .when(self._is_invalid_identifier) \
            .inject(self._create_schema_message)

    def _is_invalid_identifier(self, ctx: MiddlewareContext) -> bool:
        return (
            ctx.error is not None and
            "invalid identifier" in ctx.error and
            not self._schema_injected
        )

    def _create_schema_message(self, ctx: MiddlewareContext) -> HumanMessage:
        self._schema_injected = True
        return HumanMessage(
            content=f"Schema database:\n{self.database_schema}\nCorreggi la query."
        )
```

### Esempio 3: Arricchimento errori

Arricchire gli errori Oracle con documentazione ufficiale:

```python
class SQLGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(self._has_oracle_docs_url) \
            .transform(self._enrich_with_docs)

    def _has_oracle_docs_url(self, ctx: MiddlewareContext) -> bool:
        result = ctx.tool_result
        return (
            result is not None and
            result.status == ToolStatus.ERROR and
            result.error is not None and
            "docs.oracle.com" in result.error
        )

    async def _enrich_with_docs(self, result: ToolResult, ctx: MiddlewareContext) -> ToolResult:
        docs = await self._fetch_oracle_docs(result.error)
        if docs:
            enriched = f"{result.error} | Cause: {docs['cause']}"
            return result.model_copy(update={"error": enriched})
        return result
```

### Esempio 4: Logging iterazioni

Loggare ogni iterazione del ciclo agent:

```python
agent.on(AgentEvent.BEFORE_LLM_CALL) \
    .do(lambda ctx: print(f"[Iterazione {ctx.iteration}] Chiamata LLM..."))

agent.on(AgentEvent.AFTER_LLM_CALL) \
    .do(lambda ctx: print(f"[Iterazione {ctx.iteration}] Risposta ricevuta"))

agent.on(AgentEvent.ON_MAX_ITERATIONS) \
    .do(lambda ctx: print(f"[WARNING] Limite {ctx.iteration} iterazioni raggiunto"))
```

### Esempio 5: Middleware condizionale su iterazione

Iniettare un warning dopo troppe iterazioni:

```python
agent.on(AgentEvent.BEFORE_LLM_CALL) \
    .when(lambda ctx: ctx.iteration >= 3) \
    .inject(lambda ctx: HumanMessage(
        content="ATTENZIONE: Stai impiegando troppe iterazioni. Semplifica l'approccio."
    ))
```

---

## Note Tecniche

### Ordine di esecuzione

I middleware registrati sullo stesso evento vengono eseguiti in ordine di registrazione. Il risultato di un middleware puo essere passato al successivo (chain pattern).

### Callback sincroni e asincroni

Tutti i metodi del middleware (`when`, `inject`, `transform`, `do`) accettano sia funzioni sincrone che asincrone. Il sistema gestisce automaticamente l'await se necessario.

### Accesso alla conversation history

La conversation history e' accessibile tramite:
- `ctx.conversation_history` (property del contesto)
- `ctx.agent.conversation_history` (accesso diretto)

Le modifiche alla history (append, insert) sono immediatamente visibili.
