# Piano Shared Memory Graph + Orchestrator Awareness

> **Stato**: da implementare
> **Ultimo aggiornamento**: post-refactor architettura esagonale, eliminazione decoratori `@orchestrator`/`@subagent`
> **Ispirazione esterna**: progetto Parlant (`C:\Users\GLoverde\PycharmProjects\Parlant`) per l'uso di NetworkX

---

## Indice

1. [Contesto e Obiettivo](#1-contesto-e-obiettivo)
2. [Architettura Attuale (riferimenti al codice)](#2-architettura-attuale-riferimenti-al-codice)
3. [Comportamento Desiderato (esempio concreto)](#3-comportamento-desiderato-esempio-concreto)
4. [Libreria Grafo: NetworkX (analisi Parlant)](#4-libreria-grafo-networkx-analisi-parlant)
5. [Design: SharedMemoryGraph](#5-design-sharedmemorygraph)
6. [Strategia di Iniezione (BEFORE_LLM_CALL)](#6-strategia-di-iniezione-before_llm_call)
7. [Strategia di Pubblicazione (BEFORE_FINAL_RESPONSE)](#7-strategia-di-pubblicazione-before_final_response)
8. [Messaggio Awareness per Orchestratore](#8-messaggio-awareness-per-orchestratore)
9. [Integrazione con AgentFactory](#9-integrazione-con-agentfactory)
10. [Modifiche a BaseAgent](#10-modifiche-a-baseagent)
11. [Flusso Runtime Completo](#11-flusso-runtime-completo)
12. [Comportamento con SubAgentWrapper (stateless/stateful)](#12-comportamento-con-subagentwrapper-statelessstateful)
13. [Esempio Completo End-to-End](#13-esempio-completo-end-to-end)
14. [Step di Implementazione](#14-step-di-implementazione)
15. [Decisioni Aperte](#15-decisioni-aperte)

---

## 1. Contesto e Obiettivo

### Il problema

Oggi i sub-agent sono registrati tramite `BaseAgent.register_agent()` e wrappati in `SubAgentWrapper`. Ogni sub-agent e' **isolato**: quando viene eseguito, riceve solo la query passata dall'orchestratore. Se il sub-agent B ha bisogno dell'output del sub-agent A, l'unico modo e' che l'orchestratore passi manualmente quel contesto nella query di B, facendo da "postino".

### L'obiettivo

Creare un sistema di **memoria condivisa temporanea** (in-memory) basato su un **grafo diretto di dipendenze** tra agent. Il sistema deve:

1. **Propagare automaticamente** le risposte finali tra agent collegati nel grafo, senza intervento dell'orchestratore
2. Condividere **solo il contenuto testuale della risposta finale** (no tool calls, no history completa)
3. Iniettare il contesto condiviso come `SystemMessage` nella history del sub-agent **prima della prima chiamata LLM** (pull-on-start)
4. Fornire all'orchestratore un **system message aggiuntivo** che descriva l'ordine di dipendenza tra i sub-agent

### Cosa NON fa

- NON persiste nulla su disco (sara' aggiunto in una fase futura)
- NON modifica il funzionamento base degli agent senza grafo (zero overhead)
- NON richiede che l'orchestratore sappia del grafo a runtime (il grafo opera tramite hook)

---

## 2. Architettura Attuale (riferimenti al codice)

### Struttura directory

```
src/domain/
  agent/
    __init__.py              -> esporta BaseAgent, SubAgentWrapper
    base_agent.py            -> 661 righe, loop principale, hook system
    hooks.py                 -> AgentEvent, Hook, AgentStatus, HookDecision, Outcome
    event_contracts.py       -> EventContract per ogni AgentEvent
    subagent_wrapper.py      -> Bridge BaseAgent <-> ToolBase
    agent_factory.py         -> AgentFactory con register()/create()
  model/
    __init__.py              -> esporta tutti i tipi messaggio
    system_message.py        -> SystemMessage (Pydantic BaseModel)
    human_message.py         -> HumanMessage
    assistant_message.py     -> AssistantMessage, AssistantResponse
    tool_message.py          -> ToolCall, ToolResult, ToolStatus, ToolMessage
    standard_message.py      -> Union type di tutti i messaggi
    usage.py                 -> Usage, AgentUsage
  tool/
    tool_base.py             -> ToolBase (ABC)
    tool_decorator.py        -> @tool decorator
```

### BaseAgent - Punti rilevanti

**File**: `src/domain/agent/base_agent.py`

**Costruttore** (riga 24-68):
```python
class BaseAgent:
    def __init__(
        self,
        system_message: str,
        provider: Optional[AbstractLLMProvider] = None,
        max_iterations: int = 15,
        tools: Optional[...] = None,
        tool_policy: Optional[List[ToolRequirement]] = None,
        exit_on_success: Optional[List[str]] = None,
    ):
        self.system_message = SystemMessage(content=system_message)
        self.conversation_history: List[StandardMessage] = [self.system_message]
        self.registered_tools: List[ToolBase] = []
        self._hooks: Dict[AgentEvent, List[Hook]] = {event: [] for event in AgentEvent}
```

**Attributi chiave** che la shared memory dovra' usare:
- `self.conversation_history` (riga 40): lista mutabile, inizia con `[SystemMessage]`
- `self._hooks` (riga 45): dizionario `AgentEvent -> List[Hook]`
- `self.registered_tools` (riga 39): lista dei tool (inclusi SubAgentWrapper)

**Metodo `register_agent()`** (riga 620-651):
```python
def register_agent(self, agent: 'BaseAgent', *, name: str, description: str, stateless: bool = False):
    wrapper = SubAgentWrapper(agent, name=name, description=description, stateless=stateless)
    self.registered_tools.append(wrapper)
```
Questo metodo vive direttamente su `BaseAgent`. Non serve nessun decoratore.

**Metodo `on()`** (riga 94-100):
```python
def on(self, event: AgentEvent) -> Hook:
    hook = Hook(event)
    self._hooks[event].append(hook)
    return hook
```
Restituisce un `Hook` con API fluent (`when()`, `handle()`).

### Hook System - Punti rilevanti

**File**: `src/domain/agent/hooks.py`

**AgentEvent** (riga 28-54): gli eventi che ci interessano sono:
- `BEFORE_LLM_CALL` = prima di `provider.invoke()` -> qui **inietteremo** la memoria condivisa
- `BEFORE_FINAL_RESPONSE` = prima di restituire la risposta -> qui **pubblicheremo** sul grafo

**HookDecision** (riga 57-62):
- `CONTINUE`: prosegui normalmente
- `RETRY`: riesegui la chiamata LLM (solo per eventi retryable)
- `FAIL`: interrompi con errore
- `STOP`: esci immediatamente con un valore

**AgentStatus** (riga 73-91): il contesto passato a ogni hook:
```python
@dataclass
class AgentStatus:
    event: AgentEvent
    agent: 'BaseAgent'            # riferimento all'agent (accesso a conversation_history)
    iteration: int = 0
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    assistant_message: Optional[AssistantMessage] = None
    error: Optional[str] = None

    @property
    def conversation_history(self) -> List['StandardMessage']:
        return self.agent.conversation_history   # riferimento diretto alla lista
```

**Hook.handle()** (riga 125-149): definisce decisione + effetti:
```python
def handle(self, decision: HookDecision, value=None, effects=None) -> 'Hook':
    self._decision = decision
    self._value = value
    self._effects = effects or []
```

**Hook.execute()** (riga 151-198): esegue gli effetti, supporta effetti **async**:
```python
for i, effect in enumerate(self._effects):
    effect_result = effect(agent_status)
    if asyncio.iscoroutine(effect_result):
        await effect_result
```

### Event Contracts - Vincoli

**File**: `src/domain/agent/event_contracts.py`

| Evento | input_type | output_type | retryable | stop_output |
|--------|-----------|-------------|-----------|-------------|
| `BEFORE_LLM_CALL` | `None` | `None` | `False` | `AssistantMessage` |
| `BEFORE_FINAL_RESPONSE` | `AssistantMessage` | `AssistantMessage` | `True` | `AssistantMessage` |
| `QUERY_END` | `AssistantResponse\|None` | `AssistantResponse\|None` | `False` | `None` |

**Implicazione per la shared memory**:
- L'hook su `BEFORE_LLM_CALL` NON riceve un `current_value` (e' `None`). Per iniettare messaggi, l'effetto deve modificare direttamente `status.agent.conversation_history`.
- L'hook su `BEFORE_FINAL_RESPONSE` riceve `assistant_msg` come `current_value` e `status.assistant_message`. Per pubblicare, l'effetto legge `status.assistant_message.content`.

### SystemMessage - Campo metadata

**File**: `src/domain/model/system_message.py`

```python
class SystemMessage(BaseModel):
    role: MessageRole = Field(default=MessageRole.SYSTEM)
    content: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

Il campo `metadata` e' **gia' presente** ed e' un dizionario libero. Lo useremo per taggare i messaggi iniettati dalla shared memory (chiave `"shared_memory_source"`) ed evitare duplicazioni.

### SubAgentWrapper - Come esegue i sub-agent

**File**: `src/domain/agent/subagent_wrapper.py`

**Modalita' stateless** (riga 72-96): copia l'agent con `copy.copy()` + copia la history:
```python
async def _execute_stateless(self, tool_call):
    agent_copy = copy.copy(self._agent)
    agent_copy.conversation_history = self._agent.conversation_history.copy()
    full_query = self._build_query(tool_call)
    response = await agent_copy.execute_query_async(query=full_query)
```

**Dettaglio critico**: `copy.copy()` e' shallow. Significa che:
- `agent_copy._hooks` punta allo **stesso dizionario** dell'originale -> gli hook registrati sull'originale **funzionano anche sulla copia**
- `agent_copy.memory_graph` sara' lo **stesso oggetto** -> la copia condivide il grafo (comportamento desiderato)
- `agent_copy.agent_id` sara' lo **stesso valore** -> la copia sa chi e' (comportamento desiderato)
- `agent_copy.conversation_history` e' una **copia nuova** -> l'iniezione avviene solo nella history della copia

**Modalita' stateful** (riga 98-119): usa l'agent originale con `asyncio.Lock()`:
```python
async def _execute_stateful(self, tool_call):
    full_query = self._build_query(tool_call)
    response = await self._agent.execute_query_async(query=full_query)
```

### AgentFactory - Stato attuale

**File**: `src/domain/agent/agent_factory.py`

**NON ha ancora** il metodo `with_memory_graph()`. Va aggiunto.

Metodi rilevanti:
- `register()` (riga 66-107): registra un `AgentSpec` con `cls`, `subagent_name`, `subagent_description`, `stateless`, `defaults`
- `create()` (riga 109-159): crea l'agent, chiama `_attach_subagents()` se servono sub-agent
- `_attach_from_registry()` (riga 177-204): crea il sub-agent e chiama `agent.register_agent()`

---

## 3. Comportamento Desiderato (esempio concreto)

### Scenario

Orchestratore "coordinator" con 3 sub-agent:
- `requirements`: estrae requisiti (nessun predecessore)
- `designer`: progetta architettura (dipende da requirements)
- `implementer`: implementa (dipende da requirements E designer)

### Grafo

```
requirements ──> designer
requirements ──> implementer
designer ────> implementer
```

### Flusso

**1) L'orchestratore parte.**

Nella sua `conversation_history`, dopo il system message originale, trova un **secondo SystemMessage** (iniettato dalla factory):

```
You are coordinating sub-agents with dependencies.

Dependency order (call upstream before downstream):
  requirements -> designer
  requirements -> implementer
  designer -> implementer

Recommended execution order: requirements, designer, implementer

Guideline: do not call an agent before its prerequisites have been executed.
```

L'LLM dell'orchestratore vede questo messaggio e decide di chiamare `requirements` per primo.

**2) `requirements` viene eseguito.**

- `BEFORE_LLM_CALL` hook scatta -> `graph.pull_for("requirements")` -> **lista vuota** (nessun predecessore) -> nessuna iniezione
- L'agent lavora normalmente, produce la risposta: *"Requisiti principali: autenticazione OAuth, database PostgreSQL, API REST"*
- `BEFORE_FINAL_RESPONSE` hook scatta -> `graph.publish("requirements", "Requisiti principali: ...")` -> il grafo salva l'output

**3) L'orchestratore chiama `designer`.**

- `BEFORE_LLM_CALL` hook scatta -> `graph.pull_for("designer")` -> trova l'edge `requirements -> designer` -> recupera l'output di `requirements`
- Hook crea un `SystemMessage`:
  ```
  Shared context from requirements:
  Requisiti principali: autenticazione OAuth, database PostgreSQL, API REST
  ```
- Il `SystemMessage` viene inserito in `conversation_history` alla posizione 1 (dopo il system message dell'agent, prima della HumanMessage con la query)
- L'LLM del designer vede il contesto di requirements **senza che l'orchestratore l'abbia passato**
- Il designer produce la sua risposta e la pubblica sul grafo

**4) L'orchestratore chiama `implementer`.**

- `BEFORE_LLM_CALL` hook scatta -> `graph.pull_for("implementer")` -> trova DUE edge: `requirements -> implementer` e `designer -> implementer`
- Hook inietta **due** SystemMessage nella history dell'implementer, uno per predecessore
- L'LLM dell'implementer vede il contesto di entrambi i predecessori

### Regole

- Si condivide **solo** il contenuto testuale di `AssistantMessage.content` della risposta finale
- La memoria viene recuperata (pull) solo al `BEFORE_LLM_CALL` e solo se non gia' iniettata
- L'orchestratore **non** deve fare relay di contesto manualmente
- Se un predecessore non ha ancora pubblicato (non e' ancora stato eseguito), il suo output viene semplicemente ignorato

---

## 4. Libreria Grafo: NetworkX (analisi Parlant)

### Perche' NetworkX

Il progetto **Parlant** (`C:\Users\GLoverde\PycharmProjects\Parlant\src\parlant`) usa NetworkX per gestire relazioni tra guideline e agent. Dopo analisi del codice, abbiamo identificato pattern direttamente applicabili.

**Dipendenza**: `networkx>=3.3` (da aggiungere a `pyproject.toml`)

### Come Parlant usa NetworkX

**File**: `parlant/core/relationships.py`

Parlant mantiene un dizionario di grafi diretti, uno per tipo di relazione:

```python
self._graphs: dict[RelationshipKind, networkx.DiGraph] = {}
```

Ogni grafo e' inizializzato con `strict=True` per prevenire loop:

```python
g = networkx.DiGraph()
g.graph["strict"] = True
```

I nodi sono ID di entity, gli edge hanno metadata (nel nostro caso: policy di propagazione).

### Pattern adottati da Parlant

**1. Lazy loading e cache in-memory**

Parlant costruisce i grafi on-demand e li tiene in cache. Noi facciamo lo stesso: il grafo viene costruito prima dell'esecuzione e resta in memoria.

**2. BFS per relazioni indirette**

Parlant usa `networkx.bfs_edges()` per trovare tutte le relazioni transitive:

```python
for _, ancestor in networkx.bfs_edges(graph.reverse(), source_id):
    # elabora relazioni indirette
```

Noi adottiamo questo per `pull_for_indirect()`: se A -> B -> C, C puo' ottenere anche l'output di A (non solo di B).

**3. Topological sort per ordinamento**

Parlant usa topological sort per determinare l'ordine di valutazione dei nodi. Noi lo usiamo in `get_topological_order()` per suggerire all'orchestratore l'ordine di esecuzione dei sub-agent.

**4. Tipo di relazione come enum**

Parlant definisce `RelationshipKind` (ENTAILMENT, PRIORITY, DEPENDENCY, ecc.). Noi definiamo `PropagationPolicy` (per ora solo `FINAL_RESPONSE_ONLY`, espandibile in futuro).

---

## 5. Design: SharedMemoryGraph

### Nuovo file

```
src/domain/agent/shared_memory.py
```

### Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Tuple
import asyncio
import networkx as nx


class PropagationPolicy(str, Enum):
    """Come il contenuto viene condiviso lungo un edge.

    Per ora solo FINAL_RESPONSE_ONLY.
    In futuro: TOOL_RESULTS, FULL_HISTORY, ecc.
    """
    FINAL_RESPONSE_ONLY = "final_response_only"


@dataclass
class MemoryItem:
    """Un pezzo di contesto condiviso recuperato dal grafo.

    Viene restituito da pull_for() e usato dal hook di iniezione
    per creare il SystemMessage da inserire nella history.
    """
    source_id: str                    # nome dell'agent sorgente
    content: str                      # testo della risposta finale
    timestamp: datetime               # quando e' stato pubblicato
    policy: PropagationPolicy         # policy dell'edge


@dataclass
class NodeData:
    """Dati memorizzati per ogni nodo (agent) nel grafo.

    last_final: ultimo contenuto pubblicato (None se l'agent non ha ancora risposto)
    timestamp: quando e' stato pubblicato
    metadata: dati extra opzionali (future-proofing)
    """
    last_final: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
```

### Classe SharedMemoryGraph

```python
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SharedMemoryGraph:
    """Grafo diretto di dipendenze tra agent con memoria condivisa.

    Backed da NetworkX DiGraph. Pattern ispirato a Parlant
    (parlant/core/relationships.py): un'istanza DiGraph, operazioni
    lazy, BFS per relazioni indirette.

    Thread-safety:
    - publish() usa asyncio.Lock (scrittura concorrente)
    - pull_for() e' read-only su snapshot immutabili (no lock)
    - add_agent()/add_edge() sono chiamati in fase di setup (no concorrenza)
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ─── Costruzione grafo ─────────────────────────────────────────────────

    def add_agent(self, node_id: str) -> None:
        """Aggiunge un nodo agent al grafo. Idempotente.

        Se il nodo esiste gia', non fa nulla.
        Viene chiamato implicitamente da add_edge(), quindi e' opzionale.
        """
        if not self._graph.has_node(node_id):
            self._graph.add_node(node_id, data=NodeData())
            logger.debug(f"SharedMemoryGraph: nodo '{node_id}' aggiunto")

    def add_edge(
        self,
        src: str,
        dst: str,
        policy: PropagationPolicy = PropagationPolicy.FINAL_RESPONSE_ONLY,
    ) -> None:
        """Aggiunge un edge di dipendenza: l'output di src fluisce verso dst.

        Crea automaticamente i nodi se non esistono.
        Significato: quando dst viene eseguito, ricevera' l'output di src.

        Args:
            src: nome dell'agent sorgente (chi produce l'output)
            dst: nome dell'agent destinazione (chi riceve l'output)
            policy: come il contenuto viene condiviso (default: solo risposta finale)
        """
        self.add_agent(src)
        self.add_agent(dst)
        self._graph.add_edge(src, dst, policy=policy)
        logger.debug(f"SharedMemoryGraph: edge '{src}' -> '{dst}' aggiunto (policy={policy.value})")

    def has_node(self, node_id: str) -> bool:
        """Controlla se un nodo esiste nel grafo."""
        return self._graph.has_node(node_id)

    # ─── Publish (scrittura) ───────────────────────────────────────────────

    async def publish(self, node_id: str, content: str, metadata: dict | None = None) -> None:
        """Salva la risposta finale di un agent nel grafo.

        Chiamato dall'hook BEFORE_FINAL_RESPONSE dopo che l'agent ha prodotto
        la sua risposta. Thread-safe tramite asyncio.Lock.

        Args:
            node_id: nome dell'agent che ha prodotto la risposta
            content: testo della risposta finale (AssistantMessage.content)
            metadata: dati extra opzionali
        """
        async with self._lock:
            self.add_agent(node_id)
            node_data: NodeData = self._graph.nodes[node_id]["data"]
            node_data.last_final = content
            node_data.timestamp = datetime.now()
            if metadata:
                node_data.metadata.update(metadata)
            logger.info(f"SharedMemoryGraph: '{node_id}' ha pubblicato ({len(content)} chars)")

    # ─── Pull (lettura) ───────────────────────────────────────────────────

    def pull_for(self, node_id: str) -> List[MemoryItem]:
        """Recupera la memoria dai predecessori DIRETTI che hanno pubblicato.

        Chiamato dall'hook BEFORE_LLM_CALL prima della prima chiamata LLM.
        Restituisce solo i predecessori che hanno gia' un last_final (cioe'
        che sono gia' stati eseguiti e hanno pubblicato).

        Esempio: se il grafo ha A->C e B->C, e solo A ha pubblicato,
        restituisce solo il MemoryItem di A.

        Args:
            node_id: nome dell'agent che sta per essere eseguito

        Returns:
            Lista di MemoryItem, uno per ogni predecessore che ha pubblicato.
            Lista vuota se il nodo non esiste o non ha predecessori.
        """
        if not self._graph.has_node(node_id):
            return []

        items = []
        for pred in self._graph.predecessors(node_id):
            node_data: NodeData = self._graph.nodes[pred]["data"]
            if node_data.last_final is None:
                continue
            edge_data = self._graph.edges[pred, node_id]
            items.append(MemoryItem(
                source_id=pred,
                content=node_data.last_final,
                timestamp=node_data.timestamp,
                policy=edge_data.get("policy", PropagationPolicy.FINAL_RESPONSE_ONLY),
            ))
        return items

    def pull_for_indirect(self, node_id: str) -> List[MemoryItem]:
        """Recupera la memoria da TUTTI i predecessori transitivi (BFS).

        Ispirato a Parlant che usa networkx.bfs_edges() per risolvere
        relazioni indirette. Se A -> B -> C, chiamando
        pull_for_indirect("C") si ottiene sia A che B.

        Usa graph.reverse() + BFS per risalire l'albero dei predecessori.

        Args:
            node_id: nome dell'agent

        Returns:
            Lista di MemoryItem da tutti gli antenati che hanno pubblicato.
        """
        if not self._graph.has_node(node_id):
            return []

        items = []
        reversed_graph = self._graph.reverse()
        for _, ancestor in nx.bfs_edges(reversed_graph, node_id):
            node_data: NodeData = self._graph.nodes[ancestor]["data"]
            if node_data.last_final is None:
                continue
            items.append(MemoryItem(
                source_id=ancestor,
                content=node_data.last_final,
                timestamp=node_data.timestamp,
                policy=PropagationPolicy.FINAL_RESPONSE_ONLY,
            ))
        return items

    # ─── Query helper ──────────────────────────────────────────────────────

    def get_edges_for_nodes(self, node_ids: list[str]) -> List[Tuple[str, str]]:
        """Restituisce tutti gli edge dove ENTRAMBI gli endpoint sono in node_ids.

        Usato dalla factory per generare il messaggio awareness
        dell'orchestratore. Filtra solo gli edge rilevanti per i
        sub-agent registrati.

        Args:
            node_ids: lista dei nomi dei sub-agent

        Returns:
            Lista di tuple (src, dst) con gli edge filtrati.
        """
        node_set = set(node_ids)
        return [
            (u, v) for u, v in self._graph.edges()
            if u in node_set and v in node_set
        ]

    def get_topological_order(self, node_ids: list[str] | None = None) -> list[str]:
        """Restituisce i nodi in ordine topologico (rispettando le dipendenze).

        Usato dalla factory per suggerire all'orchestratore l'ordine
        di esecuzione. Se ci sono cicli, lancia NetworkXUnfeasible.

        Args:
            node_ids: se specificato, ordina solo questi nodi (subgraph).
                      Se None, ordina tutti i nodi del grafo.

        Returns:
            Lista di nomi agent in ordine topologico.

        Raises:
            networkx.NetworkXUnfeasible: se il grafo contiene cicli.
        """
        if node_ids:
            subgraph = self._graph.subgraph(node_ids)
            return list(nx.topological_sort(subgraph))
        return list(nx.topological_sort(self._graph))

    def clear_published_data(self) -> None:
        """Resetta tutti i dati pubblicati, mantiene la struttura del grafo.

        Utile per rieseguire una pipeline dallo stesso orchestratore
        senza ricreare il grafo.
        """
        for node_id in self._graph.nodes():
            self._graph.nodes[node_id]["data"] = NodeData()
        logger.info("SharedMemoryGraph: tutti i dati pubblicati resettati")
```

---

## 6. Strategia di Iniezione (BEFORE_LLM_CALL)

### Dove si aggancia

Evento: `AgentEvent.BEFORE_LLM_CALL`

Questo evento scatta in `BaseAgent._async_execute_query()` (riga 230):
```python
for iteration in range(1, self.max_iterations + 1):
    outcome = await self._run_hooks(AgentEvent.BEFORE_LLM_CALL, iteration=iteration)
```

L'evento scatta ad **ogni iterazione** del loop. Il `current_value` e' `None` (da contract). L'effetto deve modificare direttamente `status.agent.conversation_history`.

### Comportamento

1. L'hook verifica che `self.memory_graph` e `self.agent_id` esistano. Se non esistono, non fa nulla (zero overhead per agent senza shared memory).
2. Chiama `self.memory_graph.pull_for(self.agent_id)` per ottenere i `MemoryItem` dai predecessori diretti.
3. Per ogni `MemoryItem`, verifica che non sia gia' stato iniettato controllando `metadata["shared_memory_source"]` nei messaggi della history.
4. Se non e' duplicato, crea un `SystemMessage` con:
   - `content`: `"Shared context from {source_id}:\n{content}"`
   - `metadata`: `{"shared_memory": True, "shared_memory_source": source_id}`
5. Inserisce il `SystemMessage` alla posizione 1 nella `conversation_history` (dopo il system message dell'agent, prima di tutto il resto).

### Deduplicazione

Poiche' `BEFORE_LLM_CALL` scatta ad ogni iterazione del loop, e un agent puo' avere piu' iterazioni (tool calls, retry, ecc.), l'iniezione deve avvenire **una sola volta**. La deduplicazione si basa sul campo `metadata["shared_memory_source"]`:

```python
# Controlla se gia' iniettato
already_injected = any(
    isinstance(msg, SystemMessage)
    and msg.metadata.get("shared_memory_source") == mem.source_id
    for msg in self.conversation_history
)
```

Se il messaggio e' gia' presente, salta l'iniezione per quella sorgente.

### Perche' SystemMessage e non HumanMessage

- `SystemMessage` ha un peso semantico piu' forte per l'LLM: viene trattato come istruzione/contesto, non come query dell'utente
- Evita confusione con il messaggio dell'utente reale (la query)
- E' gia' la posizione naturale per il contesto (system prompt e' il primo messaggio)

### Posizione di inserimento

Posizione 1 nella `conversation_history`:
```
[0] SystemMessage originale dell'agent ("You are a designer...")
[1] SystemMessage shared memory da requirements  <-- iniettato qui
[2] SystemMessage shared memory da altro_agent   <-- iniettato qui
[3] HumanMessage con la query
...
```

---

## 7. Strategia di Pubblicazione (BEFORE_FINAL_RESPONSE)

### Dove si aggancia

Evento: `AgentEvent.BEFORE_FINAL_RESPONSE`

Questo evento scatta in `BaseAgent._async_execute_query()` in **tre punti** (riga 272, 297, 320), tutti con lo stesso pattern:
```python
outcome = await self._run_hooks(
    AgentEvent.BEFORE_FINAL_RESPONSE,
    current_value=assistant_msg,
    iteration=iteration,
    assistant_message=assistant_msg
)
```

Il `current_value` e' l'`AssistantMessage` finale. `status.assistant_message` contiene lo stesso oggetto.

### Comportamento

1. L'hook verifica che `self.memory_graph` e `self.agent_id` esistano.
2. Legge `status.assistant_message.content` (il testo della risposta finale).
3. Se il contenuto non e' vuoto, chiama `await self.memory_graph.publish(self.agent_id, content)`.
4. `publish()` e' async perche' usa `asyncio.Lock` internamente.

### Cosa viene pubblicato

**Solo** il campo `content` di `AssistantMessage`. Questo significa:
- NO tool_calls (non vengono propagati)
- NO tool_results (non vengono propagati)
- NO usage/metadata (non vengono propagati)
- Solo il testo leggibile della risposta finale dell'LLM

### Timing

La pubblicazione avviene **prima** che `_build_final_response()` costruisca l'`AssistantResponse`. Questo e' corretto perche':
1. Il contenuto e' gia' disponibile in `assistant_msg.content`
2. L'effetto e' asincrono e viene awaited dal sistema di hook
3. Non modifica il valore di ritorno (la decisione resta `CONTINUE`)

---

## 8. Messaggio Awareness per Orchestratore

### Requisito

Quando un agent orchestratore viene creato con sub-agent che hanno dipendenze nel grafo, deve ricevere un **system message aggiuntivo** che descrive l'ordine delle dipendenze.

Il messaggio:
- **NON** elenca i nomi dei sub-agent (il function calling li espone gia')
- **NON** descrive cosa fa ogni sub-agent (la description nel tool schema lo fa gia')
- Descrive **solo** le dipendenze e l'ordine consigliato
- Include l'output di `get_topological_order()` per chiarezza

### Contenuto esempio

```
You are coordinating sub-agents with dependencies.

Dependency order (call upstream before downstream):
  requirements -> designer
  requirements -> implementer
  designer -> implementer

Recommended execution order: requirements, designer, implementer

Guideline: do not call an agent before its prerequisites have been executed.
```

### Dove viene inserito

Il messaggio viene inserito dalla `AgentFactory` nel metodo `create()`, **dopo** aver registrato tutti i sub-agent. Viene inserito alla posizione 1 nella `conversation_history` dell'orchestratore:

```
[0] SystemMessage originale dell'orchestratore ("You coordinate tasks...")
[1] SystemMessage awareness dipendenze  <-- inserito dalla factory
[2] ... (messaggi futuri)
```

### Quando NON viene inserito

- Se il grafo non ha edge tra i sub-agent registrati
- Se il grafo non e' stato configurato sulla factory
- Se non ci sono sub-agent

---

## 9. Integrazione con AgentFactory

### Modifica: nuovo attributo e metodo

**File**: `src/domain/agent/agent_factory.py`

Aggiungere all'`__init__`:
```python
def __init__(self, global_defaults=None):
    self._global_defaults = global_defaults or {}
    self._registry: Dict[str, AgentSpec] = {}
    self._memory_graph: Optional['SharedMemoryGraph'] = None   # <-- NUOVO
```

Aggiungere il metodo:
```python
def with_memory_graph(self, graph: 'SharedMemoryGraph') -> 'AgentFactory':
    """Configura il grafo di memoria condivisa.

    Tutti gli agent creati da questa factory riceveranno
    un riferimento allo STESSO grafo.

    Args:
        graph: istanza di SharedMemoryGraph gia' configurata con nodi e edge.

    Returns:
        self per method chaining.
    """
    self._memory_graph = graph
    return self
```

### Modifica: `create()` attacca il grafo

Nel metodo `create()` (riga 109), **dopo** la creazione dell'agent e **prima** del return, aggiungere:

```python
def create(self, name, *, subagents=None, subagent_config=None, **overrides):
    # ... codice esistente che crea l'agent ...
    agent = spec.cls(**config)

    # NUOVO: attacca il memory graph all'agent principale
    if self._memory_graph:
        agent.memory_graph = self._memory_graph
        agent.agent_id = name

    if subagents is not None:
        self._attach_subagents(agent, subagents, subagent_config or {})

        # NUOVO: inietta il messaggio awareness se ci sono dipendenze
        if self._memory_graph:
            self._inject_dependency_awareness(agent, subagents)

    return agent
```

### Modifica: `_attach_from_registry()` attacca il grafo ai sub-agent

Nel metodo `_attach_from_registry()` (riga 177), **dopo** la creazione del sub-agent e **prima** della registrazione:

```python
def _attach_from_registry(self, agent, sub_name, extra_config):
    # ... codice esistente ...
    config = {**self._global_defaults, **sub_spec.defaults, **extra_config}
    sub_agent = sub_spec.cls(**config)

    # NUOVO: attacca il memory graph al sub-agent
    if self._memory_graph:
        sub_agent.memory_graph = self._memory_graph
        sub_agent.agent_id = sub_name

    agent.register_agent(
        sub_agent,
        name=sub_spec.subagent_name or sub_name,
        description=sub_spec.subagent_description,
        stateless=sub_spec.stateless,
    )
```

### Nuovo metodo: `_inject_dependency_awareness()`

```python
def _inject_dependency_awareness(
        self,
        agent: 'BaseAgent',
        subagent_refs: List[Union[str, 'BaseAgent']],
) -> None:
    """Inietta il system message di awareness delle dipendenze nell'orchestratore.

    Viene chiamato da create() dopo aver registrato tutti i sub-agent.
    Il messaggio descrive solo le dipendenze e l'ordine consigliato,
    NON elenca i sub-agent (gia' esposti dal function calling).

    Args:
        agent: l'agent orchestratore
        subagent_refs: lista di nomi dei sub-agent
    """
    from src.core.model.system_message import SystemMessage

    # Estrai solo i nomi stringa
    subagent_names = [s for s in subagent_refs if isinstance(s, str)]
    if not subagent_names:
        return

    edges = self._memory_graph.get_edges_for_nodes(subagent_names)
    if not edges:
        logger.debug("AgentFactory: nessun edge tra i sub-agent, skip awareness message")
        return

    # Calcola ordine topologico (puo' fallire se ci sono cicli)
    try:
        order = self._memory_graph.get_topological_order(subagent_names)
        order_str = f"\nRecommended execution order: {', '.join(order)}"
    except Exception as e:
        logger.warning(f"AgentFactory: impossibile calcolare ordine topologico: {e}")
        order_str = ""

    lines = [
        "You are coordinating sub-agents with dependencies.",
        "",
        "Dependency order (call upstream before downstream):",
    ]
    for src, dst in edges:
        lines.append(f"  {src} -> {dst}")
    if order_str:
        lines.append(order_str)
    lines.append("")
    lines.append("Guideline: do not call an agent before its prerequisites have been executed.")

    msg = SystemMessage(
        content="\n".join(lines),
        metadata={"orchestrator_awareness": True}
    )
    agent.conversation_history.insert(1, msg)
    logger.info("AgentFactory: awareness message iniettato nell'orchestratore")
```

### Validazione nomi sub-agent nel grafo

Aggiungere in `_attach_from_registry()`, prima della creazione:

```python
if self._memory_graph and not self._memory_graph.has_node(sub_name):
    logger.warning(
        f"Sub-agent '{sub_name}' non trovato nel memory graph. "
        "Non partecipera' alla shared memory."
    )
```

Questo e' un **warning**, non un errore: non tutti i sub-agent devono essere nel grafo.

---

## 10. Modifiche a BaseAgent

### Nuovi attributi

**File**: `src/domain/agent/base_agent.py`

Aggiungere nel costruttore, dopo le inizializzazioni esistenti (dopo riga 46):

```python
def __init__(self, ...):
    # ... codice esistente ...

    # Shared memory (impostati dalla factory o manualmente, None se non usati)
    self.memory_graph: Optional['SharedMemoryGraph'] = None
    self.agent_id: Optional[str] = None

    # Registra gli hook per la shared memory
    # (controllano a runtime se memory_graph esiste)
    self._register_memory_hooks()
```

Aggiungere l'import in TYPE_CHECKING (riga 1-18):

```python
if TYPE_CHECKING:
    from src.core.agent.shared_memory import SharedMemoryGraph
```

### Nuovi metodi

```python
def _register_memory_hooks(self) -> None:
    """Registra gli hook per iniezione e pubblicazione della shared memory.

    Gli hook controllano a RUNTIME se memory_graph esiste.
    Se non esiste (agent senza shared memory), non fanno nulla.
    Questo garantisce zero overhead per agent normali.

    Due hook vengono registrati:
    1. BEFORE_LLM_CALL: inietta il contesto dai predecessori
    2. BEFORE_FINAL_RESPONSE: pubblica la propria risposta sul grafo
    """
    # Hook di iniezione: prima di ogni chiamata LLM
    self.on(AgentEvent.BEFORE_LLM_CALL).handle(
        decision=HookDecision.CONTINUE,
        effects=[self._inject_shared_memory],
    )

    # Hook di pubblicazione: prima della risposta finale
    self.on(AgentEvent.BEFORE_FINAL_RESPONSE).handle(
        decision=HookDecision.CONTINUE,
        effects=[self._publish_to_memory],
    )

def _inject_shared_memory(self, status: AgentStatus) -> None:
    """Recupera e inietta le memorie dai predecessori nel grafo.

    Chiamato dall'hook BEFORE_LLM_CALL ad ogni iterazione.
    Usa il campo metadata["shared_memory_source"] per evitare
    duplicazioni (il messaggio viene iniettato solo una volta
    per sorgente, anche se l'agent ha multiple iterazioni).

    Il SystemMessage viene inserito alla posizione 1 della
    conversation_history (dopo il system prompt, prima della query).

    Args:
        status: AgentStatus con riferimento all'agent e alla iteration corrente.
    """
    if not self.memory_graph or not self.agent_id:
        return

    memories = self.memory_graph.pull_for(self.agent_id)
    if not memories:
        return

    history = status.agent.conversation_history

    for mem in memories:
        # Deduplicazione: controlla se gia' iniettato
        already_injected = any(
            isinstance(msg, SystemMessage)
            and msg.metadata.get("shared_memory_source") == mem.source_id
            for msg in history
        )
        if already_injected:
            continue

        msg = SystemMessage(
            content=f"Shared context from {mem.source_id}:\n{mem.content}",
            metadata={
                "shared_memory": True,
                "shared_memory_source": mem.source_id,
            }
        )
        # Inserisci dopo il system message, prima della query
        history.insert(1, msg)
        logger.debug(
            f"SharedMemory: iniettato contesto da '{mem.source_id}' "
            f"in '{self.agent_id}' ({len(mem.content)} chars)"
        )

async def _publish_to_memory(self, status: AgentStatus) -> None:
    """Pubblica la risposta finale dell'agent sul grafo di memoria.

    Chiamato dall'hook BEFORE_FINAL_RESPONSE.
    L'effetto e' async perche' graph.publish() usa asyncio.Lock.
    Il sistema di hook di Obelix supporta effetti async nativamente
    (hooks.py riga 181-183: controlla asyncio.iscoroutine e fa await).

    Pubblica solo il campo content di AssistantMessage.
    Non pubblica tool_calls, tool_results, usage o metadata.

    Args:
        status: AgentStatus con assistant_message contenente la risposta.
    """
    if not self.memory_graph or not self.agent_id:
        return

    if not status.assistant_message or not status.assistant_message.content:
        return

    await self.memory_graph.publish(self.agent_id, status.assistant_message.content)
    logger.debug(
        f"SharedMemory: '{self.agent_id}' ha pubblicato "
        f"({len(status.assistant_message.content)} chars)"
    )
```

### Perche' gli hook sono registrati SEMPRE

Gli hook vengono registrati nel costruttore per TUTTI gli agent, anche quelli senza shared memory. Questo perche':
1. La factory imposta `memory_graph` e `agent_id` **dopo** la creazione dell'agent
2. Gli hook controllano `self.memory_graph` a **runtime** (if-guard nel primo riga)
3. Se `memory_graph` e' `None`, gli effetti ritornano immediatamente senza fare nulla
4. Costo: una chiamata di funzione con early-return per iterazione (trascurabile)

---

## 11. Flusso Runtime Completo

### Fase di setup (prima dell'esecuzione)

```
1. Utente crea SharedMemoryGraph e definisce le dipendenze
2. Utente crea AgentFactory e chiama with_memory_graph(graph)
3. Utente registra gli agent nella factory
4. Utente chiama factory.create("coordinator", subagents=[...])
   4a. Factory crea il coordinator
   4b. Factory imposta coordinator.memory_graph = graph
   4c. Factory imposta coordinator.agent_id = "coordinator"
   4d. Per ogni sub-agent:
       - Factory crea il sub-agent
       - Factory imposta sub_agent.memory_graph = graph (STESSO oggetto)
       - Factory imposta sub_agent.agent_id = sub_name
       - Factory chiama coordinator.register_agent(sub_agent, ...)
       - register_agent() wrappa in SubAgentWrapper e lo aggiunge ai tool
   4e. Factory inietta il messaggio awareness nell'orchestratore
5. Tutti gli agent condividono lo STESSO SharedMemoryGraph
```

### Fase di esecuzione (runtime)

```
coordinator.execute_query_async("Build auth system")
  |
  +--> [conversation_history]:
  |      [0] SystemMessage("You coordinate tasks...")
  |      [1] SystemMessage("You are coordinating sub-agents with dependencies...")
  |      [2] HumanMessage("Build auth system")
  |
  +--> BEFORE_LLM_CALL hook scatta su coordinator:
  |      _inject_shared_memory(): memory_graph esiste,
  |      ma coordinator non ha predecessori nel grafo -> noop
  |
  +--> provider.invoke() -> LLM decide di chiamare "requirements"
  |
  +--> SubAgentWrapper("requirements").execute(tool_call)
  |      |
  |      +--> [stateless] copy.copy(agent) + history.copy()
  |      |
  |      +--> agent_copy.execute_query_async("Extract requirements for auth...")
  |             |
  |             +--> BEFORE_LLM_CALL hook scatta su agent_copy:
  |             |      _inject_shared_memory():
  |             |        graph.pull_for("requirements") -> [] (nessun predecessore)
  |             |        -> noop
  |             |
  |             +--> provider.invoke() -> risposta: "Requisiti: OAuth, PostgreSQL..."
  |             |
  |             +--> BEFORE_FINAL_RESPONSE hook scatta:
  |             |      _publish_to_memory():
  |             |        await graph.publish("requirements", "Requisiti: OAuth...")
  |             |        -> grafo aggiornato
  |             |
  |             +--> return AssistantResponse
  |
  |      +--> return ToolResult(result="Requisiti: OAuth...")
  |
  +--> Orchestratore riceve il ToolResult, continua il loop
  |
  +--> BEFORE_LLM_CALL hook (iteration 2) su coordinator:
  |      _inject_shared_memory(): nessun predecessore -> noop
  |
  +--> provider.invoke() -> LLM decide di chiamare "designer"
  |
  +--> SubAgentWrapper("designer").execute(tool_call)
  |      |
  |      +--> [stateless] copy.copy(agent) + history.copy()
  |      |
  |      +--> agent_copy.execute_query_async("Design architecture...")
  |             |
  |             +--> BEFORE_LLM_CALL hook scatta su agent_copy:
  |             |      _inject_shared_memory():
  |             |        graph.pull_for("designer") ->
  |             |          [MemoryItem(source="requirements", content="Requisiti: OAuth...")]
  |             |        Controlla deduplicazione -> non presente
  |             |        Crea SystemMessage("Shared context from requirements:\n...")
  |             |        Inserisce in history alla posizione 1
  |             |
  |             +--> [conversation_history]:
  |             |      [0] SystemMessage("You are a designer...")
  |             |      [1] SystemMessage("Shared context from requirements:\n...")  <-- INIETTATO
  |             |      [2] HumanMessage("Design architecture...")
  |             |
  |             +--> provider.invoke() -> LLM vede il contesto di requirements
  |             |
  |             +--> BEFORE_FINAL_RESPONSE hook:
  |             |      _publish_to_memory():
  |             |        await graph.publish("designer", "Architecture: microservices...")
  |             |
  |             +--> return AssistantResponse
  |
  +--> [... orchestratore chiama "implementer" ...]
  |
  +--> SubAgentWrapper("implementer").execute(tool_call)
         |
         +--> BEFORE_LLM_CALL hook:
         |      graph.pull_for("implementer") ->
         |        [MemoryItem(source="requirements", ...), MemoryItem(source="designer", ...)]
         |      Inietta DUE SystemMessage
         |
         +--> [conversation_history]:
         |      [0] SystemMessage("You are an implementer...")
         |      [1] SystemMessage("Shared context from requirements:\n...")
         |      [2] SystemMessage("Shared context from designer:\n...")
         |      [3] HumanMessage("Implement the solution...")
         |
         +--> LLM vede contesto da ENTRAMBI i predecessori
```

---

## 12. Comportamento con SubAgentWrapper (stateless/stateful)

### Modalita' stateless (default raccomandato per shared memory)

Quando `stateless=True` (o `stateless=False` con la semantica attuale che copia):

1. `SubAgentWrapper._execute_stateless()` fa `copy.copy(self._agent)`
2. La copia ha gli **stessi** riferimenti a:
   - `_hooks`: stesso dizionario -> gli hook registrati funzionano sulla copia
   - `memory_graph`: stesso oggetto -> la copia condivide il grafo
   - `agent_id`: stessa stringa -> la copia sa chi e'
3. La copia ha una **nuova** `conversation_history` (`.copy()`)
4. L'hook `BEFORE_LLM_CALL` scatta sulla copia:
   - `status.agent` e' la copia
   - `status.agent.conversation_history` e' la history della copia
   - L'iniezione avviene nella history della copia (isolata)
5. L'hook `BEFORE_FINAL_RESPONSE` scatta sulla copia:
   - `status.assistant_message.content` contiene la risposta
   - `self.memory_graph.publish()` aggiorna il grafo condiviso
6. La risposta e' visibile a tutti gli agent successivi tramite il grafo

### Modalita' stateful

Quando `stateless=False` (default in `register_agent()`):

1. `SubAgentWrapper._execute_stateful()` usa l'agent originale con `asyncio.Lock()`
2. L'agent originale ha `memory_graph` e `agent_id` impostati
3. L'iniezione avviene nella history dell'agent originale
4. La pubblicazione aggiorna il grafo condiviso
5. **Attenzione**: la history dell'agent persiste tra le chiamate, quindi:
   - I messaggi di shared memory iniettati alla prima chiamata restano nella history
   - La deduplicazione via metadata previene re-iniezione
   - Ma il contesto potrebbe diventare stale (il predecessore potrebbe aver aggiornato)

**Raccomandazione**: usare `stateless=True` per sub-agent con shared memory.

---

## 13. Esempio Completo End-to-End

```python
from src.core.agent import BaseAgent
from src.core.agent.agent_factory import AgentFactory
from src.core.agent.shared_memory import SharedMemoryGraph
from src.core.tool.tool_decorator import tool
from src.core.tool.tool_base import ToolBase
from pydantic import Field


# ─── Definizione agent ─────────────────────────────────────────────────

class RequirementsAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a requirements analyst. Extract and list key requirements.",
            **kwargs
        )


class DesignerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a system architect. Design the architecture based on requirements.",
            **kwargs
        )


class ImplementerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a developer. Implement the solution based on design and requirements.",
            **kwargs
        )


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You coordinate a team of specialists to build software systems.",
            **kwargs
        )


# ─── Setup ──────────────────────────────────────────────────────────────

# 1. Crea il grafo di dipendenze
graph = SharedMemoryGraph()
graph.add_edge("requirements", "designer")  # designer dipende da requirements
graph.add_edge("requirements", "implementer")  # implementer dipende da requirements
graph.add_edge("designer", "implementer")  # implementer dipende anche da designer

# 2. Crea la factory e attacca il grafo
factory = AgentFactory()
factory.with_memory_graph(graph)

# 3. Registra gli agent
factory.register(
    "requirements",
    RequirementsAgent,
    subagent_description="Extracts and analyzes project requirements",
    stateless=True,  # raccomandato per shared memory
)
factory.register(
    "designer",
    DesignerAgent,
    subagent_description="Designs system architecture based on requirements",
    stateless=True,
)
factory.register(
    "implementer",
    ImplementerAgent,
    subagent_description="Implements the solution based on design and requirements",
    stateless=True,
)
factory.register("coordinator", CoordinatorAgent)

# 4. Crea l'orchestratore con sub-agent
coordinator = factory.create(
    "coordinator",
    subagents=["requirements", "designer", "implementer"]
)

# A questo punto:
# - coordinator.memory_graph = graph
# - coordinator.agent_id = "coordinator"
# - Ogni sub-agent (dentro i SubAgentWrapper) ha memory_graph e agent_id impostati
# - coordinator.conversation_history contiene il messaggio awareness delle dipendenze

# ─── Esecuzione ─────────────────────────────────────────────────────────

response = await coordinator.execute_query_async(
    "Build a user authentication system with OAuth support"
)

# Flusso atteso:
# 1. Orchestratore vede awareness message -> chiama "requirements" per primo
# 2. requirements produce output -> pubblicato sul grafo
# 3. Orchestratore chiama "designer"
# 4. designer riceve automaticamente output di requirements via SystemMessage
# 5. designer produce output -> pubblicato sul grafo
# 6. Orchestratore chiama "implementer"
# 7. implementer riceve automaticamente output di requirements E designer
# 8. implementer produce output -> orchestratore costruisce risposta finale

print(response.content)
```

---

## 14. Step di Implementazione

### Fase 1: Core (in-memory, nessuna persistence)

**Step 1.1: Creare `src/domain/agent/shared_memory.py`**
- Classi: `PropagationPolicy`, `MemoryItem`, `NodeData`, `SharedMemoryGraph`
- Metodi: `add_agent()`, `add_edge()`, `publish()`, `pull_for()`, `pull_for_indirect()`, `get_edges_for_nodes()`, `get_topological_order()`, `clear_published_data()`
- Dipendenza: `networkx>=3.3`

**Step 1.2: Aggiungere dipendenza NetworkX**
- Aggiungere `networkx>=3.3` a `pyproject.toml` (o `requirements.txt`)

**Step 1.3: Modificare `BaseAgent`** (`src/domain/agent/base_agent.py`)
- Aggiungere attributi: `self.memory_graph = None`, `self.agent_id = None`
- Aggiungere metodi: `_register_memory_hooks()`, `_inject_shared_memory()`, `_publish_to_memory()`
- Chiamare `_register_memory_hooks()` nel costruttore

**Step 1.4: Modificare `AgentFactory`** (`src/domain/agent/agent_factory.py`)
- Aggiungere attributo: `self._memory_graph = None`
- Aggiungere metodo: `with_memory_graph()`
- Modificare `create()`: attacca grafo all'agent principale + inietta awareness
- Modificare `_attach_from_registry()`: attacca grafo ai sub-agent
- Aggiungere metodo: `_inject_dependency_awareness()`

**Step 1.5: Aggiornare export** (`src/domain/agent/__init__.py`)
- Aggiungere `SharedMemoryGraph` agli export

**Step 1.6: Test unitari**
- Test `SharedMemoryGraph` isolato: add/edge/publish/pull/topo sort/clear
- Test `pull_for()` con predecessori che non hanno pubblicato (devono essere ignorati)
- Test `pull_for_indirect()` con catene transitive
- Test deduplicazione iniezione (BEFORE_LLM_CALL chiamato 2 volte)
- Test pubblicazione asincrona (multiple publish concorrenti)
- Test factory: awareness message generato correttamente
- Test factory: sub-agent ricevono il grafo
- Test factory: warning per sub-agent non nel grafo

**Step 1.7: Test di integrazione**
- Pipeline completa: 3 agent con dipendenze, verifica che il contesto fluisca
- Verifica che agent senza shared memory non siano impattati (zero overhead)
- Verifica compatibilita' con `SubAgentWrapper` stateless e stateful

### Fase 2: Miglioramenti

**Step 2.1: Validazione cicli**
- Aggiungere controllo su `add_edge()`: se esiste gia' un path da `dst` a `src`, rifiuta l'edge (previene cicli)
- Usare `nx.has_path(self._graph, dst, src)` per la verifica

**Step 2.2: Policy multiple**
- Estendere `PropagationPolicy` con `TOOL_RESULTS`, `FULL_HISTORY`
- Modificare `pull_for()` per restituire dati diversi a seconda della policy

**Step 2.3: Memory TTL**
- Aggiungere TTL opzionale ai nodi: dopo N secondi, il contenuto pubblicato scade
- Utile per sessioni lunghe dove il contesto diventa stale

**Step 2.4: Visualizzazione grafo**
- Metodo `to_dict()` per serializzazione
- Opzionale: `nx.drawing` per debug grafico

### Fase 3: Persistence (futuro)

**Step 3.1: Persist grafo su document store**
- Salvare struttura nodi/edge su database
- Caricare al riavvio

**Step 3.2: Persist memoria pubblicata**
- Salvare `last_final` per agent su database
- Supporto sessioni multi-esecuzione

**Step 3.3: Vector search (ispirato a Parlant)**
- Parlant usa un pattern hybrid: Document DB per struttura + Vector DB per semantic search
- Possibilita': dato il contesto di un agent, cercare semanticamente le memorie piu' rilevanti

---

## 15. Decisioni Aperte

| Decisione | Opzioni | Raccomandazione |
|-----------|---------|-----------------|
| `pull_for()` diretto vs `pull_for_indirect()` BFS come default | A: solo predecessori diretti; B: tutti gli antenati transitivi | **A** (diretto) per semplicita' iniziale. `pull_for_indirect()` disponibile se serve |
| Tipo di messaggio per l'iniezione | A: SystemMessage; B: HumanMessage | **A** (SystemMessage) per peso semantico piu' forte |
| Validazione cicli su `add_edge()` | A: eager (al momento dell'aggiunta); B: lazy (all'esecuzione) | **A** (eager) con `nx.has_path()` per fail-fast |
| Sub-agent: consigliare stateless? | A: obbligare stateless con shared memory; B: lasciare la scelta | **B** (lasciare la scelta) ma documentare i rischi di stateful |
| Memory eviction per sessioni lunghe | A: TTL-based; B: LRU; C: nessuna (in-memory infinita) | **C** (nessuna) per Fase 1, poi TTL in Fase 2 |
| Awareness message: lingue | A: solo inglese; B: configurabile | **A** (solo inglese) - e' un system prompt tecnico |