# Graph-Based Memory System per Obelix

> **Status**: Concept Development
> **Ispirazione**: Parlant Relationships & Journey Graphs
> **Obiettivo**: Memoria condivisa tra sub-agent senza stato globale monolitico

---

## 🎯 Problema da Risolvere

### Situazione Attuale
In Obelix, quando un **orchestrator** coordina multipli **sub-agent**, ogni sub-agent opera in isolamento:

```
Orchestrator
    ├─> SubAgent A (analyzer)    → conversation_history_A
    ├─> SubAgent B (validator)   → conversation_history_B
    └─> SubAgent C (reporter)    → conversation_history_C
```

**Problemi:**
1. **Ridondanza**: Se B deve validare il lavoro di A, l'orchestrator deve ri-inserire tutto il contesto di A nella query per B
2. **Perdita di contesto**: Le connessioni semantiche tra output di diversi agent sono implicite (solo nell'orchestrator)
3. **Scalabilità**: Con N sub-agent, la complessità di coordinazione cresce quadraticamente
4. **Mancanza di tracciabilità**: Non c'è modo di ricostruire "perché B ha prodotto X dato che A ha prodotto Y"

### Soluzione Alternativa 1: Stato Globale Condiviso
```python
# ❌ Approccio naive
shared_state = {
    "analyzer_result": {...},
    "validator_result": {...},
    "reporter_result": {...}
}
```

**Svantaggi:**
- Accoppiamento forte tra agent (devono conoscere le chiavi)
- Nessuna semantica sulle relazioni (tutto è "flat")
- Difficile gestire invalidazioni (se A ri-esegue, B deve saperlo?)
- Race conditions in esecuzione parallela

---

## 💡 Idea Core: Memoria come Grafo

### Visione
Invece di uno stato condiviso monolitico, ogni **output significativo** di un agent diventa un **nodo** in un grafo di memoria. Le **relazioni semantiche** tra output diventano **archi** nel grafo.

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY GRAPH                             │
│                                                             │
│   [User Input]                                              │
│        │                                                    │
│        ├─DERIVES_FROM─> [Raw Data]                         │
│        │                     │                              │
│        │                     ├─DERIVES_FROM─> [Analysis]    │
│        │                     │                     │        │
│        │                     │                     └─REQUIRES─> [Validation] │
│        │                     │                                      │        │
│        └─SHARES_CONTEXT─────┴──────────────────────────────────────┘        │
│                                                                              │
│                                    └─DERIVES_FROM─> [Final Report]          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Vantaggi
1. **Memoria Selettiva**: Ogni agent accede solo alle memorie correlate (via BFS nel grafo)
2. **Semantica Esplicita**: Le relazioni (`DERIVES_FROM`, `REQUIRES`, ecc.) rendono esplicito il "perché"
3. **Tracciabilità**: Percorsi nel grafo = audit trail (chi ha prodotto cosa, basandosi su cosa)
4. **Invalidazione Automatica**: Se un nodo viene invalidato, posso marcare tutti i discendenti come "stale"
5. **Parallelismo**: Agent possono scrivere nodi concorrentemente (no lock su stato globale)

---

## 🧩 Componenti Concettuali

### 1. MemoryNode
**Cosa rappresenta**: Un'unità atomica di informazione prodotta da un agent

**Proprietà chiave**:
- **Provenienza**: Chi l'ha creato? (agent_id)
- **Tipo**: Tool result, agent output, user input, observation
- **Contenuto**: Testo + dati strutturati (JSON)
- **Metadata**: Timestamp, tags, embedding_vector (per semantic search)

**Esempi**:
```
MemoryNode {
    id: "mem_abc123"
    agent_id: "analyzer_agent"
    type: AGENT_OUTPUT
    content: "Q4 sales increased 15% vs Q3..."
    structured_data: {revenue: 1.2M, growth: 0.15}
    timestamp: 2024-01-15T10:30:00Z
}
```

### 2. MemoryEdge (Relationship)
**Cosa rappresenta**: Una relazione semantica tra due memorie

**Tipi di relazioni** (ispirati da Parlant + estesi):

#### A. Relazioni di Derivazione
- **DERIVES_FROM**: `B` è stato calcolato/inferito da `A`
  - Esempio: `[Analysis] DERIVES_FROM [Raw Data]`
  - Uso: Tracciare lineage dei dati

- **TRANSFORMS**: `B` è una trasformazione di `A` (più specifico di DERIVES_FROM)
  - Esempio: `[Normalized Data] TRANSFORMS [Raw CSV]`
  - Uso: Pipeline di data processing

#### B. Relazioni di Dipendenza
- **REQUIRES**: `B` richiede `A` per essere comprensibile
  - Esempio: `[Validation Report] REQUIRES [Analysis]`
  - Uso: Quando validator legge analysis, crea REQUIRES

- **DEPENDS_ON**: `B` dipende funzionalmente da `A` (se A cambia, B deve ricalcolarsi)
  - Esempio: `[Forecast] DEPENDS_ON [Historical Data]`
  - Uso: Invalidazione automatica

#### C. Relazioni di Consistenza
- **INVALIDATES**: `B` rende `A` obsoleta
  - Esempio: `[Updated Analysis] INVALIDATES [Old Analysis]`
  - Uso: Gestire versioning delle memorie

- **CONTRADICTS**: `B` contraddice `A` (conflict detection)
  - Esempio: `[Validator Output] CONTRADICTS [Analyzer Output]`
  - Uso: Segnalare inconsistenze

- **CONFIRMS**: `B` conferma/valida `A`
  - Esempio: `[Validation Result] CONFIRMS [Analysis]`
  - Uso: Chain of trust

#### D. Relazioni di Contesto
- **SHARES_CONTEXT**: `A` e `B` condividono contesto (stessa query utente, stesso workflow)
  - Esempio: `[Analysis] SHARES_CONTEXT [Validation]` (entrambe sulla stessa query)
  - Uso: Raggruppamento logico

- **COMPLEMENTS**: `B` aggiunge informazioni a `A` senza invalidarla
  - Esempio: `[Risk Analysis] COMPLEMENTS [Financial Analysis]`
  - Uso: Prospettive multiple sullo stesso dato

#### E. Relazioni di Sequenza
- **FOLLOWS**: `B` segue temporalmente `A` (workflow step)
  - Esempio: `[Step 2: Validation] FOLLOWS [Step 1: Analysis]`
  - Uso: Ricostruire ordine di esecuzione

- **TRIGGERS**: `A` ha causato l'esecuzione che ha prodotto `B`
  - Esempio: `[Error Detection] TRIGGERS [Error Report]`
  - Uso: Causality tracking

### 3. MemoryGraph
**Cosa rappresenta**: Il grafo complessivo che connette tutte le memorie

**Operazioni chiave**:

#### Query Relazionali (BFS-based)
```
get_related_memories(node_id, relationship_kind, max_depth, direction)
→ Trova tutte le memorie correlate a distanza ≤ max_depth
```

**Esempi**:
- `get_related_memories("validation_123", REQUIRES, depth=2, direction="incoming")`
  → Trova cosa è stato richiesto per produrre la validazione (e cosa a sua volta era richiesto)

- `get_related_memories("analysis_456", DERIVES_FROM, depth=∞, direction="outgoing")`
  → Trova tutta la chain di derivazioni dall'analisi

#### Context Building
```
get_context_for_agent(agent_id, max_memories)
→ Trova le memorie più rilevanti per un agent
```

**Strategia**:
1. Memorie create dall'agent (ownership)
2. Memorie con relazione REQUIRES/SHARES_CONTEXT (relevance)
3. Memorie recenti nello stesso workflow (temporal proximity)
4. Ordina per rilevanza (combination of factors)

#### Semantic Search (opzionale, richiede embeddings)
```
search_memories(query_text, agent_id?, top_k)
→ Vector search sugli embedding dei nodi
```

**Uso**: "Trova memorie simili a 'revenue analysis Q4'" anche senza conoscere gli ID

---

## 🔄 Pattern di Utilizzo

### Pattern 1: Linear Pipeline
**Scenario**: Workflow sequenziale (A → B → C)

```
User Query
    ↓ DERIVES_FROM
[Raw Data Extraction] (Agent A)
    ↓ DERIVES_FROM
[Data Analysis] (Agent B)
    ↓ REQUIRES
[Validation Report] (Agent C)
```

**Come funziona**:
1. Agent A esegue, salva `raw_data` node
2. Agent B riceve query + `raw_data_id`, crea edge `analysis DERIVES_FROM raw_data`
3. Agent C riceve query + `analysis_id`, crea edge `validation REQUIRES analysis`
4. C può fare BFS per trovare anche `raw_data` (transitività)

**Vantaggio**: Non serve passare tutto il contesto a C, basta l'ID dell'analysis

### Pattern 2: Parallel Aggregation
**Scenario**: Multipli agent analizzano lo stesso input, un reporter aggrega

```
              [User Query]
                   ↓ SHARES_CONTEXT
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   [Financial] [Risk]    [Market]
   (Agent A)   (Agent B) (Agent C)
        │          │          │
        └──────────┴──────────┘
                   ↓ REQUIRES (all 3)
            [Final Report]
            (Agent D)
```

**Come funziona**:
1. A, B, C eseguono in parallelo, ognuno salva un nodo
2. Tutti creano edge `SHARES_CONTEXT` con lo stesso `user_query_id`
3. D riceve IDs di A, B, C, crea edges `report REQUIRES financial/risk/market`
4. D può query `SHARES_CONTEXT` per trovare tutto ciò che è stato prodotto per la stessa query

**Vantaggio**: Aggregazione flessibile senza conoscere a priori quanti agent partecipano

### Pattern 3: Iterative Refinement
**Scenario**: Agent raffina progressivamente un risultato

```
[Initial Analysis v1]
         ↓ INVALIDATES
[Refined Analysis v2]
         ↓ INVALIDATES
[Final Analysis v3]
```

**Come funziona**:
1. Agent produce `v1`, utente dà feedback
2. Agent produce `v2`, crea edge `v2 INVALIDATES v1`
3. Quando altro agent cerca "latest analysis", filtra per nodi non invalidati

**Vantaggio**: Versioning automatico + audit trail

### Pattern 4: Cross-Validation
**Scenario**: Validator controlla consistency tra agent

```
[Analysis A]     [Analysis B]
     │                 │
     └─────CONTRADICTS─┘ (detected by Validator)
              ↓
     [Conflict Resolution]
```

**Come funziona**:
1. Validator confronta output di A e B
2. Se trova inconsistenza, crea edge `A CONTRADICTS B`
3. Produce memoria `conflict_resolution` con edges `REQUIRES [A, B]`

**Vantaggio**: Conflict detection esplicito

---

## 🎨 Design Decisions & Trade-offs

### Decision 1: Grafi Immutabili vs Mutabili
**Opzione A (Immutable)**: I nodi non si modificano mai, solo si aggiungono (con INVALIDATES)
- ✅ Pro: Audit trail completo, time-travel possibile
- ❌ Contro: Crescita continua del grafo (serve garbage collection)

**Opzione B (Mutable)**: I nodi si aggiornano in-place
- ✅ Pro: Efficienza memoria
- ❌ Contro: Perdita di storico

**Proposta**: **Immutable con GC policy**
- Regola: Mantieni solo ultimi N nodi per agent, oppure nodi degli ultimi M giorni
- Soft delete: Marca come "archived" invece di eliminare (per compliance)

### Decision 2: Relationship Transitivity
**Domanda**: Se `A REQUIRES B` e `B REQUIRES C`, allora `A REQUIRES C`?

**Approccio Parlant**: Sì, BFS trova tutte le relazioni transitive

**Trade-off**:
- ✅ Pro: Context building automatico (trova tutte le dipendenze)
- ❌ Contro: Performance (BFS può essere costoso su grafi grandi)
- ⚠️ Ambiguità semantica: Non tutte le relazioni sono transitive (CONTRADICTS non lo è!)

**Proposta**: **Transitività configurabile per tipo di relazione**
```python
TRANSITIVE_RELATIONSHIPS = {
    DERIVES_FROM,    # Sì, transitiva
    REQUIRES,        # Sì, transitiva
    SHARES_CONTEXT,  # No, non transitiva (troppo vaga)
    CONTRADICTS      # No, non transitiva
}
```

### Decision 3: Embedding & Semantic Search
**Domanda**: Tutti i nodi devono avere embeddings per semantic search?

**Trade-off**:
- ✅ Pro: Powerful retrieval ("trova memorie simili a X")
- ❌ Contro: Costo computazionale (chiamata a embedding model), storage, latency

**Proposta**: **Opt-in per tipo di memoria**
```python
EMBED_TYPES = {
    AGENT_OUTPUT,    # Sì, utile per "trova analisi simili"
    USER_INPUT,      # Sì, utile per "query simili"
    TOOL_RESULT      # No, troppo strutturato (usa filtri esatti)
}
```

### Decision 4: Scope della Memoria
**Domanda**: Il grafo è globale o per-session?

**Opzioni**:
- **Globale**: Tutte le memorie di tutti gli agent in un unico grafo
  - ✅ Cross-session learning possibile
  - ❌ Privacy concerns, complessità query

- **Per-Session**: Ogni sessione ha il suo grafo
  - ✅ Isolamento, privacy
  - ❌ No knowledge reuse

- **Ibrido**: Grafo per-session + knowledge base globale (read-only)
  - ✅ Best of both worlds
  - ❌ Complessità architetturale

**Proposta**: **Per-Session con optional Knowledge Base**
```python
class MemoryGraph:
    session_graph: nx.DiGraph  # Read-write
    knowledge_base: nx.DiGraph  # Read-only (pre-popolato con "common knowledge")
```

### Decision 5: Persistence Strategy
**Domanda**: Dove/come persistere il grafo?

**Opzioni**:
1. **In-Memory (InMemoryStore)**: Dizionario Python + NetworkX
   - ✅ Semplicità, velocità
   - ❌ Volatile, no scalability

2. **SQL (SQLiteStore / PostgreSQL)**: Tabelle `nodes`, `edges`
   - ✅ ACID, query relazionali
   - ❌ Graph traversal inefficiente (JOIN ricorsivi)

3. **Graph DB (Neo4j / ArangoDB)**: Native graph storage
   - ✅ Performance BFS, Cypher/AQL query
   - ❌ Complessità deployment, dependency

4. **Vector DB (Chroma / Qdrant)**: Solo per semantic search
   - ✅ Embedding-based retrieval
   - ❌ No relationships native (servono ancora SQL/Graph DB per edges)

**Proposta**: **Layered Approach**
```
┌─────────────────────────────────────────┐
│  Application (MemoryGraph API)          │
├─────────────────────────────────────────┤
│  Document Store (nodes metadata)        │  ← SQLite/PostgreSQL
│  Edge Store (relationships)             │  ← SQLite/PostgreSQL
│  Vector Store (embeddings)              │  ← Chroma/Qdrant (optional)
└─────────────────────────────────────────┘
```

- **Development**: InMemoryStore (no deps)
- **Production**: PostgreSQL + Chroma (scalability + semantic search)

---

## 🔮 Possibili Evoluzioni

### Evolution 1: Automatic Relationship Inference
**Idea**: L'orchestrator inferisce relazioni automaticamente invece che richiederle esplicitamente

**Esempio**:
```python
# Invece di:
validator.create_relationship(validation_id, analysis_id, REQUIRES)

# L'orchestrator fa:
orchestrator.register_agent(ValidatorAgent(), auto_infer_relations=True)
# → Se Validator legge Analysis, crea REQUIRES automaticamente
```

**Implementazione**: Hook su `get_memory()` che traccia "chi legge cosa"

### Evolution 2: Semantic Relationship Types
**Idea**: Invece di relazioni hard-coded, permettere relazioni custom semantiche

**Esempio**:
```python
create_relationship(
    source_id,
    target_id,
    kind=CustomRelation(
        name="critiques",
        description="Source provides critical analysis of target",
        transitive=False
    )
)
```

**Uso**: Domain-specific relationships (medical: DIAGNOSES, legal: CITES)

### Evolution 3: Memory Compression
**Idea**: Automaticamente "comprimere" catene lunghe di memorie in summaries

**Esempio**:
```
[A] → [B] → [C] → [D] → [E]
         ↓ COMPRESS
[Summary(A→E)] + tombstones
```

**Trigger**: Quando catena supera threshold (e.g., 10 nodi), LLM genera summary

### Evolution 4: Conflict Resolution Agent
**Idea**: Agent specializzato che monitora il grafo per CONTRADICTS edges

**Flusso**:
1. Validator crea edge `[Analysis A] CONTRADICTS [Analysis B]`
2. ConflictResolver agent viene triggerato automaticamente
3. Analizza A e B, produce `[Resolution]` con `RESOLVES [A, B]`

### Evolution 5: Distributed Memory Graph
**Idea**: Ogni agent ha il proprio sotto-grafo, sincronizzati via gossip protocol

**Scenario**: Multi-datacenter deployment, ogni DC ha replica del grafo

**Sfide**: Consistency (eventual vs strong), conflict resolution, partitioning

---

## 📊 Metriche di Successo

Come sappiamo se il sistema funziona bene?

### Metriche Quantitative
1. **Context Efficiency**: Riduzione % di token passati tra agent
   - Before: Ogni sub-agent riceve full context (N tokens)
   - After: Ogni sub-agent riceve solo relevant IDs + BFS retrieval (M tokens)
   - Target: M < 0.5 * N

2. **Retrieval Precision**: % di memorie retrieved che sono effettivamente usate
   - Misura: Tracking di quali nodi l'agent include nella risposta finale
   - Target: > 80% delle memorie retrieved sono rilevanti

3. **Graph Density**: Numero di edges / numero di nodes
   - Troppo sparse: Poche relazioni (forse under-utilized)
   - Troppo dense: Troppe relazioni (forse over-connected, noise)
   - Target: 1.5-3 edges per node (empirico)

### Metriche Qualitative
1. **Tracciabilità**: Posso ricostruire "perché agent X ha detto Y"?
   - Test: Audit trail deve mostrare path da output a input originale

2. **Consistenza**: Gli agent producono output consistenti quando condividono memoria?
   - Test: Validator dovrebbe trovare meno CONTRADICTS quando usa memoria vs quando no

3. **Reusabilità**: Posso riusare memorie in sessioni diverse?
   - Test: Knowledge base globale migliora accuracy su nuove query?

---

## 🤔 Domande Aperte

### Q1: Privacy & Security
**Domanda**: Come gestiamo memorie sensibili nel grafo?

**Opzioni**:
- **Encryption**: Criptare `content` dei nodi sensibili
- **Access Control**: Permessi per-agent (agent A può leggere nodi di agent B?)
- **Ephemeral Memory**: Auto-delete dopo N minuti per dati PII

**Da decidere**: Policy di retention, compliance (GDPR, HIPAA)

### Q2: Multi-Tenancy
**Domanda**: In un deployment multi-utente, i grafi sono separati per user?

**Opzioni**:
- **Isolamento totale**: Ogni user ha il proprio grafo (privacy)
- **Shared Knowledge Base**: Grafo globale con memorie "pubbliche" (efficienza)

**Da decidere**: Architettura di deployment

### Q3: Real-Time Updates
**Domanda**: Se un nodo viene invalidato, gli agent che lo stanno usando devono essere notificati?

**Scenario**:
- Agent A sta usando memoria M
- Agent B invalida M (crea M' INVALIDATES M)
- A deve saperlo?

**Opzioni**:
- **Polling**: A controlla periodicamente se le sue memorie sono ancora valide
- **Push**: Event system notifica A quando M viene invalidata

### Q4: Testing & Simulation
**Domanda**: Come testiamo che il grafo si comporta come vogliamo?

**Approcci**:
- **Unit Tests**: Test per ogni tipo di relazione (DERIVES_FROM, REQUIRES, ecc.)
- **Integration Tests**: Workflow completi (3+ agent) con asserzioni sul grafo risultante
- **Simulation**: Generare grafi sintetici e validare proprietà (no cycles, max depth, ecc.)

---

## 📚 Riferimenti & Ispirazione

### Da Parlant
- **Relationships**: ENTAILMENT, PRIORITY, DEPENDENCY
  - Uso: Filtrare/ordinare guidelines matched
  - Trasposizione: Applicabile a memorie (quali memorie sono rilevanti?)

- **Journey Graphs**: Nodi + Archi per workflow multi-step
  - Uso: Navigare conversazioni strutturate
  - Trasposizione: Memorie come "journey" attraverso il workflow

- **BFS per Transitività**: Trovare relazioni indirette
  - Uso: Se A implica B e B implica C → A implica C
  - Trasposizione: Context building automatico

### Da Altri Sistemi
- **Knowledge Graphs (Wikidata, DBpedia)**: Entità + relazioni semantiche
  - Lezione: Ontologie ben definite (relationship types) sono cruciali

- **Git DAG**: Commits + parent relationships
  - Lezione: Immutabilità + branching/merging (potrebbe servire per conflitti?)

- **Blockchain**: Append-only log con hash-linking
  - Lezione: Tamper-proof audit trail (overkill per noi, ma ispirazione)

---

## 🛠️ Roadmap Implementazione (Draft)

### Fase 1: MVP (Minimum Viable Product)
**Goal**: Dimostrare valore core con minimal implementation

**Scope**:
- ✅ `MemoryNode` (dataclass)
- ✅ `MemoryEdge` (dataclass)
- ✅ `MemoryGraph` (NetworkX + BFS)
- ✅ `InMemoryStore` (dizionario)
- ✅ 3 relationship types: DERIVES_FROM, REQUIRES, SHARES_CONTEXT
- ✅ `MemoryMixin` per BaseAgent
- ✅ Hook automatico per salvare tool results
- ✅ Esempio: Linear pipeline (A → B → C)

**Non-Goals** (rinviati):
- Persistence (solo RAM)
- Semantic search (solo relational queries)
- Conflict detection

### Fase 2: Production-Ready
**Goal**: Scalare a deployment reale

**Scope**:
- 📦 Persistence: SQLite store (nodes + edges tables)
- 🔍 Query optimization: Indexing, caching
- 🧹 Garbage collection: Auto-delete old memories
- 📊 Monitoring: Metriche (graph density, retrieval precision)
- 🧪 Test suite: Unit + integration tests

### Fase 3: Advanced Features
**Goal**: Funzionalità avanzate per casi d'uso complessi

**Scope**:
- 🧠 Semantic search: Embeddings + Chroma
- 🔗 Automatic relationship inference
- 🤖 Conflict resolution agent
- 🌐 Knowledge base globale (read-only)
- 📈 Visualization dashboard (NetworkX + Plotly)

---

## 💭 Conclusioni

Il sistema di memoria basato su grafi trasforma la coordinazione tra sub-agent da **orchestrazione esplicita** (l'orchestrator passa tutto il contesto) a **discovery implicita** (ogni agent trova ciò che serve via relazioni semantiche).

**Key Insight**: Le relazioni tra memorie sono **altrettanto importanti** delle memorie stesse. Codificando esplicitamente "perché" due memorie sono connesse, rendiamo il sistema:
- **Più tracciabile** (audit trail)
- **Più efficiente** (context retrieval selettivo)
- **Più flessibile** (nuovi agent possono navigare il grafo esistente)

**Next Steps**:
1. Validare i concetti con prototyping rapido
2. Definire ontologia completa delle relationships (quali servono davvero?)
3. Implementare MVP e testare su workflow reali
4. Iterare in base a feedback

---

**Domande per Discussion**:
- Quali relationship types sono davvero essenziali vs nice-to-have?
- Per quali use case la memoria grafo è overkill vs necessaria?
- Come bilanciare automatizzazione (auto-infer relations) vs controllo esplicito?
