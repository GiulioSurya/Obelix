# Roadmap Text2SQL - Informatore Contabile Intelligente
## Progetto basato su framework Obelix

**Data inizio**: Ottobre 2025
**Obiettivo**: Sviluppare un sistema text2sql per interrogare database Oracle di contabilità pubblica (CFA) tramite linguaggio naturale

---

## Fase 0: Setup e Preparazione 
### Obiettivi
- Setup ambiente di sviluppo
- Analisi database CFA esistente
- Preparazione dataset di test

### Attività

1. **Analisi Database CFA**
   - Mappatura schema database Oracle (tabelle: `vista_bilancio_spesa`, `eseimp`, `liq`, `opere`, `fat`, etc.)
   - Identificazione query più frequenti dal manuale utente Finmatica
   - Creazione dizionario metadati (descrizioni colonne, foreign keys, business logic) se non sufficienti

2. **Setup Ambiente**
   - Configurazione connessione Oracle Database
   - Setup ambiente Python con agentic framework

3. **Dataset di Riferimento**
   - Raccolta 20-30 query SQL reali dal documento Finmatica o eventualmente aggiunte
   - Creazione dataset di test con domande in linguaggio naturale + query SQL attese
   - Categorizzazione per complessità (semplici, medie, complesse)


---

## Fase 1: PoC - BaseAgent Text2SQL 

### Obiettivi
- Creare un BaseAgent funzionante per text2sql di base
- Validare il flusso end-to-end: domanda → query SQL → risultati
- Testare su query semplici/medie

### Attività

#### 1.1 Strategia Tool per Database Oracle

**Due approcci possibili**:

**A. Tool Custom (consigliato per iniziare)**
- Creiamo tool Python che estendono `ToolBase` dal framework Obelix
- Massimo controllo su logica, validazione, error handling
- Flessibilità per adattare alle specificità del database CFA
- Più semplice da debuggare e modificare rapidamente

**Tool previsti (Fase 1)**:
1. **read_tables_tool**: Legge l'elenco delle tabelle disponibili nel database
2. **read_columns_tool**: Legge le colonne di una specifica tabella (con tipo dato, nullable, etc.)
3. **execute_query_tool**: Esegue query SQL SELECT e ritorna risultati
4. **get_table_relationships_tool**: Estrae foreign keys e relazioni tra tabelle
5. **sample_table_data_tool**: Ritorna sample dati (es. prime 5 righe) per comprendere il contenuto

**Connessione**: Tutti i tool useranno **python-oracledb** per interagire con Oracle Database

**B. MCP Server (valutare successivamente)**
- Utilizza server MCP esistenti per Oracle Database (se disponibili)
- Wrapper tool MCP tramite `MCPTool` del framework Obelix
- Pro: Standardizzato, comunità, possibili funzionalità aggiuntive
- Contro: Meno controllo, dipendenza esterna, possibile overhead

**Decisione**:
1. **Fase 1 PoC**: Tool Custom per rapidità e controllo
2. **Post-PoC**: Valutare migrazione a MCP se emersi benefici concreti

**Differenza chiave Tool Custom vs MCP**:
- **Tool Custom**: Codice Python diretto, logica interna al progetto, async nativo
- **MCP**: Processo esterno (stdio/http), protocollo standardizzato, tool riutilizzabili tra progetti

#### 1.2 BaseAgent Text2SQL 
```python
# src/text2sql/text2sql_agent.py
class Text2SQLAgent(BaseAgent):
    def __init__(self, database_schema: dict):
        system_message = """
        Sei un esperto di SQL per database Oracle di contabilità pubblica.

        # Schema Database
        {schema_description}

        # Few-shot Examples
        {examples}

        # Istruzioni
        - Traduci domande in linguaggio naturale in query SQL
        - Usa solo le tabelle fornite nello schema
        - Genera query valide per Oracle Database
        - Considera le join necessarie
        """
        super().__init__(
            system_message=system_message,
            agent_name="text2sql-agent",
            description="Agente per conversione text-to-SQL"
        )
        self.register_tool(OracleQueryTool(connection_string=...))
```

#### 1.3 Prompt Engineering 
- **Few-shot learning**: Includere 5-10 esempi rappresentativi nel system prompt
- **Schema description**: Formato ottimizzato (es. JSON, CREATE TABLE, o descrizione naturale)
- **Guardrails**: Validazione sintassi SQL prima dell'esecuzione
- **Error handling**: Retry logic con error feedback all'LLM

### Test e Validazione
- Testare su 20 query semplici (filtri base, aggregazioni semplici)
- Metriche:
  - **Execution Success Rate**: % query eseguite senza errori SQL
  - **Result Accuracy**: Confronto risultati con query gold standard
  - **Schema Compliance**: Query usa solo tabelle/colonne valide


---

## Fase 2: Output Strutturato e Presentazione

### Obiettivi
- Definire schema di output per i risultati delle query
- Implementare logica di arricchimento dati (sintesi AI, suggerimenti visualizzazione)

### Attività

#### 2.1 Schema Output
- Definizione struttura dati output completa
- Include: query originale, query SQL generata, risultati, metadati

#### 2.2 Arricchimento Risultati
- **Sintesi AI**: Generazione automatica di summary testuale dei risultati
- **Analisi dati**: Identificazione automatica di pattern, KPI, anomalie
- **Suggerimenti presentazione**: Logica per determinare modalità di visualizzazione ottimale dei dati

**Note**: L'interfaccia utente e la presentazione visuale sono responsabilità di componenti esterni e fuori scope per questo progetto.


---

## Fase 3: Query Complesse e Ottimizzazione 

### Obiettivi
- Gestire query complesse (multi-join, subquery, aggregazioni annidate)
- Ottimizzare performance su database grandi
- Migliorare accuracy su query difficili

### Attività

#### 3.1 Query Enhancement Agent

**Ruolo**: Agente di pianificazione che traduce la richiesta utente in una sequenza logica di passi **prima** della generazione SQL.

**Input**:
- Domanda utente in linguaggio naturale
- Tabelle e colonne selezionate (da SchemaAgent o analisi precedente)

**Output**:
- **Chain of Thought**: Sequenza logica di operazioni necessarie
  - Es: "1. Filtrare capitoli con descrizione contenente 'PNRR' → 2. Unire con tabella impegni → 3. Aggregare per unità organizzativa → 4. Calcolare totale"
- **Decomposizione query**: Identifica se servono subquery, CTE, window functions
- **Operazioni richieste**: JOIN necessarie, filtri WHERE, aggregazioni GROUP BY, ordinamenti
- **Vincoli logici**: Condizioni di business (es. "solo competenza", "escludi residui passivi")

**Tecniche**:
- **Chain-of-Thought prompting**: L'LLM ragiona passo-passo sulla logica della query
- **Decomposizione gerarchica**: Query complesse vengono scomposte in query più semplici
- **Mapping semantico**: Traduce concetti business in operazioni SQL (es. "pagato" → SUM(tot_mandati_comp))

**Vantaggi**:
- L'agente SQL successivo riceve una "ricetta" chiara da seguire
- Riduce errori logici nella query finale
- Facilita il debug (si può vedere dove il ragionamento fallisce)
- Permette retry mirati su step specifici

#### 3.2 Gestione Database Grandi
**Problema**: Quando il database Oracle CFA ha uno schema molto complesso (100+ tabelle, 1000+ colonne), il prompt inviato all'LLM diventa troppo grande e inefficiente. L'agente SQL riceve informazioni su centinaia di tabelle irrilevanti per la domanda specifica.

**Obiettivo**: Ridurre dinamicamente lo schema mostrato agli agenti, includendo solo le parti rilevanti per la domanda dell'utente.

**Soluzioni**:

**A. Schema Pruning Dinamico**
Invece di passare l'intero schema del database all'agente SQL, lo **SchemaAgent** (vedi Fase 4) seleziona automaticamente solo le 5-10 tabelle più probabilmente rilevanti.

Meccanismo:
- Calcola similarità semantica tra la domanda utente e le descrizioni delle tabelle del database
- Seleziona le top-k tabelle più simili (es. top-5)
- Aggiunge automaticamente le tabelle collegate tramite foreign keys (per permettere JOIN corretti)
- Passa solo questo schema ridotto all'agente SQL successivo

Esempio: se l'utente chiede "Mostrami gli impegni del capitolo X", lo SchemaAgent seleziona `vista_bilancio_spesa` ed `eseimp`, ignorando le altre 98 tabelle.

**B. Column Filtering**
Anche dopo aver selezionato le tabelle giuste, ogni tabella può avere 50+ colonne. Si filtrano ulteriormente le colonne mostrate:
- Colonne menzionate esplicitamente nella domanda (es. "capitolo", "impegnato")
- Colonne "frequently used" (chiavi primarie, colonne di totale, date)
- Tutte le altre colonne vengono nascoste dal prompt

Questo riduce ulteriormente la dimensione del prompt LLM mantenendo le informazioni essenziali.

#### 3.3 Test su Query Complesse 
- Dataset di 20 query complesse dal documento Finmatica:
  - Multi-join (3+ tabelle)
  - Subquery correlate
  - Window functions
  - CTE (Common Table Expressions)

**Esempi**:
```sql
-- Query complessa: economie di spesa per UO con join multipli
SELECT
    unita_organizzativa,
    SUM(economie) as totale_economie
FROM vista_bilancio_spesa v
JOIN riaccertamento_residui r ON ...
WHERE ...
GROUP BY unita_organizzativa
HAVING SUM(economie) > 10000
```

### Metriche Target
- Execution Success Rate: >85%
- Result Accuracy: >75%
- Avg Query Generation Time: <5s


---

## Fase 4: Multi-Agent Architecture 

### Obiettivi
- Architettura modulare con agenti specializzati
- Ogni agente gestisce un aspetto del pipeline text2sql


### Architettura Proposta

```
User Query
    ↓
[MasterAgent]
    ↓
    ├─→ [SchemaAgent] → Seleziona tabelle/colonne rilevanti
    ↓
    ├─→ [QueryPlannerAgent] → Decompone query complesse
    ↓
    ├─→ [SQLGeneratorAgent] → Genera SQL ottimizzato
    ↓
    ├─→ [ValidationAgent] → Valida sintassi e sicurezza
    ↓
    ├─→ [ExecutionAgent] → Esegue query (OracleQueryTool)
    ↓
    ├─→ [VisualizationAgent] → Genera config grafici
    ↓
    ├─→ [SummaryAgent] → Crea sintesi AI dei risultati
    ↓
SlideOutput
```

### Attività

#### 4.1 Agenti Specializzati 

**SchemaAgent**
```python
class SchemaAgent(BaseAgent):
    """
    Seleziona tabelle e colonne rilevanti dal full schema
    """
    def execute_query(self, query: str) -> AgentResponse:
        # Input: user query
        # Output: {
        #   "selected_tables": ["vista_bilancio_spesa", "eseimp"],
        #   "selected_columns": ["capitolo", "tot_impegnato_comp"],
        #   "reasoning": "..."
        # }
```

**QueryPlannerAgent**
```python
class QueryPlannerAgent(BaseAgent):
    """
    Decompone query complesse in step logici
    """
    def execute_query(self, query: str) -> AgentResponse:
        # Input: user query + selected schema
        # Output: {
        #   "query_plan": [
        #     "1. Filtra capitoli per descrizione PNRR",
        #     "2. Calcola somma impegni",
        #     "3. Raggruppa per unità organizzativa"
        #   ],
        #   "needs_subquery": true
        # }
```

**SQLGeneratorAgent**
```python
class SQLGeneratorAgent(BaseAgent):
    """
    Genera query SQL ottimizzata
    """
    def execute_query(self, query: str) -> AgentResponse:
        # Input: user query + schema + query plan
        # Output: {
        #   "sql": "SELECT ... FROM ... WHERE ...",
        #   "explanation": "Query calcola...",
        #   "estimated_complexity": "medium"
        # }
```

**ValidationAgent**
```python
class ValidationAgent(BaseAgent):
    """
    Valida query SQL per sicurezza e correttezza
    """
    def execute_query(self, query: str) -> AgentResponse:
        # Checks:
        # - Sintassi SQL corretta (parser)
        # - No SQL injection patterns
        # - Solo SELECT (no INSERT/UPDATE/DELETE)
        # - Tabelle esistono nello schema
        # - Limiti ragionevoli (no SELECT * senza WHERE su tabelle grandi)
        # Output: {"valid": true/false, "errors": [...]}
```

#### 4.2 MasterAgent Orchestrator 
```python
class Text2SQLMasterAgent(MasterAgent):
    """
    Coordina gli agenti specializzati
    """
    def __init__(self):
        super().__init__(system_message="Coordina il pipeline text2sql")

        # Registra agenti
        self.register_agent(SchemaAgent(...))
        self.register_agent(QueryPlannerAgent(...))
        self.register_agent(SQLGeneratorAgent(...))
        self.register_agent(ValidationAgent(...))
        self.register_agent(VisualizationAgent(...))
        self.register_agent(SummaryAgent(...))

    def execute_query(self, user_query: str) -> SlideOutput:
        # 1. SchemaAgent: seleziona schema rilevante
        schema_result = self.call_agent("schema-agent", user_query)

        # 2. QueryPlannerAgent: pianifica query
        plan_result = self.call_agent("query-planner", {
            "query": user_query,
            "schema": schema_result
        })

        # 3. SQLGeneratorAgent: genera SQL
        sql_result = self.call_agent("sql-generator", {
            "query": user_query,
            "schema": schema_result,
            "plan": plan_result
        })

        # 4. ValidationAgent: valida
        validation = self.call_agent("validation-agent", sql_result)
        if not validation["valid"]:
            # Retry con feedback
            ...

        # 5. Esegui query
        data = execute_sql(sql_result["sql"])

        # 6. VisualizationAgent: genera grafico
        viz_config = self.call_agent("visualization-agent", data)

        # 7. SummaryAgent: genera sintesi
        summary = self.call_agent("summary-agent", {
            "query": user_query,
            "data": data
        })

        return SlideOutput(...)
```

#### 4.3 Gestione Conversazione Multi-Turn 
**Problema**: Follow-up contestuali
- User: "Mostrami i capitoli PNRR"
- Agent: [Slide 1 con risultati]
- User: "Filtra solo quelli del settore istruzione" ← deve capire il contesto

**Soluzione**: Conversation Memory
```python
class ConversationManager:
    """
    Gestisce lo stato della conversazione
    """
    def __init__(self):
        self.slides_history: List[SlideOutput] = []
        self.context: Dict = {}

    def add_slide(self, slide: SlideOutput):
        self.slides_history.append(slide)
        self.context["last_query_sql"] = slide.query_sql
        self.context["last_data_columns"] = slide.data[0].keys()

    def enhance_followup_query(self, user_query: str) -> str:
        # Arricchisce query con contesto
        if "filtra" in user_query.lower() and self.context.get("last_query_sql"):
            return f"""
            Modifica questa query precedente:
            {self.context['last_query_sql']}

            Applicando questo filtro aggiuntivo:
            {user_query}
            """
        return user_query
```

### Test Multi-Agent
- 30 query end-to-end attraverso 
- 10 conversazioni multi-turn (3-4 domande consecutive)
- Metriche:
  - Agent orchestration success rate
  - Latency totale pipeline
  - Context retention accuracy (follow-up)


---
