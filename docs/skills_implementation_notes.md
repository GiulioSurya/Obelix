# Skills — Note di Implementazione (v1)

Note personali per il maintainer. Italiano informale, niente marketing.

---

## Stato v1 — cosa c'e'

- **Discovery esplicita**: skill caricate solo passando `skills_config` a `BaseAgent`. Nessuno scan globale del filesystem, nessun walk-up.
- **Validation a tappeto**: `run_validators()` aggrega *tutti* gli issue prima di sollevare `SkillValidationError`. Non c'e' short-circuit, cosi' l'utente vede ogni problema al primo boot.
- **Validator attivi**: `FrontmatterSchemaValidator` (Pydantic, `extra="ignore"`), `HookEventValidator`, `ArgumentUniquenessValidator`, `PlaceholderConsistencyValidator`, `BodyNonEmptyValidator`.
- **Placeholder substitution pura**: `substitute_placeholders()` e' stateless — niente IO, niente side-effect. Ordine: named-positional longest-first → `$ARGUMENTS` → meta (`${OBELIX_SKILL_DIR}`, `${OBELIX_SESSION_ID}`). Escape via `\$`.
- **Tre mode di esecuzione**:
  - `inline` (default): body reso e restituito come tool result.
  - `fork` (`context: fork`): body diventa system_message di un `BaseAgent` effimero, wrappato in `SubAgentWrapper(stateless=True)`, che eredita provider + tools + `max_iterations` del parent. Ritorna solo l'ultimo `AssistantMessage.content`.
  - `hooks`: chiave frontmatter opzionale per inline, installa hook `CONTINUE` con effetto che appende `HumanMessage` alla history al firing dell'evento.
- **MCP skills**: se `mcp_config` e' valorizzato, `MCPSkillProvider` scopre i prompts dal `ClientSessionGroup` e li espone con nome `mcp__<server>__<prompt>`. Convivono con le FS skills nello stesso `SkillManager`.
- **Tool built-in `Skill`**: costruito in `make_skill_tool()`, una istanza per agent. Field `name`, `args`. Il listing va nel system prompt via `system_prompt_fragment()`.
- **Idempotenza per-query**: `active_skills: set[str]` nel closure del tool. Seconda invocazione nella stessa query → stringa marker, niente re-iniezione.
- **Cleanup QUERY_END**: un hook si auto-installa alla prima invocazione con hooks frontmatter, rimuove gli hook skill-scoped, svuota `active_skills`, e si disinstalla da solo. Re-armabile per la query successiva.
- **Listing budget**: `DEFAULT_LISTING_BUDGET = 8_000` caratteri (~1% di 200k token). `manager.format_listing(budget)` tronca in coda se serve.

---

## Stato v1 — cosa NON c'e'

Volutamente fuori scope:

- **Auto-discovery filesystem**: niente scan di `./skills/` o directory parent. Tutto esplicito via `skills_config`. *Perche'*: principio di least surprise, e scoraggia "magia" non debuggabile.
- **Compat `~/.claude/skills/`**: nessun supporto nativo. *Perche'*: Obelix non e' Claude Code, e il costo di supportare un secondo path layout non vale la candela finche' nessuno lo chiede.
- **Conditional activation via `paths:`**: niente campo `paths:` per attivare skill in base alla working dir. *Perche'*: aggiungerlo senza filesystem walk-up richiede API aggiuntive — rimandato a v2.
- **Shell inline ``!`cmd` ``**: niente esecuzione di comandi nel body al momento della substitution. *Perche'*: superficie di attacco non trascurabile, richiede sandbox/allowlist — rimandato.
- **Override `model:` / `effort:`**: le skill non possono forzare provider/modello/reasoning_effort del parent. *Perche'*: violerebbe il contratto "il provider lo setta il chiamante". Da discutere separatamente per v2.
- **Walk-up discovery**: niente risalita da cwd a home per trovare skill dirs.
- **Classe `SkillConfig` dedicata**: accettiamo `str | Path | list`, con normalizzazione in `BaseAgent._normalize_skills_config`. *Perche'*: una dataclass aggiuntiva non porta valore per ora.
- **Plugin marketplace / remote skill search**: niente fetch da URL, niente registry centrale.
- **Skill nella `AgentFactory`**: il factory non wira `skills_config`. Per ora solo `BaseAgent` diretto. *Perche'*: se serve shared skills tra agent si passa la stessa lista di path; il factory path richiede decisioni aggiuntive (override vs merge) che non abbiamo ancora affrontato.

---

## Esempi d'uso end-to-end

### 1. Skill minimal locale

```
skills/
  greeter/
    SKILL.md
```

```markdown
---
description: Saluta l'utente in modo caloroso
when_to_use: User saluta o apre la conversazione
---
Saluta l'utente per nome, poi chiedi come puoi aiutare.
```

```python
from obelix.core.agent.base_agent import BaseAgent

agent = BaseAgent(
    system_message="Sei un assistente.",
    provider=provider,
    skills_config="./skills/greeter",
)
agent.execute_query("Ciao!")
```

### 2. Skill con fork

```markdown
---
description: Scrive un commit message da staged diff
context: fork
---
Sei un writer di conventional-commit.

1. Chiama bash `git diff --staged`.
2. Scrivi il messaggio imperativo in una riga.
3. Output: solo il messaggio.
```

```python
agent = BaseAgent(
    system_message="Sei un git helper.",
    provider=provider,
    tools=[BashTool],
    skills_config="./skills/commit-writer",
)
# La history del parent non vede nulla del loop interno — solo il messaggio finale.
agent.execute_query("Fammi un commit message.")
```

### 3. Skill con hook `on_tool_error`

```markdown
---
description: Debug sistematico
when_to_use: User riporta un bug
hooks:
  on_tool_error: "Un tool e' appena fallito. Rileggi lo schema e spiega in una riga perche'."
---
Riproduci, ipotizza, isola, fixa, aggiungi test di regressione.
```

```python
agent = BaseAgent(
    system_message="Sei un assistente di debug.",
    provider=provider,
    tools=[BashTool],
    skills_config="./skills/debugger",
)
agent.execute_query("I test falliscono con ImportError.")
# Al QUERY_END l'hook si rimuove da solo: la query successiva parte pulita.
```

### 4. Skill da server MCP

```python
from obelix.adapters.outbound.mcp.config import MCPServerConfig

agent = BaseAgent(
    system_message="Sei un assistente.",
    provider=provider,
    mcp_config=MCPServerConfig(
        name="obelix-tracer",
        command="obelix-tracer-mcp",
        args=[],
    ),
    skills_config=[],  # opzionale: mix con skill locali
)

async with agent:
    # Le prompts del server MCP sono apparse nel listing come mcp__obelix-tracer__<name>.
    await agent.async_execute_query("Usa la skill di tracing per riassumere l'ultima sessione.")
```

Nota: serve `async with agent:` perche' `MCPManager.connect()` e' asincrono. Se `skills_config` viene omesso ma `mcp_config` c'e', nessuno skill viene scoperto (il manager skill si costruisce solo se `skills_config is not None`).

---

## Edge cases & troubleshooting — mappa sintomo → file

| Sintomo | File da guardare |
|---------|------------------|
| Validazione non aggrega tutti gli issue | `src/obelix/adapters/outbound/skill/filesystem.py::discover` + `src/obelix/core/skill/validation.py::run_validators` |
| Body mal sostituito (placeholder rimasti / escape rotti) | `src/obelix/core/skill/substitution.py` |
| Listing sballato (troncato male, skill mancanti) | `src/obelix/core/skill/manager.py::format_listing` |
| Hook non rimossi dopo `QUERY_END` | `src/obelix/plugins/builtin/skill_tool.py::_install_cleanup_hook` |
| Fork non funziona / fork produce vuoto | `src/obelix/plugins/builtin/skill_tool.py::_execute_fork` + `_make_fork_agent` |
| MCP skill non scoperta | `src/obelix/adapters/outbound/mcp/manager.py::list_prompts` + `src/obelix/adapters/outbound/skill/mcp.py` |
| "Skill X already active" quando non dovrebbe | closure `active_skills` in `make_skill_tool` — controllare che il cleanup hook stia girando (`QUERY_END` fired?) |
| `ValueError: skills_config string cannot be empty` | `BaseAgent._normalize_skills_config` — passare `None` non stringa vuota |
| `SkillInvocationError: skill expects N argument(s), got M` | LLM passa troppi args — rivedere `description`/`when_to_use` perche' il modello capisca il formato |

---

## Prossime iterazioni probabili (v2+)

1. **Conditional activation `paths:`** — attiva skill solo se cwd matcha un glob. Richiede walk-up o riferimento esplicito alla working dir.
2. **Shell inline ``!`cmd` ``** — sostituzione con output comando, sandboxato (timeout, allowlist, cwd = `${OBELIX_SKILL_DIR}`). Necessita policy esplicita.
3. **Override `model:` / `effort:`** — permettere alla skill di richiedere un modello diverso. Implica che il provider sia swappable per-query, non banale.
4. **Skill nella `AgentFactory`** — wiring dichiarativo di `skills_config` su `register()`, con merge chiaro tra global e per-agent.
5. **Hot-reload / watch mode** — invalidare cache `SkillManager` su cambio file, utile in dev.
6. **Skill-level tool filtering** — rispettare `allowed_tools:` restringendo davvero i tool visibili al sub-agent in fork (oggi e' informativo).
7. **Marketplace / remote skills** — fetch da URL / registry, firma, versioning.
