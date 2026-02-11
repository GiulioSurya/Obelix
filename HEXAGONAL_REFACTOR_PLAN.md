# Piano Refactor Hexagonal Architecture

**Checkpoint rollback**: commit `b8b17e9` su branch `develop`

---

## Struttura Target

```
src/
├── domain/
│   ├── agent/                       # da base_agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── hooks.py
│   │   ├── event_contracts.py
│   │   ├── orchestrator_decorator.py
│   │   ├── subagent_decorator.py
│   │   └── agent_factory.py
│   ├── model/                       # da obelix_types/
│   │   ├── __init__.py
│   │   ├── roles.py
│   │   ├── usage.py
│   │   ├── standard_message.py
│   │   ├── system_message.py
│   │   ├── human_message.py
│   │   ├── assistant_message.py
│   │   └── tool_message.py
│   └── tool/                        # da tools/tool_base + tool_decorator
│       ├── __init__.py
│       ├── tool_base.py
│       └── tool_decorator.py
├── ports/
│   └── outbound/
│       ├── __init__.py
│       ├── llm_provider.py          # da client_adapters/llm_abstraction.py
│       ├── llm_connection.py        # da connections/llm_connection/base_llm_connection.py
│       └── embedding_provider.py    # da embedding_providers/abstract_embedding_provider.py
├── adapters/
│   └── outbound/
│       ├── anthropic/
│       │   ├── __init__.py
│       │   ├── provider.py          # da client_adapters/anthropic_provider.py
│       │   └── connection.py        # da connections/llm_connection/anthropic_connection.py
│       ├── openai/
│       │   ├── __init__.py
│       │   ├── provider.py
│       │   └── connection.py
│       ├── oci/
│       │   ├── __init__.py
│       │   ├── provider.py
│       │   ├── connection.py
│       │   └── strategies/
│       │       ├── __init__.py
│       │       ├── base_strategy.py
│       │       ├── generic_strategy.py
│       │       └── cohere_strategy.py
│       ├── ibm/
│       │   ├── __init__.py
│       │   ├── provider.py
│       │   └── connection.py
│       ├── ollama/
│       │   ├── __init__.py
│       │   └── provider.py          # no connection
│       ├── vllm/
│       │   ├── __init__.py
│       │   └── provider.py          # no connection
│       ├── embedding/
│       │   ├── __init__.py
│       │   └── oci_embedding.py
│       └── _legacy/
│           ├── __init__.py
│           ├── mapping_mixin.py     # da client_adapters/_legacy_mapping_mixin.py
│           ├── provider_mapping.py  # da mapping/provider_mapping.py
│           └── tool_extr_fall_back.py
├── infrastructure/
│   ├── __init__.py
│   ├── config.py                    # da src/config.py
│   ├── providers.py                 # da src/providers.py
│   ├── logging.py                   # da src/logging_config.py
│   ├── k8s.py                       # da src/k8s_config.py
│   └── utility/
│       ├── __init__.py
│       └── pydantic_validation.py
└── plugins/
    ├── mcp/                         # da tools/mcp/
    │   ├── __init__.py
    │   ├── mcp_tool.py
    │   ├── mcp_client_manager.py
    │   ├── mcp_validator.py
    │   └── run_time_manager.py
    └── builtin/
        ├── __init__.py
        └── ask_user_question_tool.py
```

---

## Strategia di Esecuzione

Per ogni fase: **script bash** con `mkdir -p`, `git mv`, e `sed -i` per bulk replace degli import. Niente edit file-per-file.

Pattern:
```bash
# 1. Crea directory
mkdir -p src/domain/model

# 2. Sposta file
git mv src/obelix_types/roles.py src/domain/model/roles.py

# 3. Bulk replace import in TUTTI i file .py
find src/ -name "*.py" -exec sed -i 's/src\.obelix_types/src.domain.model/g' {} +

# 4. Verifica
grep -rn "from src.obelix_types" src/   # deve dare 0 risultati
python -c "from src.domain.model import SystemMessage; print('OK')"

# 5. Commit
git add -A && git commit -m "refactor: phase N - description"
```

---

## Fase 1: `obelix_types/` → `domain/model/`

Il modulo più importato (~21 file dipendono). Muoverlo per primo.

### File da spostare

| Origine | Destinazione |
|---------|-------------|
| `obelix_types/__init__.py` | `domain/model/__init__.py` |
| `obelix_types/roles.py` | `domain/model/roles.py` |
| `obelix_types/usage.py` | `domain/model/usage.py` |
| `obelix_types/standard_message.py` | `domain/model/standard_message.py` |
| `obelix_types/system_message.py` | `domain/model/system_message.py` |
| `obelix_types/human_message.py` | `domain/model/human_message.py` |
| `obelix_types/assistant_message.py` | `domain/model/assistant_message.py` |
| `obelix_types/tool_message.py` | `domain/model/tool_message.py` |

### Import da aggiornare

Un solo sed globale: `src.obelix_types` → `src.domain.model`

File impattati:
- **domain/model/ interni** (5 file): `__init__.py`, `standard_message.py`, `system_message.py`, `human_message.py`, `assistant_message.py`, `tool_message.py`
- **base_agent/**: `base_agent.py`, `hooks.py`, `event_contracts.py`, `orchestrator_decorator.py`, `subagent_decorator.py`
- **client_adapters/**: `llm_abstraction.py`, `anthropic_provider.py`, `openai_provider.py`, `oci_provider.py`, `ibm_provider.py`, `ollama_provider.py`, `vllm_provider.py`, `_legacy_mapping_mixin.py`
- **client_adapters/oci_strategies/**: `base_strategy.py`, `generic_strategy.py`, `cohere_strategy.py`
- **tools/**: `tool_decorator.py`, `mcp/mcp_tool.py`
- **mapping/**: `tool_extr_fall_back.py`
- **logging_config.py** (lazy import dentro `format_message_for_trace`)
- **demo_factory.py** (root)

### Script

```bash
mkdir -p src/domain/model
touch src/domain/__init__.py

git mv src/obelix_types/roles.py src/domain/model/roles.py
git mv src/obelix_types/usage.py src/domain/model/usage.py
git mv src/obelix_types/standard_message.py src/domain/model/standard_message.py
git mv src/obelix_types/system_message.py src/domain/model/system_message.py
git mv src/obelix_types/human_message.py src/domain/model/human_message.py
git mv src/obelix_types/assistant_message.py src/domain/model/assistant_message.py
git mv src/obelix_types/tool_message.py src/domain/model/tool_message.py
git mv src/obelix_types/__init__.py src/domain/model/__init__.py
git rm -r src/obelix_types/

# Bulk replace (src/ + demo_factory.py)
find src/ -name "*.py" -exec sed -i 's/src\.obelix_types/src.domain.model/g' {} +
sed -i 's/src\.obelix_types/src.domain.model/g' demo_factory.py
```

### Verifica

```bash
grep -rn "src\.obelix_types" src/ demo_factory.py   # 0 risultati
python -c "from src.domain.model import SystemMessage, ToolCall, Usage; print('OK')"
```

---

## Fase 2: `tools/tool_base.py` + `tools/tool_decorator.py` → `domain/tool/`

Solo 2 file core. MCP e builtin restano in `tools/` fino alla Fase 6.

### File da spostare

| Origine | Destinazione |
|---------|-------------|
| `tools/tool_base.py` | `domain/tool/tool_base.py` |
| `tools/tool_decorator.py` | `domain/tool/tool_decorator.py` |

### Import da aggiornare

Due sed globali:
- `src.tools.tool_base` → `src.domain.tool.tool_base`
- `src.tools.tool_decorator` → `src.domain.tool.tool_decorator`

File impattati:
- **tools/__init__.py** (aggiornare re-export)
- **tools/mcp/mcp_tool.py**
- **tools/tool/ask_user_question_tool.py**
- **base_agent/**: `base_agent.py`, `orchestrator_decorator.py`
- **client_adapters/**: `llm_abstraction.py`, `anthropic_provider.py`, `ibm_provider.py`, `ollama_provider.py`, `vllm_provider.py`, `_legacy_mapping_mixin.py`
- **client_adapters/oci_strategies/**: `base_strategy.py`, `generic_strategy.py`, `cohere_strategy.py`
- **demo_factory.py**

### Script

```bash
mkdir -p src/domain/tool
touch src/domain/tool/__init__.py

git mv src/tools/tool_base.py src/domain/tool/tool_base.py
git mv src/tools/tool_decorator.py src/domain/tool/tool_decorator.py

find src/ -name "*.py" -exec sed -i 's/src\.tools\.tool_base/src.domain.tool.tool_base/g' {} +
find src/ -name "*.py" -exec sed -i 's/src\.tools\.tool_decorator/src.domain.tool.tool_decorator/g' {} +
sed -i 's/src\.tools\.tool_base/src.domain.tool.tool_base/g' demo_factory.py
sed -i 's/src\.tools\.tool_decorator/src.domain.tool.tool_decorator/g' demo_factory.py

# Aggiornare tools/__init__.py per re-export dal nuovo path
```

### Nota su `tools/__init__.py`

Questo file diventa:
```python
from src.domain.tool.tool_base import ToolBase
from src.domain.tool.tool_decorator import tool
```
Così `from src.tools import ToolBase, tool` continua a funzionare fino alla Fase 6.

### Verifica

```bash
grep -rn "from src\.tools\.tool_base\|from src\.tools\.tool_decorator" src/ demo_factory.py  # 0 (tranne tools/__init__.py re-export)
python -c "from src.domain.tool.tool_base import ToolBase; print('OK')"
```

---

## Fase 3: `base_agent/` → `domain/agent/`

### File da spostare

| Origine | Destinazione |
|---------|-------------|
| `base_agent/__init__.py` | `domain/agent/__init__.py` |
| `base_agent/base_agent.py` | `domain/agent/base_agent.py` |
| `base_agent/hooks.py` | `domain/agent/hooks.py` |
| `base_agent/event_contracts.py` | `domain/agent/event_contracts.py` |
| `base_agent/orchestrator_decorator.py` | `domain/agent/orchestrator_decorator.py` |
| `base_agent/subagent_decorator.py` | `domain/agent/subagent_decorator.py` |
| `base_agent/agent_factory.py` | `domain/agent/agent_factory.py` |

### Import da aggiornare

Un sed globale: `src.base_agent` → `src.domain.agent`

File impattati:
- **domain/agent/ interni** (tutti si importano tra loro)
- **demo_factory.py**

Pochi riferimenti esterni perché `base_agent` è in cima alla catena.

### Script

```bash
mkdir -p src/domain/agent

git mv src/base_agent/base_agent.py src/domain/agent/base_agent.py
git mv src/base_agent/hooks.py src/domain/agent/hooks.py
git mv src/base_agent/event_contracts.py src/domain/agent/event_contracts.py
git mv src/base_agent/orchestrator_decorator.py src/domain/agent/orchestrator_decorator.py
git mv src/base_agent/subagent_decorator.py src/domain/agent/subagent_decorator.py
git mv src/base_agent/agent_factory.py src/domain/agent/agent_factory.py
git mv src/base_agent/__init__.py src/domain/agent/__init__.py
git rm -r src/base_agent/

find src/ -name "*.py" -exec sed -i 's/src\.base_agent/src.domain.agent/g' {} +
sed -i 's/src\.base_agent/src.domain.agent/g' demo_factory.py
```

### Verifica

```bash
grep -rn "src\.base_agent" src/ demo_factory.py   # 0 risultati
python -c "from src.domain.agent import BaseAgent; print('OK')"
```

---

## Fase 4: ABC → `ports/outbound/`

Estrae le 3 interfacce astratte.

### File da spostare

| Origine | Destinazione |
|---------|-------------|
| `client_adapters/llm_abstraction.py` | `ports/outbound/llm_provider.py` |
| `connections/llm_connection/base_llm_connection.py` | `ports/outbound/llm_connection.py` |
| `embedding_providers/abstract_embedding_provider.py` | `ports/outbound/embedding_provider.py` |

### Import da aggiornare

Tre sed:
- `src.client_adapters.llm_abstraction` → `src.ports.outbound.llm_provider`
- `src.connections.llm_connection.base_llm_connection` → `src.ports.outbound.llm_connection`
- `src.embedding_providers.abstract_embedding_provider` → `src.ports.outbound.embedding_provider`

**`AbstractLLMProvider`** (~10 file):
- `client_adapters/__init__.py`, tutti i provider, `providers.py`, `domain/agent/base_agent.py`

**`AbstractLLMConnection`** (~6 file):
- `connections/llm_connection/__init__.py`, tutte le connection concrete, `config.py`

**`AbstractEmbeddingProvider`** (~2 file):
- `embedding_providers/__init__.py`, `oci_embedding_provider.py`

### Script

```bash
mkdir -p src/ports/outbound
touch src/ports/__init__.py src/ports/outbound/__init__.py

git mv src/client_adapters/llm_abstraction.py src/ports/outbound/llm_provider.py
git mv src/connections/llm_connection/base_llm_connection.py src/ports/outbound/llm_connection.py
git mv src/embedding_providers/abstract_embedding_provider.py src/ports/outbound/embedding_provider.py

find src/ -name "*.py" -exec sed -i 's/src\.client_adapters\.llm_abstraction/src.ports.outbound.llm_provider/g' {} +
find src/ -name "*.py" -exec sed -i 's/src\.connections\.llm_connection\.base_llm_connection/src.ports.outbound.llm_connection/g' {} +
find src/ -name "*.py" -exec sed -i 's/src\.embedding_providers\.abstract_embedding_provider/src.ports.outbound.embedding_provider/g' {} +
```

### Verifica

```bash
grep -rn "llm_abstraction\|base_llm_connection\|abstract_embedding_provider" src/  # 0
python -c "from src.ports.outbound.llm_provider import AbstractLLMProvider; print('OK')"
```

---

## Fase 5: Provider + Connection → `adapters/outbound/{vendor}/`

La fase più ampia. Ogni provider viene raggruppato con la sua connection.

### File da spostare

| Origine | Destinazione |
|---------|-------------|
| `client_adapters/anthropic_provider.py` | `adapters/outbound/anthropic/provider.py` |
| `connections/llm_connection/anthropic_connection.py` | `adapters/outbound/anthropic/connection.py` |
| `client_adapters/openai_provider.py` | `adapters/outbound/openai/provider.py` |
| `connections/llm_connection/openai_connection.py` | `adapters/outbound/openai/connection.py` |
| `client_adapters/oci_provider.py` | `adapters/outbound/oci/provider.py` |
| `connections/llm_connection/oci_connection.py` | `adapters/outbound/oci/connection.py` |
| `client_adapters/oci_strategies/*` | `adapters/outbound/oci/strategies/*` |
| `client_adapters/ibm_provider.py` | `adapters/outbound/ibm/provider.py` |
| `connections/llm_connection/ibm_connection.py` | `adapters/outbound/ibm/connection.py` |
| `client_adapters/ollama_provider.py` | `adapters/outbound/ollama/provider.py` |
| `client_adapters/vllm_provider.py` | `adapters/outbound/vllm/provider.py` |
| `embedding_providers/oci_embedding_provider.py` | `adapters/outbound/embedding/oci_embedding.py` |
| `client_adapters/_legacy_mapping_mixin.py` | `adapters/outbound/_legacy/mapping_mixin.py` |
| `mapping/provider_mapping.py` | `adapters/outbound/_legacy/provider_mapping.py` |
| `mapping/tool_extr_fall_back.py` | `adapters/outbound/_legacy/tool_extr_fall_back.py` |

### Import da aggiornare

Sed multipli. I pattern principali:

```bash
# Provider
s/src\.client_adapters\.anthropic_provider/src.adapters.outbound.anthropic.provider/g
s/src\.client_adapters\.openai_provider/src.adapters.outbound.openai.provider/g
s/src\.client_adapters\.oci_provider/src.adapters.outbound.oci.provider/g
s/src\.client_adapters\.ibm_provider/src.adapters.outbound.ibm.provider/g
s/src\.client_adapters\.ollama_provider/src.adapters.outbound.ollama.provider/g
s/src\.client_adapters\.vllm_provider/src.adapters.outbound.vllm.provider/g

# Connection
s/src\.connections\.llm_connection\.anthropic_connection/src.adapters.outbound.anthropic.connection/g
s/src\.connections\.llm_connection\.openai_connection/src.adapters.outbound.openai.connection/g
s/src\.connections\.llm_connection\.oci_connection/src.adapters.outbound.oci.connection/g
s/src\.connections\.llm_connection\.ibm_connection/src.adapters.outbound.ibm.connection/g

# Package-level connection imports (from src.connections.llm_connection import X)
# Questi vanno gestiti MANUALMENTE perché il package __init__.py scompare

# OCI strategies
s/src\.client_adapters\.oci_strategies/src.adapters.outbound.oci.strategies/g

# Legacy
s/src\.client_adapters\._legacy_mapping_mixin/src.adapters.outbound._legacy.mapping_mixin/g
s/src\.mapping\.provider_mapping/src.adapters.outbound._legacy.provider_mapping/g
s/src\.mapping\.tool_extr_fall_back/src.adapters.outbound._legacy.tool_extr_fall_back/g
s/src\.mapping/src.adapters.outbound._legacy/g

# Embedding
s/src\.embedding_providers\.oci_embedding_provider/src.adapters.outbound.embedding.oci_embedding/g
s/src\.embedding_providers/src.adapters.outbound.embedding/g
```

### Attenzione: import dal package `connections.llm_connection`

Diversi file importano così:
```python
from src.connections.llm_connection import AnthropicConnection
```

Questo usa il `__init__.py` del package come hub. Dopo lo spostamento, ogni connection è in una cartella diversa. Soluzioni:

**Opzione A**: Creare `adapters/outbound/__init__.py` che re-esporta tutte le connection:
```python
from src.adapters.outbound.anthropic.connection import AnthropicConnection
from src.adapters.outbound.openai.connection import OpenAIConnection
# ...
```

**Opzione B**: Aggiornare ogni import individualmente nei file che li usano (`config.py`, `providers.py`, `demo_factory.py`, ciascun provider).

→ **Scelta: Opzione B** (import espliciti, più pulito per hexagonal).

### Pulizia

```bash
git rm -r src/client_adapters/
git rm -r src/connections/
git rm -r src/embedding_providers/
git rm -r src/mapping/
```

### Verifica

```bash
grep -rn "src\.client_adapters\|src\.connections\|src\.embedding_providers\|src\.mapping" src/ demo_factory.py  # 0
python -c "from src.adapters.outbound.anthropic.provider import AnthropicProvider; print('OK')"
```

---

## Fase 6: Infrastructure + Plugins

### File da spostare (infrastructure)

| Origine | Destinazione |
|---------|-------------|
| `src/config.py` | `infrastructure/config.py` |
| `src/providers.py` | `infrastructure/providers.py` |
| `src/logging_config.py` | `infrastructure/logging.py` |
| `src/k8s_config.py` | `infrastructure/k8s.py` |
| `utility/pydantic_validation.py` | `infrastructure/utility/pydantic_validation.py` |
| `utility/__init__.py` | `infrastructure/utility/__init__.py` |

### File da spostare (plugins)

| Origine | Destinazione |
|---------|-------------|
| `tools/mcp/*` | `plugins/mcp/*` |
| `tools/tool/ask_user_question_tool.py` | `plugins/builtin/ask_user_question_tool.py` |

### Import da aggiornare

```bash
# Infrastructure
s/src\.config/src.infrastructure.config/g
s/src\.providers/src.infrastructure.providers/g
s/src\.logging_config/src.infrastructure.logging/g
s/src\.k8s_config/src.infrastructure.k8s/g
s/src\.utility\.pydantic_validation/src.infrastructure.utility.pydantic_validation/g
s/src\.utility/src.infrastructure.utility/g

# Plugins
s/src\.tools\.mcp/src.plugins.mcp/g
s/src\.tools\.tool\.ask_user_question_tool/src.plugins.builtin.ask_user_question_tool/g
```

File impattati da `src.logging_config` (~15 file): quasi tutti i provider, base_agent, agent_factory, strategies, oci_connection.

File impattati da `src.config` (~3 file): `base_agent.py`, `llm_provider.py` (lazy), `demo_factory.py`.

File impattati da `src.providers` (~9 file): tutti i provider, `config.py`, `provider_mapping.py`, `_legacy_mapping_mixin.py`.

### Pulizia

```bash
git rm -r src/tools/
git rm -r src/utility/
git rm src/config.py src/providers.py src/logging_config.py src/k8s_config.py
```

### Verifica

```bash
grep -rn "src\.logging_config\|src\.config\b\|src\.providers\b\|src\.k8s_config\|src\.utility\|src\.tools" src/ demo_factory.py  # 0
python -c "from src.infrastructure.logging import get_logger; print('OK')"
```

---

## Fase 7: Cleanup Finale

1. **Aggiornare `CLAUDE.md`**: tabella moduli, path, import examples
2. **Aggiornare `MEMORY.md`**: key patterns con nuovi path
3. **Spostare `PROVIDER_MIGRATION_STATUS.md`** se esiste in `client_adapters/`
4. **Eliminare directory vuote residue**
5. **Verificare `setup.py`**: `find_packages(where="src")` funziona automaticamente
6. **Eliminare `obelix_types/TODO.md`** se presente
7. **Test finale completo**:
   ```bash
   python -c "
   from src.domain.model import SystemMessage, HumanMessage, AssistantMessage, ToolMessage, ToolCall
   from src.domain.tool.tool_base import ToolBase
   from src.domain.agent import BaseAgent
   from src.ports.outbound.llm_provider import AbstractLLMProvider
   from src.infrastructure.logging import get_logger
   from src.infrastructure.config import GlobalConfig
   print('ALL OK')
   "
   ```

---

## Rischi e Mitigazioni

| Rischio | Mitigazione |
|---------|-------------|
| Import circolare durante stati intermedi | Ordine foglie-per-prime (model → tool → agent) |
| `__init__.py` mancanti | `touch` subito per ogni nuova directory |
| Lazy import dentro function body | `sed` cattura anche import indentati |
| `__pycache__` stale | `find . -name __pycache__ -exec rm -rf {} +` prima di ogni verifica |
| `from src.connections.llm_connection import X` (package import) | Gestione manuale nella Fase 5 |
| `sed` troppo aggressivo (match parziali) | Ordine sed dal più specifico al più generico |

---

## Stima

~170 import statement da aggiornare in ~40 file. Nessun cambio di logica. 7 commit atomici.