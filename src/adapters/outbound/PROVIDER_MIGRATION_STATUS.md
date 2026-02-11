# Provider Migration: Self-Contained Architecture

## Obiettivo Finale

Eliminare `src/mapping/provider_mapping.py` e `src/mapping/tool_extr_fall_back.py`.
Ogni provider deve essere self-contained: conversione messaggi, tool, e parsing response inline.

---

## Stato Attuale

| Provider | Self-contained? | Usa LegacyMappingMixin? | File |
|----------|----------------|------------------------|------|
| **OCI** | SI (via strategy) | NO | `oci_provider.py` + `oci_strategies/` |
| **Anthropic** | SI (inline) | NO | `anthropic_provider.py` |
| **OpenAI** | SI (inline) | NO | `openai_provider.py` |
| **IBM Watson** | NO | SI | `ibm_provider.py` |
| **Ollama** | NO | SI | `ollama_provider.py` |
| **vLLM** | NO | SI | `vllm_provider.py` |

---

## Cosa e' stato fatto (Step 1)

### 1. `llm_abstraction.py` -> Puro contratto ABC

Rimossi i metodi condivisi che usavano `ProviderRegistry`:
- `_convert_messages_to_provider_format()` (RIMOSSO)
- `_convert_tools_to_provider_format()` (RIMOSSO)
- `_extract_tool_calls()` (RIMOSSO)

Restano:
- `invoke()` - abstract, unico contratto pubblico
- `provider_type` - abstract property
- `_get_connection_from_global_config()` - utility statica condivisa

### 2. `_legacy_mapping_mixin.py` (TEMPORANEO)

Creato mixin con i 3 metodi rimossi dalla base class.
I provider non migrati (OpenAI, IBM, Ollama, vLLM) ora ereditano:
```python
class OpenAIProvider(LegacyMappingMixin, AbstractLLMProvider):
```

**Questo file va eliminato dopo aver migrato tutti i provider.**

### 3. `anthropic_provider.py` - Self-contained

Struttura interna (convenzione, non abstract):
- `_convert_messages(messages) -> Tuple[Optional[str], List[dict]]`
  - Antropic-specifico: separa system message (parametro) dai conversation messages (array)
- `_convert_tools(tools) -> List[dict]`
- `_extract_tool_calls(response) -> List[ToolCall]` con validazione, raises `ToolCallExtractionError`
- `_extract_content(response) -> str`
- `_extract_usage(response) -> Optional[Usage]`
- `_convert_response_to_assistant_message(response) -> AssistantMessage`
- `_get_block_attribute(block, attr)` - helper per dict/object notation

Pattern retry: `MAX_EXTRACTION_RETRIES = 3` con feedback HumanMessage all'LLM.

### 4. Mapping files

- `provider_mapping.py`: rimossa sezione ANTHROPIC + relativo `ProviderRegistry.register()`
- `tool_extr_fall_back.py`: rimossa `_extract_tool_calls_anthropic()`

---

## Step 2: Migrare i restanti 4 provider

### Ordine consigliato

1. ~~**OpenAI** - COMPLETATO~~ (self-contained inline, rimosso da mapping)
2. **IBM Watson** - Simile a OpenAI (OpenAI-compatible format)
3. **Ollama** - Simile a OpenAI (OpenAI-compatible format)
4. **vLLM** - Caso speciale: offline inference, `_extract_tool_calls` riceve anche `tools` come parametro

### Per ogni provider, la checklist e':

1. **Rimuovere `LegacyMappingMixin`** dalla classe:
   ```python
   # DA:
   class OpenAIProvider(LegacyMappingMixin, AbstractLLMProvider):
   # A:
   class OpenAIProvider(AbstractLLMProvider):
   ```

2. **Implementare metodi inline** (stessa convenzione di Anthropic/OCI):
   - `_convert_messages(messages) -> List[dict]`
   - `_convert_tools(tools) -> List[dict]`
   - `_extract_tool_calls(response) -> List[ToolCall]` con validazione
   - `_extract_content(response) -> str`
   - `_extract_usage(response) -> Optional[Usage]`

3. **Aggiungere `ToolCallExtractionError`** locale + retry loop in `invoke()`

4. **Rimuovere mapping** da `provider_mapping.py`:
   - Sezione OPENAI/IBM_WATSON/OLLAMA/VLLM
   - `ProviderRegistry.register()` corrispondente

5. **Rimuovere extractor** da `tool_extr_fall_back.py`:
   - `_extract_tool_calls_openai`, `_extract_tool_calls_ibm_watson_hybrid`, etc.

### Dove trovare la logica da inlineare

Per ogni provider, la logica da copiare inline e' in DUE file:

| Provider | Mapping (conversione msg/tool) | Extractor (tool calls) |
|----------|-------------------------------|----------------------|
| OpenAI | `provider_mapping.py` sezione `OPENAI` (riga ~416) | `tool_extr_fall_back.py` `_extract_tool_calls_openai()` |
| IBM | `provider_mapping.py` sezione `IBM_WATSON` (riga ~82) | `tool_extr_fall_back.py` `_extract_tool_calls_ibm_watson_hybrid()` |
| Ollama | `provider_mapping.py` sezione `OLLAMA` (riga ~299) | `tool_extr_fall_back.py` `_extract_tool_calls_ollama()` |
| vLLM | `provider_mapping.py` sezione `VLLM` (riga ~357) | `tool_extr_fall_back.py` `_extract_tool_calls_vllm()` |

### Note specifiche per provider

**OpenAI:**
- Response: `response.choices[0].message.tool_calls[i].function.{name, arguments}`
- Arguments e' stringa JSON -> serve `json.loads()`
- Usage: `response.usage.{prompt_tokens, completion_tokens, total_tokens}`

**IBM Watson:**
- Response e' un dict (non oggetto): `response["choices"][0]["message"]["tool_calls"]`
- Arguments e' stringa JSON -> serve `json.loads()`
- Ha fallback che parsa tool call dal content (da ELIMINARE, rimpiazzare con retry)

**Ollama:**
- Response e' oggetto: `response.message.tool_calls[i].function.{name, arguments}`
- Arguments puo' essere stringa JSON o dict
- Ha fallback che parsa tool call dal content (da ELIMINARE, rimpiazzare con retry)

**vLLM (caso speciale):**
- Offline inference, nessun client HTTP
- `_extract_tool_calls` riceve ANCHE `tools` come secondo parametro
- Response: `output.outputs[0].text` contiene JSON array di tool calls
- Richiede parsing puro da testo -> piu' delicato, tenere retry robusto

---

## Pulizia finale (dopo Step 2)

Quando TUTTI i provider sono migrati:

1. **Eliminare** `src/mapping/provider_mapping.py`
2. **Eliminare** `src/mapping/tool_extr_fall_back.py`
3. **Eliminare** `src/mapping/__init__.py` (o svuotarlo)
4. **Eliminare** `src/llm_providers/_legacy_mapping_mixin.py`
5. **Rimuovere `ProviderRegistry`** da `src/providers.py` (se non usata altrove)
6. **Aggiornare** `CLAUDE.md` sezione "Pattern Architetturali":
   - Rimuovere "Registry" pattern
   - Aggiungere nota su provider self-contained

---

## Architettura Target

```
AbstractLLMProvider (puro ABC)
    |
    |-- invoke() [abstract]
    |-- provider_type [abstract property]
    |-- _get_connection_from_global_config() [static utility]
    |
    +-- AnthropicProvider (self-contained, inline)
    |       _convert_messages() -> (system, messages)
    |       _convert_tools()
    |       _extract_tool_calls() + ToolCallExtractionError
    |       _extract_content()
    |       _extract_usage()
    |
    +-- OCILLm (self-contained, via strategy pattern)
    |       strategy.convert_messages()
    |       strategy.convert_tools()
    |       strategy.extract_tool_calls() + ToolCallExtractionError
    |       _extract_content()
    |       _extract_usage()
    |
    +-- OpenAIProvider (self-contained, inline) [DONE]
    +-- IBMWatsonXLLm (self-contained, inline) [TODO]
    +-- OllamaProvider (self-contained, inline) [TODO]
    +-- VLLMProvider (self-contained, inline) [TODO]
```

Ogni provider:
- Nessuna dipendenza da `src/mapping/`
- Logica di conversione co-locata nel file del provider
- `ToolCallExtractionError` + retry loop per tool call malformati
- Nessun fallback da content testuale (rimpiazzato da retry con feedback LLM)