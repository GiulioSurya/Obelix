# TODO - Messages Module

## ToolMessage.content ridondante

**Data analisi**: 2026-01-14

### Problema
Il campo `content` in `ToolMessage` viene auto-generato da `_generate_content_summary()` ma **non viene mai usato**. Tutti i provider mapping iterano direttamente su `tool_results`.

### Evidenze

**Provider mapping analizzati** (in `src/mapping/provider_mapping.py`):
- Anthropic: usa `result.tool_call_id`, `result.result`
- OpenAI: usa `result.tool_call_id`, `result.result`, `result.error`
- OCI Generic: usa `result.tool_call_id`, `result.result`, `result.error`
- OCI Cohere: usa `result.tool_name`, `result.result`, `result.error`
- Ollama: usa `result.tool_call_id`, `result.result`, `result.error`
- vLLM: usa `result.tool_call_id`, `result.result`, `result.error`
- WatsonX: usa `result.tool_call_id`, `result.result`, `result.error`

**Nessun mapping usa `msg.content`** per ToolMessage.

**Verifica su SophIA_IC**: stesso risultato, `content` mai usato.

### Azione intrapresa
- Commentato il codice di auto-generazione in `__init__`
- Marcato `_generate_content_summary` come DEPRECATED
- Il metodo resta disponibile per backward compatibility

### Possibili azioni future

1. **Rimuovere completamente** `_generate_content_summary` in una release major

2. **Convertire in @property** per lazy evaluation (on-demand):
   ```python
   @property
   def summary(self) -> str:
       return "; ".join(...)
   ```

3. **Valutare rimozione di BaseMessage** - altri campi potenzialmente ridondanti:
   - `role`: non usato per routing (si usa `isinstance`)
   - `timestamp`: mai usato nel codice, solo serializzazione
   - `metadata`: mai usato nel codice, solo serializzazione

### Note
Il routing dei messaggi in `_convert_messages_to_provider_format()` usa `isinstance()` sulle classi concrete, non il campo `role`.
