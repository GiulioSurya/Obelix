# Hooks System Guide

Hooks are interceptors that let you observe and modify agent behavior at specific lifecycle moments. They enable validation, error recovery, context injection, and flow control without changing the core agent logic.

---

## Core Concepts

### What is a Hook?

A hook is a named lifecycle event where you can:
1. **Observe** the current state via `AgentStatus`
2. **Change state** via side effects (modifying conversation history, flags, etc.)
3. **Transform values** that flow through the pipeline
4. **Control flow** with decisions (CONTINUE, RETRY, FAIL, STOP)

### Hook vs Effect vs Value

- **Hook**: The registration point (what event, when to trigger, what to do)
- **Effect**: A function that mutates state (e.g., injecting a message into history). Returns None.
- **Value**: A function that transforms the pipeline data (e.g., modifying a message). Returns the modified value.

**Rule of thumb:**
- Use **effects** to change agent memory or state
- Use **value** to transform what flows through the pipeline

---

## Lifecycle Events

### BEFORE_LLM_CALL

Fires before each LLM invocation (at the start of every iteration).

- **current_value**: `None`
- **Retryable**: No
- **Typical use**: Inject real-time context, validate history

```python
self.on(AgentEvent.BEFORE_LLM_CALL).handle(
    decision=HookDecision.CONTINUE,
    effects=[self._inject_context],
)
```

### AFTER_LLM_CALL

Fires after the LLM returns with a message.

- **current_value**: `AssistantMessage`
- **Retryable**: Yes
- **Typical use**: Validate LLM output, transform response format

```python
self.on(AgentEvent.AFTER_LLM_CALL).when(
    lambda s: "INVALID" in (s.assistant_message.content or "")
).handle(
    decision=HookDecision.RETRY,
    effects=[self._inject_correction_guidance],
)
```

### BEFORE_TOOL_EXECUTION

Fires before executing each tool call.

- **current_value**: `ToolCall`
- **Retryable**: No
- **Typical use**: Log tool invocations, validate tool arguments

```python
self.on(AgentEvent.BEFORE_TOOL_EXECUTION).handle(
    decision=HookDecision.CONTINUE,
    effects=[self._log_tool_call],
)
```

### AFTER_TOOL_EXECUTION

Fires after a tool completes successfully.

- **current_value**: `ToolResult`
- **Retryable**: No
- **Typical use**: Enrich tool results, log execution

```python
self.on(AgentEvent.AFTER_TOOL_EXECUTION).handle(
    decision=HookDecision.CONTINUE,
    value=self._enrich_result,  # Transform the ToolResult
)
```

### ON_TOOL_ERROR

Fires when a tool returns an error (including validation failures).

- **current_value**: `ToolResult` (with error status)
- **Retryable**: No
- **Typical use**: Provide error context, suggest recovery actions

```python
self.on(AgentEvent.ON_TOOL_ERROR).when(
    lambda s: "database" in (s.error or "").lower()
).handle(
    decision=HookDecision.CONTINUE,
    effects=[self._inject_database_schema],
)
```

### BEFORE_FINAL_RESPONSE

Fires before building the final response (when the LLM stops requesting tools).

- **current_value**: `AssistantMessage`
- **Retryable**: Yes
- **Typical use**: Validate output format, enforce required fields

```python
self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
    lambda s: "<answer>" not in (s.assistant_message.content or "")
).handle(
    decision=HookDecision.RETRY,
    effects=[self._inject_format_guidance],
)
```

### QUERY_END

Fires after execution completes (successful or failed).

- **current_value**: `AssistantResponse | None`
- **Retryable**: No
- **Typical use**: Cleanup, logging, telemetry

```python
self.on(AgentEvent.QUERY_END).handle(
    decision=HookDecision.CONTINUE,
    effects=[self._log_execution],
)
```

---

## Hook Decisions

Every hook must return a decision that controls the pipeline:

| Decision | Effect | Allowed Events |
|----------|--------|----------------|
| `CONTINUE` | Proceed normally | All |
| `RETRY` | Restart LLM phase | `AFTER_LLM_CALL`, `BEFORE_FINAL_RESPONSE` |
| `FAIL` | Stop with error | All |
| `STOP` | Return immediately with value | All (but requires specific value type) |

### CONTINUE

The default. Execution proceeds as normal:

```python
self.on(AgentEvent.BEFORE_LLM_CALL).handle(decision=HookDecision.CONTINUE)
```

### RETRY

Restart the LLM phase. Only valid for retryable events:

```python
self.on(AgentEvent.AFTER_LLM_CALL).when(
    self._output_invalid
).handle(decision=HookDecision.RETRY)
```

If used on a non-retryable event, raises `RuntimeError`.

### FAIL

Stop immediately with an error:

```python
self.on(AgentEvent.BEFORE_LLM_CALL).when(
    lambda s: not self._has_context(s)
).handle(decision=HookDecision.FAIL)
```

### STOP

Return immediately with a provided value. Each event has specific `STOP` constraints:

```python
self.on(AgentEvent.BEFORE_LLM_CALL).when(
    lambda s: self._cache_hit(s)
).handle(
    decision=HookDecision.STOP,
    value=self._cached_response(),  # Must be AssistantMessage
)
```

---

## AgentStatus - The Hook Context

Every hook receives an `AgentStatus` object with the current state:

```python
@dataclass
class AgentStatus:
    event: AgentEvent               # Which event fired
    agent: BaseAgent                # The agent instance
    iteration: int                  # Current iteration number
    tool_call: Optional[ToolCall]  # Set for tool execution events
    tool_result: Optional[ToolResult]  # Set for tool result events
    assistant_message: Optional[AssistantMessage]  # Set for LLM events
    error: Optional[str]            # Set for error events
```

Access conversation history via `status.agent.conversation_history`.

---

## Hook Registration API

### Basic Pattern

```python
self.on(event)                  # Register hook for event
    .when(condition)            # Optional: when to trigger
    .handle(
        decision=...,           # How to proceed
        value=...,              # Optional: transform value
        effects=...,            # Optional: side effects
    )
```

### when(condition)

Optional condition function. If provided, hook only triggers when `condition(status)` returns `True`:

```python
self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
    lambda status: "<answer>" not in (status.assistant_message.content or "")
).handle(
    decision=HookDecision.RETRY,
    effects=[...],
)
```

Condition must be a callable with signature:
```python
def condition(status: AgentStatus) -> bool:
    ...
```

Can be sync or async.

### handle(decision, value=None, effects=None)

Define what happens when the hook triggers:

| Parameter | Type | Description |
|-----------|------|-------------|
| `decision` | `HookDecision` | `CONTINUE`, `RETRY`, `FAIL`, or `STOP` |
| `value` | callable or direct value | Optional: transform pipeline data |
| `effects` | `List[callable]` | Optional: side effects to execute |

---

## Effects and Values

### Effects (State Mutations)

Effects modify agent state. They receive `status` and return `None`:

```python
def _inject_guidance(status: AgentStatus) -> None:
    """Add a message to conversation history."""
    status.agent.conversation_history.append(
        SystemMessage(content="Follow this format...")
    )

self.on(AgentEvent.BEFORE_FINAL_RESPONSE).handle(
    decision=HookDecision.CONTINUE,
    effects=[_inject_guidance],
)
```

Effects can be sync or async:

```python
async def _fetch_and_inject_context(status: AgentStatus) -> None:
    """Async effect that fetches data before injecting."""
    context = await self._fetch_from_db()
    status.agent.conversation_history.append(
        SystemMessage(content=f"Context: {context}")
    )

self.on(AgentEvent.BEFORE_LLM_CALL).handle(
    decision=HookDecision.CONTINUE,
    effects=[_fetch_and_inject_context],  # Async OK
)
```

### Values (Transformations)

Values transform the current pipeline value. They receive `(status, current_value)` and return the transformed value:

```python
def _uppercase_response(status: AgentStatus, msg: AssistantMessage) -> AssistantMessage:
    """Transform response to uppercase."""
    msg.content = (msg.content or "").upper()
    return msg

self.on(AgentEvent.BEFORE_FINAL_RESPONSE).handle(
    decision=HookDecision.CONTINUE,
    value=_uppercase_response,
)
```

The returned value type must match the event contract. Signature:
```python
def transform(status: AgentStatus, current_value: T) -> T:
    ...
```

Can be sync or async.

---

## Complete Hook Examples

### Example 1: Output Validation

Ensure final response contains required tags:

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.system_message import SystemMessage


class ValidatingAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            self._missing_answer_tag
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_format_guidance],
        )

    def _missing_answer_tag(self, status) -> bool:
        content = status.assistant_message.content or ""
        return "<answer>" not in content

    def _inject_format_guidance(self, status) -> None:
        status.agent.conversation_history.append(
            SystemMessage(content="Format your answer with <answer>...</answer> tags.")
        )
```

### Example 2: Error Recovery

Provide schema when database errors occur:

```python
class RobustAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.ON_TOOL_ERROR).when(
            lambda s: "database" in (s.error or "").lower()
        ).handle(
            decision=HookDecision.CONTINUE,
            effects=[self._inject_schema],
        )

    def _inject_schema(self, status) -> None:
        schema = "Tables: users (id, name), orders (id, user_id, total)"
        status.agent.conversation_history.append(
            SystemMessage(content=f"Database schema:\n{schema}")
        )
```

### Example 3: Context Injection

Inject real-time data before each LLM call:

```python
class ContextAwareAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.BEFORE_LLM_CALL).handle(
            decision=HookDecision.CONTINUE,
            effects=[self._inject_current_context],
        )

    async def _inject_current_context(self, status) -> None:
        # Fetch real-time data
        timestamp = datetime.now().isoformat()
        stock_price = await self._fetch_stock_price()

        msg = SystemMessage(
            content=f"Current time: {timestamp}, Stock price: ${stock_price}"
        )
        status.agent.conversation_history.append(msg)

    async def _fetch_stock_price(self) -> float:
        # Connect to API or database
        ...
```

### Example 4: Conditional Early Exit

Return cached response without calling LLM:

```python
class CachedAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")
        self._cache = {}

        self.on(AgentEvent.BEFORE_LLM_CALL).when(
            self._cache_hit
        ).handle(
            decision=HookDecision.STOP,
            value=self._get_cached_response,
        )

    def _cache_hit(self, status) -> bool:
        query = status.agent.conversation_history[-1].content
        return query in self._cache

    def _get_cached_response(self, status):
        query = status.agent.conversation_history[-1].content
        cached_text = self._cache[query]
        return AssistantMessage(content=cached_text)
```

### Example 5: Result Enrichment

Add additional information to tool results:

```python
class EnrichingAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.AFTER_TOOL_EXECUTION).handle(
            decision=HookDecision.CONTINUE,
            value=self._enrich_result,
        )

    def _enrich_result(self, status, result: ToolResult) -> ToolResult:
        if result.tool_name == "fetch_data":
            # Add metadata
            result.result["enriched_at"] = datetime.now().isoformat()
            result.result["count"] = len(result.result.get("items", []))
        return result
```

---

## Tips and Best Practices

1. **Use `when()` to limit scope**: Only run expensive effects when needed.

   ```python
   self.on(AgentEvent.ON_TOOL_ERROR).when(
       lambda s: "database" in (s.error or "")
   ).handle(...)
   ```

2. **Keep effects simple**: Complex logic makes debugging harder. Use hooks for cross-cutting concerns.

3. **Use retryable events for control flow**: Only AFTER_LLM_CALL and BEFORE_FINAL_RESPONSE support RETRY.

4. **Async effects for I/O**: Write async effects when fetching external data.

   ```python
   async def fetch_and_inject(status):
       data = await self._fetch_data()
       status.agent.conversation_history.append(...)
   ```

5. **Be careful with STOP**: It bypasses normal execution. Use only for well-defined scenarios like caching.

6. **Test hooks independently**: Create test fixtures that exercise each hook.

---

## Event Contract Reference

Quick reference of what's available in each event:

| Event | iteration | tool_call | tool_result | assistant_message | error |
|-------|:---------:|:---------:|:-----------:|:-----------------:|:-----:|
| BEFORE_LLM_CALL | x | - | - | - | - |
| AFTER_LLM_CALL | x | - | - | x | - |
| BEFORE_TOOL_EXECUTION | x | x | - | - | - |
| AFTER_TOOL_EXECUTION | x | - | x | - | - |
| ON_TOOL_ERROR | x | - | x | - | x |
| BEFORE_FINAL_RESPONSE | x | - | - | x | - |
| QUERY_END | x | - | - | - | - |

---

## See Also

- [BaseAgent Guide](base_agent.md) - Creating and using agents
- [Agent Factory](agent_factory.md) - Composing agents
- [README](../README.md) - Installation and quick start
