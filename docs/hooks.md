# Hooks API (User Guide)

This guide explains how to **use hooks** in the agent system. It is written for developers who want to understand **what hooks represent conceptually** and **how to implement them correctly**.

---

## 1) What is a Hook (conceptually)
A hook is an **interceptor** in the agent lifecycle. It lets you:
- **observe** a specific moment (event)
- **change agent state** (effects)
- **transform the pipeline value** (value)
- **control the flow** (decision)

Think of the agent as a pipeline that produces an answer. A hook is a point where you can step in, inspect the state, and decide what happens next.

---

## 2) Events (where hooks run)
An event is a **named lifecycle moment**. Each event has a **contract** (see Section 6).

Supported events:
- before_llm_call
- after_llm_call
- before_tool_execution
- after_tool_execution
- on_tool_error
- before_final_response
- query_end

### Event meaning (conceptual)
- **before_llm_call**: just before the LLM is invoked.
- **after_llm_call**: immediately after the LLM returned a message.
- **before_tool_execution**: right before a tool call is executed.
- **after_tool_execution**: right after a tool returns a result.
- **on_tool_error**: only when a tool returns an error.
- **before_final_response**: right before building the final response.
- **query_end**: after a response has been prepared (end of execution).

---

## 3) Context (AgentStatus)
Hooks receive a **context** object that represents the current state.

### What it represents
- A snapshot of the **agent state** at the moment of the event.
- A way to access or mutate the conversation history.
- Event-specific data (tool call, tool result, assistant message).

### Fields
- event: current event
- agent: agent instance
- iteration: current iteration number
- tool_call / tool_result / assistant_message / error: event-specific data
- conversation_history: list of messages

**Note:** Only some fields are populated depending on the event. Others are None.

---

## 4) Decision (flow control)
Every hook returns a **decision**, which is the control signal for the pipeline:
- **CONTINUE**: keep going to the next hook.
- **RETRY**: rerun the LLM phase (allowed only on retryable events).
- **FAIL**: stop execution with an error.
- **STOP**: stop execution and return a specific value.

Conceptually:
- **RETRY** is for forcing the LLM to try again (e.g., missing <plan> tag).
- **STOP** is for early exit when you already have a final answer (e.g., cached response).

---

## 5) Value vs Effects
This is the most important distinction.

### Value (pipeline data)
- Represents **the data that flows through the event** (e.g., AssistantMessage, ToolResult).
- Used when you want to **transform** the output of a stage.

### Effects (state mutation)
- Represent **side-effects** on agent state (history, flags, logs).
- Do not change the pipeline value.

**Rule of thumb:**
- Use **value** to change what the pipeline consumes next.
- Use **effects** to change the agent's memory or state.

---

## 6) Event Contracts (very important)
Each event has a contract that defines:
- **input type** (what value comes in)
- **output type** (what value must come out)
- **retryable** (whether RETRY is allowed)
- **stop_output** (what type STOP must return)

These rules are enforced at runtime. If you break them, you get a clear error.

### Contracts summary
- before_llm_call: input=None, output=None, retryable=False, stop_output=AssistantMessage
- after_llm_call: input=AssistantMessage, output=AssistantMessage, retryable=True, stop_output=AssistantMessage
- before_tool_execution: input=ToolCall, output=ToolCall, retryable=False, stop_output=None
- after_tool_execution: input=ToolResult, output=ToolResult, retryable=False, stop_output=None
- on_tool_error: input=ToolResult, output=ToolResult, retryable=False, stop_output=None
- before_final_response: input=AssistantMessage, output=AssistantMessage, retryable=True, stop_output=AssistantMessage
- query_end: input=AssistantResponse or None, output=AssistantResponse or None, retryable=False, stop_output=None

### Why contracts matter
They prevent mistakes such as:
- returning the wrong value type
- retrying an event that cannot be retried
- using STOP where it is not meaningful

---

## 7) How to Build a Callable (mandatory signatures)
This section defines the **required input/output** for every callable you pass to hooks.

### 7.1 Condition callable
Used in `when(...)`.

**Signature**:
```
condition(status) -> bool
```
- Input: `status` (AgentStatus)
- Output: bool
- Sync or async allowed

### 7.2 Value callable
Used in `handle(value=...)`.

**Signature**:
```
value_fn(status, current_value) -> new_value
```
- Inputs:
  - `status` (AgentStatus)
  - `current_value` (pipeline value for the event)
- Output: **new_value** of the type required by the event contract
- Sync or async allowed

If you return the wrong type, execution fails with a runtime error.

### 7.3 Effect callable
Used in `handle(effects=[...])`.

**Signature**:
```
effect_fn(status) -> None
```
- Input: `status` (AgentStatus)
- Output: None (return value is ignored)
- Sync or async allowed

Effects must mutate state through `status` (e.g., history, flags).

---

## 8) Available Values by Event (what you can read/write)
This list tells you **which pipeline value exists** and which **context fields are populated**.

### before_llm_call
- current_value: None
- populated: status.iteration, status.agent, status.conversation_history

### after_llm_call
- current_value: AssistantMessage
- populated: status.assistant_message

### before_tool_execution
- current_value: ToolCall
- populated: status.tool_call

### after_tool_execution
- current_value: ToolResult
- populated: status.tool_result

### on_tool_error
- current_value: ToolResult (error)
- populated: status.tool_result, status.error

### before_final_response
- current_value: AssistantMessage
- populated: status.assistant_message

### query_end
- current_value: AssistantResponse or None
- populated: none specific (only core fields)

---

## 9) API Usage (when + handle)
The hook API is intentionally simple:

```
agent.on(event)
    .when(condition)
    .handle(decision=..., value=..., effects=[...])
```

### when(condition)
- **Mandatory parameterization**: condition must accept `status`.
- Signature: `condition(status) -> bool`
- Can be sync or async.

### handle(...)
Parameters:
- decision: CONTINUE | RETRY | FAIL | STOP
- value: optional
  - direct value
  - or function `value_fn(status, current_value) -> new_value`
- effects: optional list of functions `effect_fn(status) -> None`

---

## 10) Practical Examples

### Example 1: Validate <plan> and force retry
Event: before_final_response
```
self.on(AgentEvent.BEFORE_FINAL_RESPONSE) \
    .when(self._missing_plan_tag) \
    .handle(
        decision=HookDecision.RETRY,
        effects=[self._inject_plan_guidance]
    )
```

### Example 2: Enrich tool error
Event: after_tool_execution
```
self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
    .when(self._is_oracle_error_with_docs) \
    .handle(
        decision=HookDecision.CONTINUE,
        value=self._enrich_with_oracle_docs
    )
```

### Example 3: Inject schema on tool error
Event: on_tool_error
```
self.on(AgentEvent.ON_TOOL_ERROR) \
    .when(self._is_invalid_identifier_error) \
    .handle(
        decision=HookDecision.CONTINUE,
        effects=[self._inject_schema_message]
    )
```

### Example 4: Early exit before LLM
Event: before_llm_call
```
self.on(AgentEvent.BEFORE_LLM_CALL) \
    .when(self._cache_hit) \
    .handle(
        decision=HookDecision.STOP,
        value=self._cached_assistant_message
    )
```

---

## 11) Best Practices
- Use **effects** for state changes (history, flags).
- Use **value** only to transform pipeline data.
- Use **RETRY** only where it makes sense (after_llm_call, before_final_response).
- Do not use STOP on events that do not allow it.
- Keep hooks small and single-purpose.

---

## 12) Common Errors (and why)
- **RETRY on non-retryable event** -> explicit runtime error.
- **STOP without proper value** -> explicit runtime error.
- **Returning wrong value type** -> explicit runtime error.

These errors are intentional guardrails to prevent silent bugs.
