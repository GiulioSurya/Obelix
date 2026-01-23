# BaseAgent (Usage Guide)

This document explains how to use `BaseAgent`, how the execution flow works, and how to build agents on top of it.

---

## 1) What BaseAgent Represents
`BaseAgent` is the core execution engine for an LLM-driven agent. It:
- manages conversation history
- calls the LLM provider
- executes tools (if any)
- coordinates hooks
- builds the final response

You subclass `BaseAgent` to create specialized agents, then register hooks and tools.

---

## 2) Constructor Parameters (detailed)

### system_message (str)
- **What it is:** the system prompt that defines the agent's identity and behavior.
- **How it is used:** inserted as the first message in `conversation_history`.
- **How to parameterize:** you can format it with variables before passing it.

### provider (AbstractLLMProvider | None)
- **What it is:** the LLM provider instance.
- **How it is used:** if None, the global provider from config is used.
- **How to parameterize:** pass a custom provider instance if you want a specific model or backend.

### agent_comment (bool)
- **What it is:** whether the agent should generate a textual response after tool execution.
- **How it is used:** if False and tools return without error, the agent may stop after tool results.
- **How to parameterize:** set to False for tool-only workflows, True for chat-style responses.

### max_iterations (int)
- **What it is:** maximum LLM/tool loop iterations per query.
- **How it is used:** limits retries and tool chaining.
- **How to parameterize:** increase for complex workflows, decrease for latency control.

### max_attempts (int)
- **What it is:** number of retries for fatal errors (e.g., provider failures).
- **How it is used:** wraps the full execution in a retry loop.
- **How to parameterize:** keep small (3-5) for production; larger for unstable environments.

### tools (ToolBase | Tool class | List)
- **What it is:** tools to auto-register at construction.
- **How it is used:** tool instances are registered into `registered_tools`.
- **How to parameterize:**
  - pass a tool instance
  - pass a tool class (it will be instantiated)
  - pass a list of mixed instances/classes

### tool_policy (List[ToolRequirement] | None)
- **What it is:** rules enforcing that certain tools must be called before final response.
- **How it is used:** when violated, the agent injects a SystemMessage and retries or fails.
- **How to parameterize:**
  - `tool_name`: tool identifier to enforce
  - `min_calls`: minimum number of calls required
  - `require_success`: whether those calls must succeed
  - `error_message`: custom message injected when violated

---

## 3) Construction Examples

### Minimal example
```
agent = BaseAgent(
    system_message="You are a helpful assistant.",
)
```

### Full constructor
```
agent = BaseAgent(
    system_message="You are a specialized agent.",
    provider=my_provider,
    agent_comment=True,
    max_iterations=10,
    max_attempts=3,
    tools=[MyTool(), OtherTool],
    tool_policy=[
        ToolRequirement(tool_name="sql_query_executor", min_calls=1, require_success=True)
    ],
)
```

---

## 4) Execution Flow (Conceptual)

1) Input validation
2) LLM loop (up to max_iterations):
   - before_llm_call hooks
   - LLM invocation
   - after_llm_call hooks
   - tool calls (if any)
   - before_final_response hooks
   - query_end hooks
3) Final response built

The loop can be controlled with hook decisions (RETRY/STOP/FAIL).

---

## 5) Tool Execution
Tools are executed in parallel with asyncio.gather. Each tool call runs through:
- before_tool_execution hooks
- tool execution
- after_tool_execution hooks
- on_tool_error hooks (only if error)

---

## 6) Tool Policy
Tool policy enforces that certain tools must be called before a final response.

Example:
```
policy = [
    ToolRequirement(tool_name="sql_query_executor", min_calls=1, require_success=True)
]
agent = BaseAgent(system_message="...", tool_policy=policy)
```

If the policy is violated, the agent injects a system message and either retries or fails (based on iteration count).

---

## 7) Building a Custom Agent

### Example skeleton
```
class MyAgent(BaseAgent):
    def __init__(self, provider=None):
        super().__init__(
            system_message="You are a specialized agent.",
            provider=provider,
        )

        # Register hooks
        self.on(AgentEvent.BEFORE_FINAL_RESPONSE) \
            .when(self._missing_plan) \
            .handle(
                decision=HookDecision.RETRY,
                effects=[self._inject_plan_message]
            )

        # Register tools (optional)
        self.register_tool(MyTool())

    def _missing_plan(self, status: AgentStatus) -> bool:
        return "<plan>" not in (status.assistant_message.content or "")

    def _inject_plan_message(self, status: AgentStatus) -> None:
        status.conversation_history.append(SystemMessage(content="Add <plan> tag."))
```

---

## 8) Decorators: @subagent and @orchestrator

These decorators are **optional** and are used when you want agent coordination.
They do not change hook behavior, but they allow agents to be registered and orchestrated.

### @subagent
Marks a BaseAgent subclass as registrable by orchestrators.

Example:
```
from src.base_agent import subagent

@subagent(name="sql_analyzer", description="Analyzes SQL errors")
class SqlAnalyzerAgent(BaseAgent):
    ...
```

**Rules:**
- `name` is required
- `description` is required
- used for discovery/registration

### @orchestrator
Marks a BaseAgent subclass as an orchestrator that can register subagents.

Example:
```
from src.base_agent import orchestrator

@orchestrator(name="coordinator", description="Coordinates tasks")
class CoordinatorAgent(BaseAgent):
    def __init__(self, provider=None):
        super().__init__(system_message="You coordinate tasks", provider=provider)

    def setup(self):
        # register subagents
        self.register_agent(SqlAnalyzerAgent())
        self.register_agent(OtherSubAgent())
```

### Registering a subagent (explicit)
```
coordinator = CoordinatorAgent()
coordinator.register_agent(SqlAnalyzerAgent())
```

---

## 9) Running Queries

### Async
```
response = await agent.execute_query_async("user query")
```

### Sync
```
response = agent.execute_query("user query")
```

`response` is an `AssistantResponse` containing:
- content
- tool results (if any)
- error (if any)

---

## 10) Common Pitfalls
- Using RETRY on non-retryable events (runtime error).
- Returning the wrong value type for an event (runtime error).
- Forgetting to register required tools when tool_policy is enabled.

---

## 11) Where to Look Next
- `src/base_agent/hooks.py`: hook API
- `src/base_agent/event_contracts.py`: event contracts
- `src/messages/tool_message.py`: ToolRequirement
- `src/base_agent/subagent_decorator.py`: @subagent
- `src/base_agent/orchestrator_decorator.py`: @orchestrator
