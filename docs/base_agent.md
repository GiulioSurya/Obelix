# BaseAgent Guide

BaseAgent is the core execution engine for all agents in Obelix. It manages conversation history, invokes LLM providers, executes tools, and coordinates hooks throughout the agent lifecycle.

---

## What is BaseAgent?

BaseAgent is the foundation of every agent in Obelix. It:
- Maintains conversation history across multiple LLM calls
- Invokes the LLM provider with appropriate messages and tools
- Executes tool calls (potentially in parallel)
- Coordinates hooks at every lifecycle phase
- Builds the final response from the LLM output
- Supports async execution natively

You create specialized agents by subclassing BaseAgent and registering tools and hooks.

---

## Constructor Parameters

### system_message (str) - Required

The system prompt that defines the agent's identity, role, and behavior.

```python
agent = BaseAgent(
    system_message="You are a helpful data analysis assistant.",
)
```

This message is inserted as the first message in `conversation_history` and remains throughout the agent's lifetime (unless explicitly cleared).

### provider (AbstractLLMProvider | None)

The LLM provider instance to use.

```python
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider

provider = AnthropicProvider(...)
agent = BaseAgent(
    system_message="...",
    provider=provider,
)
```

### agent_comment (bool)

Whether the agent should generate a textual response after tool execution. Default: `True`.

- `True` (default): Agent always generates a text response, even after tool calls
- `False`: Agent may skip generating text if tools return successfully

### max_iterations (int)

Maximum number of LLM/tool loop iterations. Default: `15`.

Prevents infinite loops caused by repeated tool calls or retries. If the limit is reached, the agent returns with an error.

```python
agent = BaseAgent(
    system_message="...",
    max_iterations=10,  # Lower for latency-critical applications
)
```

### max_attempts (int)

Number of retries for fatal errors (provider failures, timeouts, etc.). Default: `3`.

Wraps the entire query execution in a retry loop with exponential backoff.

```python
agent = BaseAgent(
    system_message="...",
    max_attempts=5,  # Higher for unstable networks
)
```

### tools (Tool | Tool class | List) - Optional

Tools to register automatically at construction. Can be:
- A single tool instance: `CalculatorTool()`
- A single tool class (auto-instantiated): `CalculatorTool`
- A list of mixed instances and classes: `[CalculatorTool, WeatherTool()]`

```python
agent = BaseAgent(
    system_message="...",
    tools=[CalculatorTool, WeatherTool()],
)
```

### tool_policy (List[ToolRequirement] | None)

Rules enforcing that certain tools must be called before the agent can return a final response.

```python
from obelix.core.model.tool_message import ToolRequirement

agent = BaseAgent(
    system_message="You are a SQL analyst.",
    tool_policy=[
        ToolRequirement(
            tool_name="sql_executor",
            min_calls=1,
            require_success=True,
            error_message="You must execute the SQL query before responding."
        )
    ]
)
```

If a tool policy is violated, the agent injects guidance and retries, or fails if max iterations is reached.

### exit_on_success (List[str] | None)

List of tool names that, when called successfully, immediately end the agent loop.

```python
agent = BaseAgent(
    system_message="...",
    tools=[RetrieveTool, ProcessTool],
    exit_on_success=["retrieve_tool"],  # Exit after retrieval succeeds
)
```

---

## Creating Custom Agents

Subclass BaseAgent to create specialized agents:

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.tool.tool_decorator import tool
from pydantic import Field


@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")

    async def execute(self) -> dict:
        ops = {
            "add": lambda: self.a + self.b,
            "subtract": lambda: self.a - self.b,
            "multiply": lambda: self.a * self.b,
            "divide": lambda: self.a / self.b if self.b != 0 else None,
        }
        return {"result": ops[self.operation]()}


class MathAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are a helpful math assistant. Use the calculator tool for arithmetic.",
            tools=[CalculatorTool],
        )


# Use the agent
agent = MathAgent()
response = agent.execute_query("What is 42 * 7?")
print(response.content)
```

---

## Registering Tools

### Option 1: Constructor Parameter

```python
agent = BaseAgent(
    system_message="...",
    tools=[CalculatorTool, WeatherTool()],
)
```

### Option 2: register_tool() Method

```python
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")
        self.register_tool(CalculatorTool())
        self.register_tool(WeatherTool())
```

### Option 3: Mixed Approach

```python
agent = BaseAgent(
    system_message="...",
    tools=[CalculatorTool],  # Auto-instantiated
)
agent.register_tool(WeatherTool())  # Manual instantiation with dependencies
```

---

## Registering Sub-Agents

Any agent can register other agents as tools via `register_agent()`:

```python
class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You coordinate specialized tasks.")


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You analyze data.")


# Register sub-agent
coordinator = CoordinatorAgent()
coordinator.register_agent(
    AnalyzerAgent(),
    name="analyzer",
    description="Analyzes datasets and generates insights",
    stateless=False,  # Preserve conversation history
)

response = coordinator.execute_query("Analyze the Q4 sales data")
print(response.content)
```

### register_agent() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `BaseAgent` | Required | The sub-agent to register |
| `name` | `str` | Required | Unique name for the sub-agent (used in tool calling) |
| `description` | `str` | Required | Description of the sub-agent's capabilities |
| `stateless` | `bool` | `False` | If `True`, each execution gets a fresh copy. If `False`, history persists. |

---

## Using the Hook System

Hooks allow you to intercept and modify agent behavior at specific lifecycle moments.

### Basic Hook Example

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.system_message import SystemMessage


class ValidatingAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You are a helpful assistant.")

        # Register a hook that validates the output format
        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            self._missing_answer_tag
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_format_guidance],
        )

    def _missing_answer_tag(self, status) -> bool:
        """Condition: check if response is missing required tag."""
        content = status.assistant_message.content or ""
        return "<answer>" not in content

    def _inject_format_guidance(self, status) -> None:
        """Effect: inject guidance to fix the format."""
        status.agent.conversation_history.append(
            SystemMessage(content="Your response must include <answer>...</answer> tags.")
        )
```

When the agent's final response doesn't contain `<answer>` tags, the hook injects guidance and triggers a retry.

### Available Events

| Event | When It Fires | current_value | Retryable |
|-------|---------------|---------------|-----------|
| `BEFORE_LLM_CALL` | Before each LLM invocation | `None` | No |
| `AFTER_LLM_CALL` | After LLM returns | `AssistantMessage` | Yes |
| `BEFORE_TOOL_EXECUTION` | Before executing a tool | `ToolCall` | No |
| `AFTER_TOOL_EXECUTION` | After tool completes | `ToolResult` | No |
| `ON_TOOL_ERROR` | When a tool fails | `ToolResult` (error) | No |
| `BEFORE_FINAL_RESPONSE` | Before building final response | `AssistantMessage` | Yes |
| `QUERY_END` | After execution completes | `AssistantResponse` | No |

### Hook Decisions

| Decision | Effect |
|----------|--------|
| `CONTINUE` | Proceed normally (default) |
| `RETRY` | Restart the current LLM phase (only for retryable events) |
| `FAIL` | Stop immediately with error |
| `STOP` | Return immediately with a provided value |

---

## Execution Methods

### Synchronous Execution

Use for CLI applications and scripts:

```python
response = agent.execute_query("Your question here")
print(response.content)
```

### Asynchronous Execution

Use in async contexts (FastAPI, asyncio, etc.):

```python
import asyncio

async def main():
    response = await agent.execute_query_async("Your question here")
    print(response.content)

asyncio.run(main())
```

### Response Object

Both methods return an `AssistantResponse` with:

```python
response.content      # Text response from the LLM
response.tool_results  # List of ToolResult objects (if any tools were called)
response.error        # Error message if the query failed
```

---

## Execution Flow

The agent executes in a loop up to `max_iterations`:

```
1. Start new iteration
2. Run BEFORE_LLM_CALL hooks
3. Invoke LLM with conversation history + registered tools
4. Run AFTER_LLM_CALL hooks
5. If LLM returns tool calls:
   a. Run BEFORE_TOOL_EXECUTION hooks for each tool
   b. Execute tools (potentially in parallel)
   c. Run AFTER_TOOL_EXECUTION or ON_TOOL_ERROR hooks
   d. Add ToolResult to conversation history
   e. Return to step 2 (unless exit_on_success triggered)
6. If LLM returns final response:
   a. Run BEFORE_FINAL_RESPONSE hooks
   b. Build AssistantResponse
   c. Run QUERY_END hooks
   d. Return response
```

---

## Tool Policy Enforcement

Tool policies guarantee that specific tools are called before the agent can return a final answer:

```python
from obelix.core.model.tool_message import ToolRequirement

agent = BaseAgent(
    system_message="You are a database consultant.",
    tool_policy=[
        ToolRequirement(
            tool_name="query_executor",
            min_calls=1,
            require_success=True,
            error_message="Always execute the SQL query before giving recommendations."
        ),
    ]
)
```

If the policy is violated:
1. The agent receives an injected system message explaining the requirement
2. The agent retries (if iterations remain)
3. If max iterations is exceeded, the agent fails with an error

---

## Accessing Conversation History

The `conversation_history` list contains all messages exchanged:

```python
# Access the current history
print(agent.conversation_history)

# Check the history after a query
response = agent.execute_query("Tell me a joke")
print(f"History has {len(agent.conversation_history)} messages")

# Get a copy (safe from modifications)
history_copy = agent.get_conversation_history()

# Clear history (keeps system message)
agent.clear_conversation_history()
```

---

## Common Patterns

### Error Recovery with Hooks

```python
class RobustAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You are a helpful assistant.")

        self.on(AgentEvent.ON_TOOL_ERROR).when(
            lambda status: "database" in (status.error or "").lower()
        ).handle(
            decision=HookDecision.CONTINUE,
            effects=[self._inject_db_help],
        )

    def _inject_db_help(self, status):
        """Provide database schema when a tool fails."""
        status.agent.conversation_history.append(
            SystemMessage(content="Here is the database schema...")
        )
```

### Context Injection Before LLM Calls

```python
class ContextAwareAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.BEFORE_LLM_CALL).handle(
            decision=HookDecision.CONTINUE,
            effects=[self._inject_context],
        )

    async def _inject_context(self, status):
        """Inject real-time context before each LLM call."""
        context = await self._fetch_context()
        status.agent.conversation_history.append(
            SystemMessage(content=f"Current context: {context}")
        )

    async def _fetch_context(self):
        # Fetch real-time data, query a database, etc.
        return "Today is Monday, Q4 revenue is up 20%"
```

### Output Transformation

```python
class TransformingAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.QUERY_END).handle(
            decision=HookDecision.CONTINUE,
            value=self._transform_response,
        )

    def _transform_response(self, status, response):
        """Transform the final response."""
        if response and response.content:
            response.content = response.content.upper()
        return response
```

---

## Tips and Best Practices

1. **Use Pydantic Fields for Tools**: Always define tool parameters with `Field()` for automatic schema generation and validation.

2. **Prefer Async**: Write tool `execute()` methods as async when possible for better scalability.

3. **Keep System Prompts Focused**: Be specific about the agent's role and constraints in the system message.

4. **Use Hooks Sparingly**: Hooks are powerful but can make debugging harder. Use them for cross-cutting concerns, not business logic.

5. **Stateless for Parallelism**: When registering sub-agents that may be called many times in parallel, use `stateless=True`.

6. **Monitor Iterations**: If agents frequently hit `max_iterations`, your system prompt or tool definitions may need adjustment.

---

## Full Example

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.tool.tool_decorator import tool
from obelix.core.model.system_message import SystemMessage
from pydantic import Field


@tool(name="database_query", description="Execute a SQL query")
class DatabaseQueryTool:
    query: str = Field(..., description="SQL query to execute")

    async def execute(self) -> dict:
        # Simulate database query
        return {"result": "Query executed successfully"}


class DataAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are a data analyst. Use the database_query tool to retrieve data.",
            tools=[DatabaseQueryTool],
            max_iterations=10,
        )

        # Hook: Validate that a query was executed
        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            self._must_have_queried
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_query_reminder],
        )

    def _must_have_queried(self, status) -> bool:
        """Check if database_query tool was called."""
        return not any(
            "database_query" in str(msg)
            for msg in status.agent.conversation_history
        )

    def _inject_query_reminder(self, status):
        """Remind the agent to use the database_query tool."""
        status.agent.conversation_history.append(
            SystemMessage(content="Remember to query the database before analyzing.")
        )


# Use the agent
agent = DataAnalystAgent()
response = agent.execute_query("Analyze Q4 sales trends")
print(response.content)
```

---

## See Also

- [Agent Factory](agent_factory.md) - Composing agents with factories
- [Hooks API](hooks.md) - Detailed hook system documentation
- [README](../README.md) - Installation and quick start
