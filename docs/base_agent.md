# BaseAgent Guide

BaseAgent is the core execution engine for all agents in Obelix. It manages conversation history, invokes LLM providers, executes tools (in parallel when possible), coordinates hooks, and supports real-time token streaming.

---

## Quick Start

```python
from obelix.core.agent import BaseAgent
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider

connection = AnthropicConnection()  # reads ANTHROPIC_API_KEY from env
provider = AnthropicProvider(connection=connection, model_id="claude-sonnet-4-20250514")

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
)

response = agent.execute_query("Hello!")
print(response.content)
```

---

## Constructor Parameters

```python
BaseAgent(
    system_message: str,              # Required
    provider: AbstractLLMProvider,     # Required
    max_iterations: int = 15,
    tools: Tool | list[Tool] | None = None,
    tool_policy: list[ToolRequirement] | None = None,
    exit_on_success: list[str] | None = None,
    response_schema: type[BaseModel] | None = None,
    tracer: Tracer | None = None,
    planning: bool = False,
)
```

### system_message (str) — Required

The system prompt that defines the agent's identity, role, and behavior. Stored as the first entry in `conversation_history` and persists for the agent's lifetime.

```python
agent = BaseAgent(
    system_message="You are a data analyst. Always back up claims with numbers.",
    provider=provider,
)
```

> **Note**: Tools can enrich the system message automatically via `system_prompt_fragment()`. See [System Prompt Fragments](#system-prompt-fragments).

### provider (AbstractLLMProvider) — Required

The LLM provider instance to use. See [README — Using Providers](../README.md#using-providers) for all supported providers.

```python
from obelix.adapters.outbound.litellm import LiteLLMProvider

provider = LiteLLMProvider(model_id="openai/gpt-4o")
agent = BaseAgent(system_message="...", provider=provider)
```

### max_iterations (int)

Maximum number of LLM/tool loop iterations. Default: `15`. Prevents infinite loops caused by repeated tool calls. When exceeded, the agent returns an error response.

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    max_iterations=5,  # stricter limit for latency-critical apps
)
```

### tools

Tools to register at construction time. Accepts a single tool instance, a tool class (auto-instantiated), or a list of both.

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[CalculatorTool, WeatherTool()],  # class or instance
)
```

See [Tools Guide](tools.md) for how to create tools with the `@tool` decorator.

### tool_policy

Rules that force the agent to call specific tools before returning a final response. If violated, the agent injects guidance and retries.

```python
from obelix.core.model.tool_message import ToolRequirement

agent = BaseAgent(
    system_message="You are a SQL analyst.",
    provider=provider,
    tool_policy=[
        ToolRequirement(
            tool_name="sql_executor",
            min_calls=1,
            require_success=True,
            error_message="You must execute the SQL query before responding.",
        )
    ],
)
```

### exit_on_success

List of tool names that, when all tools in an iteration are in this set and succeed, immediately end the agent loop without waiting for a text response.

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[RetrieveTool, StoreTool],
    exit_on_success=["store_data"],  # exit as soon as store succeeds
)
```

### response_schema

A Pydantic `BaseModel` subclass that constrains the LLM's final response to a specific JSON structure. The schema is passed to the provider (requires provider support for structured output).

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    summary: str = Field(..., description="Brief summary")
    confidence: float = Field(..., description="Confidence score 0-1")
    sources: list[str] = Field(default_factory=list)

agent = BaseAgent(
    system_message="You analyze documents.",
    provider=provider,
    response_schema=AnalysisResult,
)

response = agent.execute_query("Analyze this quarterly report...")
# response.content is a JSON string matching AnalysisResult schema
```

### tracer

Optional `Tracer` instance for observability. When set, the agent emits spans for every LLM call, tool execution, and query lifecycle event. Zero overhead when `None`.

The simplest setup uses `ConsoleExporter`, which prints a colored trace to the terminal:

```python
from obelix.core.tracer import Tracer, ConsoleExporter

tracer = Tracer(
    exporter=ConsoleExporter(verbosity=2),  # 1=minimal, 2=standard, 3=debug
    service_name="my-service",
)

agent = BaseAgent(system_message="...", provider=provider, tracer=tracer)
```

The console output shows the full agent lifecycle: LLM calls with token usage, tool executions with arguments/results, sub-agent invocations, and a summary footer with duration and stats.

### planning

When `True`, appends a structured planning protocol to the system message that instructs the agent to: **ANALYZE** the request, **DECOMPOSE** into steps, **EXECUTE** step by step, **REVISE** if needed, **RESPOND** with a synthesis.

```python
agent = BaseAgent(
    system_message="You are a research assistant.",
    provider=provider,
    planning=True,
)
```

For best results, pair with high reasoning effort on the provider:
- Anthropic: `thinking_mode=True`
- OpenAI/LiteLLM: `reasoning_effort="high"`

---

## Execution Methods

### Synchronous

For scripts and CLI applications:

```python
response = agent.execute_query("What is 42 * 7?")
print(response.content)
```

### Asynchronous

For async contexts (FastAPI, asyncio):

```python
response = await agent.execute_query_async("What is 42 * 7?")
print(response.content)
```

### Streaming

Yields tokens in real time as they arrive from the LLM. Intermediate iterations (tool calls) are handled internally — only the final response tokens are streamed to the consumer.

```python
async for event in agent.execute_query_stream("Explain quantum computing"):
    if event.token:
        print(event.token, end="", flush=True)
    if event.is_final:
        response = event.assistant_response
        print()  # newline after streaming completes
```

If the provider does not support streaming, the agent falls back to `invoke()` automatically and emits the full response as a single token event.

### Query Input Types

All execution methods accept either a plain string or a message list:

```python
# String (most common)
response = agent.execute_query("Hello!")

# Message list (for injecting extra context alongside the user query)
from obelix.core.model import HumanMessage, SystemMessage

response = agent.execute_query([
    SystemMessage(content="The user is a premium subscriber."),
    HumanMessage(content="Help me with my account."),
])
```

Message lists must contain exactly one `HumanMessage`.

### Response Object

All methods return an `AssistantResponse`:

```python
response.content        # str — final text from the LLM
response.agent_name     # str — class name of the agent
response.tool_results   # list[ToolResult] | None — all tool results collected
response.error          # str | None — error description if something failed
```

### StreamEvent

When using `execute_query_stream()`, each yielded event is a `StreamEvent`:

```python
event.token                 # str | None — a text chunk (None on non-text events)
event.is_final              # bool — True on the last event
event.assistant_message     # AssistantMessage | None — set on final event
event.assistant_response    # AssistantResponse | None — set on final event
event.deferred_tool_calls   # list[ToolCall] | None — set when deferred tools stop the loop
```

---

## Execution Flow

The agent runs in a loop (up to `max_iterations`):

```
1. BEFORE_LLM_CALL hooks
2. Invoke LLM (with conversation history + registered tools)
3. AFTER_LLM_CALL hooks
4. If tool calls:
   a. BEFORE_TOOL_EXECUTION hook (per tool)
   b. Execute tools in parallel
   c. AFTER_TOOL_EXECUTION / ON_TOOL_ERROR hooks
   d. If deferred tools detected → stop loop, yield deferred_tool_calls
   e. If exit_on_success matches → go to step 6
   f. Otherwise → back to step 1
5. If text response:
   a. BEFORE_FINAL_RESPONSE hooks (can trigger RETRY)
6. Build AssistantResponse
7. QUERY_END hooks
8. Return
```

---

## Registering Tools

### At Construction

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[CalculatorTool, WeatherTool()],
)
```

### After Construction

```python
agent = BaseAgent(system_message="...", provider=provider)
agent.register_tool(CalculatorTool())
agent.register_tool(WeatherTool())
```

### System Prompt Fragments

When a tool defines a `system_prompt_fragment()` method, `register_tool()` automatically appends the returned text to the agent's system message. This allows tools to inject environment context (shell info, DB schema, API docs) without the caller having to know about it.

```python
@tool(name="database_query", description="Executes SQL against the warehouse")
class DatabaseQueryTool:
    query: str = Field(..., description="SQL query")

    async def execute(self) -> dict:
        ...

    def system_prompt_fragment(self) -> str:
        return (
            "\n\n## Database Environment\n"
            "- Engine: PostgreSQL 16\n"
            "- Schema: public (tables: users, orders, products)\n"
            "- Max result rows: 1000"
        )
```

When this tool is registered, the agent's system message automatically gains the `## Database Environment` block. The LLM sees it as part of its instructions and can generate better queries.

Built-in example: `BashTool` injects a `## Shell Environment` fragment with platform, shell path, working directory, and drive mounts. See [Tools Guide](tools.md) for details.

---

## Registering Sub-Agents

Any agent can register other agents as callable tools via `register_agent()`:

```python
coordinator = BaseAgent(system_message="You coordinate tasks.", provider=provider)

analyzer = BaseAgent(
    system_message="You analyze datasets.",
    provider=provider,
    tools=[SQLTool],
)

coordinator.register_agent(
    analyzer,
    name="analyzer",
    description="Analyzes datasets and generates insights",
    stateless=False,
)

response = coordinator.execute_query("Analyze the Q4 sales data")
```

### register_agent() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `BaseAgent` | Required | The sub-agent instance |
| `name` | `str` | Required | Tool name the LLM uses to call this sub-agent |
| `description` | `str` | Required | What the sub-agent does (shown to LLM in tool list) |
| `stateless` | `bool` | `False` | `True`: each call gets a fresh copy (parallel-safe). `False`: conversation history persists across calls. |

For large multi-agent systems, consider using `AgentFactory` instead — it handles registration, shared memory, and dependency wiring. See [Agent Factory Guide](agent_factory.md).

---

## Deferred Tools and Resume

Some tools are **deferred** — they don't execute on the server but delegate to the client (e.g., the client's terminal for `BashTool`, or a human for `RequestUserInputTool`).

When the LLM calls a deferred tool:

1. The agent loop **stops** and yields a `StreamEvent` with `deferred_tool_calls` set
2. The caller (e.g., A2A executor or CLI runner) executes the tool client-side
3. The caller injects the `ToolMessage` result into `conversation_history`
4. The caller resumes the loop with `resume_after_deferred()`

```python
async for event in agent.execute_query_stream("List files in /tmp"):
    if event.deferred_tool_calls:
        # Client-side: execute the deferred tool
        for call in event.deferred_tool_calls:
            result = execute_locally(call)  # your logic
            agent.conversation_history.append(
                ToolMessage(tool_results=[result])
            )

        # Resume the agent loop
        async for resumed_event in agent.resume_after_deferred():
            if resumed_event.token:
                print(resumed_event.token, end="")
            if resumed_event.is_final:
                response = resumed_event.assistant_response

    elif event.token:
        print(event.token, end="")

    elif event.is_final:
        response = event.assistant_response
```

See [Tools Guide — Deferred Tools](tools.md) for how to create deferred tools.

---

## Hooks

Hooks intercept the agent lifecycle at specific events. Use them for validation, error recovery, context injection, and output transformation.

### Quick Example

```python
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.system_message import SystemMessage


class ValidatingAgent(BaseAgent):
    def __init__(self, provider):
        super().__init__(system_message="You are a helpful assistant.", provider=provider)

        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            lambda status: "<answer>" not in (status.assistant_message.content or "")
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_format_guidance],
        )

    def _inject_format_guidance(self, status):
        status.agent.conversation_history.append(
            SystemMessage(content="Wrap your answer in <answer>...</answer> tags.")
        )
```

### Available Events

| Event | When | Retryable |
|-------|------|-----------|
| `BEFORE_LLM_CALL` | Before each LLM invocation | No |
| `AFTER_LLM_CALL` | After LLM returns | Yes |
| `BEFORE_TOOL_EXECUTION` | Before executing a tool | No |
| `AFTER_TOOL_EXECUTION` | After tool completes | No |
| `ON_TOOL_ERROR` | When a tool fails | No |
| `BEFORE_FINAL_RESPONSE` | Before building final response | Yes |
| `QUERY_END` | After execution completes | No |

### Hook Decisions

| Decision | Effect |
|----------|--------|
| `CONTINUE` | Proceed normally (default) |
| `RETRY` | Restart the current LLM phase (only retryable events) |
| `STOP` | Return immediately with a provided value |
| `FAIL` | Stop with error |

For the full hook API, conditions, effects, and advanced patterns, see [Hooks Guide](hooks.md).

---

## Conversation History

The agent maintains a `conversation_history` list containing all messages exchanged (system, human, assistant, tool).

```python
# Access history
print(len(agent.conversation_history))

# Get a safe copy
history = agent.get_conversation_history

# Clear history (keeps system message by default)
agent.clear_conversation_history()

# Clear everything including system message
agent.clear_conversation_history(keep_system_message=False)
```

---

## Creating Custom Agents

Subclass `BaseAgent` to create reusable specialized agents:

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.system_message import SystemMessage
from obelix.core.tool.tool_decorator import tool
from pydantic import Field


@tool(name="sql_query", description="Execute a SQL query")
class SQLQueryTool:
    query: str = Field(..., description="SQL query to execute")

    async def execute(self) -> dict:
        # Your database logic here
        return {"rows": [...], "count": 42}


class DataAnalystAgent(BaseAgent):
    def __init__(self, provider):
        super().__init__(
            system_message=(
                "You are a data analyst. Use the sql_query tool to retrieve data. "
                "Always show the SQL you executed."
            ),
            provider=provider,
            tools=[SQLQueryTool],
            max_iterations=10,
            planning=True,
        )

        # Ensure the agent actually queries before responding
        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            self._no_query_executed
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._remind_to_query],
        )

    def _no_query_executed(self, status) -> bool:
        return not any(
            isinstance(msg, ToolMessage)
            for msg in self.conversation_history
        )

    def _remind_to_query(self, status):
        status.agent.conversation_history.append(
            SystemMessage(content="You must query the database before responding.")
        )


# Usage
agent = DataAnalystAgent(provider=provider)
response = agent.execute_query("What were Q4 sales trends?")
print(response.content)
```

---

## See Also

- [Tools Guide](tools.md) — creating tools, deferred tools, OutputSchema, built-in tools
- [Agent Factory](agent_factory.md) — composing agents with shared memory
- [A2A Server](a2a_server.md) — exposing agents as HTTP services
- [Hooks Guide](hooks.md) — full hook API and patterns