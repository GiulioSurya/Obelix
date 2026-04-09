---
name: creating-agents
description: Use when creating, configuring, or composing Obelix agents — covers BaseAgent, tools, hooks, sub-agents, AgentFactory, shared memory, streaming, tool policy, and MCP integration. Trigger keywords - agent, tool, hook, sub-agent, factory, orchestration, shared memory, streaming
---

# Creating Agents with Obelix

## Overview

Obelix is a Python framework for building **multi-provider LLM agents** with tools, hooks, sub-agent orchestration, and shared memory. The core loop is:

```
User Query -> Agent -> Provider -> LLM -> Tool Execution -> Response
                |
         Hooks intercept every phase
```

**Requires Python >= 3.13.** Install: `uv sync` (core) or `uv sync --all-extras` (all providers).

---

## 1. Providers

A provider connects your agent to an LLM. Every provider implements `AbstractLLMProvider` with `invoke()` and `provider_type`.

### Anthropic (native)

```python
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider

connection = AnthropicConnection(api_key="sk-ant-...")
provider = AnthropicProvider(
    connection=connection,
    model_id="claude-sonnet-4-20250514",  # or claude-haiku-4-5-20251001
    max_tokens=3000,
    temperature=0.1,
    # thinking_mode=True,         # Extended thinking (forces temperature=1)
    # thinking_params={"budget_tokens": 2000},
)
```

### OpenAI (native)

```python
from obelix.adapters.outbound.openai.connection import OpenAIConnection
from obelix.adapters.outbound.openai.provider import OpenAIProvider

connection = OpenAIConnection(api_key="sk-...", base_url=None)
provider = OpenAIProvider(
    connection=connection,
    model_id="gpt-4o",
    max_tokens=4096,
    temperature=0.1,
    # tool_choice="auto",          # "auto", "required", "none"
    # parallel_tool_calls=True,
    # reasoning_effort="high",     # "low", "medium", "high"
)
```

### LiteLLM (universal — 100+ providers)

```python
from obelix.adapters.outbound.litellm.provider import LiteLLMProvider

# Routes via model_id prefix: anthropic/, openai/, ollama/, azure/, etc.
provider = LiteLLMProvider(
    model_id="anthropic/claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",      # or set env vars
    # base_url="http://...",   # for proxies/self-hosted
    max_tokens=4096,
    drop_params=True,          # silently drop unsupported params (recommended)
)

# Ollama local
provider = LiteLLMProvider(model_id="ollama/llama3", base_url="http://localhost:11434")
```

Install: `uv sync --extra litellm`

### Other native providers

| Provider | Install extra | Connection class | Provider class |
|----------|--------------|-----------------|----------------|
| OCI Generative AI | `--extra oci` | `OCIConnection` | `OCILLm` |
| IBM Watson | `--extra ibm` | `IBMConnection` | `IBMProvider` |
| Ollama | `--extra ollama` | — | `OllamaProvider` |
| vLLM | `--extra vllm` | — | `VLLMProvider` |

All providers are self-contained: message conversion is inline, no central registry.

---

## 2. BaseAgent

```python
from obelix.core.agent import BaseAgent

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
    # max_iterations=15,              # max agentic loop iterations
    # tools=[MyTool(), OtherTool],    # accepts instances OR classes
    # tool_policy=[ToolRequirement(...)],
    # exit_on_success=["tool_a"],     # stop loop after these tools succeed
    # response_schema=MyOutputModel,  # force structured JSON output
    # planning=True,                  # plan-before-act protocol
    # tracer=tracer,                  # distributed tracing
    # agent_name="my_agent",          # custom name (default: class name)
)
```

### Execution modes

```python
# Synchronous (uses asyncio.run internally — do NOT call from async context)
response = agent.execute_query("What is 2+2?")

# Asynchronous (for FastAPI, async scripts)
response = await agent.execute_query_async("What is 2+2?")

# Streaming (real-time tokens)
async for event in agent.execute_query_stream("Explain quantum physics"):
    if event.token:
        print(event.token, end="", flush=True)
    if event.is_final:
        response = event.assistant_response
```

### AssistantResponse

```python
response.content        # str — final text
response.agent_name     # str
response.tool_results   # list[ToolResult] | None
response.error          # str | None
```

### Multi-turn conversations

```python
from obelix.core.model import HumanMessage

# Pass a list of messages for multi-turn
response = agent.execute_query([
    HumanMessage(content="My name is Alice"),
    HumanMessage(content="What is my name?"),
])

# Or build on previous conversation (history is preserved automatically)
agent.execute_query("My name is Alice")
agent.execute_query("What is my name?")  # agent remembers

# Reset history
agent.clear_conversation_history(keep_system_message=True)
```

### Structured output

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int
    country: str

agent = BaseAgent(
    system_message="Extract city information.",
    provider=provider,
    response_schema=CityInfo,  # LLM forced to return this JSON shape
)
response = agent.execute_query("Tell me about Rome")
# response.content is valid JSON matching CityInfo
```

### Cancellation

```python
import asyncio

async def run_with_timeout():
    task = asyncio.create_task(agent.execute_query_async("Long task..."))
    await asyncio.sleep(5)
    agent.cancel()  # cooperative — checked 3 times per iteration
    response = await task
    # response may be partial or contain cancellation message
```

### Subclassing pattern

```python
class MyAgent(BaseAgent):
    """Custom agent with fixed configuration."""

    # Pydantic Fields become sub-agent input parameters when registered
    context: str = Field(default="", description="Additional context")

    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a specialist agent.",
            **kwargs,  # provider, max_iterations, etc. passed through
        )
        self.register_tool(MyTool())
```

---

## 3. Tools

### @tool decorator

```python
from pydantic import Field
from obelix.core.tool.tool_decorator import tool

@tool(name="weather", description="Get weather for a city")
class WeatherTool:
    city: str = Field(..., description="City name")
    units: str = Field(default="celsius", description="Temperature units")

    async def execute(self) -> dict:
        # self.city and self.units are already populated from LLM arguments
        return {"city": self.city, "temp": 22, "units": self.units}
```

**Key behaviors:**
- `execute()` can be **sync or async** (sync runs in `asyncio.to_thread`)
- Fields are Pydantic — validation happens automatically, errors returned to LLM
- Tools in the same iteration execute **in parallel** (thread-safe via `copy.copy`)
- Errors auto-truncated to 2000 chars (head + tail)

### Tool registration

```python
# Via constructor — accepts BOTH instances and classes (classes auto-instantiated)
agent = BaseAgent(tools=[WeatherTool(), CalculatorTool], ...)

# Dynamic registration — accepts ONLY instances (not classes)
agent.register_tool(WeatherTool())
# agent.register_tool(WeatherTool)  # ERROR: register_tool requires an instance
```

### Dependency injection in tools

Tools can have a custom `__init__` for dependencies — Fields are still extracted from class annotations:

```python
@tool(name="db_query", description="Query the database")
class DbQueryTool:
    sql: str = Field(..., description="SQL query to execute")

    def __init__(self, db_connection):
        self.db = db_connection  # injected dependency

    async def execute(self) -> dict:
        # self.sql populated by LLM, self.db from __init__
        rows = await self.db.execute(self.sql)
        return {"rows": rows, "count": len(rows)}

# Usage
tool = DbQueryTool(db_connection=my_db)
agent.register_tool(tool)
```

### system_prompt_fragment

Tools can enrich the agent's system prompt automatically:

```python
@tool(name="search", description="Search the knowledge base")
class SearchTool:
    query: str = Field(..., description="Search query")

    def system_prompt_fragment(self) -> str | None:
        """Appended to agent's system message on registration."""
        return "\n\nWhen using the search tool, prefer specific queries over broad ones."

    async def execute(self) -> dict:
        return {"results": [...]}
```

The fragment is appended to `agent.system_message.content` when `register_tool()` is called.

### Deferred tools (client-executed)

For tools that must run on the client side (e.g., shell commands in A2A):

```python
from pydantic import BaseModel

@tool(name="bash", description="Execute shell command", is_deferred=True)
class RemoteBashTool:
    command: str = Field(..., description="Shell command")

    class OutputSchema(BaseModel):
        """Validates client response. Optional but recommended."""
        stdout: str
        stderr: str
        exit_code: int

    def execute(self) -> None:
        return None  # returning None signals: defer to client
```

Deferred tools stop the agent loop. The client executes the tool, then resumes with `agent.resume_after_deferred()`.

### Tool inspection

```python
WeatherTool.print_schema()  # prints JSON schema to stdout (classmethod)
```

---

## 4. Hooks

Hooks intercept agent lifecycle events with a fluent API.

### Events

| Event | When | Can RETRY? | Pipeline value |
|-------|------|-----------|----------------|
| `BEFORE_LLM_CALL` | Before `provider.invoke()` | No | — |
| `AFTER_LLM_CALL` | After LLM response | Yes | `AssistantMessage` |
| `BEFORE_TOOL_EXECUTION` | Before each `tool.execute()` | Yes | `ToolCall` |
| `AFTER_TOOL_EXECUTION` | After each tool execution | Yes | `ToolResult` |
| `ON_TOOL_ERROR` | Tool returned ERROR status | Yes | `ToolResult` |
| `BEFORE_FINAL_RESPONSE` | Before building response | Yes | `AssistantMessage` |
| `QUERY_END` | End of execution | No | — |

### Fluent API

```python
from obelix.core.agent import AgentEvent, HookDecision

agent.on(AgentEvent.AFTER_LLM_CALL)\
    .when(condition_func)\           # Optional: activate only when True
    .handle(
        decision=HookDecision.RETRY, # CONTINUE | RETRY | FAIL | STOP | REJECT
        value=transform_func,        # Optional: transform pipeline value
        effects=[side_effect_1],     # Optional: run before value transform
    )
```

### Decisions

| Decision | Effect |
|----------|--------|
| `CONTINUE` | Normal flow (default) |
| `RETRY` | Re-run the iteration (only on retryable events) |
| `FAIL` | Terminate with error |
| `STOP` | Stop loop and return value |
| `REJECT` | Reject task (maps to A2A `TaskState.rejected`) |

### AgentStatus (passed to conditions and effects)

```python
@dataclass
class AgentStatus:
    event: AgentEvent
    agent: BaseAgent           # full agent access
    iteration: int = 0
    tool_call: ToolCall | None
    tool_result: ToolResult | None
    assistant_message: AssistantMessage | None
    error: str | None

    @property
    def conversation_history(self) -> list[StandardMessage]:
        """Direct access to agent's conversation history"""
```

### Examples

**Logging:**
```python
agent.on(AgentEvent.BEFORE_LLM_CALL).handle(
    HookDecision.CONTINUE,
    effects=[lambda s: print(f"LLM call #{s.iteration}")]
)
```

**Retry on bad response:**
```python
agent.on(AgentEvent.BEFORE_FINAL_RESPONSE)\
    .when(lambda s: "I don't know" in (s.assistant_message.content or ""))\
    .handle(
        HookDecision.RETRY,
        effects=[lambda s: s.agent.conversation_history.append(
            HumanMessage(content="Try harder. Do not say 'I don't know'.")
        )]
    )
```

**Reject dangerous queries:**
```python
agent.on(AgentEvent.BEFORE_LLM_CALL)\
    .when(lambda s: "hack" in s.conversation_history[-1].content.lower())\
    .reject("Cannot assist with hacking-related queries.")
```

Catch rejections:
```python
from obelix.core.agent import TaskRejectedError

try:
    response = agent.execute_query("hack the system")
except TaskRejectedError as e:
    print(f"Rejected: {e}")
```

**Transform tool results:**
```python
agent.on(AgentEvent.AFTER_TOOL_EXECUTION)\
    .when(lambda s: s.tool_result.tool_name == "search")\
    .handle(
        HookDecision.CONTINUE,
        value=lambda s, result: ToolResult(
            tool_name=result.tool_name,
            tool_call_id=result.tool_call_id,
            result={"filtered": result.result["data"][:5]},
            status=result.status,
        )
    )
```

**Conditions can be async:**
```python
agent.on(AgentEvent.BEFORE_LLM_CALL)\
    .when(async_check_rate_limit)\
    .handle(HookDecision.FAIL, value="Rate limit exceeded")
```

---

## 5. Sub-Agent Orchestration

### Direct registration (without factory)

```python
researcher = BaseAgent(system_message="You research topics.", provider=provider)
writer = BaseAgent(system_message="You write reports.", provider=provider)

coordinator = BaseAgent(system_message="Delegate to your agents.", provider=provider)
coordinator.register_agent(researcher, name="researcher", description="Finds information on any topic")
coordinator.register_agent(writer, name="writer", description="Formats research into reports")

response = coordinator.execute_query("Write a report about Python")
# Coordinator calls researcher and writer as tools
```

### Stateless vs Stateful

```python
# Stateless (default=False → stateful): each call gets isolated copy, parallel-safe
coordinator.register_agent(agent, name="a", description="...", stateless=True)

# Stateful: serialized access (asyncio.Lock), shared conversation history
coordinator.register_agent(agent, name="b", description="...", stateless=False)
```

**Use stateless=True** when the sub-agent is called multiple times with independent queries.
**Use stateless=False** when the sub-agent needs to remember context across calls.

### Sub-agent input fields

When you subclass BaseAgent with Pydantic Fields, those become extra input parameters:

```python
class AnalysisAgent(BaseAgent):
    language: str = Field(default="en", description="Output language")
    depth: str = Field(default="brief", description="Analysis depth: brief or detailed")

    def __init__(self, **kwargs):
        super().__init__(system_message="You analyze data.", **kwargs)

# When registered as sub-agent, the LLM sees:
# - query (str, required) — always added automatically
# - language (str, optional)
# - depth (str, optional)
```

---

## 6. AgentFactory

For declarative multi-agent composition with shared configuration.

```python
from obelix.core.agent import AgentFactory

factory = AgentFactory(global_defaults={"provider": provider, "max_iterations": 20})
```

### Registration

```python
factory.register(
    "researcher",
    ResearcherAgent,
    subagent_description="Finds factual information on any topic",  # REQUIRED for sub-agents
    stateless=True,
    defaults={"max_iterations": 10},  # per-agent overrides
)

factory.register(
    "writer",
    WriterAgent,
    subagent_name="report_writer",  # custom tool name (default: registration name)
    subagent_description="Produces polished reports from research data",
)

factory.register("coordinator", CoordinatorAgent)

# Chaining supported
factory.register("a", A, subagent_description="...").register("b", B, subagent_description="...")
```

### Creation

```python
coordinator = factory.create(
    "coordinator",
    subagents=["researcher", "writer"],
    # subagent_config={"researcher": {"max_iterations": 5}},  # per-subagent overrides
    # **overrides: any BaseAgent constructor param
)
```

The factory automatically:
- Injects coordination protocol into the orchestrator's system message
- Injects sub-agent protocol into each sub-agent's system message
- Computes execution order from memory graph (topological sort)

### Standalone agents

```python
# Factory can create agents without sub-agents too
agent = factory.create("researcher")
```

### Introspection

```python
factory.get_registered_names()   # ["researcher", "writer", "coordinator"]
factory.is_registered("writer")  # True
factory.get_spec("researcher")   # AgentSpec dataclass
```

---

## 7. Shared Memory

Enables data flow between sub-agents via a directed acyclic graph.

```python
from obelix.core.agent import SharedMemoryGraph
from obelix.core.agent.shared_memory import PropagationPolicy

memory = SharedMemoryGraph()

# add_edge auto-creates nodes — no need to call add_agent()
memory.add_edge("researcher", "writer", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
memory.add_edge("researcher", "reviewer", policy=PropagationPolicy.LAST_TOOL_RESULT)
# Raises ValueError if edge would create a cycle
```

### Propagation policies

| Policy | What flows |
|--------|-----------|
| `FINAL_RESPONSE_ONLY` (default) | Agent's final text response |
| `LAST_TOOL_RESULT` | Last successful tool result |

### Wiring to factory

```python
factory.with_memory_graph(memory)
# Now all agents created by factory share this memory graph
# Hooks auto-register: BEFORE_LLM_CALL injects context, BEFORE_FINAL_RESPONSE publishes
```

### Manual usage (without factory)

```python
memory = SharedMemoryGraph()
memory.add_edge("a", "b")

agent_a = BaseAgent(system_message="...", provider=provider)
agent_a._memory_graph = memory
agent_a._agent_id = "a"

# publish/pull manually
await memory.publish("a", "result content", kind="final")
items = memory.pull_for("b")  # returns list[MemoryItem] from a
```

---

## 8. Tool Policy

Force the agent to use specific tools before responding.

```python
from obelix.core.model.tool_message import ToolRequirement

policy = [
    ToolRequirement(
        tool_name="search",
        min_calls=1,              # must call at least once
        require_success=True,     # calls must succeed (status=SUCCESS)
        error_message="You must search before answering.",  # feedback to LLM
    ),
    ToolRequirement(tool_name="validate", min_calls=1),
]

agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[SearchTool(), ValidateTool()],
    tool_policy=policy,
)
```

Enforcement: if the agent tries to respond without satisfying requirements, a RETRY is triggered with an error message injected into conversation. Fails after `max_iterations`.

---

## 9. Planning Protocol

Appends structured instructions to the system message for plan-before-act behavior.

```python
agent = BaseAgent(
    system_message="You are a research assistant.",
    provider=provider,
    planning=True,
)
```

The protocol instructs the LLM to:
1. ANALYZE the request
2. DECOMPOSE into discrete steps
3. EXECUTE step by step
4. REVISE if failures occur
5. RESPOND with final answer

---

## 10. Streaming

Real-time token output with full tool execution transparency.

```python
from obelix.core.model.assistant_message import StreamEvent

async for event in agent.execute_query_stream("Explain relativity"):
    if event.token:
        print(event.token, end="", flush=True)
    if event.deferred_tool_calls:
        # Client must execute these, then call resume_after_deferred()
        for tc in event.deferred_tool_calls:
            print(f"Deferred: {tc.name}({tc.arguments})")
    if event.canceled:
        print("Agent was canceled")
    if event.is_final:
        response = event.assistant_response
```

**StreamEvent fields:**
- `token: str | None` — text chunk
- `is_final: bool` — last event, carries complete response
- `assistant_response: AssistantResponse | None` — on final event
- `deferred_tool_calls: list[ToolCall] | None` — pending client tools
- `canceled: bool` — cancellation flag

Tokens are yielded only on iterations without tool calls. Tool execution happens silently between streaming chunks.

Requires provider to implement `invoke_stream()`. Falls back to `invoke()` automatically if not supported.

---

## 11. MCP Integration

Connect to MCP servers and use their tools as agent tools.

```python
from obelix.plugins.mcp.mcp_client_manager import MCPConfig
from obelix.plugins.mcp.run_time_manager import MCPRuntimeManager

# STDIO transport — local MCP server
config = MCPConfig(
    name="my_server",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
    env={"API_KEY": "..."},
)

# Streamable HTTP — remote MCP server
config = MCPConfig(
    name="tavily",
    transport="streamable-http",
    url="https://mcp.tavily.com/mcp/?tavilyApiKey=...",
    # headers={"Authorization": "Bearer ..."},  # for auth
)

# Create runtime manager (synchronous interface, thread-safe)
manager = MCPRuntimeManager(config)

# Get tools and register with agent
for mcp_tool in manager.get_tools():
    agent.register_tool(mcp_tool)
```

MCPRuntimeManager:
- Auto-connects on first use
- Thread-safe with internal queue
- Auto-cleanup at program exit
- Automatic reconnection on errors

---

## 12. Tracing

Instrument agent execution for observability.

```python
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import ConsoleExporter, HTTPExporter

# Console (development)
tracer = Tracer(exporter=ConsoleExporter(verbosity=3), service_name="my-app")

# HTTP (production — sends to obelix-tracer backend)
tracer = Tracer(exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1"))

# Wire to single agent
agent = BaseAgent(system_message="...", provider=provider, tracer=tracer)

# Wire to factory (all agents)
factory.with_tracer(tracer)
```

Span types: `AGENT`, `LLM`, `TOOL`, `SUB_AGENT`, `HOOK`, `MEMORY`.

---

## 13. Messages & Attachments

```python
from obelix.core.model import SystemMessage, HumanMessage, AssistantMessage, ToolMessage
from obelix.core.model.content import TextContent, FileContent, DataContent

# Text message
msg = HumanMessage(content="Hello")

# Message with image attachment
msg = HumanMessage(
    content="What's in this image?",
    attachments=[
        FileContent(data="base64...", mime_type="image/png", filename="screenshot.png")
    ]
)

# Message with structured data
msg = HumanMessage(
    content="Analyze this data",
    attachments=[
        DataContent(data={"users": 100, "revenue": 50000})
    ]
)

# Message with URL
msg = HumanMessage(
    content="Describe this",
    attachments=[
        FileContent(data="https://example.com/image.png", mime_type="image/png", is_url=True)
    ]
)
```

---

## Quick Reference

| What | How |
|------|-----|
| Create agent | `BaseAgent(system_message=..., provider=...)` |
| Run sync | `agent.execute_query("...")` |
| Run async | `await agent.execute_query_async("...")` |
| Stream | `async for event in agent.execute_query_stream("...")` |
| Add tool | `agent.register_tool(MyTool())` or `tools=[MyTool()]` |
| Add sub-agent | `agent.register_agent(sub, name=..., description=...)` |
| Add hook | `agent.on(AgentEvent.XXX).when(...).handle(...)` |
| Reject | `agent.on(...).when(...).reject("reason")` |
| Cancel | `agent.cancel()` |
| Structured output | `response_schema=MyModel` |
| Force tool use | `tool_policy=[ToolRequirement(...)]` |
| Plan mode | `planning=True` |
| Shared memory | `SharedMemoryGraph().add_edge(src, dst)` |
| MCP tools | `MCPRuntimeManager(config).get_tools()` |
| Trace | `Tracer(exporter=ConsoleExporter())` |
| Token usage | `agent.agent_usage` (input_tokens, output_tokens, requests) |
| History | `agent.get_conversation_history()` / `agent.clear_conversation_history()` |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Calling `execute_query()` from async context | Use `execute_query_async()` — sync version calls `asyncio.run()` internally |
| Forgetting `subagent_description` in factory | Required when agent is listed in `subagents=[]` |
| Using RETRY on non-retryable event | Only `AFTER_LLM_CALL`, `BEFORE/AFTER_TOOL_EXECUTION`, `ON_TOOL_ERROR`, `BEFORE_FINAL_RESPONSE` support RETRY |
| Mutating tool state across calls | Tools are `copy.copy()`-ed per execution — don't rely on shared mutable state |
| Creating cycles in SharedMemoryGraph | `add_edge()` raises ValueError — redesign the flow |
| Not handling `TaskRejectedError` | Catch it when using rejection hooks |
| Using `is_deferred=True` without returning None | Deferred tools MUST return None from execute() to signal deferral |
| Passing a class to `register_tool()` | `register_tool()` requires instances; only the `tools=` constructor param auto-instantiates classes |
