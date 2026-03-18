# Agent Factory Guide

The Agent Factory is the centralized API for creating and composing agents. It handles agent registration, sub-agent wiring, shared memory graphs, tracing, and A2A server deployment.

---

## Quick Start

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory


class AnalyzerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(system_message="You analyze data.", **kwargs)


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(system_message="You coordinate tasks.", **kwargs)


factory = AgentFactory()
factory.register("analyzer", AnalyzerAgent, subagent_description="Analyzes data")
factory.register("coordinator", CoordinatorAgent)

coordinator = factory.create("coordinator", subagents=["analyzer"])
response = coordinator.execute_query("Analyze the Q4 report")
print(response.content)
```

---

## Factory Initialization

```python
from obelix.core.agent.agent_factory import AgentFactory

# Basic
factory = AgentFactory()

# With global defaults applied to all agents
factory = AgentFactory(global_defaults={"max_iterations": 10})
```

Global defaults are overridden by per-agent defaults, subagent config, and create-time overrides:

```
global_defaults < spec.defaults < subagent_config[name] < create(**overrides)
```

---

## Registering Agents

```python
factory.register(
    name="analyzer",
    cls=AnalyzerAgent,
    subagent_description="Analyzes datasets and generates insights",
    stateless=True,
    defaults={"max_iterations": 5},
)
```

### register() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Unique registry key |
| `cls` | type[BaseAgent] | Required | Agent subclass |
| `subagent_description` | str | None | Description for LLM (required when used as sub-agent) |
| `subagent_name` | str | None | Tool name when used as sub-agent (defaults to `name`) |
| `stateless` | bool | `False` | `True`: isolated copy per call (parallel-safe). `False`: shared history |
| `defaults` | dict | None | Default constructor kwargs for this agent |

### Method Chaining

All configuration methods return `self`:

```python
factory.register("analyzer", AnalyzerAgent, subagent_description="Analyzes data") \
       .register("designer", DesignerAgent, subagent_description="Designs systems") \
       .register("coordinator", CoordinatorAgent)
```

---

## Creating Agents

### Standalone

```python
analyzer = factory.create("analyzer")
response = analyzer.execute_query("Analyze this dataset")
```

### With Overrides

```python
analyzer = factory.create("analyzer", max_iterations=5)
```

### Orchestrator with Sub-Agents

```python
coordinator = factory.create(
    "coordinator",
    subagents=["analyzer", "designer"],
)
```

Each sub-agent is automatically registered as a tool on the orchestrator. The LLM can invoke them via function calling.

### Per-Subagent Configuration

```python
coordinator = factory.create(
    "coordinator",
    subagents=["analyzer", "designer"],
    subagent_config={
        "analyzer": {"max_iterations": 3},
        "designer": {"max_iterations": 10, "provider": custom_provider},
    },
)
```

> **Note**: Instance-based sub-agents (passing a `BaseAgent` instance in the `subagents` list) are not supported via factory. Use `agent.register_agent()` directly for pre-built instances.

---

## Stateless vs Stateful Sub-Agents

| `stateless` | History | Parallel Calls | Use Case |
|-------------|---------|----------------|----------|
| `False` (default) | Persists across calls | Serialized (lock) | Context-aware agents |
| `True` | Fresh copy per call | Concurrent | One-shot tasks (translation, validation) |

```python
# Stateful: remembers previous interactions
factory.register("analyst", AnalystAgent, subagent_description="...", stateless=False)

# Stateless: isolated, parallel-safe
factory.register("translator", TranslatorAgent, subagent_description="...", stateless=True)
```

---

## Tracer Integration

Attach a tracer to instrument all agents created by the factory:

```python
from obelix.core.tracer import Tracer, ConsoleExporter

tracer = Tracer(exporter=ConsoleExporter(verbosity=2), service_name="my-service")

factory = AgentFactory()
factory.with_tracer(tracer) \
       .register("analyzer", AnalyzerAgent, subagent_description="...") \
       .register("coordinator", CoordinatorAgent)

# All created agents get the tracer automatically
coordinator = factory.create("coordinator", subagents=["analyzer"])
```

The tracer is injected into every agent's constructor unless explicitly overridden in `create()`.

---

## Shared Memory

Agents can share context through a directed dependency graph. Upstream agent outputs are automatically propagated as system messages to downstream agents.

### Setup

```python
from obelix.core.agent.shared_memory import SharedMemoryGraph, PropagationPolicy

graph = SharedMemoryGraph()
graph.add_edge("requirements", "designer")
graph.add_edge("requirements", "implementer")
graph.add_edge("designer", "implementer")

factory = AgentFactory()
factory.with_memory_graph(graph) \
       .register("requirements", RequirementsAgent, subagent_description="Extracts requirements", stateless=True) \
       .register("designer", DesignerAgent, subagent_description="Designs architecture", stateless=True) \
       .register("implementer", ImplementerAgent, subagent_description="Implements solution", stateless=True) \
       .register("coordinator", CoordinatorAgent)

coordinator = factory.create("coordinator", subagents=["requirements", "designer", "implementer"])
```

### How It Works

When the coordinator executes:

1. `requirements` runs → its final response is stored in the graph
2. `designer` runs → automatically receives the requirements output as a system message
3. `implementer` runs → automatically receives outputs from both `requirements` and `designer`

The factory injects a **Coordination Protocol** into the orchestrator's system message with:
- **Execution order**: topological sort of the dependency graph
- **Data flow**: which agent feeds into which, with propagation policy

Sub-agents receive a **Sub-Agent Protocol** telling them they operate in a pipeline and their output will be forwarded downstream.

### PropagationPolicy

Controls what data flows along each edge:

| Policy | Propagates | Use Case |
|--------|-----------|----------|
| `FINAL_RESPONSE_ONLY` (default) | Agent's final text response | Most cases |
| `LAST_TOOL_RESULT` | Last tool result (raw) | When downstream needs structured data, not prose |

```python
from obelix.core.agent.shared_memory import PropagationPolicy

graph.add_edge("sql_agent", "chart_agent", policy=PropagationPolicy.LAST_TOOL_RESULT)
graph.add_edge("sql_agent", "report_agent", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
```

In this example, `chart_agent` gets the raw SQL result (structured data for chart generation), while `report_agent` gets the SQL agent's textual summary.

### Cycle Detection

The graph rejects edges that would create cycles:

```python
graph.add_edge("a", "b")
graph.add_edge("b", "a")  # raises ValueError: would create a cycle
```

### Thread Safety

- `publish()` uses `asyncio.Lock` for concurrent writes
- `pull_for()` is read-only (no lock needed)
- `add_agent()` / `add_edge()` are called during setup (before concurrency)

---

## A2A Server

Expose an agent as an HTTP service using the [A2A protocol](https://a2a-protocol.org/).

```python
factory.a2a_serve(
    "coordinator",
    host="0.0.0.0",
    port=8000,
    subagents=["analyzer"],
    description="Data analysis coordinator",
)
```

### a2a_serve() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | str | Required | Registered agent name |
| `host` | str | `"0.0.0.0"` | Bind address |
| `port` | int | `8000` | Bind port |
| `endpoint` | str | None | Public URL for Agent Card (e.g. `"https://my-agent.prod.example.com"`) |
| `version` | str | `"0.1.0"` | Version in Agent Card |
| `description` | str | None | Description in Agent Card (falls back to system message) |
| `provider_name` | str | `"Obelix"` | Organization name |
| `provider_url` | str | None | Organization URL |
| `log_level` | str | `"info"` | Uvicorn log level |
| `subagents` | list[str] | None | Sub-agents to attach |
| `subagent_config` | dict | None | Per-subagent constructor overrides |
| `**create_overrides` | Any | — | Extra kwargs forwarded to `create()` |

### What It Does

1. Creates the agent via `factory.create()` (with sub-agents if specified)
2. Auto-registers `RequestUserInputTool` on each agent instance (for input-required flow)
3. Builds an Agent Card from the agent's tools, sub-agents, and system message
4. Starts a FastAPI + Uvicorn server with a2a-sdk handling protocol compliance

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /.well-known/agent.json` | Agent Card (capabilities, skills) |
| `POST /` | JSON-RPC 2.0 (`message/send`, `message/stream`) |

### Requirements

```bash
uv sync --extra serve  # fastapi, uvicorn, a2a-sdk
```

For full A2A details, see [A2A Server Guide](a2a_server.md).

---

## Utility Methods

```python
factory.get_registered_names()  # ['analyzer', 'coordinator']
factory.is_registered("analyzer")  # True
factory.get_spec("analyzer")  # AgentSpec(cls=AnalyzerAgent, ...)
factory.unregister("analyzer")  # True (removed) / False (not found)
factory.clear()  # Remove all
```

---

## Complete Example

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.shared_memory import SharedMemoryGraph, PropagationPolicy
from obelix.core.tracer import Tracer, ConsoleExporter
from obelix.core.tool.tool_decorator import tool
from pydantic import Field


@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="add, subtract, multiply, divide")

    async def execute(self) -> dict:
        ops = {"add": lambda: self.a + self.b, "multiply": lambda: self.a * self.b}
        return {"result": ops[self.operation]()}


class MathAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a math expert.",
            tools=[CalculatorTool],
            **kwargs,
        )


class ReportAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You write clear reports from data.",
            **kwargs,
        )


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You coordinate math and reporting tasks.",
            planning=True,
            **kwargs,
        )


# Shared memory: math results flow to report
graph = SharedMemoryGraph()
graph.add_edge("math", "report", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)

# Tracer
tracer = Tracer(exporter=ConsoleExporter(verbosity=2), service_name="demo")

# Factory
factory = AgentFactory(global_defaults={"max_iterations": 10})
factory.with_tracer(tracer) \
       .with_memory_graph(graph) \
       .register("math", MathAgent, subagent_description="Performs calculations", stateless=True) \
       .register("report", ReportAgent, subagent_description="Writes reports", stateless=True) \
       .register("coordinator", CoordinatorAgent)

# Create and run
coordinator = factory.create("coordinator", subagents=["math", "report"])
response = coordinator.execute_query("Calculate 123 * 456 and write a summary report")
print(response.content)
```

---

## Error Handling

| Condition | Error |
|-----------|-------|
| Duplicate name | `ValueError: Agent 'X' is already registered` |
| Agent not registered | `ValueError: Agent 'X' not registered. Available: [...]` |
| Missing subagent_description | `ValueError: subagent_description is required when used as subagent` |
| Invalid subagent_config key | `ValueError: subagent_config contains keys not in subagents list` |
| Instance in subagents list | `ValueError: Instance-based subagents are not supported via factory` |
| Cycle in memory graph | `ValueError: Adding edge would create a cycle` |

---

## See Also

- [BaseAgent Guide](base_agent.md) — agent creation, tools, hooks, streaming
- [Tools Guide](tools.md) — creating and registering tools
- [A2A Server Guide](a2a_server.md) — A2A protocol details
- [Hooks Guide](hooks.md) — intercepting agent behavior
