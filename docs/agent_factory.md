# Agent Factory Guide

The Agent Factory is the centralized API for creating and composing agents in Obelix. It enables declarative agent registration, standalone agent creation, and hierarchical sub-agent orchestration with optional shared memory support.

---

## Quick Start

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory


# Define your agents
class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You are a data analyst.")


class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You coordinate tasks.")


# Create factory and register agents
factory = AgentFactory()
factory.register("analyzer", AnalyzerAgent, subagent_description="Analyzes data")
factory.register("coordinator", CoordinatorAgent)

# Create agents
analyzer = factory.create("analyzer")
coordinator = factory.create("coordinator", subagents=["analyzer"])

# Use the agents
response = coordinator.execute_query("Analyze the Q4 report")
print(response.content)
```

---

## Factory Initialization

### Basic Factory

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
```

### Factory with Global Defaults

Set constructor defaults that apply to all registered agents:

```python
factory = AgentFactory(
    global_defaults={
        "max_iterations": 10,
        "agent_comment": True,
    }
)
```

Constructor arguments passed when creating agents override global defaults:

```python
# This agent uses max_iterations=15 (overrides default of 10)
agent = factory.create("analyzer", max_iterations=15)
```

---

## Registering Agents

### Basic Registration

Register an agent class that will be available for creation:

```python
factory.register(
    name="analyzer",
    cls=AnalyzerAgent,
)
```

### Registration with Subagent Configuration

Make the agent available to be used as a sub-agent:

```python
factory.register(
    name="analyzer",
    cls=AnalyzerAgent,
    subagent_description="Analyzes datasets and generates insights",  # Required if exposed as subagent
)
```

### Registration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique identifier in registry (used in `create()` and `subagents` lists) |
| `cls` | `Type[BaseAgent]` | Required | The BaseAgent subclass to register |
| `subagent_description` | `str` | `None` | Description of capabilities (required if agent will be used as sub-agent) |
| `stateless` | `bool` | `False` | If `True`, each execution is isolated. If `False`, conversation history persists |
| `defaults` | `dict` | `None` | Constructor defaults for this specific agent |

### Method Chaining

Registrations return `self` for method chaining:

```python
factory.register("analyzer", AnalyzerAgent, subagent_description="Analyzes data") \
       .register("designer", DesignerAgent, subagent_description="Designs systems") \
       .register("implementer", ImplementerAgent, subagent_description="Implements code")
```

---

## Creating Agents

### Standalone Agent

Create a single agent without sub-agents:

```python
analyzer = factory.create("analyzer")
response = analyzer.execute_query("Analyze this dataset")
```

### Standalone with Overrides

Override constructor parameters at creation time:

```python
analyzer = factory.create(
    "analyzer",
    max_iterations=5,          # Lower iteration limit
    agent_comment=False,       # Skip comment generation
)
```

### Orchestrator with Sub-Agents

Create an agent with registered sub-agents. Pass `subagents` as a list of registered names:

```python
coordinator = factory.create(
    "coordinator",
    subagents=["analyzer", "designer", "implementer"],
)

response = coordinator.execute_query("Build a complete system")
```

When an agent has sub-agents:
- Each sub-agent is automatically registered as a tool on the main agent
- The LLM can invoke sub-agents via function calling
- Sub-agents receive the query passed by the orchestrator

### Empty Orchestrator

Create an orchestrator without sub-agents (useful for testing or dynamic composition):

```python
empty_orchestrator = factory.create("coordinator", subagents=[])
# Can register agents later via coordinator.register_agent()
```

### Per-Subagent Configuration

Override constructor parameters for specific sub-agents:

```python
coordinator = factory.create(
    "coordinator",
    subagents=["analyzer", "designer"],
    subagent_config={
        "analyzer": {"max_iterations": 3},
        "designer": {"max_iterations": 10, "provider": custom_provider},
    }
)
```

### Shared Sub-Agent Instance

Pass an already-created agent instance as a sub-agent. Useful for sharing state across multiple orchestrators:

```python
# Create a shared instance
shared_analyzer = factory.create("analyzer")

# Use in multiple orchestrators
coordinator1 = factory.create("coordinator", subagents=[shared_analyzer])
coordinator2 = factory.create("coordinator", subagents=[shared_analyzer])

# Both orchestrators share the exact same analyzer instance (and its history)
```

---

## Configuration Merge Order

Constructor arguments are merged in this order (later overwrites earlier):

```
global_defaults < agent_spec.defaults < subagent_config[name] < create_overrides
```

For the main agent:
```python
# For agent "coordinator" with global_defaults and spec.defaults
config = {**global_defaults, **spec.defaults, **create_overrides}
coordinator = CoordinatorAgent(**config)
```

For sub-agents:
```python
# For sub-agent "analyzer" with registry and subagent_config
sub_config = {**global_defaults, **sub_spec.defaults, **subagent_config.get("analyzer", {})}
analyzer = AnalyzerAgent(**sub_config)
```

---

## Stateless vs Stateful Sub-Agents

The `stateless` parameter controls how sub-agents handle execution and state:

| `stateless` | Conversation History | Parallel Calls | Use Case |
|-------------|----------------------|----------------|----------|
| `False` (default) | Persists across calls | Serialized (with lock) | Context-aware agents (remember previous interactions) |
| `True` | Fresh copy each call | Yes (concurrent) | One-shot tasks (translations, validations) |

### Stateful (default)

```python
factory.register(
    "analyst",
    AnalystAgent,
    subagent_description="Analyzes data with context",
    stateless=False,  # Each call adds to the same history
)
```

When the orchestrator calls `analyst` multiple times, each call adds to the agent's conversation history. The agent remembers all previous interactions.

**Trade-off**: Better context, but calls are serialized (can be slow for concurrent calls).

### Stateless

```python
factory.register(
    "translator",
    TranslatorAgent,
    subagent_description="Translates text",
    stateless=True,  # Each call gets a fresh copy with forked history
)
```

When the orchestrator calls `translator` multiple times, each call gets a shallow copy with a forked (but isolated) conversation history. The history is discarded after execution.

**Trade-off**: No context persistence, but calls can run in parallel.

---

## Utility Methods

### get_registered_names()

List all registered agent names:

```python
names = factory.get_registered_names()
# ['analyzer', 'designer', 'implementer', 'coordinator']
```

### is_registered(name)

Check if an agent name is registered:

```python
if factory.is_registered("analyzer"):
    print("Analyzer is available")
```

### get_spec(name)

Get the `AgentSpec` for an agent (for introspection):

```python
spec = factory.get_spec("analyzer")
print(spec.cls)                   # <class 'AnalyzerAgent'>
print(spec.subagent_description)  # "Analyzes datasets..."
print(spec.stateless)             # False
print(spec.defaults)              # {"max_iterations": 10}
```

### unregister(name)

Remove an agent from the registry:

```python
factory.unregister("analyzer")  # Returns True if found, False if not registered
```

### clear()

Remove all agents from the registry:

```python
factory.clear()
```

---

## Shared Memory Support

Agents can share a graph-based memory to propagate outputs between dependent agents. This is useful for orchestrators where downstream agents need context from upstream agents.

### Using Shared Memory

```python
from obelix.core.agent.shared_memory import SharedMemoryGraph

# Create a graph of dependencies
graph = SharedMemoryGraph()
graph.add_edge("requirements", "designer")  # requirements -> designer
graph.add_edge("requirements", "implementer")
graph.add_edge("designer", "implementer")

# Attach the graph to the factory
factory = AgentFactory()
factory.with_memory_graph(graph)

# Register agents
factory.register("requirements", RequirementsAgent, subagent_description="Extracts requirements", stateless=True)
factory.register("designer", DesignerAgent, subagent_description="Designs architecture", stateless=True)
factory.register("implementer", ImplementerAgent, subagent_description="Implements solution", stateless=True)

# Create orchestrator
coordinator = factory.create(
    "coordinator",
    subagents=["requirements", "designer", "implementer"],
)

# When executed:
# 1. requirements runs -> outputs stored in graph
# 2. designer runs -> automatically receives requirements output via SystemMessage
# 3. implementer runs -> automatically receives outputs from both upstream agents
```

**Note**: Shared memory is optional. Agents without a memory graph work normally.

---

## Complete Example

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tool.tool_decorator import tool
from pydantic import Field


# Define tools
@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="add, subtract, multiply, divide")

    async def execute(self) -> dict:
        ops = {
            "add": lambda: self.a + self.b,
            "subtract": lambda: self.a - self.b,
            "multiply": lambda: self.a * self.b,
            "divide": lambda: self.a / self.b if self.b != 0 else None,
        }
        return {"result": ops[self.operation]()}


# Define agents
class MathAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a math expert. Use the calculator tool.",
            tools=[CalculatorTool],
            **kwargs
        )


class VerifierAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You verify mathematical results.",
            **kwargs
        )


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You coordinate math tasks.",
            **kwargs
        )


# Set up factory
factory = AgentFactory(
    global_defaults={"max_iterations": 15}
)

# Register agents
factory.register("math", MathAgent, subagent_description="Performs math calculations", stateless=True) \
       .register("verifier", VerifierAgent, subagent_description="Verifies results", stateless=True) \
       .register("coordinator", CoordinatorAgent)

# Create orchestrator
coordinator = factory.create(
    "coordinator",
    subagents=["math", "verifier"],
    subagent_config={
        "math": {"max_iterations": 10},
    }
)

# Execute
response = coordinator.execute_query("Calculate 123 * 456 and verify the result")
print(response.content)
```

---

## Error Handling

The factory raises clear errors for invalid configurations:

| Condition | Error |
|-----------|-------|
| Duplicate agent name | `ValueError: Agent 'X' is already registered` |
| Agent not registered | `ValueError: Agent 'X' not registered. Available: [...]` |
| Sub-agent not registered | `ValueError: Agent 'X' not registered or not exposed as subagent` |
| Invalid subagent_config key | `ValueError: subagent_config contains keys not in subagents list` |

---

## Tips and Best Practices

1. **Use Method Chaining**: Chain `register()` calls for cleaner setup.

   ```python
   factory.register("a", AgentA, ...) \
          .register("b", AgentB, ...) \
          .register("c", AgentC, ...)
   ```

2. **Separate Concerns**: Use different factories for different domains:

   ```python
   analytics_factory = AgentFactory()
   analytics_factory.register("analyzer", AnalyzerAgent, ...)

   marketing_factory = AgentFactory()
   marketing_factory.register("writer", WriterAgent, ...)
   ```

3. **Use Defaults Wisely**: Set factory defaults only for settings that apply to all agents.

   ```python
   factory = AgentFactory(
       global_defaults={
           "max_iterations": 10,  # Reasonable default for all
           "agent_comment": True,  # Always generate comments
       }
   )
   ```

4. **Prefer Stateless for Parallelism**: When sub-agents can be called concurrently, use `stateless=True`.

   ```python
   factory.register("translator", TranslatorAgent, stateless=True)
   ```

5. **Test Agent Creation**: Always test that agents can be instantiated:

   ```python
   # This catches missing dependencies early
   test_agent = factory.create("analyzer")
   ```

---

## See Also

- [BaseAgent Guide](base_agent.md) - Creating and using agents
- [Hooks API](hooks.md) - Hook system for agent customization
- [SharedMemoryGraph](../README.md#shared-memory-support) - Sharing context between agents
- [README](../README.md) - Installation and quick start
