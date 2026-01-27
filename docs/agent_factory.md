# Agent Factory

The Agent Factory is the centralized API for creating and composing agents in Obelix. It handles agent registration, instantiation, and sub-agent orchestration without requiring manual decorator usage.

## Quick Start

```python
from src.base_agent import BaseAgent
from src.base_agent.agent_factory import AgentFactory

# Create factory
factory = AgentFactory()

# Register agents
factory.register("weather", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_description="Provides weather forecasts")
factory.register("planner", PlannerAgent)

# Create standalone agent
weather = factory.create("weather")

# Create orchestrator with sub-agents
planner = factory.create("planner", subagents=["weather"])
```

---

## AgentFactory

### Constructor

```python
AgentFactory(global_defaults: Optional[Dict[str, Any]] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_defaults` | `Dict[str, Any]` | `None` | Default constructor arguments applied to ALL agents created by this factory. Can be overridden per-agent. |

**Example:**
```python
factory = AgentFactory(global_defaults={"max_iterations": 10})
```

---

## Registration

### `register()`

Registers an agent class with the factory.

```python
factory.register(
    name: str,
    cls: Type[BaseAgent],
    *,
    expose_as_subagent: bool = False,
    subagent_name: Optional[str] = None,
    subagent_description: Optional[str] = None,
    stateless: bool = False,
    override_decorator: bool = False,
    defaults: Optional[Dict[str, Any]] = None,
) -> AgentFactory
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Unique identifier in the registry. Used in `create()` and `subagents` lists. |
| `cls` | `Type[BaseAgent]` | required | The BaseAgent subclass to register. |
| `expose_as_subagent` | `bool` | `False` | If `True`, this agent can be used as a sub-agent in orchestrators. |
| `subagent_name` | `str` | `name` | Tool name exposed to LLM when used as sub-agent. Defaults to registry `name`. |
| `subagent_description` | `str` | required* | Description of sub-agent capabilities for LLM. *Required if `expose_as_subagent=True` and class lacks `@subagent` decorator. |
| `stateless` | `bool` | `False` | If `False` (default), conversation history preserved across calls. If `True`, each execution is isolated (parallel-safe). |
| `override_decorator` | `bool` | `False` | If `True`, factory values override any existing `@subagent` decorator on the class. |
| `defaults` | `Dict[str, Any]` | `{}` | Default constructor arguments for this specific agent. |

### Returns

Returns `self` for method chaining.

### Raises

- `ValueError`: If `name` is already registered.
- `ValueError`: If `expose_as_subagent=True` but no description is available.

### Examples

**Basic registration:**
```python
factory.register("weather", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_description="Provides weather forecasts")
```

**With defaults:**
```python
factory.register("sql", SQLAgent,
                 expose_as_subagent=True,
                 subagent_description="Executes SQL queries",
                 defaults={"max_iterations": 15})
```

**Method chaining:**
```python
factory.register("weather", WeatherAgent, expose_as_subagent=True, subagent_description="...")
       .register("sql", SQLAgent, expose_as_subagent=True, subagent_description="...")
       .register("planner", PlannerAgent)
```

---

## Creation

### `create()`

Creates an agent instance from a registered class.

```python
factory.create(
    name: str,
    *,
    subagents: Optional[List[Union[str, BaseAgent]]] = None,
    subagent_config: Optional[Dict[str, Dict[str, Any]]] = None,
    **overrides: Any,
) -> BaseAgent
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Registered agent name. |
| `subagents` | `List[str \| BaseAgent]` | `None` | List of sub-agents to attach. If provided (even empty `[]`), creates an orchestrator. Accepts registry names (strings) or existing `BaseAgent` instances. |
| `subagent_config` | `Dict[str, Dict]` | `None` | Per-subagent constructor overrides. Keys are registry names, values are dicts of kwargs. Only applies to string-based subagents. |
| `**overrides` | `Any` | - | Override constructor parameters for the main agent. |

### Returns

Configured `BaseAgent` instance (standalone or orchestrator).

### Raises

- `ValueError`: If agent not registered.
- `ValueError`: If subagent not registered or not exposed as subagent.
- `ValueError`: If `subagent_config` contains keys not in `subagents` list.

### Agent Types

The presence of `subagents` determines the agent type:

| `subagents` | Result |
|-------------|--------|
| `None` (not provided) | Standalone agent |
| `[]` (empty list) | Empty orchestrator |
| `["a", "b"]` | Orchestrator with sub-agents |

### Examples

**Standalone agent:**
```python
weather = factory.create("weather")
```

**With overrides:**
```python
weather = factory.create("weather", max_iterations=5, provider=custom_provider)
```

**Orchestrator with sub-agents:**
```python
planner = factory.create("planner", subagents=["weather", "sql"])
```

**Per-subagent configuration:**
```python
planner = factory.create(
    "planner",
    subagents=["weather", "sql"],
    subagent_config={
        "weather": {"max_iterations": 3},
        "sql": {"max_iterations": 10, "provider": powerful_provider}
    }
)
```

**Shared sub-agent instance:**
```python
shared_weather = factory.create("weather")

orch1 = factory.create("planner", subagents=[shared_weather])
orch2 = factory.create("analyst", subagents=[shared_weather])
# Both orchestrators share the SAME weather instance
```

---

## Configuration Merge Order

Constructor arguments are merged in this order (later wins):

```
global_defaults < spec.defaults < subagent_config[name] < overrides
```

**For main agent:**
```python
config = {**global_defaults, **spec.defaults, **overrides}
```

**For sub-agents (from string name):**
```python
config = {**global_defaults, **spec.defaults, **(subagent_config.get(name, {}))}
```

**For sub-agents (passed as instance):**
No merge. Instance is used as-is.

---

## Stateless vs Stateful Sub-Agents

The `stateless` parameter controls how sub-agents handle concurrent calls:

| `stateless` | Parallel Calls | Conversation History |
|-------------|----------------|---------------------|
| `False` (default) | Serialized (lock) | Preserved across calls |
| `True` | Yes (concurrent) | Forked and discarded |

**Stateful (default):**
- Calls are serialized with an async lock
- Conversation history accumulates across calls
- Use for agents that need context from previous interactions

**Stateless:**
- Each call gets a shallow copy with forked history
- Copy is discarded after execution
- Multiple calls can run in parallel
- Use for one-shot tasks (translations, validations)

```python
# Stateful (default) - good for context-aware agents
factory.register("analyst", AnalystAgent,
                 expose_as_subagent=True,
                 subagent_description="Analyzes data with context")

# Stateless - good for parallel one-shot tasks
factory.register("translator", TranslatorAgent,
                 expose_as_subagent=True,
                 subagent_description="Translates text",
                 stateless=True)
```

---

## Sub-Agent Input Fields

Sub-agents can define additional input fields using Pydantic `Field`:

```python
from pydantic import Field

class SQLAnalyzerAgent(BaseAgent):
    # These fields become part of the sub-agent's input schema
    error_context: str = Field(default="", description="Error context from database")
    schema_info: str = Field(default="", description="Optional schema information")

    def __init__(self, **kwargs):
        super().__init__(system_message="You are a SQL expert.", **kwargs)
```

When the LLM calls this sub-agent, it can provide:
```json
{
  "name": "sql_analyzer",
  "arguments": {
    "query": "Why is this failing?",
    "error_context": "ORA-00942: table not found",
    "schema_info": "Tables: users, orders"
  }
}
```

The `query` field is always added automatically as the first parameter.

---

## Utility Methods

### `get_registered_names()`

Returns all registered agent names.

```python
names = factory.get_registered_names()
# ['weather', 'sql', 'planner']
```

### `is_registered(name)`

Checks if a name is registered.

```python
if factory.is_registered("weather"):
    ...
```

### `get_spec(name)`

Returns the `AgentSpec` for introspection.

```python
spec = factory.get_spec("weather")
print(spec.cls)  # <class 'WeatherAgent'>
print(spec.expose_as_subagent)  # True
```

### `unregister(name)`

Removes an agent from the registry.

```python
factory.unregister("weather")  # Returns True if found
```

### `clear()`

Removes all registered agents.

```python
factory.clear()
```

---

## Complete Example

```python
from src.base_agent import BaseAgent
from src.base_agent.agent_factory import AgentFactory
from src.tools import ToolBase, tool
from pydantic import Field

# Define a tool
@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool(ToolBase):
    operation: str = Field(...)
    a: float = Field(...)
    b: float = Field(...)

    async def execute(self) -> dict:
        ops = {"add": lambda: self.a + self.b, "multiply": lambda: self.a * self.b}
        return {"result": ops[self.operation]()}


# Define agents (no decorators needed!)
class MathAgent(BaseAgent):
    context: str = Field(default="", description="Additional context")

    def __init__(self, **kwargs):
        kwargs.setdefault("system_message", "You are a math expert.")
        super().__init__(**kwargs)
        self.register_tool(CalculatorTool())


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("system_message", "You coordinate math tasks.")
        super().__init__(**kwargs)


# Setup factory
factory = AgentFactory(global_defaults={"max_iterations": 10})

factory.register("math", MathAgent,
                 expose_as_subagent=True,
                 subagent_description="Performs calculations",
                 stateless=True)

factory.register("coordinator", CoordinatorAgent)


# Create and use
coordinator = factory.create("coordinator", subagents=["math"])

response = coordinator.execute_query("Calculate 15 * 7 and 100 + 50")
print(response.content)
```

---

## Error Messages

| Condition | Error |
|-----------|-------|
| Duplicate registration | `ValueError: Agent 'X' is already registered` |
| Missing description | `ValueError: Agent 'X': expose_as_subagent=True requires subagent_description...` |
| Unknown agent | `ValueError: Agent 'X' not registered. Available: [...]` |
| Not exposed as subagent | `ValueError: Agent 'X' is not exposed as subagent...` |
| Invalid subagent_config key | `ValueError: subagent_config contains keys not in subagents list: {...}` |
| Instance not subagent-capable | `ValueError: Instance X must be subagent-capable...` |