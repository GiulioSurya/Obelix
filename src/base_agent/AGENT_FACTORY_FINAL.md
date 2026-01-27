# Agent Factory - Final Design Specification

This document is the **definitive specification** for the Agent Factory implementation in `src/base_agent/`. It supersedes all previous design documents and contains all information needed for implementation without assumptions.

---

## 1. Purpose and Goals

### 1.1 Why a Factory

The current system requires users to:
- Manually apply `@subagent` decorator to expose agents as tools
- Manually apply `@orchestrator` decorator to enable agent coordination
- Manually call `register_agent()` to wire subagents to orchestrators

The factory provides a **single, centralized API** for:
- Registering agent classes once
- Creating agent instances (standalone or orchestrator) via configuration
- Automatic wiring of subagents without manual decorator/registration boilerplate

### 1.2 Core Principle

**All agents are created exclusively through the factory.** Direct instantiation (`MyAgent()`) is not supported in production code.

### 1.3 Compatibility Requirement

The factory must be compatible with the future shared memory system described in `docs/shared_memory_plan.md`. This means:
- Stub methods for `memory_graph` attachment
- Agent identification via registry name
- No design decisions that would block shared memory integration

---

## 2. Agent Types

The factory produces two types of agents, determined **solely by the presence of the `subagents` parameter**:

| `subagents` parameter | Agent type | Capabilities |
|-----------------------|------------|--------------|
| Not provided (`None`) | Standalone agent | Handles queries, uses tools |
| Provided (list, even if empty `[]`) | Orchestrator agent | All standalone capabilities + can call subagents as tools |

There is **no separate flag** like `mode` or `as_orchestrator`. The presence of `subagents` is the only discriminator.

---

## 3. User-Facing API

### 3.1 Factory Instantiation

```python
from src.base_agent.agent_factory import AgentFactory

factory = AgentFactory(global_defaults={"max_iterations": 10})
```

**Parameters:**
- `global_defaults: Optional[Dict[str, Any]]` — Default constructor arguments applied to ALL agents created by this factory. Can be overridden per-agent.

---

### 3.2 Agent Registration

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
) -> AgentFactory  # Returns self for chaining
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `str` | Yes | — | Unique identifier in the registry. Used to reference this agent in `create()` and `subagents` lists. |
| `cls` | `Type[BaseAgent]` | Yes | — | The agent class to register. Must be a `BaseAgent` subclass. |
| `expose_as_subagent` | `bool` | No | `False` | If `True`, this agent can be used as a subagent in orchestrators. |
| `subagent_name` | `str` | No | Same as `name` | The tool name exposed to the LLM when used as subagent. |
| `subagent_description` | `str` | Conditional | — | Description of subagent capabilities for LLM. **Required if `expose_as_subagent=True` AND class is NOT already decorated with `@subagent`.** |
| `stateless` | `bool` | No | `False` | If `False` (default), executions are serialized and conversation history is preserved across calls. If `True`, subagent executions are isolated (copy of agent, forked history, discarded after). |
| `override_decorator` | `bool` | No | `False` | If `True`, factory values (`subagent_name`, `subagent_description`, `stateless`) override any existing `@subagent` decorator on the class. If `False`, decorator values take precedence. |
| `defaults` | `Dict[str, Any]` | No | `{}` | Default constructor arguments for this specific agent. Merged with `global_defaults`. |

**Validation (fail-fast at registration time):**
- If `expose_as_subagent=True` and class lacks `@subagent` decorator and `subagent_description` is not provided → `ValueError`
- If `name` is already registered → `ValueError` (no silent overwrite)

**Returns:** `self` for method chaining.

**Example:**
```python
factory.register("weather", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_description="Provides weather forecasts")
       .register("sql", SQLAgent,
                 expose_as_subagent=True,
                 subagent_description="Executes SQL queries")
       .register("planner", PlannerAgent,
                 defaults={"max_iterations": 20})
```

---

### 3.3 Agent Creation

```python
factory.create(
    name: str,
    *,
    subagents: Optional[List[Union[str, BaseAgent]]] = None,
    subagent_config: Optional[Dict[str, Dict[str, Any]]] = None,
    **overrides: Any,
) -> BaseAgent
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `str` | Yes | — | Registry name of the agent to create. |
| `subagents` | `List[Union[str, BaseAgent]]` | No | `None` | List of subagents to attach. If provided (even empty `[]`), creates an orchestrator. Accepts registry names (strings) or existing `BaseAgent` instances. |
| `subagent_config` | `Dict[str, Dict[str, Any]]` | No | `None` | Per-subagent constructor overrides. Keys are registry names, values are dicts of kwargs. **Only applies to string-based subagents, not instances.** |
| `**overrides` | `Any` | No | — | Override constructor parameters for the main agent being created. |

**Behavior:**

1. **Standalone agent** (when `subagents` is `None`):
   ```python
   weather = factory.create("weather")
   weather = factory.create("weather", max_iterations=5)  # with override
   ```

2. **Orchestrator with subagents** (when `subagents` is provided):
   ```python
   planner = factory.create("planner", subagents=["weather", "sql"])
   ```

3. **Orchestrator with per-subagent config**:
   ```python
   planner = factory.create(
       "planner",
       subagents=["weather", "sql"],
       subagent_config={
           "weather": {"max_iterations": 3},
           "sql": {"max_iterations": 10, "agent_comment": False}
       }
   )
   ```

4. **Orchestrator with shared instance**:
   ```python
   shared_weather = factory.create("weather")

   orch1 = factory.create("planner", subagents=[shared_weather])
   orch2 = factory.create("analyst", subagents=[shared_weather])
   # Both orchestrators share the SAME weather instance
   ```

5. **Empty orchestrator** (for later manual registration):
   ```python
   orch = factory.create("planner", subagents=[])
   # orch.register_agent(...) can be called later
   ```

**Validation:**
- If `name` not in registry → `ValueError`
- If string in `subagents` not in registry → `ValueError`
- If string in `subagents` has `expose_as_subagent=False` → `ValueError`
- If instance in `subagents` lacks subagent capabilities → `ValueError`
- If key in `subagent_config` not in `subagents` list → `ValueError` (typo protection)

**Returns:** Configured `BaseAgent` instance (standalone or orchestrator-capable).

---

## 4. Configuration Merge Order

Constructor arguments are merged in this order (later wins):

```
global_defaults < spec.defaults < subagent_config[name] < overrides
```

**For main agent:**
```python
config = {**global_defaults, **spec.defaults, **overrides}
```

**For subagents (created from string name):**
```python
config = {**global_defaults, **spec.defaults, **(subagent_config.get(name, {}))}
```

**For subagents (passed as instance):**
No merge. Instance is used as-is.

---

## 5. Decorator Handling

### 5.1 Classes WITHOUT existing decorators

If a class is not decorated with `@subagent` or `@orchestrator`, the factory applies the decorator dynamically **to a subclass**, never mutating the original class.

```python
# Original class untouched
class WeatherAgent(BaseAgent): ...

# Factory creates:
# - WeatherAgentSubAgent (dynamic subclass with @subagent applied)
# - WeatherAgentOrchestrator (dynamic subclass with @orchestrator applied)
```

### 5.2 Classes WITH existing `@subagent` decorator

**Default behavior (`override_decorator=False`):**
- Decorator values are used (name, description, stateless)
- Factory values are ignored
- No dynamic subclass created

**Override behavior (`override_decorator=True`):**
- Factory values override decorator values
- Dynamic subclass created with factory's `@subagent` applied

### 5.3 Classes WITH existing `@orchestrator` decorator

- Class is used as-is
- No dynamic subclass needed

### 5.4 Why dynamic subclasses?

To avoid mutating the original class. If `WeatherAgent` is used in multiple contexts (tests, different factory instances), mutations would cause unexpected behavior.

```python
# BAD: Mutates original
cls = subagent(...)(cls)

# GOOD: Creates isolated subclass
subclass = type(f"{cls.__name__}SubAgent", (cls,), {})
decorated = subagent(...)(subclass)
```

---

## 6. Instance Sharing

### 6.1 Default behavior (string names)

Each `create()` call instantiates **new** subagent instances:

```python
orch1 = factory.create("planner", subagents=["weather"])
orch2 = factory.create("planner", subagents=["weather"])
# orch1's weather ≠ orch2's weather (different instances)
```

### 6.2 Shared instances

Pass an existing instance instead of a string:

```python
shared = factory.create("weather")

orch1 = factory.create("planner", subagents=[shared])
orch2 = factory.create("planner", subagents=[shared])
# orch1's weather == orch2's weather (same instance)
```

### 6.3 Mixed

```python
shared = factory.create("weather")

orch = factory.create("planner", subagents=[shared, "sql"])
# weather = shared instance
# sql = new instance
```

### 6.4 Warning for shared stateful subagents

If sharing an instance with `stateless=False`, the factory logs a warning:
```
WARNING: Sharing stateful subagent 'weather' between orchestrators.
Calls will be serialized and conversation history shared across orchestrators.
```

---

## 7. Utility Methods

```python
factory.get_registered_names() -> List[str]
# Returns list of all registered agent names

factory.is_registered(name: str) -> bool
# Returns True if name is in registry

factory.get_spec(name: str) -> Optional[AgentSpec]
# Returns AgentSpec for introspection, or None if not registered
```

---

## 8. Future: Shared Memory Integration

The factory includes stub methods for future `shared_memory_plan.md` integration:

```python
factory.with_memory_graph(graph: SharedMemoryGraph) -> AgentFactory
```

**Current behavior:** Stores graph reference but does not attach to agents (stub).

**Future behavior:** All created agents will receive:
- `agent.memory_graph = graph`
- `agent.agent_id = registry_name`

The factory design does NOT block this integration. No changes to the public API will be needed.

---

## 9. Complete Reference Implementation

```python
# src/base_agent/agent_factory.py
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Any, Union, TYPE_CHECKING

from src.base_agent.subagent_decorator import subagent
from src.base_agent.orchestrator_decorator import orchestrator
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class AgentSpec:
    """Specification for a registered agent."""
    cls: Type['BaseAgent']
    expose_as_subagent: bool = False
    subagent_name: Optional[str] = None
    subagent_description: Optional[str] = None
    stateless: bool = False
    override_decorator: bool = False
    defaults: Dict[str, Any] = field(default_factory=dict)

    def is_already_subagent(self) -> bool:
        return hasattr(self.cls, 'subagent_name')

    def is_already_orchestrator(self) -> bool:
        return getattr(self.cls, '_is_orchestrator', False)


class AgentFactory:
    """
    Factory for creating and composing agents.

    All agents should be created through this factory.
    """

    def __init__(self, global_defaults: Optional[Dict[str, Any]] = None):
        self._global_defaults = global_defaults or {}
        self._registry: Dict[str, AgentSpec] = {}
        self._memory_graph = None  # Future: SharedMemoryGraph

    def register(
        self,
        name: str,
        cls: Type['BaseAgent'],
        *,
        expose_as_subagent: bool = False,
        subagent_name: Optional[str] = None,
        subagent_description: Optional[str] = None,
        stateless: bool = False,
        override_decorator: bool = False,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> 'AgentFactory':
        """Register an agent class."""
        # Validation: no duplicate names
        if name in self._registry:
            raise ValueError(f"Agent '{name}' is already registered")

        # Validation: subagent needs description from somewhere
        if expose_as_subagent:
            has_decorator = hasattr(cls, 'subagent_name')
            if not has_decorator and not subagent_description:
                raise ValueError(
                    f"Agent '{name}': expose_as_subagent=True requires subagent_description "
                    f"because class {cls.__name__} is not decorated with @subagent"
                )

        self._registry[name] = AgentSpec(
            cls=cls,
            expose_as_subagent=expose_as_subagent,
            subagent_name=subagent_name or name,
            subagent_description=subagent_description,
            stateless=stateless,
            override_decorator=override_decorator,
            defaults=defaults or {},
        )

        logger.info(f"AgentFactory: registered '{name}' ({cls.__name__})")
        return self

    def create(
        self,
        name: str,
        *,
        subagents: Optional[List[Union[str, 'BaseAgent']]] = None,
        subagent_config: Optional[Dict[str, Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> 'BaseAgent':
        """Create an agent instance."""
        if name not in self._registry:
            raise ValueError(f"Agent '{name}' not registered. Available: {list(self._registry.keys())}")

        spec = self._registry[name]
        config = {**self._global_defaults, **spec.defaults, **overrides}

        # Validate subagent_config keys
        if subagent_config:
            subagent_names = {s for s in (subagents or []) if isinstance(s, str)}
            invalid_keys = set(subagent_config.keys()) - subagent_names
            if invalid_keys:
                raise ValueError(
                    f"subagent_config contains keys not in subagents list: {invalid_keys}"
                )

        if subagents is not None:
            agent = self._create_orchestrator(spec, config, subagents, subagent_config or {})
        else:
            agent = self._create_agent(spec, config)

        # Future: attach memory graph
        # if self._memory_graph:
        #     agent.memory_graph = self._memory_graph
        #     agent.agent_id = name

        return agent

    def _create_agent(self, spec: AgentSpec, config: Dict[str, Any]) -> 'BaseAgent':
        """Create a standalone agent."""
        return spec.cls(**config)

    def _create_orchestrator(
        self,
        spec: AgentSpec,
        config: Dict[str, Any],
        subagents: List[Union[str, 'BaseAgent']],
        subagent_config: Dict[str, Dict[str, Any]]
    ) -> 'BaseAgent':
        """Create an orchestrator with subagents."""
        cls = self._ensure_orchestrator_class(spec)
        agent = cls(**config)

        for sub in subagents:
            if isinstance(sub, str):
                sub_agent = self._create_subagent_instance(sub, subagent_config.get(sub, {}))
            else:
                # Validate instance has subagent capabilities
                if not hasattr(sub, 'subagent_name'):
                    raise ValueError(
                        f"Instance {sub.__class__.__name__} must be subagent-capable. "
                        f"Create it via factory with expose_as_subagent=True."
                    )
                # Warn if sharing stateful subagent
                if not getattr(sub, 'subagent_stateless', True):
                    logger.warning(
                        f"Sharing stateful subagent '{sub.subagent_name}' between orchestrators. "
                        "Calls will be serialized and conversation history shared."
                    )
                sub_agent = sub

            agent.register_agent(sub_agent)

        return agent

    def _ensure_orchestrator_class(self, spec: AgentSpec) -> Type['BaseAgent']:
        """Get or create orchestrator-capable class."""
        if spec.is_already_orchestrator():
            return spec.cls

        # Create dynamic subclass
        orchestrator_cls = type(
            f"{spec.cls.__name__}Orchestrator",
            (spec.cls,),
            {'__module__': spec.cls.__module__}
        )
        return orchestrator(orchestrator_cls)

    def _create_subagent_instance(
        self,
        name: str,
        extra_config: Dict[str, Any]
    ) -> 'BaseAgent':
        """Create a subagent instance from registry."""
        if name not in self._registry:
            raise ValueError(f"Subagent '{name}' not registered")

        spec = self._registry[name]

        if not spec.expose_as_subagent:
            raise ValueError(
                f"Agent '{name}' is not exposed as subagent. "
                f"Set expose_as_subagent=True in register()"
            )

        cls = self._ensure_subagent_class(spec, name)
        config = {**self._global_defaults, **spec.defaults, **extra_config}

        return cls(**config)

    def _ensure_subagent_class(self, spec: AgentSpec, registry_name: str) -> Type['BaseAgent']:
        """Get or create subagent-capable class."""
        # Use original if decorated AND not overriding
        if spec.is_already_subagent() and not spec.override_decorator:
            return spec.cls

        # Create dynamic subclass
        subagent_cls = type(
            f"{spec.cls.__name__}SubAgent",
            (spec.cls,),
            {'__module__': spec.cls.__module__}
        )

        return subagent(
            name=spec.subagent_name or registry_name,
            description=spec.subagent_description or f"Subagent {registry_name}",
            stateless=spec.stateless,
        )(subagent_cls)

    # ─── Future: Shared Memory ───────────────────────────────────

    def with_memory_graph(self, graph: Any) -> 'AgentFactory':
        """Attach shared memory graph (stub for future implementation)."""
        self._memory_graph = graph
        logger.info("AgentFactory: memory graph attached (stub)")
        return self

    # ─── Utility Methods ─────────────────────────────────────────

    def get_registered_names(self) -> List[str]:
        """Return all registered agent names."""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def get_spec(self, name: str) -> Optional[AgentSpec]:
        """Get AgentSpec for introspection."""
        return self._registry.get(name)
```

---

## 10. Usage Examples

### 10.1 Basic usage

```python
from src.base_agent.agent_factory import AgentFactory
from myagents import WeatherAgent, SQLAgent, PlannerAgent

# Create factory
factory = AgentFactory()

# Register agents
factory.register("weather", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_description="Provides weather information")

factory.register("sql", SQLAgent,
                 expose_as_subagent=True,
                 subagent_description="Executes SQL queries")

factory.register("planner", PlannerAgent)

# Create standalone agent
weather = factory.create("weather")

# Create orchestrator with subagents
planner = factory.create("planner", subagents=["weather", "sql"])
```

### 10.2 With configuration overrides

```python
# Global defaults
factory = AgentFactory(global_defaults={"max_iterations": 10})

# Per-agent defaults at registration
factory.register("weather", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_description="Weather",
                 defaults={"max_iterations": 5})

# Override at creation
fast_weather = factory.create("weather", max_iterations=2)

# Per-subagent config
planner = factory.create(
    "planner",
    subagents=["weather", "sql"],
    subagent_config={
        "weather": {"max_iterations": 3},
        "sql": {"agent_comment": False}
    }
)
```

### 10.3 Shared subagent instance

```python
# Create shared instance
shared_weather = factory.create("weather")

# Use in multiple orchestrators
orch1 = factory.create("planner", subagents=[shared_weather])
orch2 = factory.create("analyst", subagents=[shared_weather])

# Both share the same weather instance
```

### 10.4 Override decorator metadata

```python
@subagent(name="weather_en", description="English weather")
class WeatherAgent(BaseAgent): ...

# Use decorator values (default)
factory.register("weather", WeatherAgent, expose_as_subagent=True)
# → subagent_name="weather_en", description="English weather"

# Override decorator values
factory.register("meteo", WeatherAgent,
                 expose_as_subagent=True,
                 subagent_name="meteo_it",
                 subagent_description="Meteo italiano",
                 override_decorator=True)
# → subagent_name="meteo_it", description="Meteo italiano"
```

---

## 11. Error Messages

The factory provides clear error messages:

| Condition | Error |
|-----------|-------|
| Duplicate registration | `ValueError: Agent 'X' is already registered` |
| Missing subagent description | `ValueError: Agent 'X': expose_as_subagent=True requires subagent_description because class Y is not decorated with @subagent` |
| Unknown agent in create | `ValueError: Agent 'X' not registered. Available: [...]` |
| Unknown subagent | `ValueError: Subagent 'X' not registered` |
| Subagent not exposed | `ValueError: Agent 'X' is not exposed as subagent. Set expose_as_subagent=True in register()` |
| Invalid subagent_config key | `ValueError: subagent_config contains keys not in subagents list: {...}` |
| Instance without subagent capability | `ValueError: Instance X must be subagent-capable. Create it via factory with expose_as_subagent=True.` |

---

## 12. What This Design Does NOT Do

To avoid scope creep, explicitly out of scope:

1. **Agent lifecycle management** — Factory creates agents but doesn't manage their lifecycle (start, stop, restart)
2. **Dependency injection container** — Not a full DI framework, just agent creation
3. **Agent discovery** — No automatic class scanning; all agents must be explicitly registered
4. **Serialization** — No agent serialization/deserialization
5. **Agent pooling** — No instance reuse/pooling beyond explicit instance sharing

---

## 13. File Location

Implementation: `src/base_agent/agent_factory.py`
This specification: `src/base_agent/AGENT_FACTORY_FINAL.md`