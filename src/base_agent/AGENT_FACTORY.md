# Agent Factory Design and Usage (Base Agent)

This document describes the proposed **Agent Factory** for `src/base_agent` in Obelix. It is intentionally **very detailed** to serve as a complete reference for design decisions, usage, edge cases, and implementation considerations.

---

## 1) Context: Why a Factory

The current system already supports:
- `BaseAgent` with tools, hooks, and provider abstraction.
- `@subagent` for exposing a `BaseAgent` as a registrable tool.
- `@orchestrator` for enabling `register_agent()` on a coordinating agent.

However, configuring these pieces requires **multiple separate steps** and **different modes** that a user must remember. This can lead to:
- Repetition (three styles: normal, subagent, orchestrator)
- Boilerplate (manual decorator usage, manual tool registration)
- Confusion (when does an agent become a tool?)

**Goal:** simplify to **two user-facing modes**:
1) **agent** — a normal standalone `BaseAgent`
2) **orchestrator** — a `BaseAgent` that can register subagents

The factory centralizes creation, configuration, and subagent registration so users do **not** need to repeat or manage three separate configurations.

---

## 2) High-Level Idea

**Factory responsibilities:**
- Store a registry of agent classes and defaults
- Create a normal agent (mode=`agent`)
- Create an orchestrator (mode=`orchestrator`)
- Register subagents into orchestrators
- (Optional) auto-apply `@subagent` metadata if not present

**User responsibilities:**
- Define agent classes (like today)
- Register them once in the factory
- Ask the factory to create agent or orchestrator instances

---

## 3) Behavior Model

### 3.1 Agent roles
- **Normal agent**: only handles user queries.
- **Orchestrator agent**: can call subagents as tools.
- **Subagent**: a normal agent that is *exposed as a tool*.

### 3.2 What changes with orchestration?
No changes to core `BaseAgent` logic. Orchestration is purely about:
- registering subagents as tools
- adding schema metadata so LLM can call them
- enforcing stateless or stateful execution inside the wrapper

### 3.3 Where the factory fits
- The factory is **not a new agent type**.
- It **returns normal `BaseAgent` instances** (or orchestrator-enabled `BaseAgent`).
- Subagents are registered through the existing `register_agent()` method of the orchestrator.

---

## 4) Proposed API (User View)

### 4.1 Register agents
```python
factory.register(
    name="weather_agent",
    cls=WeatherAgent,
    expose_as_subagent=True,
    subagent_description="Weather expert",
)

factory.register(
    name="planner",
    cls=PlannerAgent,
)
```

### 4.2 Create a normal agent
```python
weather = factory.create(
    name="weather_agent",
    mode="agent",
)
```

### 4.3 Create an orchestrator with subagents
```python
planner = factory.create(
    name="planner",
    mode="orchestrator",
    subagents=["weather_agent"],
)
```

---

## 5) Detailed Design

### 5.1 Factory registry
The factory holds a registry of `AgentSpec` entries:
- `cls`: the agent class
- `defaults`: optional agent-specific default configuration
- `subagent_enabled`: should it be exposed as a tool
- `subagent_name`, `subagent_description`, `subagent_stateless`

This lets one centralized place define **how each agent should behave**.

### 5.2 Handling `@subagent`
Current `@subagent` decorator:
- **Does not modify runtime behavior**
- Adds only metadata + schema

The factory can auto-apply it to avoid forcing user to use it. This provides:
- Single config point
- Consistent metadata
- No extra boilerplate

If the class is already decorated, the factory should **not** re-apply.

### 5.3 Orchestrator creation
The orchestrator is just a `BaseAgent` decorated with `@orchestrator`.
The factory creates an instance and then registers subagents.

---

## 6) Reference Implementation (Proposed)

```python
# src/base_agent/factory.py
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Any

from src.base_agent.base_agent import BaseAgent
from src.base_agent.subagent_decorator import subagent
from src.base_agent.orchestrator_decorator import orchestrator


@orchestrator
class OrchestratorAgent(BaseAgent):
    pass


@dataclass
class AgentSpec:
    cls: Type[BaseAgent]
    defaults: Dict[str, Any] = field(default_factory=dict)
    subagent_enabled: bool = False
    subagent_name: Optional[str] = None
    subagent_description: Optional[str] = None
    subagent_stateless: bool = False


class AgentFactory:
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        self._defaults = defaults or {}
        self._registry: Dict[str, AgentSpec] = {}

    def register(
        self,
        name: str,
        cls: Type[BaseAgent],
        *,
        defaults: Optional[Dict[str, Any]] = None,
        expose_as_subagent: bool = False,
        subagent_name: Optional[str] = None,
        subagent_description: Optional[str] = None,
        subagent_stateless: bool = False,
    ) -> None:
        # Auto-decorate if needed
        if expose_as_subagent and not hasattr(cls, "subagent_name"):
            cls = subagent(
                name=subagent_name or name,
                description=subagent_description or f"Subagent {name}",
                stateless=subagent_stateless,
            )(cls)

        self._registry[name] = AgentSpec(
            cls=cls,
            defaults=defaults or {},
            subagent_enabled=expose_as_subagent,
            subagent_name=subagent_name or name,
            subagent_description=subagent_description,
            subagent_stateless=subagent_stateless,
        )

    def create(
        self,
        name: str,
        *,
        mode: str = "agent",
        subagents: Optional[List[str]] = None,
        **overrides: Any,
    ) -> BaseAgent:
        if name not in self._registry:
            raise ValueError(f"Agent '{name}' not registered")

        spec = self._registry[name]
        cfg = {**self._defaults, **spec.defaults, **overrides}

        if mode == "orchestrator":
            agent = OrchestratorAgent(**cfg)
            for sub_name in (subagents or []):
                sub_spec = self._registry.get(sub_name)
                if not sub_spec:
                    raise ValueError(f"Subagent '{sub_name}' not registered")

                sub_cfg = {**self._defaults, **sub_spec.defaults}
                sub_agent = sub_spec.cls(**sub_cfg)
                agent.register_agent(sub_agent)
            return agent

        if mode == "agent":
            return spec.cls(**cfg)

        raise ValueError(f"Unknown mode: {mode}")
```

---

## 7) How It Works Internally

1. **Registration:**
   - Factory stores a spec for each agent class.
   - Optional `@subagent` metadata applied automatically.

2. **Creation (agent):**
   - Merges defaults (factory + agent + overrides).
   - Instantiates the agent normally.

3. **Creation (orchestrator):**
   - Instantiates an `OrchestratorAgent`.
   - For each subagent name:
     - instantiates the class
     - calls `register_agent()` which wraps it via `_SubAgentWrapper` and registers as tool

---

## 8) Potential Issues / Open Questions

### 8.1 Decorator duplication
If a class is already decorated with `@subagent`, auto-decorating again could overwrite metadata or throw errors. The factory should check `hasattr(cls, "subagent_name")` before applying.

### 8.2 Schema differences
If the class was decorated manually, its schema might not match the factory config. Decide if factory overrides or respects class defaults.

### 8.3 Stateless vs stateful subagents
- `stateless=True` duplicates the agent and uses a forked conversation history.
- `stateless=False` serializes with a lock and shares memory.

This choice impacts correctness and concurrency. The factory should make this explicit in config.

### 8.4 Tool registration order
The orchestrator’s registered tools list could have tools already set in its constructor. If subagents are added after, order might matter for some providers. If ordering is important, enforce it in factory.

### 8.5 Provider requirements
If subagents use a different provider/model, the factory needs a way to override per-agent defaults. The `defaults` param solves this, but the user must be careful.

### 8.6 Conversation history sharing
For stateful subagents, the wrapper locks and uses the same instance. This can lead to cross-request contamination if a subagent is reused in multiple orchestrations. Stateless subagents avoid this but lose memory continuity.

### 8.7 Hook interference
Subagents execute their own hooks. When used as tools, they run hooks with their own history. If hooks depend on global state, they may behave unexpectedly.

### 8.8 Error propagation
Tool execution errors are wrapped into `ToolResult`. The orchestrator should decide whether to surface or retry. The factory does not change this behavior, but it affects how subagents are used.

---

## 9) Recommended Usage Patterns

- Use **mode=agent** for standalone agents.
- Use **mode=orchestrator** when you need coordination.
- Use **expose_as_subagent=True** for any agent that should be callable.
- Default to `stateless=True` for safe parallel execution.

---

## 10) Summary

The Agent Factory provides a **single entry point** for creating all agents while keeping user configuration minimal. It removes the need for three different creation paths and allows orchestration by configuration rather than boilerplate decorators. The factory does not change core agent behavior; it only standardizes setup and orchestration wiring.

This document should give enough detail to implement the factory, understand its flow, and anticipate potential edge cases or bugs.

---

# REVISION NOTES (Added after code review)

> **IMPORTANT**: The sections below (11-14) were added after a detailed code review of the original proposal.
> They identify critical bugs and propose a revised implementation that maintains the same user-facing API
> while fixing architectural issues and ensuring compatibility with `docs/shared_memory_plan.md`.

---

## 11) CRITICAL ISSUES IDENTIFIED IN ORIGINAL DESIGN

The original implementation in Section 6 contains several bugs and architectural problems that would cause incorrect behavior or runtime errors.

### 11.1 BUG: OrchestratorAgent ignores registered class

**Location**: `create()` method, lines 202-203

```python
if mode == "orchestrator":
    agent = OrchestratorAgent(**cfg)  # <-- BUG: ignores spec.cls!
```

**Problem**: When `mode="orchestrator"` is requested, the factory creates a generic `OrchestratorAgent` instead of using the registered class (`spec.cls`). This means:

- The registered class's `system_message` is lost
- Any tools registered in the class's `__init__` are lost
- Any hooks configured in the class are lost
- Any `tool_policy` is lost
- The class's custom behavior is completely ignored

**Example of incorrect behavior**:
```python
factory.register("planner", PlannerAgent)  # PlannerAgent has system_message, tools, hooks

planner = factory.create("planner", mode="orchestrator")
# planner is OrchestratorAgent, NOT PlannerAgent!
# All PlannerAgent configuration is lost
```

---

### 11.2 BUG: Auto-decoration mutates original class

**Location**: `register()` method, lines 172-177

```python
if expose_as_subagent and not hasattr(cls, "subagent_name"):
    cls = subagent(...)(cls)  # <-- BUG: mutates cls permanently
```

**Problem**: This modifies the original class object permanently. If the same class is used elsewhere (tests, different factory instance, direct instantiation), it will have unexpected `@subagent` metadata attached.

**Side effects**:
- Class behavior changes globally, not just for this factory
- Tests may fail unexpectedly
- Difficult to debug

---

### 11.3 BUG: Duplicate/conflicting configuration sources

**Problem**: Subagent metadata can be defined in two places:
1. In the `@subagent` decorator on the class
2. In `factory.register(..., subagent_description=...)`

If both are present with different values, behavior is undefined. The factory stores `subagent_description` in the spec but doesn't use it if the class is already decorated.

**Example of confusion**:
```python
@subagent(name="weather", description="Weather from decorator")
class WeatherAgent(BaseAgent): ...

factory.register(
    "weather",
    WeatherAgent,
    subagent_description="Weather from factory"  # <-- Which one wins?
)
```

---

### 11.4 BUG: `stateless` parameter not propagated correctly

**Location**: `register()` method

**Problem**: If a class is already decorated with `@subagent(stateless=True)`, the factory's `subagent_stateless=False` is ignored. Conversely, if auto-decorating, the factory's value is used. This inconsistency leads to unpredictable behavior.

---

### 11.5 ISSUE: `mode` string lacks type safety

**Problem**: Using `mode="agent"` vs `mode="orchestrator"` as strings:
- No IDE autocompletion
- Typos cause runtime errors (`mode="orchestrtor"`)
- Not discoverable

---

### 11.6 ISSUE: No validation for required `system_message`

**Problem**: `BaseAgent.__init__` requires `system_message` as a positional argument. The factory merges defaults but doesn't validate that `system_message` is present. This causes cryptic `TypeError` at instantiation.

---

### 11.7 ISSUE: Incompatibility with `shared_memory_plan.md`

The future shared memory system requires:
- `memory_graph` attached to agents
- `agent_id` for node identification
- Dependency edges between subagents (A -> B -> C)

**The original design has no support for any of these**:
- No `memory_graph` parameter
- No `agent_id` concept
- `subagents=["a", "b"]` is a flat list, not a dependency graph

---

## 12) REVISED DESIGN PRINCIPLES

The revised implementation maintains the **same user-facing API** while fixing all identified issues.

### 12.1 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Always use registered class** | Never create generic `OrchestratorAgent`; use `spec.cls` and apply decorator to dynamic subclass if needed |
| **Never mutate original classes** | Create dynamic subclasses via `type()` when decorators need to be applied |
| **Single source of truth** | Decorated class metadata takes precedence; factory can only extend, not override |
| **Fail-fast validation** | Validate required parameters at `register()` time, not `create()` time |
| **Shared memory ready** | Include `with_memory_graph()` stub and `agent_id` assignment |

### 12.2 Key Changes

1. **`as_orchestrator=True` instead of `mode="orchestrator"`**
   - More explicit boolean flag
   - Better IDE support
   - `mode` string eliminated

2. **Dynamic subclass creation**
   ```python
   # Instead of mutating cls:
   cls = subagent(...)(cls)  # BAD

   # Create a subclass:
   subclass = type(f"{cls.__name__}SubAgent", (cls,), {})
   decorated = subagent(...)(subclass)  # GOOD - original untouched
   ```

3. **Orchestrator uses registered class**
   ```python
   # Instead of:
   agent = OrchestratorAgent(**cfg)  # BAD

   # Use registered class:
   cls = self._ensure_orchestrator_class(spec)  # Returns spec.cls or decorated subclass
   agent = cls(**cfg)  # GOOD
   ```

4. **Validation at registration**
   ```python
   if expose_as_subagent and not hasattr(cls, 'subagent_name'):
       if not subagent_description:
           raise ValueError("subagent_description required")  # Fail fast
   ```

---

## 13) REVISED IMPLEMENTATION

```python
# src/base_agent/agent_factory.py
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Any, TYPE_CHECKING

from src.base_agent.subagent_decorator import subagent
from src.base_agent.orchestrator_decorator import orchestrator
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class AgentSpec:
    """Immutable specification for a registered agent."""
    cls: Type['BaseAgent']

    # Subagent config (used only if expose_as_subagent=True AND class not already decorated)
    expose_as_subagent: bool = False
    subagent_name: Optional[str] = None
    subagent_description: Optional[str] = None
    stateless: bool = True

    # Constructor defaults
    defaults: Dict[str, Any] = field(default_factory=dict)

    def is_already_subagent(self) -> bool:
        """Check if class is already decorated with @subagent."""
        return hasattr(self.cls, 'subagent_name')

    def is_already_orchestrator(self) -> bool:
        """Check if class is already decorated with @orchestrator."""
        return getattr(self.cls, '_is_orchestrator', False)


class AgentFactory:
    """
    Factory for creating and composing agents.

    Provides a centralized way to register agent classes and create instances
    with consistent configuration. Supports both standalone agents and
    orchestrator agents with subagents.

    Usage:
        factory = AgentFactory()

        # Register agents
        factory.register("weather", WeatherAgent, expose_as_subagent=True,
                         subagent_description="Weather expert")
        factory.register("planner", PlannerAgent)

        # Create standalone agent
        weather = factory.create("weather")

        # Create orchestrator with subagents
        planner = factory.create(
            "planner",
            as_orchestrator=True,
            subagents=["weather"]
        )
    """

    def __init__(self, global_defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize the factory.

        Args:
            global_defaults: Default constructor arguments applied to all agents
        """
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
        stateless: bool = True,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> 'AgentFactory':
        """
        Register an agent class with the factory.

        Args:
            name: Unique identifier for this agent in the registry
            cls: The BaseAgent subclass to register
            expose_as_subagent: If True, this agent can be used as a subagent
            subagent_name: Name when used as subagent (defaults to `name`)
            subagent_description: Description when used as subagent
                (required if expose_as_subagent=True and class not decorated with @subagent)
            stateless: If True, subagent executions are isolated (default: True)
            defaults: Default constructor arguments for this agent

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If validation fails (e.g., missing required description)
        """
        # Validation: subagent needs description from somewhere
        if expose_as_subagent and not hasattr(cls, 'subagent_name'):
            if not subagent_description:
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
            defaults=defaults or {},
        )

        logger.info(f"AgentFactory: registered '{name}' ({cls.__name__})")
        return self

    def create(
        self,
        name: str,
        *,
        as_orchestrator: bool = False,
        subagents: Optional[List[str]] = None,
        **overrides: Any,
    ) -> 'BaseAgent':
        """
        Create an agent instance from a registered class.

        Args:
            name: Registered agent name
            as_orchestrator: If True, enable orchestration capabilities on the agent
            subagents: List of registered agent names to attach as subagents
                (implies as_orchestrator=True)
            **overrides: Override constructor parameters

        Returns:
            Configured BaseAgent instance

        Raises:
            ValueError: If agent not registered or configuration invalid
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Agent '{name}' not registered. Available: {available}")

        spec = self._registry[name]

        # Merge configs: global < spec.defaults < overrides
        config = {**self._global_defaults, **spec.defaults, **overrides}

        # Determine if orchestration is needed
        needs_orchestrator = as_orchestrator or bool(subagents)

        if needs_orchestrator:
            agent = self._create_orchestrator(spec, config, subagents or [])
        else:
            agent = self._create_agent(spec, config)

        # Future: attach memory graph and agent_id
        # if self._memory_graph:
        #     agent.memory_graph = self._memory_graph
        #     agent.agent_id = name

        return agent

    def _create_agent(self, spec: AgentSpec, config: Dict[str, Any]) -> 'BaseAgent':
        """Create a standalone agent instance using the registered class."""
        return spec.cls(**config)

    def _create_orchestrator(
        self,
        spec: AgentSpec,
        config: Dict[str, Any],
        subagent_names: List[str]
    ) -> 'BaseAgent':
        """
        Create an orchestrator with registered subagents.

        IMPORTANT: This method uses the REGISTERED CLASS (spec.cls), not a generic
        OrchestratorAgent. If the class is not already decorated with @orchestrator,
        a dynamic subclass is created and decorated to avoid mutating the original.
        """
        # Get orchestrator-capable class (original or dynamic subclass)
        cls = self._ensure_orchestrator_class(spec)

        # Create instance using the correct class
        agent = cls(**config)

        # Register each subagent
        for sub_name in subagent_names:
            sub_agent = self._create_subagent_instance(sub_name)
            agent.register_agent(sub_agent)

        return agent

    def _ensure_orchestrator_class(self, spec: AgentSpec) -> Type['BaseAgent']:
        """
        Ensure the class has orchestrator capabilities.

        If already decorated with @orchestrator, returns the class as-is.
        Otherwise, creates a dynamic subclass and applies the decorator.
        This avoids mutating the original class.

        Returns:
            Class with orchestrator capabilities
        """
        if spec.is_already_orchestrator():
            return spec.cls

        # Create dynamic subclass to avoid mutating original
        orchestrator_cls = type(
            f"{spec.cls.__name__}Orchestrator",
            (spec.cls,),
            {'__module__': spec.cls.__module__}
        )
        return orchestrator(orchestrator_cls)

    def _create_subagent_instance(self, name: str) -> 'BaseAgent':
        """
        Create a subagent-capable instance from registry.

        Validates that the agent is registered and exposed as subagent.
        """
        if name not in self._registry:
            raise ValueError(f"Subagent '{name}' not registered")

        spec = self._registry[name]

        if not spec.expose_as_subagent:
            raise ValueError(
                f"Agent '{name}' is not exposed as subagent. "
                f"Set expose_as_subagent=True in register()"
            )

        # Get subagent-capable class
        cls = self._ensure_subagent_class(spec, name)

        # Merge configs for subagent
        config = {**self._global_defaults, **spec.defaults}

        return cls(**config)

    def _ensure_subagent_class(self, spec: AgentSpec, registry_name: str) -> Type['BaseAgent']:
        """
        Ensure the class has subagent capabilities.

        If already decorated with @subagent, returns the class as-is.
        Otherwise, creates a dynamic subclass and applies the decorator.
        This avoids mutating the original class.

        Returns:
            Class with subagent capabilities
        """
        if spec.is_already_subagent():
            return spec.cls

        # Create dynamic subclass to avoid mutating original
        subagent_cls = type(
            f"{spec.cls.__name__}SubAgent",
            (spec.cls,),
            {'__module__': spec.cls.__module__}
        )

        return subagent(
            name=spec.subagent_name or registry_name,
            description=spec.subagent_description,
            stateless=spec.stateless,
        )(subagent_cls)

    # ─────────────────────────────────────────────────────────────
    # Future: Shared Memory Support (stubs for forward compatibility)
    # ─────────────────────────────────────────────────────────────

    def with_memory_graph(self, graph: Any) -> 'AgentFactory':
        """
        Attach a shared memory graph to the factory.

        All agents created after this call will receive the memory graph.

        Args:
            graph: SharedMemoryGraph instance (from src/base_agent/shared_memory.py)

        Returns:
            self (for method chaining)

        Note:
            This is a stub for future implementation per docs/shared_memory_plan.md.
            Currently stores the graph but does not attach it to agents.
        """
        self._memory_graph = graph
        logger.info("AgentFactory: memory graph attached (stub - not yet implemented)")
        return self

    # ─────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────

    def get_registered_names(self) -> List[str]:
        """Return list of all registered agent names."""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if an agent name is registered."""
        return name in self._registry

    def get_spec(self, name: str) -> Optional[AgentSpec]:
        """Get the AgentSpec for a registered agent (for introspection)."""
        return self._registry.get(name)
```

---

## 14) COMPARISON: Original vs Revised

| Aspect | Original Implementation | Revised Implementation |
|--------|------------------------|------------------------|
| Orchestrator class | Generic `OrchestratorAgent` | Uses `spec.cls` (registered class) |
| Class mutation | `cls = subagent(...)(cls)` mutates original | Dynamic subclass via `type()` |
| Mode selection | `mode="agent"` / `mode="orchestrator"` string | `as_orchestrator=True` boolean |
| Validation | None (fails at instantiation) | Fail-fast at `register()` |
| Shared memory | Not supported | Stub ready (`with_memory_graph()`) |
| Config precedence | Undefined | Clear: global < spec.defaults < overrides |
| Type safety | String-based mode | Boolean flag, IDE-friendly |

---

## 15) MIGRATION NOTES

If code was written against the original API (`mode="agent"`), update as follows:

```python
# Original API
agent = factory.create("name", mode="agent")
orch = factory.create("name", mode="orchestrator", subagents=["sub"])

# Revised API
agent = factory.create("name")  # mode="agent" is now default (no flag needed)
orch = factory.create("name", as_orchestrator=True, subagents=["sub"])
```

The `register()` API remains unchanged.
