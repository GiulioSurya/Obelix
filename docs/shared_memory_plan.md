# Shared Memory Graph + Orchestrator Awareness (Plan)

## Context & Goal
We need a shared, temporary (in-memory) memory/state system across agents in this repo (`C:\Users\GLoverde\PycharmProjects\Obelix`).
Currently, sub-agents are registered via `@orchestrator` and executed as tools; they are isolated and only receive context via the orchestrator?s prompt.
Goal: allow direct propagation of **final responses only** between agents based on explicit relationships (graph), without forcing the orchestrator to relay that context.

Additionally, the orchestrator should receive an extra **system message** that makes it aware of its coordination responsibilities and **dependency ordering** between sub-agents (but **not** list sub-agents, since function calling already exposes them).

This doc captures the plan and the expected behavior so work can resume later.

---

## Current Code Touchpoints
Relevant files in `src/base_agent`:
- `src/base_agent/base_agent.py`
  - Holds `conversation_history` and LLM loop
  - Hooks system (`AgentEvent.*`)
  - `_build_final_response()` appends AssistantMessage to history and returns `AssistantResponse`
- `src/base_agent/orchestrator_decorator.py`
  - `@orchestrator` adds `register_agent()`
  - sub-agents are wrapped in `_SubAgentWrapper`, executed as tools
  - sub-agent is currently isolated (stateless copy or stateful lock)
- `src/base_agent/subagent_decorator.py`
  - adds schema metadata
- `src/base_agent/hooks.py`
  - lifecycle events (BEFORE_LLM_CALL, BEFORE_FINAL_RESPONSE, QUERY_END, etc.)

This architecture already has hook points for injecting memory and publishing memory.

---

## Desired Behavior (Concrete Example)
Agents: Orchestrator + 3 sub-agents
- AgentA is prerequisite for B and C
- AgentB is prerequisite for C

Graph:
- A -> B
- A -> C
- B -> C

Flow:
1) AgentA runs and produces final response:
   - "Ho estratto i requisiti principali: X, Y, Z."
   - Graph stores A.last_final

2) AgentB runs later:
   - Before LLM call, it pulls memory for B:
     - Finds edge A -> B, injects A.last_final into history
   - AgentB produces final response
   - Graph stores B.last_final

3) AgentC runs:
   - Pulls memory for C:
     - Finds A -> C and B -> C
     - Injects both A.last_final and B.last_final into history
   - AgentC responds using that shared context

Key rules:
- **Only final response** content is shared (no tool calls/results)
- Memory is only pulled at the **start of execution** (pull-on-start)
- Orchestrator does NOT need to pass context manually

---

## Design: Shared Memory Graph (Temporary In-Memory)
### New module
Create a new module (suggested):
- `src/base_agent/shared_memory.py`

### Responsibilities
- Maintain a directed graph of agent dependencies
- Store last published memory for each agent (final response only)
- Provide ?pull? function to retrieve memory for a given agent

### Data model (conceptual)
- Node: agent_id / agent_name
  - last_final: string
  - metadata: timestamp, etc. (optional)
- Edge: (src -> dst)
  - policy: currently only `final_response_only`

### API proposal
```
class SharedMemoryGraph:
    def add_agent(node_id: str) -> None
    def add_edge(src: str, dst: str, policy: str = "final_response_only") -> None

    def publish(node_id: str, content: str, metadata: dict | None = None) -> None

    def pull_for(node_id: str) -> list[MemoryItem]
        # returns list of MemoryItem from all direct predecessors

@dataclass
class MemoryItem:
    source_id: str
    content: str
    timestamp: float | datetime
    policy: str
```

### Thread safety
- Use a simple asyncio Lock or threading Lock around graph updates (future-proofing for async sub-agent execution).

---

## Injection Strategy (Agent-side)
### Where to inject
- Hook into `AgentEvent.BEFORE_LLM_CALL`.

### Behavior
- If agent has a memory graph and any incoming edges, call `pull_for(agent_id)`.
- For each memory item, create a **SystemMessage** or **AssistantMessage** (prefer SystemMessage for stronger grounding).
- Prepend to `conversation_history` for this run.
- Must avoid duplication (e.g. tag injected items in metadata).

Example injected message:
```
System: Shared context from AgentA:
"Ho estratto i requisiti principali: X, Y, Z."
```

---

## Publish Strategy (Agent-side)
### Where to publish
- Hook into `AgentEvent.BEFORE_FINAL_RESPONSE` or right after `_build_final_response()`.

### Behavior
- Publish only `AssistantResponse.content` (not tool outputs)
- Update `SharedMemoryGraph.publish(agent_id, content)`

---

## Orchestrator Awareness Message
### Requirement
When an agent is decorated with `@orchestrator`, it should receive an additional **system message** describing **dependency ordering** among sub-agents.
- It should NOT list sub-agent names or schemas (function calling already exposes them).
- It should simply warn: ?respect dependencies; do not call downstream before upstream?.

### Content example (minimal)
```
System: You are coordinating multiple sub-agents. Use them respecting the dependency graph.
Dependencies:
A -> B
A -> C
B -> C
Guideline: do not call an agent before its prerequisites have been executed.
```

### Insertion point
- Prefer injecting on the **first execution** (e.g., in `BaseAgent.execute_query_async`) if `_is_orchestrator == True` and not already injected.
- Alternatively, update in `register_agent()` when graph edges are set.

---

## Proposed Minimal Integration Steps (no persistence yet)
1) Create `src/base_agent/shared_memory.py`
   - Implement in-memory directed graph
   - Store last_final per node
2) Extend `BaseAgent` to optionally attach a `memory_graph` and `agent_id`
3) Add hook or direct logic:
   - BEFORE_LLM_CALL: inject shared memories
   - BEFORE_FINAL_RESPONSE or _build_final_response: publish own final response
4) In `@orchestrator` decorator:
   - Ensure orchestrator has `memory_graph`
   - When registering sub-agents, register them as nodes
   - Provide API to define edges (A->B, etc.)
5) Add orchestrator system message injection (dependency-only, no agent list)

---

## Open Decisions (for next session)
- How to identify agent_id (class name vs decorator name)
- How edges are defined (manual API vs inferred by orchestrator config)
- Whether injection should be SystemMessage or AssistantMessage
- Avoid duplicate injections within the same run
- Whether sub-agents need awareness message too (currently only orchestrator)

---

## Notes from Discussion
- Memory is **temporary** in-memory for now; persistence will be added later.
- Graph is a ?good idea? for relationship modeling.
- AgentB sees AgentA?s output **only when AgentB is invoked after AgentA** has published.
- Orchestrator does not need to relay context manually.

---

## File Location
This plan is stored at:
`docs/shared_memory_plan.md`

---

# ADDENDUM: Integration with Agent Factory

> This section was added after the Agent Factory design was finalized.
> See: `src/base_agent/AGENT_FACTORY_FINAL.md`

---

## Factory as Central Integration Point

The Agent Factory becomes the **single point** where shared memory is wired to agents. This simplifies integration and ensures consistency.

### Key Design Decisions Resolved

| Open Decision | Resolution |
|---------------|------------|
| How to identify `agent_id` | **Registry name** from `factory.register(name, ...)`. Each agent gets `agent_id = name`. |
| How edges are defined | **On the SharedMemoryGraph directly**, before attaching to factory. Factory does not manage edges. |
| Where memory_graph is attached | **In `factory.create()`**, automatically to all created agents (orchestrator and subagents). |

---

## Integration Flow

### Step 1: Create and configure the SharedMemoryGraph

```python
from src.base_agent.shared_memory import SharedMemoryGraph

# Create graph
graph = SharedMemoryGraph()

# Define agent nodes (optional, auto-created on first edge/publish)
graph.add_agent("requirements")
graph.add_agent("designer")
graph.add_agent("implementer")

# Define dependency edges
graph.add_edge("requirements", "designer")      # requirements -> designer
graph.add_edge("requirements", "implementer")   # requirements -> implementer
graph.add_edge("designer", "implementer")       # designer -> implementer
```

### Step 2: Attach graph to factory

```python
from src.base_agent.agent_factory import AgentFactory

factory = AgentFactory()

# Attach memory graph - all agents created will receive it
factory.with_memory_graph(graph)

# Register agents (names MUST match graph node IDs)
factory.register("requirements", RequirementsAgent, expose_as_subagent=True, ...)
factory.register("designer", DesignerAgent, expose_as_subagent=True, ...)
factory.register("implementer", ImplementerAgent, expose_as_subagent=True, ...)
factory.register("coordinator", CoordinatorAgent)
```

### Step 3: Create orchestrator with subagents

```python
coordinator = factory.create(
    "coordinator",
    subagents=["requirements", "designer", "implementer"]
)
```

**What happens internally:**

1. Factory creates `CoordinatorAgent` instance
2. Factory attaches: `coordinator.memory_graph = graph`
3. Factory attaches: `coordinator.agent_id = "coordinator"`
4. For each subagent name:
   - Factory creates subagent instance (e.g., `RequirementsAgent`)
   - Factory attaches: `subagent.memory_graph = graph` (same graph!)
   - Factory attaches: `subagent.agent_id = "requirements"`
   - Factory calls `coordinator.register_agent(subagent)`

All agents share the **same** `SharedMemoryGraph` instance.

---

## Runtime Behavior

### When subagent executes (e.g., "requirements")

```
┌─────────────────────────────────────────────────────────────┐
│ RequirementsAgent.execute_query_async()                     │
├─────────────────────────────────────────────────────────────┤
│ 1. BEFORE_LLM_CALL hook fires                               │
│    → Hook checks: agent.memory_graph exists?                │
│    → Hook calls: graph.pull_for("requirements")             │
│    → Returns: [] (no predecessors for requirements)         │
│    → No injection needed                                    │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM processes query, uses tools, produces response       │
├─────────────────────────────────────────────────────────────┤
│ 3. BEFORE_FINAL_RESPONSE hook fires                         │
│    → Hook calls: graph.publish("requirements", content)     │
│    → Graph stores: requirements.last_final = content        │
├─────────────────────────────────────────────────────────────┤
│ 4. Response returned to orchestrator                        │
└─────────────────────────────────────────────────────────────┘
```

### When dependent subagent executes (e.g., "designer")

```
┌─────────────────────────────────────────────────────────────┐
│ DesignerAgent.execute_query_async()                         │
├─────────────────────────────────────────────────────────────┤
│ 1. BEFORE_LLM_CALL hook fires                               │
│    → Hook calls: graph.pull_for("designer")                 │
│    → Returns: [MemoryItem(source="requirements", content=...)]│
│    → Hook injects SystemMessage into conversation_history:  │
│      "Shared context from requirements: ..."                │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM sees injected context + current query                │
│    → Produces response informed by requirements output      │
├─────────────────────────────────────────────────────────────┤
│ 3. BEFORE_FINAL_RESPONSE hook fires                         │
│    → Hook calls: graph.publish("designer", content)         │
├─────────────────────────────────────────────────────────────┤
│ 4. Response returned to orchestrator                        │
└─────────────────────────────────────────────────────────────┘
```

### When "implementer" executes (depends on both)

```
┌─────────────────────────────────────────────────────────────┐
│ ImplementerAgent.execute_query_async()                      │
├─────────────────────────────────────────────────────────────┤
│ 1. BEFORE_LLM_CALL hook fires                               │
│    → Hook calls: graph.pull_for("implementer")              │
│    → Returns: [                                             │
│        MemoryItem(source="requirements", content=...),      │
│        MemoryItem(source="designer", content=...)           │
│      ]                                                      │
│    → Hook injects TWO SystemMessages                        │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM sees context from BOTH predecessors                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Factory Implementation Changes

The factory already has a stub for `with_memory_graph()`. The full implementation:

```python
# In AgentFactory.create()
def create(self, name: str, *, subagents=None, ...):
    # ... existing logic ...

    agent = self._create_agent(spec, config)  # or _create_orchestrator

    # Attach memory graph if configured
    if self._memory_graph:
        agent.memory_graph = self._memory_graph
        agent.agent_id = name

    return agent

# In AgentFactory._create_subagent_instance()
def _create_subagent_instance(self, name: str, extra_config: Dict):
    # ... existing logic ...

    agent = cls(**config)

    # Attach memory graph to subagent too
    if self._memory_graph:
        agent.memory_graph = self._memory_graph
        agent.agent_id = name  # Uses registry name as agent_id

    return agent
```

---

## BaseAgent Changes Required

Add optional attributes and hook registration:

```python
# In BaseAgent.__init__
def __init__(self, ...):
    # ... existing code ...

    # Shared memory (set by factory, None if not using shared memory)
    self.memory_graph: Optional['SharedMemoryGraph'] = None
    self.agent_id: Optional[str] = None

    # Auto-register memory hooks if memory will be attached later
    # (hooks check for memory_graph presence at runtime)
    self._register_memory_hooks()

def _register_memory_hooks(self):
    """Register hooks for shared memory injection/publishing."""

    # Inject shared context before LLM call
    self.on(AgentEvent.BEFORE_LLM_CALL).do(self._inject_shared_memory)

    # Publish final response to graph
    self.on(AgentEvent.BEFORE_FINAL_RESPONSE).do(self._publish_to_memory)

def _inject_shared_memory(self, status: AgentStatus):
    """Pull and inject memories from predecessors."""
    if not self.memory_graph or not self.agent_id:
        return

    memories = self.memory_graph.pull_for(self.agent_id)
    for mem in memories:
        # Avoid duplicate injection (check if already injected this run)
        if self._is_memory_already_injected(mem):
            continue

        msg = SystemMessage(
            content=f"Shared context from {mem.source_id}:\n{mem.content}",
            metadata={"shared_memory": True, "source": mem.source_id}
        )
        # Insert after system message, before user obelix_types
        self.conversation_history.insert(1, msg)

def _publish_to_memory(self, status: AgentStatus):
    """Publish final response to memory graph."""
    if not self.memory_graph or not self.agent_id:
        return

    # Get the final response content (from assistant_message in status)
    content = status.assistant_message.content
    if content:
        self.memory_graph.publish(self.agent_id, content)
```

---

## Orchestrator Awareness Message

The orchestrator receives a system message about dependency ordering. This is injected by the factory when creating an orchestrator with a memory graph:

```python
# In AgentFactory._create_orchestrator(), after registering subagents:
if self._memory_graph and subagent_names:
    # Build dependency message from graph
    edges = self._memory_graph.get_edges_for_nodes(subagent_names)
    if edges:
        dep_msg = self._build_dependency_message(edges)
        agent.conversation_history.insert(1, SystemMessage(content=dep_msg))

def _build_dependency_message(self, edges: List[Tuple[str, str]]) -> str:
    lines = ["You are coordinating sub-agents with dependencies.", ""]
    lines.append("Dependency order (call upstream before downstream):")
    for src, dst in edges:
        lines.append(f"  {src} -> {dst}")
    lines.append("")
    lines.append("Respect this order: do not call an agent before its prerequisites.")
    return "\n".join(lines)
```

---

## Validation

The factory validates that subagent names match graph nodes:

```python
# In AgentFactory.create(), if memory_graph is set:
if self._memory_graph and subagents:
    for sub in subagents:
        if isinstance(sub, str):
            if not self._memory_graph.has_node(sub):
                logger.warning(
                    f"Subagent '{sub}' not found in memory graph. "
                    "It will not participate in shared memory."
                )
```

This is a **warning**, not an error, because not all subagents need to be in the graph.

---

## Complete Example

```python
from src.base_agent.agent_factory import AgentFactory
from src.base_agent.shared_memory import SharedMemoryGraph

# 1. Create memory graph with dependencies
graph = SharedMemoryGraph()
graph.add_edge("requirements", "designer")
graph.add_edge("requirements", "implementer")
graph.add_edge("designer", "implementer")

# 2. Create factory with memory
factory = AgentFactory()
factory.with_memory_graph(graph)

# 3. Register agents
factory.register("requirements", RequirementsAgent,
                 expose_as_subagent=True,
                 subagent_description="Extracts and analyzes requirements")
factory.register("designer", DesignerAgent,
                 expose_as_subagent=True,
                 subagent_description="Designs system architecture")
factory.register("implementer", ImplementerAgent,
                 expose_as_subagent=True,
                 subagent_description="Implements the solution")
factory.register("coordinator", CoordinatorAgent)

# 4. Create orchestrator
coordinator = factory.create(
    "coordinator",
    subagents=["requirements", "designer", "implementer"]
)

# 5. Execute
response = await coordinator.execute_query_async(
    "Build a user authentication system with OAuth support"
)

# Flow:
# - Orchestrator sees dependency message in system prompt
# - Orchestrator calls "requirements" first
# - requirements publishes its output to graph
# - Orchestrator calls "designer"
# - designer receives requirements' output via injection
# - designer publishes its output
# - Orchestrator calls "implementer"
# - implementer receives BOTH requirements and designer outputs
# - Final response returned
```

---

## Summary: What Changes Where

| Component | Change |
|-----------|--------|
| `SharedMemoryGraph` | New module (`src/base_agent/shared_memory.py`) |
| `AgentFactory` | Implement `with_memory_graph()`, attach to agents, inject dependency message |
| `BaseAgent` | Add `memory_graph`, `agent_id` attributes; add memory hooks |
| Hooks | Use existing `BEFORE_LLM_CALL` and `BEFORE_FINAL_RESPONSE` |
| Decorators | No changes needed |
