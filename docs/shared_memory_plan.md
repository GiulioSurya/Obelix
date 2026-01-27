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
