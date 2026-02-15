# Obelix Documentation

Welcome to the Obelix documentation. Obelix is a multi-provider LLM agent framework with tool support, hooks system, and seamless integration with major AI providers.

This documentation covers how to build, compose, and deploy agents using Obelix.

---

## Quick Links

- **[README](../README.md)** - Installation, quick start, and project overview
- **[BaseAgent Guide](base_agent.md)** - Creating and using individual agents
- **[Agent Factory Guide](agent_factory.md)** - Composing agents and managing sub-agents
- **[Hooks API](hooks.md)** - Intercepting and modifying agent behavior

---

## User Guides

### Core Concepts

| Document | Description |
|----------|-------------|
| [BaseAgent Guide](base_agent.md) | Complete guide to `BaseAgent`: constructors, execution, tools, sub-agents, and hooks |
| [Agent Factory Guide](agent_factory.md) | How to register agents and compose them into orchestrators with optional shared memory |
| [Hooks API](hooks.md) | Understanding and implementing hooks for agent customization (validation, error recovery, context injection) |

### Key Features

**Tools**: Define custom tools that agents can invoke using the `@tool` decorator. See [BaseAgent Guide - Registering Tools](base_agent.md#registering-tools).

**Sub-Agents**: Register other agents as tools for hierarchical composition. See [BaseAgent Guide - Registering Sub-Agents](base_agent.md#registering-sub-agents).

**Hooks**: Intercept agent lifecycle events for cross-cutting concerns. See [Hooks API](hooks.md).

**Shared Memory**: Propagate context between dependent agents. See [Agent Factory Guide - Shared Memory Support](agent_factory.md#shared-memory-support).

**Providers**: Use any of 6 LLM providers (OpenAI, Anthropic, OCI, IBM Watson, Ollama, vLLM). See [README - Using Providers](../README.md#using-providers).

---

## Design & Architecture Documentation

### Design Plans (In Progress)

These documents describe planned features and design decisions. They are for reference and future implementation:

| Document | Description |
|----------|-------------|
| [Shared Memory Graph Plan](shared_memory_plan.md) | Design for agent-to-agent memory sharing via directed graph (partially implemented in `obelix.core.agent.shared_memory`) |
| [Inbound A2A Architecture Plan](INBOUND_ARCHITECTURE_PLAN.md) | Design for Agent-to-Agent (A2A) protocol support with HTTP/A2A adapters (not yet implemented) |

---

## Code Organization

The source code follows hexagonal architecture:

```
src/obelix/
├── core/                    # Domain logic (agents, tools, messages, hooks)
├── ports/                   # Abstract interfaces (inbound/outbound contracts)
├── adapters/                # Concrete implementations (LLM providers)
├── infrastructure/          # Cross-cutting concerns (config, logging, etc.)
└── plugins/                 # Optional extensions (MCP, built-in tools)
```

All imports use the `obelix` package name:
- `from obelix.core.agent import BaseAgent`
- `from obelix.core.tool.tool_decorator import tool`
- `from obelix.adapters.outbound.anthropic.provider import AnthropicProvider`

---

## Installation & Setup

For installation and quick start, see [README.md](../README.md#installation).

For tooling (uv, ruff, pytest), see [README.md - Tooling](../README.md#tooling).

---

## Common Tasks

### Create a Simple Agent

```python
from obelix.core.agent import BaseAgent

agent = BaseAgent(system_message="You are a helpful assistant.")
response = agent.execute_query("Hello!")
print(response.content)
```

See [BaseAgent Guide - Quick Start](base_agent.md#what-is-baseagent).

### Create a Tool

```python
from obelix.core.tool.tool_decorator import tool
from pydantic import Field

@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    async def execute(self) -> dict:
        return {"result": self.a + self.b}

agent = BaseAgent(system_message="...", tools=[CalculatorTool])
```

See [BaseAgent Guide - Creating a Tool](base_agent.md#creating-a-tool) and [README - Quick Start](../README.md#creating-a-tool).

### Compose Agents (Orchestrator)

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
factory.register("analyzer", AnalyzerAgent, subagent_description="Analyzes data")
factory.register("coordinator", CoordinatorAgent)

coordinator = factory.create("coordinator", subagents=["analyzer"])
response = coordinator.execute_query("Analyze the report")
```

See [Agent Factory Guide - Quick Start](agent_factory.md#quick-start) and [README - Sub-Agent Orchestration](../README.md#sub-agent-orchestration).

### Add Custom Behavior with Hooks

```python
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.system_message import SystemMessage

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        self.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            lambda s: "<answer>" not in (s.assistant_message.content or "")
        ).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_format_guidance],
        )

    def _inject_format_guidance(self, status) -> None:
        status.agent.conversation_history.append(
            SystemMessage(content="Format with <answer>...</answer> tags.")
        )
```

See [Hooks API - Complete Hook Examples](hooks.md#complete-hook-examples).

### Use a Specific LLM Provider

```python
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider
from obelix.core.agent import BaseAgent

connection = AnthropicConnection()
provider = AnthropicProvider(
    connection=connection,
    model_id="claude-sonnet-4-20250514",
)

agent = BaseAgent(system_message="...", provider=provider)
```

See [README - Using Providers](../README.md#using-providers).

---

## Troubleshooting

### "Agent not registered"

Make sure you registered the agent before trying to use it:

```python
factory.register("my_agent", MyAgent, subagent_description="...")
agent = factory.create("my_agent")  # OK
```

### "Tool validation failed"

Check that all tool parameters have `Field()` definitions with descriptions:

```python
from pydantic import Field

@tool(name="my_tool", description="...")
class MyTool:
    param: str = Field(..., description="Parameter description")  # OK
    # param: str  # ERROR: missing Field definition
```

### "max_iterations reached"

The agent hit the iteration limit without producing a final response. This can mean:
- Tool definitions are unclear or too many tools
- System prompt is contradictory
- LLM is in a loop calling the same tool repeatedly

Try:
- Simplifying the system prompt
- Reducing `max_iterations` to catch issues faster
- Adding output validation via hooks

---

## Contributing

See [README - How to Contribute](../README.md#how-to-contribute) for development guidelines.

---

## License

Apache License 2.0 - see [LICENSE](../LICENSE) file.

---

## Related Resources

- [Obelix GitHub Repository](https://github.com/GiulioSurya/Obelix)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
