"""
Agent Factory for centralized agent creation and composition.

All agents should be created through this factory in production code.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.agent.shared_memory import SharedMemoryGraph
    from obelix.core.tracer.tracer import Tracer

logger = get_logger(__name__)


@dataclass
class AgentSpec:
    """
    Specification for a registered agent.

    Stores the class and configuration for creating agent instances.
    """

    cls: type["BaseAgent"]
    subagent_name: str | None = None
    subagent_description: str | None = None
    stateless: bool = False
    defaults: dict[str, Any] = field(default_factory=dict)


class AgentFactory:
    """
    Factory for creating and composing agents.

    Provides a centralized API for:
    - Registering agent classes with configuration
    - Creating standalone agent instances
    - Creating orchestrator agents with subagents

    Usage:
        factory = AgentFactory()

        factory.register("weather", WeatherAgent,
                         subagent_description="Provides weather forecasts")
        factory.register("planner", PlannerAgent)

        # Standalone agent
        weather = factory.create("weather")

        # Orchestrator with subagents
        planner = factory.create("planner", subagents=["weather"])
    """

    def __init__(self, global_defaults: dict[str, Any] | None = None):
        """
        Initialize the factory.

        Args:
            global_defaults: Default constructor arguments applied to ALL agents
                created by this factory. Can be overridden per-agent in register()
                or per-instance in create().
        """
        self._global_defaults = global_defaults or {}
        self._registry: dict[str, AgentSpec] = {}
        self._memory_graph: SharedMemoryGraph | None = None
        self._tracer: Tracer | None = None

    def register(
        self,
        name: str,
        cls: type["BaseAgent"],
        *,
        subagent_name: str | None = None,
        subagent_description: str | None = None,
        stateless: bool = False,
        defaults: dict[str, Any] | None = None,
    ) -> "AgentFactory":
        """
        Register an agent class with the factory.

        Args:
            name: Unique identifier in the registry. Used in create() and subagents lists.
            cls: The BaseAgent subclass to register.
            subagent_name: Tool name when used as subagent (defaults to `name`).
            subagent_description: Description for LLM when used as subagent.
                Required when this agent is referenced in a subagents list.
            stateless: If False (default), conversation history preserved across calls.
                If True, each execution is isolated (parallel-safe).
            defaults: Default constructor arguments for this agent.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If name already registered.
        """
        if name in self._registry:
            raise ValueError(f"Agent '{name}' is already registered")

        self._registry[name] = AgentSpec(
            cls=cls,
            subagent_name=subagent_name or name,
            subagent_description=subagent_description,
            stateless=stateless,
            defaults=defaults or {},
        )

        logger.info(f"AgentFactory: registered '{name}' ({cls.__name__})")
        return self

    def with_tracer(self, tracer: "Tracer") -> "AgentFactory":
        """Configure a tracer for all agents created by this factory.

        Returns:
            self for method chaining.
        """
        self._tracer = tracer
        return self

    def with_memory_graph(self, graph: "SharedMemoryGraph") -> "AgentFactory":
        """Configure the shared memory graph.

        All agents created by this factory will share the SAME graph instance.

        Returns:
            self for method chaining.
        """
        self._memory_graph = graph
        return self

    def create(
        self,
        name: str,
        *,
        subagents: list[Union[str, "BaseAgent"]] | None = None,
        subagent_config: dict[str, dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> "BaseAgent":
        """
        Create an agent instance from a registered class.

        Args:
            name: Registered agent name.
            subagents: List of subagents to attach. Accepts:
                - Strings: registry names, new instance created
                - BaseAgent instances: used directly (shared state).
                  Requires subagent_meta dict with 'name' and 'description'.
            subagent_config: Per-subagent constructor overrides.
                Keys are registry names, values are dicts of kwargs.
                Only applies to string-based subagents.
            **overrides: Override constructor parameters for the main agent.

        Returns:
            Configured BaseAgent instance.

        Raises:
            ValueError: If agent not registered or validation fails.
        """
        if name not in self._registry:
            raise ValueError(
                f"Agent '{name}' not registered. Available: {list(self._registry.keys())}"
            )

        spec = self._registry[name]
        config = {**self._global_defaults, **spec.defaults, **overrides}

        if self._tracer and "tracer" not in config:
            config["tracer"] = self._tracer

        # Validate subagent_config keys
        if subagent_config:
            subagent_names = {s for s in (subagents or []) if isinstance(s, str)}
            invalid_keys = set(subagent_config.keys()) - subagent_names
            if invalid_keys:
                raise ValueError(
                    f"subagent_config contains keys not in subagents list: {invalid_keys}"
                )

        agent = spec.cls(**config)

        # Attach shared memory graph
        if self._memory_graph:
            agent.memory_graph = self._memory_graph
            agent.agent_id = name

        if subagents is not None:
            self._attach_subagents(agent, subagents, subagent_config or {})

            # Inject dependency awareness message
            if self._memory_graph:
                self._inject_dependency_awareness(agent, subagents)

        return agent

    def _attach_subagents(
        self,
        agent: "BaseAgent",
        subagents: list[Union[str, "BaseAgent"]],
        subagent_config: dict[str, dict[str, Any]],
    ) -> None:
        """Register subagents on the orchestrator agent."""
        for sub in subagents:
            if isinstance(sub, str):
                self._attach_from_registry(agent, sub, subagent_config.get(sub, {}))
            else:
                raise ValueError(
                    "Instance-based subagents are not supported via factory. "
                    "Use agent.register_agent() directly for pre-built instances."
                )

    def _attach_from_registry(
        self,
        agent: "BaseAgent",
        sub_name: str,
        extra_config: dict[str, Any],
    ) -> None:
        """Create a subagent from registry and register it on the agent."""
        if sub_name not in self._registry:
            raise ValueError(f"Subagent '{sub_name}' not registered")

        sub_spec = self._registry[sub_name]

        if not sub_spec.subagent_description:
            raise ValueError(
                f"Agent '{sub_name}': subagent_description is required "
                f"when used as a subagent in create()"
            )

        # Merge config: global < spec.defaults < extra_config
        config = {**self._global_defaults, **sub_spec.defaults, **extra_config}
        if self._tracer and "tracer" not in config:
            config["tracer"] = self._tracer
        sub_agent = sub_spec.cls(**config)

        # Attach shared memory graph to sub-agent
        if self._memory_graph:
            sub_agent.memory_graph = self._memory_graph
            sub_agent.agent_id = sub_name
            if not self._memory_graph.has_node(sub_name):
                logger.warning(
                    f"Sub-agent '{sub_name}' not found in memory graph. "
                    "It won't participate in shared memory."
                )

        agent.register_agent(
            sub_agent,
            name=sub_spec.subagent_name or sub_name,
            description=sub_spec.subagent_description,
            stateless=sub_spec.stateless,
        )

    def _inject_dependency_awareness(
        self,
        agent: "BaseAgent",
        subagent_refs: list,
    ) -> None:
        """Inject dependency awareness system message into orchestrator."""
        from obelix.core.model.system_message import SystemMessage

        subagent_names = [s for s in subagent_refs if isinstance(s, str)]
        if not subagent_names:
            return

        edges = self._memory_graph.get_edges_for_nodes(subagent_names)
        if not edges:
            logger.debug(
                "AgentFactory: no edges between sub-agents, skip awareness message"
            )
            return

        try:
            order = self._memory_graph.get_topological_order(subagent_names)
            order_str = f"\nRecommended execution order: {', '.join(order)}"
        except Exception as e:
            logger.warning(f"AgentFactory: cannot compute topological order: {e}")
            order_str = ""

        lines = [
            "You are coordinating sub-agents with dependencies.",
            "",
            "Dependency order (call upstream before downstream):",
        ]
        for src, dst in edges:
            lines.append(f"  {src} -> {dst}")
        if order_str:
            lines.append(order_str)
        lines.append("")
        lines.append(
            "Guideline: do not call an agent before its prerequisites have been executed."
        )

        msg = SystemMessage(
            content="\n".join(lines), metadata={"orchestrator_awareness": True}
        )
        agent.conversation_history.insert(1, msg)
        logger.info("AgentFactory: awareness message injected into orchestrator")

    # ─── Utility Methods ─────────────────────────────────────────────────────

    def get_registered_names(self) -> list[str]:
        """Return all registered agent names."""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def get_spec(self, name: str) -> AgentSpec | None:
        """Get AgentSpec for introspection."""
        return self._registry.get(name)

    def unregister(self, name: str) -> bool:
        """
        Remove an agent from the registry.

        Returns:
            True if removed, False if not found.
        """
        if name in self._registry:
            del self._registry[name]
            logger.info(f"AgentFactory: unregistered '{name}'")
            return True
        return False

    def clear(self) -> None:
        """Remove all registered agents."""
        self._registry.clear()
        logger.info("AgentFactory: registry cleared")
