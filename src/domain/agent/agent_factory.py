"""
Agent Factory for centralized agent creation and composition.

All agents should be created through this factory in production code.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Any, Union, TYPE_CHECKING

from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from src.domain.agent.base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class AgentSpec:
    """
    Specification for a registered agent.

    Stores the class and configuration for creating agent instances.
    """
    cls: Type['BaseAgent']
    subagent_name: Optional[str] = None
    subagent_description: Optional[str] = None
    stateless: bool = False
    defaults: Dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, global_defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize the factory.

        Args:
            global_defaults: Default constructor arguments applied to ALL agents
                created by this factory. Can be overridden per-agent in register()
                or per-instance in create().
        """
        self._global_defaults = global_defaults or {}
        self._registry: Dict[str, AgentSpec] = {}

    def register(
        self,
        name: str,
        cls: Type['BaseAgent'],
        *,
        subagent_name: Optional[str] = None,
        subagent_description: Optional[str] = None,
        stateless: bool = False,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> 'AgentFactory':
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

    def create(
        self,
        name: str,
        *,
        subagents: Optional[List[Union[str, 'BaseAgent']]] = None,
        subagent_config: Optional[Dict[str, Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> 'BaseAgent':
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

        # Validate subagent_config keys
        if subagent_config:
            subagent_names = {s for s in (subagents or []) if isinstance(s, str)}
            invalid_keys = set(subagent_config.keys()) - subagent_names
            if invalid_keys:
                raise ValueError(
                    f"subagent_config contains keys not in subagents list: {invalid_keys}"
                )

        agent = spec.cls(**config)

        if subagents is not None:
            self._attach_subagents(agent, subagents, subagent_config or {})

        return agent

    def _attach_subagents(
        self,
        agent: 'BaseAgent',
        subagents: List[Union[str, 'BaseAgent']],
        subagent_config: Dict[str, Dict[str, Any]],
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
        agent: 'BaseAgent',
        sub_name: str,
        extra_config: Dict[str, Any],
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
        sub_agent = sub_spec.cls(**config)

        agent.register_agent(
            sub_agent,
            name=sub_spec.subagent_name or sub_name,
            description=sub_spec.subagent_description,
            stateless=sub_spec.stateless,
        )

    # ─── Utility Methods ─────────────────────────────────────────────────────

    def get_registered_names(self) -> List[str]:
        """Return all registered agent names."""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def get_spec(self, name: str) -> Optional[AgentSpec]:
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