# src/base_agent/agent_factory.py
"""
Agent Factory for centralized agent creation and composition.

All agents should be created through this factory in production code.
See AGENT_FACTORY_FINAL.md for full specification.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Any, Union, TYPE_CHECKING

from src.domain.agent.subagent_decorator import subagent
from src.domain.agent.orchestrator_decorator import orchestrator
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
    expose_as_subagent: bool = False
    subagent_name: Optional[str] = None
    subagent_description: Optional[str] = None
    stateless: bool = False  # Default: preserve conversation history
    override_decorator: bool = False
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

    Provides a centralized API for:
    - Registering agent classes with configuration
    - Creating standalone agent instances
    - Creating orchestrator agents with subagents
    - Automatic decorator application without mutating original classes

    Usage:
        factory = AgentFactory()

        factory.register("weather", WeatherAgent,
                         expose_as_subagent=True,
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
        expose_as_subagent: bool = False,
        subagent_name: Optional[str] = None,
        subagent_description: Optional[str] = None,
        stateless: bool = False,
        override_decorator: bool = False,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> 'AgentFactory':
        """
        Register an agent class with the factory.

        Args:
            name: Unique identifier in the registry. Used in create() and subagents lists.
            cls: The BaseAgent subclass to register.
            expose_as_subagent: If True, this agent can be used as a subagent.
            subagent_name: Tool name when used as subagent (defaults to `name`).
            subagent_description: Description for LLM when used as subagent.
                Required if expose_as_subagent=True and class not decorated with @subagent.
            stateless: If False (default), conversation history preserved across calls.
                If True, each execution is isolated.
            override_decorator: If True, factory values override existing @subagent decorator.
            defaults: Default constructor arguments for this agent.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If name already registered or validation fails.
        """
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
        """
        Create an agent instance from a registered class.

        Args:
            name: Registered agent name.
            subagents: List of subagents to attach. If provided (even empty []),
                creates an orchestrator. Accepts:
                - Strings: registry names, new instance created
                - BaseAgent instances: used directly (shared state)
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

        # Validate subagent_config keys match subagents list
        if subagent_config:
            subagent_names = {s for s in (subagents or []) if isinstance(s, str)}
            invalid_keys = set(subagent_config.keys()) - subagent_names
            if invalid_keys:
                raise ValueError(
                    f"subagent_config contains keys not in subagents list: {invalid_keys}"
                )

        # Create orchestrator if subagents provided, otherwise standalone
        if subagents is not None:
            agent = self._create_orchestrator(spec, config, subagents, subagent_config or {})
        else:
            agent = self._create_agent(spec, config)

        return agent

    def _create_agent(self, spec: AgentSpec, config: Dict[str, Any]) -> 'BaseAgent':
        """Create a standalone agent instance."""
        return spec.cls(**config)

    def _create_orchestrator(
        self,
        spec: AgentSpec,
        config: Dict[str, Any],
        subagents: List[Union[str, 'BaseAgent']],
        subagent_config: Dict[str, Dict[str, Any]]
    ) -> 'BaseAgent':
        """
        Create an orchestrator with subagents.

        Uses the registered class (not a generic OrchestratorAgent).
        Applies @orchestrator decorator to dynamic subclass if needed.
        """
        # Get or create orchestrator-capable class
        cls = self._ensure_orchestrator_class(spec)

        # Create orchestrator instance
        agent = cls(**config)

        # Register each subagent
        for sub in subagents:
            if isinstance(sub, str):
                # Create new instance from registry
                sub_agent = self._create_subagent_instance(sub, subagent_config.get(sub, {}))
            else:
                # Use provided instance directly
                sub_agent = self._validate_subagent_instance(sub)

            agent.register_agent(sub_agent)

        return agent

    def _ensure_orchestrator_class(self, spec: AgentSpec) -> Type['BaseAgent']:
        """
        Get or create orchestrator-capable class.

        If already decorated with @orchestrator, returns class as-is.
        Otherwise, creates a dynamic subclass with decorator applied.
        This avoids mutating the original class.
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

    def _create_subagent_instance(
        self,
        name: str,
        extra_config: Dict[str, Any]
    ) -> 'BaseAgent':
        """
        Create a subagent instance from registry.

        Args:
            name: Registry name of the subagent.
            extra_config: Additional constructor overrides from subagent_config.
        """
        if name not in self._registry:
            raise ValueError(f"Subagent '{name}' not registered")

        spec = self._registry[name]

        if not spec.expose_as_subagent:
            raise ValueError(
                f"Agent '{name}' is not exposed as subagent. "
                f"Set expose_as_subagent=True in register()"
            )

        # Get or create subagent-capable class
        cls = self._ensure_subagent_class(spec, name)

        # Merge config: global < spec.defaults < extra_config
        config = {**self._global_defaults, **spec.defaults, **extra_config}

        # Create instance
        agent = cls(**config)

        return agent

    def _ensure_subagent_class(
        self,
        spec: AgentSpec,
        registry_name: str
    ) -> Type['BaseAgent']:
        """
        Get or create subagent-capable class.

        If already decorated with @subagent and not overriding, returns class as-is.
        Otherwise, creates a dynamic subclass with decorator applied.
        """
        # Use original if decorated AND not overriding
        if spec.is_already_subagent() and not spec.override_decorator:
            return spec.cls

        # Create dynamic subclass to avoid mutating original
        subagent_cls = type(
            f"{spec.cls.__name__}SubAgent",
            (spec.cls,),
            {'__module__': spec.cls.__module__}
        )

        # Apply @subagent decorator
        return subagent(
            name=spec.subagent_name or registry_name,
            description=spec.subagent_description or f"Subagent {registry_name}",
            stateless=spec.stateless,
        )(subagent_cls)

    def _validate_subagent_instance(self, instance: 'BaseAgent') -> 'BaseAgent':
        """
        Validate that an instance has subagent capabilities.

        Args:
            instance: BaseAgent instance to validate.

        Returns:
            The validated instance.

        Raises:
            ValueError: If instance lacks subagent capabilities.
        """
        if not hasattr(instance, 'subagent_name'):
            raise ValueError(
                f"Instance {instance.__class__.__name__} must be subagent-capable. "
                f"Create it via factory with expose_as_subagent=True."
            )

        # Warn if sharing stateful subagent
        if not getattr(instance, 'subagent_stateless', True):
            logger.warning(
                f"Sharing stateful subagent '{instance.subagent_name}' between orchestrators. "
                "Calls will be serialized and conversation history shared."
            )

        return instance

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

        Args:
            name: Registry name to remove.

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
