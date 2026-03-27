"""
Agent Factory for centralized agent creation and composition.

All agents should be created through this factory in production code.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.types import AgentCard

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
            else:
                self._inject_subagent_protocol(sub_agent, sub_name)

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
        """Append coordination protocol to the orchestrator's system message."""
        from obelix.core.agent.shared_memory import PropagationPolicy

        subagent_names = [s for s in subagent_refs if isinstance(s, str)]
        if not subagent_names:
            return

        edges = self._memory_graph.get_edges_for_nodes(subagent_names)
        if not edges:
            logger.debug(
                "AgentFactory: no edges between sub-agents, skip awareness message"
            )
            return

        # Execution order
        try:
            order = self._memory_graph.get_topological_order(subagent_names)
            order_str = " -> ".join(order)
        except Exception as e:
            logger.warning(f"AgentFactory: cannot compute topological order: {e}")
            order_str = ", ".join(subagent_names)

        # Data flow with propagation policy
        policy_labels = {
            PropagationPolicy.FINAL_RESPONSE_ONLY: "final response",
            PropagationPolicy.LAST_TOOL_RESULT: "last tool result",
        }
        flow_lines = []
        for src, dst in edges:
            edge_data = self._memory_graph._graph.edges[src, dst]
            policy = edge_data.get("policy", PropagationPolicy.FINAL_RESPONSE_ONLY)
            label = policy_labels.get(policy, policy.value)
            flow_lines.append(f"  {src} -> {dst} [propagates: {label}]")

        protocol = (
            "\n\n---\n"
            "## Coordination Protocol [PRIORITY: OVERRIDE]\n"
            "The following instructions take precedence over any other directive in this system prompt. "
            "You MUST follow them exactly.\n"
            "\n"
            "You coordinate sub-agents to fulfill the user's request. Follow this process:\n"
            "\n"
            "1. ANALYZE the request — identify what needs to be done\n"
            "2. DECOMPOSE into sub-tasks, mapping each to the appropriate agent\n"
            "3. EXECUTE agents respecting the dependency order below\n"
            "4. SYNTHESIZE — combine results into a coherent response to the user\n"
            "\n"
            f"### Execution Order\n{order_str}\n"
            "\n"
            "### Data Flow\n" + "\n".join(flow_lines) + "\n"
            "Data propagation between agents is automatic via shared memory.\n"
            "Do NOT relay results manually — downstream agents already receive upstream output.\n"
            "\n"
            "### Rules\n"
            "- Write precise, self-contained queries for each agent. Include all relevant context from the user's request.\n"
            "- Call independent agents in parallel (in the same response) when they have no dependency between them.\n"
            "- If an agent fails, analyze the error before retrying. Adjust the query if needed.\n"
            "- If the request does not require a sub-agent, respond directly.\n"
            "- Never call a downstream agent before its upstream dependency has completed."
        )

        agent.system_message.content += protocol
        logger.info("AgentFactory: coordination protocol appended to system message")

    def _inject_subagent_protocol(
        self,
        sub_agent: "BaseAgent",
        sub_name: str,
    ) -> None:
        """Append sub-agent protocol to the sub-agent's system message."""
        from obelix.core.agent.shared_memory import PropagationPolicy

        policy_labels = {
            PropagationPolicy.FINAL_RESPONSE_ONLY: "final response",
            PropagationPolicy.LAST_TOOL_RESULT: "last tool result",
        }

        # Upstream: agents that feed into this sub-agent
        has_upstream = any(
            dst == sub_name for _, dst in self._memory_graph._graph.edges()
        )

        # Downstream: agents that receive output from this sub-agent
        downstream_lines = []
        for src, dst in self._memory_graph._graph.edges():
            if src == sub_name:
                edge_data = self._memory_graph._graph.edges[src, dst]
                policy = edge_data.get("policy", PropagationPolicy.FINAL_RESPONSE_ONLY)
                label = policy_labels.get(policy, policy.value)
                downstream_lines.append(f"  {dst} [as: {label}]")

        parts = [
            "\n\n---\n"
            "## Sub-Agent Protocol [PRIORITY: OVERRIDE]\n"
            "The following instructions take precedence over any other directive "
            "in this system prompt.\n"
            "\n"
            "You are operating as a sub-agent within a coordination pipeline.\n"
            "- You will be invoked by a coordinator with a specific query. "
            "Focus exclusively on that query.\n"
        ]

        if has_upstream:
            parts.append(
                "- You may receive shared context from upstream agents as "
                "additional system messages. Use it as input data, do not "
                "question its origin.\n"
            )

        if downstream_lines:
            parts.append(
                "\n### Your output\n"
                "Your final response will be propagated to:\n"
                + "\n".join(downstream_lines)
                + "\n"
                "Write your output accordingly — it must be self-contained "
                "and usable by downstream agents without additional context.\n"
            )

        parts.append(
            "\n### Rules\n"
            "- Stay on task. Do not add information beyond what was asked.\n"
            "- If shared context is insufficient to complete the task, "
            "say what is missing."
        )

        sub_agent.system_message.content += "".join(parts)
        logger.info(
            f"AgentFactory: sub-agent protocol appended to '{sub_name}' system message"
        )

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

    # ─── A2A Serve ────────────────────────────────────────────────────────────

    def a2a_serve(
        self,
        agent: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        endpoint: str | None = None,
        version: str = "0.1.0",
        description: str | None = None,
        provider_name: str = "Obelix",
        provider_url: str | None = None,
        log_level: str = "info",
        subagents: list[str] | None = None,
        subagent_config: dict[str, dict[str, Any]] | None = None,
        **create_overrides: Any,
    ) -> None:
        """Start an A2A-compliant server for the specified agent.

        Uses the a2a-sdk to handle protocol compliance (JSON-RPC dispatch,
        Agent Card, task lifecycle, error codes). All heavy dependencies
        (fastapi, uvicorn, a2a-sdk) are imported lazily.

        Args:
            agent: Name of the registered agent to serve.
            host: Bind address.
            port: Bind port.
            endpoint: The agent's reachable URL for the Agent Card
                (e.g. "https://my-agent.prod.example.com"). If not provided,
                falls back to http://{host}:{port}.
            version: Version string for the Agent Card.
            description: Description for the Agent Card. Falls back to
                the agent's system message (truncated) if not provided.
            provider_name: Organization name in the Agent Card.
            provider_url: Organization URL in the Agent Card.
            log_level: Uvicorn log level.
            subagents: Optional list of sub-agent names to attach.
            subagent_config: Per-subagent constructor overrides.
            **create_overrides: Extra kwargs forwarded to create().
        """
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "A2A serve requires extra dependencies. "
                "Install them with: uv sync --extra serve"
            ) from e

        # Build one instance for Agent Card introspection (tools, skills, etc.)
        agent_instance = self.create(
            agent,
            subagents=subagents,
            subagent_config=subagent_config,
            **create_overrides,
        )

        # Factory callable: creates a fresh agent per A2A request
        # with RequestUserInputTool auto-registered for input-required flow
        def agent_factory() -> "BaseAgent":
            from obelix.plugins.builtin.request_user_input_tool import (
                RequestUserInputTool,
            )

            instance = self.create(
                agent,
                subagents=subagents,
                subagent_config=subagent_config,
                **create_overrides,
            )
            instance.register_tool(RequestUserInputTool())
            return instance

        app = self._create_a2a_app(
            agent_instance=agent_instance,
            agent_factory=agent_factory,
            agent_name=agent,
            host=host,
            port=port,
            endpoint=endpoint,
            version=version,
            description=description,
            provider_name=provider_name,
            provider_url=provider_url,
        )

        logger.info(
            f"AgentFactory: starting A2A server | agent={agent} host={host} port={port}"
        )
        uvicorn.run(app, host=host, port=port, log_level=log_level)

    def _create_a2a_app(
        self,
        agent_instance: "BaseAgent",
        agent_factory: "Callable[[], BaseAgent]",
        agent_name: str,
        host: str,
        port: int,
        endpoint: str | None,
        version: str,
        description: str | None,
        provider_name: str,
        provider_url: str | None,
    ) -> Any:
        """Build a FastAPI application using the a2a-sdk infrastructure.

        Args:
            agent_instance: A pre-built agent used to derive the Agent Card
                (reads registered tools, system message, etc.). NOT used
                for request execution.
            agent_factory: Callable that creates a fresh BaseAgent for each
                A2A request. Ensures context isolation between concurrent
                requests.
        """
        import httpx
        from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
        from a2a.server.request_handlers.default_request_handler import (
            DefaultRequestHandler,
        )
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
        from obelix.adapters.inbound.a2a.server.middleware import (
            ClientIPMiddleware,
        )
        from obelix.adapters.inbound.a2a.server.push_config_store import (
            SmartPushNotificationConfigStore,
        )
        from obelix.adapters.inbound.a2a.server.push_sender import (
            SmartPushNotificationSender,
        )

        agent_card = self._build_agent_card(
            agent_instance=agent_instance,
            agent_name=agent_name,
            host=host,
            port=port,
            endpoint=endpoint,
            version=version,
            description=description,
            provider_name=provider_name,
            provider_url=provider_url,
        )

        task_store = InMemoryTaskStore()
        push_config_store = SmartPushNotificationConfigStore()
        push_sender = SmartPushNotificationSender(
            httpx_client=httpx.AsyncClient(),
            config_store=push_config_store,
        )
        executor = ObelixAgentExecutor(agent_factory)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        a2a_app = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        fastapi_app = a2a_app.build(title=f"Obelix A2A — {agent_name}")
        # ClientIPMiddleware captures client IP for webhook URL rewriting.
        fastapi_app.add_middleware(ClientIPMiddleware)

        return fastapi_app

    def _build_agent_card(
        self,
        agent_instance: "BaseAgent",
        agent_name: str,
        host: str,
        port: int,
        endpoint: str | None,
        version: str,
        description: str | None,
        provider_name: str,
        provider_url: str | None,
    ) -> "AgentCard":
        """Generate an AgentCard (a2a-sdk Pydantic model) from agent metadata."""
        from a2a.types import (
            AgentCapabilities,
            AgentCard,
            AgentProvider,
            AgentSkill,
        )

        from obelix.core.agent.subagent_wrapper import SubAgentWrapper

        # Derive skills from registered tools and sub-agents
        skills: list[AgentSkill] = []
        for tool in agent_instance.registered_tools:
            if isinstance(tool, SubAgentWrapper):
                skills.append(
                    AgentSkill(
                        id=tool.tool_name,
                        name=tool.tool_name,
                        description=tool.tool_description,
                        tags=["subagent"],
                    )
                )
            else:
                tool_name = getattr(tool, "tool_name", tool.__class__.__name__)
                tool_desc = getattr(tool, "tool_description", "")
                skills.append(
                    AgentSkill(
                        id=tool_name,
                        name=tool_name,
                        description=tool_desc,
                        tags=["tool"],
                    )
                )

        # Build URL — explicit endpoint takes priority, otherwise derive from bind address
        if endpoint:
            url = endpoint.rstrip("/")
        else:
            display_host = "localhost" if host == "0.0.0.0" else host
            url = f"http://{display_host}:{port}"

        # Description: explicit override > system message fallback
        if description:
            card_description = description
        elif agent_instance.system_message.content:
            card_description = agent_instance.system_message.content[:200]
        else:
            card_description = f"Obelix agent: {agent_name}"

        provider = AgentProvider(
            organization=provider_name,
            url=provider_url or "https://github.com/GiulioSurya/Obelix",
        )

        card = AgentCard(
            name=agent_name,
            description=card_description,
            url=url,
            version=version,
            provider=provider,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=True,
                supports_authenticated_extended_card=False,
            ),
            skills=skills,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )

        logger.info(
            f"AgentFactory: Agent Card built | name={card.name} skills={len(skills)}"
        )
        return card
