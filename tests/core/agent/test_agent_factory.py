"""Tests for obelix.core.agent.agent_factory.AgentFactory.

Covers registration, creation, config merging, subagent attachment,
unregister, clear, utility methods, with_tracer, and with_memory_graph.
"""

from unittest.mock import MagicMock

import pytest

from obelix.core.agent.agent_factory import AgentFactory, AgentSpec
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.subagent_wrapper import SubAgentWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockProvider:
    """Minimal provider for test agent construction."""

    model_id = "test-model"
    provider_type = "test"

    def __init__(self):
        from unittest.mock import AsyncMock

        self.invoke = AsyncMock()


class _WorkerAgent(BaseAgent):
    """Concrete agent class for factory tests."""

    pass


class _CoordinatorAgent(BaseAgent):
    """Concrete orchestrator agent class for factory tests."""

    pass


def _make_provider():
    return _MockProvider()


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestFactoryRegister:
    """Tests for AgentFactory.register."""

    def test_register_returns_self(self):
        """register() returns self for chaining."""
        factory = AgentFactory()
        result = factory.register(
            "worker", _WorkerAgent, subagent_description="does work"
        )
        assert result is factory

    def test_register_chaining(self):
        """Multiple register() calls can be chained."""
        factory = AgentFactory()
        result = factory.register("a", _WorkerAgent, subagent_description="A").register(
            "b", _CoordinatorAgent, subagent_description="B"
        )
        assert result is factory
        assert factory.is_registered("a")
        assert factory.is_registered("b")

    def test_register_duplicate_raises_value_error(self):
        """Registering the same name twice raises ValueError."""
        factory = AgentFactory()
        factory.register("worker", _WorkerAgent)
        with pytest.raises(ValueError, match="already registered"):
            factory.register("worker", _WorkerAgent)

    def test_register_stores_spec(self):
        """After register, the spec is retrievable."""
        factory = AgentFactory()
        factory.register(
            "worker",
            _WorkerAgent,
            subagent_description="desc",
            stateless=True,
            defaults={"max_iterations": 3},
        )
        spec = factory.get_spec("worker")
        assert spec is not None
        assert spec.cls is _WorkerAgent
        assert spec.subagent_description == "desc"
        assert spec.stateless is True
        assert spec.defaults == {"max_iterations": 3}

    def test_subagent_name_defaults_to_name(self):
        """When subagent_name is not given, it defaults to the registration name."""
        factory = AgentFactory()
        factory.register("my_worker", _WorkerAgent)
        spec = factory.get_spec("my_worker")
        assert spec.subagent_name == "my_worker"

    def test_subagent_name_custom(self):
        """Custom subagent_name overrides the default."""
        factory = AgentFactory()
        factory.register("worker", _WorkerAgent, subagent_name="custom_tool")
        spec = factory.get_spec("worker")
        assert spec.subagent_name == "custom_tool"


# ---------------------------------------------------------------------------
# create (basic)
# ---------------------------------------------------------------------------


class TestFactoryCreate:
    """Tests for AgentFactory.create basic functionality."""

    def test_create_returns_base_agent(self):
        """create() returns a BaseAgent instance."""
        factory = AgentFactory()
        provider = _make_provider()
        factory.register("worker", _WorkerAgent)
        agent = factory.create("worker", system_message="hi", provider=provider)
        assert isinstance(agent, _WorkerAgent)

    def test_create_unregistered_raises_value_error(self):
        """create() with unregistered name raises ValueError."""
        factory = AgentFactory()
        with pytest.raises(ValueError, match="not registered"):
            factory.create("nonexistent")

    def test_create_applies_overrides(self):
        """Overrides in create() are passed to the agent constructor."""
        factory = AgentFactory()
        provider = _make_provider()
        factory.register("worker", _WorkerAgent)
        agent = factory.create(
            "worker",
            system_message="custom",
            provider=provider,
            max_iterations=7,
        )
        assert agent.max_iterations == 7

    def test_create_merges_global_spec_override(self):
        """Config merging: global_defaults < spec.defaults < overrides."""
        provider = _make_provider()
        factory = AgentFactory(global_defaults={"max_iterations": 10})
        factory.register(
            "worker",
            _WorkerAgent,
            defaults={"max_iterations": 5},
        )
        # spec.defaults (5) should override global (10)
        agent = factory.create("worker", system_message="x", provider=provider)
        assert agent.max_iterations == 5

    def test_create_override_wins_over_defaults(self):
        """Direct overrides in create() win over all defaults."""
        provider = _make_provider()
        factory = AgentFactory(global_defaults={"max_iterations": 10})
        factory.register("worker", _WorkerAgent, defaults={"max_iterations": 5})
        agent = factory.create(
            "worker",
            system_message="x",
            provider=provider,
            max_iterations=2,
        )
        assert agent.max_iterations == 2

    def test_create_global_defaults_used_when_no_spec_defaults(self):
        """Global defaults are used when spec has no defaults for that key."""
        provider = _make_provider()
        factory = AgentFactory(
            global_defaults={"system_message": "global", "provider": provider}
        )
        factory.register("worker", _WorkerAgent)
        agent = factory.create("worker")
        assert agent.system_message.content == "global"


# ---------------------------------------------------------------------------
# create with subagents
# ---------------------------------------------------------------------------


class TestFactoryCreateWithSubagents:
    """Tests for AgentFactory.create with subagent composition."""

    def test_subagent_registered_on_agent(self):
        """create() with subagents attaches SubAgentWrapper to the agent."""
        provider = _make_provider()
        factory = AgentFactory()
        factory.register(
            "worker",
            _WorkerAgent,
            subagent_description="Worker agent",
            # subagent needs its own constructor args; pass them as defaults
            defaults={"system_message": "I am a worker", "provider": provider},
        )
        factory.register("coordinator", _CoordinatorAgent)
        agent = factory.create(
            "coordinator",
            subagents=["worker"],
            system_message="coord",
            provider=provider,
        )
        assert len(agent.registered_tools) >= 1
        wrappers = [t for t in agent.registered_tools if isinstance(t, SubAgentWrapper)]
        assert len(wrappers) == 1
        assert wrappers[0].tool_name == "worker"

    def test_subagent_not_registered_raises_value_error(self):
        """Referencing a subagent name not in registry raises ValueError."""
        provider = _make_provider()
        factory = AgentFactory()
        factory.register("coordinator", _CoordinatorAgent)
        with pytest.raises(ValueError, match="not registered"):
            factory.create(
                "coordinator",
                subagents=["ghost"],
                system_message="x",
                provider=provider,
            )

    def test_subagent_missing_description_raises_value_error(self):
        """Subagent without subagent_description raises ValueError."""
        provider = _make_provider()
        factory = AgentFactory()
        factory.register("worker", _WorkerAgent)  # no subagent_description
        factory.register("coordinator", _CoordinatorAgent)
        with pytest.raises(ValueError, match="subagent_description is required"):
            factory.create(
                "coordinator",
                subagents=["worker"],
                system_message="x",
                provider=provider,
            )

    def test_instance_based_subagent_raises_value_error(self):
        """Passing a BaseAgent instance in subagents list raises ValueError."""
        provider = _make_provider()
        factory = AgentFactory()
        factory.register("coordinator", _CoordinatorAgent)
        child = _WorkerAgent(system_message="x", provider=provider)
        with pytest.raises(
            ValueError, match="Instance-based subagents are not supported"
        ):
            factory.create(
                "coordinator",
                subagents=[child],
                system_message="x",
                provider=provider,
            )

    def test_subagent_config_invalid_key_raises_value_error(self):
        """subagent_config with a key not in subagents raises ValueError."""
        provider = _make_provider()
        factory = AgentFactory()
        factory.register("worker", _WorkerAgent, subagent_description="w")
        factory.register("coordinator", _CoordinatorAgent)
        with pytest.raises(ValueError, match="subagent_config contains keys"):
            factory.create(
                "coordinator",
                subagents=["worker"],
                subagent_config={"nonexistent": {"max_iterations": 1}},
                system_message="x",
                provider=provider,
            )

    def test_subagent_config_applied_to_subagent(self):
        """subagent_config overrides are applied to the subagent."""
        provider = _make_provider()
        factory = AgentFactory(
            global_defaults={"system_message": "default", "provider": provider}
        )
        factory.register(
            "worker",
            _WorkerAgent,
            subagent_description="Worker",
            defaults={"max_iterations": 10},
        )
        factory.register("coordinator", _CoordinatorAgent)
        agent = factory.create(
            "coordinator",
            subagents=["worker"],
            subagent_config={"worker": {"max_iterations": 2}},
        )
        # The subagent wrapper should have been created; we verify it exists
        wrappers = [t for t in agent.registered_tools if isinstance(t, SubAgentWrapper)]
        assert len(wrappers) == 1
        # The wrapped agent should have max_iterations=2 (from subagent_config)
        assert wrappers[0]._agent.max_iterations == 2


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


class TestFactoryUnregister:
    """Tests for AgentFactory.unregister."""

    def test_unregister_existing_returns_true(self):
        """unregister returns True when name was registered."""
        factory = AgentFactory()
        factory.register("worker", _WorkerAgent)
        assert factory.unregister("worker") is True
        assert not factory.is_registered("worker")

    def test_unregister_nonexistent_returns_false(self):
        """unregister returns False when name was not registered."""
        factory = AgentFactory()
        assert factory.unregister("ghost") is False


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestFactoryClear:
    """Tests for AgentFactory.clear."""

    def test_clear_empties_registry(self):
        """clear() removes all registrations."""
        factory = AgentFactory()
        factory.register("a", _WorkerAgent)
        factory.register("b", _CoordinatorAgent)
        factory.clear()
        assert factory.get_registered_names() == []

    def test_clear_then_register(self):
        """After clear(), new registrations work."""
        factory = AgentFactory()
        factory.register("old", _WorkerAgent)
        factory.clear()
        factory.register("new", _CoordinatorAgent)
        assert factory.is_registered("new")
        assert not factory.is_registered("old")


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------


class TestFactoryUtilities:
    """Tests for get_registered_names, is_registered, get_spec."""

    def test_get_registered_names(self):
        """get_registered_names returns all registered names."""
        factory = AgentFactory()
        factory.register("alpha", _WorkerAgent)
        factory.register("beta", _CoordinatorAgent)
        names = factory.get_registered_names()
        assert set(names) == {"alpha", "beta"}

    def test_get_registered_names_empty(self):
        """get_registered_names returns empty list when nothing registered."""
        factory = AgentFactory()
        assert factory.get_registered_names() == []

    def test_is_registered_true(self):
        """is_registered returns True for a registered name."""
        factory = AgentFactory()
        factory.register("x", _WorkerAgent)
        assert factory.is_registered("x") is True

    def test_is_registered_false(self):
        """is_registered returns False for an unregistered name."""
        factory = AgentFactory()
        assert factory.is_registered("nope") is False

    def test_get_spec_existing(self):
        """get_spec returns AgentSpec for registered name."""
        factory = AgentFactory()
        factory.register("w", _WorkerAgent, stateless=True)
        spec = factory.get_spec("w")
        assert isinstance(spec, AgentSpec)
        assert spec.cls is _WorkerAgent
        assert spec.stateless is True

    def test_get_spec_nonexistent_returns_none(self):
        """get_spec returns None for unregistered name."""
        factory = AgentFactory()
        assert factory.get_spec("ghost") is None


# ---------------------------------------------------------------------------
# with_tracer / with_memory_graph
# ---------------------------------------------------------------------------


class TestFactoryWithTracer:
    """Tests for AgentFactory.with_tracer."""

    def test_with_tracer_returns_self(self):
        """with_tracer() returns self for chaining."""
        factory = AgentFactory()
        mock_tracer = MagicMock()
        result = factory.with_tracer(mock_tracer)
        assert result is factory

    def test_with_tracer_stores_tracer(self):
        """with_tracer() stores the tracer on the factory."""
        factory = AgentFactory()
        mock_tracer = MagicMock()
        factory.with_tracer(mock_tracer)
        assert factory._tracer is mock_tracer

    def test_tracer_injected_into_created_agent(self):
        """Tracer is passed to created agents as 'tracer' kwarg."""
        provider = _make_provider()
        factory = AgentFactory()
        mock_tracer = MagicMock()
        factory.with_tracer(mock_tracer)
        factory.register("worker", _WorkerAgent)
        agent = factory.create("worker", system_message="x", provider=provider)
        assert agent._tracer is mock_tracer


class TestFactoryWithMemoryGraph:
    """Tests for AgentFactory.with_memory_graph."""

    def test_with_memory_graph_returns_self(self):
        """with_memory_graph() returns self for chaining."""
        factory = AgentFactory()
        mock_graph = MagicMock()
        result = factory.with_memory_graph(mock_graph)
        assert result is factory

    def test_with_memory_graph_stores_graph(self):
        """with_memory_graph() stores the graph on the factory."""
        factory = AgentFactory()
        mock_graph = MagicMock()
        factory.with_memory_graph(mock_graph)
        assert factory._memory_graph is mock_graph

    def test_memory_graph_injected_into_created_agent(self):
        """Memory graph and agent_id are set on created agent."""
        provider = _make_provider()
        factory = AgentFactory()
        mock_graph = MagicMock()
        factory.with_memory_graph(mock_graph)
        factory.register("worker", _WorkerAgent)
        agent = factory.create("worker", system_message="x", provider=provider)
        assert agent.memory_graph is mock_graph
        assert agent.agent_id == "worker"

    def test_memory_graph_set_on_subagent(self):
        """Memory graph is propagated to subagents created through the factory."""
        provider = _make_provider()
        factory = AgentFactory(
            global_defaults={"system_message": "default", "provider": provider}
        )
        mock_graph = MagicMock()
        mock_graph.has_node.return_value = True
        mock_graph.get_edges_for_nodes.return_value = []
        factory.with_memory_graph(mock_graph)
        factory.register("worker", _WorkerAgent, subagent_description="w")
        factory.register("coord", _CoordinatorAgent)
        agent = factory.create("coord", subagents=["worker"])
        wrappers = [t for t in agent.registered_tools if isinstance(t, SubAgentWrapper)]
        assert len(wrappers) == 1
        assert wrappers[0]._agent.memory_graph is mock_graph
        assert wrappers[0]._agent.agent_id == "worker"


# ---------------------------------------------------------------------------
# AgentSpec dataclass
# ---------------------------------------------------------------------------


class TestA2AOpenShellDeploy:
    """a2a_openshell_deploy() creates deployer and blocks until interrupted."""

    def test_creates_deployer_and_calls_deploy(self):
        """Verify the method creates an OpenShellDeployer and calls deploy()."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from obelix.core.agent.agent_factory import AgentFactory
        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.model import SystemMessage

        class DummyAgent(BaseAgent):
            def __init__(self, **kwargs):
                super().__init__(
                    system_message=SystemMessage(content="test"),
                    provider=MagicMock(),
                    **kwargs,
                )

        factory = AgentFactory()
        factory.register("my_agent", DummyAgent)

        mock_deployer_instance = MagicMock()
        mock_deployer_instance.deploy = AsyncMock()
        mock_deployer_instance.destroy = AsyncMock()

        with patch(
            "obelix.core.agent.agent_factory.OpenShellDeployer",
            return_value=mock_deployer_instance,
        ) as mock_cls:
            # Simulate SIGINT by making deploy raise KeyboardInterrupt
            mock_deployer_instance.deploy.side_effect = KeyboardInterrupt

            factory.a2a_openshell_deploy("my_agent", port=9000, policy="p.yaml")

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs[0][0] is factory  # agent_factory
            assert call_kwargs[0][1] == "my_agent"  # agent_name
            assert call_kwargs[1]["port"] == 9000
            assert call_kwargs[1]["policy"] == "p.yaml"

            # destroy called on KeyboardInterrupt
            mock_deployer_instance.destroy.assert_called_once()

    def test_passes_all_params(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from obelix.core.agent.agent_factory import AgentFactory
        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.model import SystemMessage

        class DummyAgent(BaseAgent):
            def __init__(self, **kwargs):
                super().__init__(
                    system_message=SystemMessage(content="test"),
                    provider=MagicMock(),
                    **kwargs,
                )

        factory = AgentFactory()
        factory.register("agent", DummyAgent)

        mock_deployer = MagicMock()
        mock_deployer.deploy = AsyncMock(side_effect=KeyboardInterrupt)
        mock_deployer.destroy = AsyncMock()

        with patch(
            "obelix.core.agent.agent_factory.OpenShellDeployer",
            return_value=mock_deployer,
        ) as mock_cls:
            factory.a2a_openshell_deploy(
                "agent",
                port=9000,
                policy="p.yaml",
                providers=["anthropic"],
                dockerfile="Dockerfile",
                gateway="gw:8080",
                tls_cert_dir="/certs",
                endpoint="https://prod.example.com",
                version="2.0.0",
                description="My agent",
                provider_name="Acme",
                subagents=["sub1"],
                subagent_config={"sub1": {}},
            )

            kwargs = mock_cls.call_args[1]
            assert kwargs["port"] == 9000
            assert kwargs["policy"] == "p.yaml"
            assert kwargs["providers"] == ["anthropic"]
            assert kwargs["dockerfile"] == "Dockerfile"
            assert kwargs["gateway"] == "gw:8080"
            assert kwargs["endpoint"] == "https://prod.example.com"
            assert kwargs["version"] == "2.0.0"
            assert kwargs["description"] == "My agent"
            assert kwargs["provider_name"] == "Acme"
            assert kwargs["subagents"] == ["sub1"]


# ---------------------------------------------------------------------------
# AgentSpec dataclass
# ---------------------------------------------------------------------------


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_defaults(self):
        """AgentSpec has correct defaults."""
        spec = AgentSpec(cls=_WorkerAgent)
        assert spec.subagent_name is None
        assert spec.subagent_description is None
        assert spec.stateless is False
        assert spec.defaults == {}

    def test_full_construction(self):
        """AgentSpec stores all provided fields."""
        spec = AgentSpec(
            cls=_CoordinatorAgent,
            subagent_name="coord",
            subagent_description="Coordinates",
            stateless=True,
            defaults={"max_iterations": 5},
        )
        assert spec.cls is _CoordinatorAgent
        assert spec.subagent_name == "coord"
        assert spec.subagent_description == "Coordinates"
        assert spec.stateless is True
        assert spec.defaults == {"max_iterations": 5}
