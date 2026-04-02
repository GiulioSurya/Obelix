"""Tests for OpenShellDeployer.

All tests mock the openshell SDK and CLI — no real Gateway or sandbox required.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, dataclass
from unittest.mock import MagicMock, patch

import pytest

from obelix.core.agent.agent_factory import AgentFactory

# ---------------------------------------------------------------------------
# Mock helpers (reused across all test classes)
# ---------------------------------------------------------------------------


@dataclass
class FakeExecResult:
    """Mimics openshell ExecResult."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class FakeSandboxRef:
    """Mimics openshell SandboxRef."""

    id: str = "sb-123"
    name: str = "test-sandbox"


def _make_mock_client(
    exec_result: FakeExecResult | None = None,
    sandbox_ref: FakeSandboxRef | None = None,
) -> MagicMock:
    """Create a mock SandboxClient with sensible defaults."""
    client = MagicMock()
    client.exec.return_value = exec_result or FakeExecResult(exit_code=0, stdout="ok\n")
    ref = sandbox_ref or FakeSandboxRef()
    client.get.return_value = ref
    client.create.return_value = ref
    client.wait_ready.return_value = None
    client.delete.return_value = None
    client.health.return_value = True
    client.close.return_value = None
    return client


def _patch_sdk(mock_client: MagicMock):
    """Patch the openshell SDK in the deployer module."""
    mock_class = MagicMock()
    mock_class.return_value = mock_client
    mock_class.from_active_cluster.return_value = mock_client
    return patch(
        "obelix.adapters.outbound.openshell.deployer.SandboxClient",
        mock_class,
    )


def _make_factory() -> AgentFactory:
    """Create a factory with a minimal agent registered."""
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
    factory.register("test_agent", DummyAgent)
    return factory


class TestConstructor:
    """OpenShellDeployer constructor stores config, does not connect."""

    def test_minimal_params(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        assert deployer._port == 8002
        assert deployer._policy is None
        assert deployer._providers is None
        assert deployer._dockerfile is None
        assert deployer._image is None

    def test_all_params(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(
            factory,
            "test_agent",
            port=9000,
            policy="policy.yaml",
            providers=["anthropic"],
            dockerfile="Dockerfile",
            gateway="gw:8080",
            tls_cert_dir="/certs",
            endpoint="https://prod.example.com",
            version="1.0.0",
            description="My agent",
            provider_name="Acme",
            subagents=["sub1"],
            subagent_config={"sub1": {"key": "val"}},
        )
        assert deployer._port == 9000
        assert deployer._policy == "policy.yaml"
        assert deployer._providers == ["anthropic"]
        assert deployer._dockerfile == "Dockerfile"


class TestValidation:
    """_validate() checks SDK, CLI, gateway, and param conflicts."""

    def test_dockerfile_and_image_conflict(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(
            factory, "test_agent", dockerfile="Dockerfile", image="my-image"
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            asyncio.get_event_loop().run_until_complete(deployer._validate())

    def test_sdk_not_installed(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        with patch("obelix.adapters.outbound.openshell.deployer.SandboxClient", None):
            with pytest.raises(ImportError, match="openshell"):
                asyncio.get_event_loop().run_until_complete(deployer._validate())

    def test_cli_not_in_path(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client), patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="CLI"):
                asyncio.get_event_loop().run_until_complete(deployer._validate())

    def test_gateway_unreachable(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        mock_client = _make_mock_client()
        mock_client.health.side_effect = Exception("connection refused")
        with (
            _patch_sdk(mock_client),
            patch("shutil.which", return_value="/usr/bin/openshell"),
        ):
            with pytest.raises(RuntimeError, match="gateway"):
                asyncio.get_event_loop().run_until_complete(deployer._validate())

    def test_validation_passes(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        mock_client = _make_mock_client()
        with (
            _patch_sdk(mock_client),
            patch("shutil.which", return_value="/usr/bin/openshell"),
        ):
            asyncio.get_event_loop().run_until_complete(deployer._validate())
            assert deployer._client is not None


class TestDeploymentInfo:
    """DeploymentInfo is a frozen dataclass with sandbox_name, endpoint, port."""

    def test_construction(self):
        from obelix.adapters.outbound.openshell.deployer import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="test-sb",
            endpoint="http://localhost:8002",
            port=8002,
        )
        assert info.sandbox_name == "test-sb"
        assert info.endpoint == "http://localhost:8002"
        assert info.port == 8002

    def test_frozen(self):
        from obelix.adapters.outbound.openshell.deployer import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="sb", endpoint="http://localhost:8002", port=8002
        )
        with pytest.raises(FrozenInstanceError):
            info.sandbox_name = "other"

    def test_importable_from_package(self):
        from obelix.adapters.outbound.openshell import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="sb", endpoint="http://localhost:8002", port=8002
        )
        assert info.sandbox_name == "sb"
