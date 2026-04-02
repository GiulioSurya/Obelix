"""Tests for OpenShellDeployer.

All tests mock the openshell SDK and CLI — no real Gateway or sandbox required.
"""

from __future__ import annotations

import asyncio
import tempfile  # noqa: F401  (used in TestBuildImage)
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
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


class TestRunCli:
    """_run_cli wraps openshell CLI calls as async subprocess."""

    @pytest.fixture
    def deployer(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        return OpenShellDeployer(_make_factory(), "test_agent")

    def test_success(self, deployer):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = asyncio.get_event_loop().run_until_complete(
                deployer._run_cli(["provider", "get", "anthropic"])
            )
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "openshell"
        assert cmd[1:] == ["provider", "get", "anthropic"]
        assert result == mock_result

    def test_failure_raises(self, deployer):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "not found"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="not found"):
                asyncio.get_event_loop().run_until_complete(
                    deployer._run_cli(["provider", "get", "missing"])
                )

    def test_failure_check_false(self, deployer):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not found"
        with patch("subprocess.run", return_value=mock_result):
            result = asyncio.get_event_loop().run_until_complete(
                deployer._run_cli(["provider", "get", "x"], check=False)
            )
        assert result.returncode == 1


class TestEnsureProviders:
    """_ensure_providers registers LLM providers in the OpenShell gateway."""

    @pytest.fixture
    def deployer(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        return OpenShellDeployer(_make_factory(), "test_agent", providers=["anthropic"])

    def test_provider_already_exists_skips_create(self, deployer):
        """If 'openshell provider get <name>' succeeds, skip create."""
        get_result = MagicMock(returncode=0, stdout="anthropic", stderr="")
        with patch("subprocess.run", return_value=get_result) as mock_run:
            asyncio.get_event_loop().run_until_complete(deployer._ensure_providers())
        calls = mock_run.call_args_list
        assert len(calls) == 1
        assert "get" in calls[0][0][0]

    def test_provider_not_found_creates(self, deployer):
        """If 'get' fails, 'create --from-existing' is called."""
        get_result = MagicMock(returncode=1, stdout="", stderr="not found")
        create_result = MagicMock(returncode=0, stdout="created", stderr="")

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "get" in cmd:
                return get_result
            return create_result

        with patch("subprocess.run", side_effect=side_effect):
            asyncio.get_event_loop().run_until_complete(deployer._ensure_providers())
        assert call_count == 2

    def test_no_providers_is_noop(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", providers=None)
        with patch("subprocess.run") as mock_run:
            asyncio.get_event_loop().run_until_complete(deployer._ensure_providers())
        mock_run.assert_not_called()

    def test_create_failure_raises(self, deployer):
        """If create also fails, RuntimeError is raised."""
        fail = MagicMock(returncode=1, stdout="", stderr="auth error")

        with patch("subprocess.run", return_value=fail):
            with pytest.raises(RuntimeError, match="auth error"):
                asyncio.get_event_loop().run_until_complete(
                    deployer._ensure_providers()
                )


class TestBuildImage:
    """_build_image generates or uses a Dockerfile, returns the --from arg."""

    def test_prebuilt_image_returns_image_ref(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", image="registry.io/agent:v1"
        )
        result = asyncio.get_event_loop().run_until_complete(deployer._build_image())
        assert result == "registry.io/agent:v1"

    def test_custom_dockerfile_returns_path(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", dockerfile="/app/Dockerfile"
        )
        result = asyncio.get_event_loop().run_until_complete(deployer._build_image())
        assert result == "/app/Dockerfile"

    def test_auto_generate_creates_temp_dockerfile(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        result = asyncio.get_event_loop().run_until_complete(deployer._build_image())
        assert Path(result).name == "Dockerfile"
        content = Path(result).read_text()
        assert "python:3.13" in content
        assert "uv" in content
        assert "pyproject.toml" in content

    def test_auto_generate_includes_entrypoint(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", port=9000)
        result = asyncio.get_event_loop().run_until_complete(deployer._build_image())
        content = Path(result).read_text()
        assert "test_agent" in content
        assert "9000" in content


class TestCreateSandbox:
    """_create_sandbox creates a sandbox via CLI with --from and waits via SDK."""

    def test_creates_sandbox_with_from_arg(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        deployer._client = mock_client

        create_result = MagicMock(returncode=0, stdout="sandbox-abc created", stderr="")
        with patch("subprocess.run", return_value=create_result):
            asyncio.get_event_loop().run_until_complete(
                deployer._create_sandbox("/tmp/Dockerfile")
            )

        assert deployer._sandbox_name is not None
        mock_client.wait_ready.assert_called_once()

    def test_sandbox_name_includes_obelix_prefix(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        deployer._client = mock_client

        create_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=create_result):
            asyncio.get_event_loop().run_until_complete(
                deployer._create_sandbox("my-image:latest")
            )

        assert deployer._sandbox_name.startswith("obelix-")

    def test_with_provider_and_policy(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(),
            "test_agent",
            providers=["anthropic"],
            policy="policy.yaml",
        )
        mock_client = _make_mock_client()
        deployer._client = mock_client

        create_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=create_result) as mock_run:
            asyncio.get_event_loop().run_until_complete(
                deployer._create_sandbox("my-image:latest")
            )

        cmd = mock_run.call_args[0][0]
        assert "--provider" in cmd
        assert "anthropic" in cmd
        assert "--policy" in cmd
        assert "policy.yaml" in cmd

    def test_no_policy_warning(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", policy=None)
        mock_client = _make_mock_client()
        deployer._client = mock_client

        create_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=create_result):
            asyncio.get_event_loop().run_until_complete(
                deployer._create_sandbox("my-image")
            )

    def test_create_failure_raises(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        deployer._client = mock_client

        fail = MagicMock(returncode=1, stdout="", stderr="spec invalid")
        with patch("subprocess.run", return_value=fail):
            with pytest.raises(RuntimeError, match="spec invalid"):
                asyncio.get_event_loop().run_until_complete(
                    deployer._create_sandbox("bad-image")
                )


class TestStartServer:
    """_start_server starts the A2A server inside the sandbox via SDK exec."""

    def test_exec_called_with_correct_command(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", port=8002)
        mock_client = _make_mock_client()
        ref = FakeSandboxRef(id="sb-123", name="obelix-abc")
        mock_client.get.return_value = ref
        deployer._client = mock_client
        deployer._sandbox_name = "obelix-abc"

        asyncio.get_event_loop().run_until_complete(deployer._start_server())

        mock_client.exec.assert_called_once()
        call_args = mock_client.exec.call_args
        assert call_args.kwargs["sandbox_id"] == "sb-123"
        cmd = call_args.kwargs["command"]
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        assert "test_agent" in cmd[2]
        assert "8002" in cmd[2]

    def test_exec_failure_raises(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        mock_client.get.return_value = FakeSandboxRef(id="sb-123", name="obelix-abc")
        mock_client.exec.side_effect = Exception("entrypoint failed")
        deployer._client = mock_client
        deployer._sandbox_name = "obelix-abc"

        with pytest.raises(RuntimeError, match="entrypoint failed"):
            asyncio.get_event_loop().run_until_complete(deployer._start_server())


class TestPortForwarding:
    """_start_forward and _stop_forward manage port forwarding via CLI."""

    @pytest.fixture
    def deployer(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        d = OpenShellDeployer(_make_factory(), "test_agent", port=8002)
        d._sandbox_name = "obelix-abc"
        return d

    def test_start_forward(self, deployer):
        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result) as mock_run:
            asyncio.get_event_loop().run_until_complete(deployer._start_forward())
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "openshell",
            "forward",
            "start",
            "8002",
            "obelix-abc",
            "-d",
        ]

    def test_stop_forward(self, deployer):
        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result) as mock_run:
            asyncio.get_event_loop().run_until_complete(deployer._stop_forward())
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "openshell",
            "forward",
            "stop",
            "8002",
            "obelix-abc",
        ]

    def test_stop_forward_failure_does_not_raise(self, deployer):
        """Cleanup is best-effort — stop_forward should not raise."""
        fail = MagicMock(returncode=1, stdout="", stderr="no forward found")
        with patch("subprocess.run", return_value=fail):
            asyncio.get_event_loop().run_until_complete(deployer._stop_forward())


class TestDestroy:
    """destroy() stops forward, deletes sandbox, closes client. Idempotent."""

    def test_full_cleanup_sequence(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", port=8002)
        mock_client = _make_mock_client()
        deployer._client = mock_client
        deployer._sandbox_name = "obelix-abc"

        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result):
            asyncio.get_event_loop().run_until_complete(deployer.destroy())

        mock_client.delete.assert_called_once_with("obelix-abc")
        mock_client.close.assert_called_once()
        assert deployer._destroyed is True

    def test_idempotent(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        deployer._client = mock_client
        deployer._sandbox_name = "obelix-abc"

        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result):
            asyncio.get_event_loop().run_until_complete(deployer.destroy())
            asyncio.get_event_loop().run_until_complete(deployer.destroy())

        assert mock_client.delete.call_count == 1

    def test_no_sandbox_created_skips_cleanup(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        deployer._client = None
        deployer._sandbox_name = None

        asyncio.get_event_loop().run_until_complete(deployer.destroy())
        assert deployer._destroyed is True

    def test_delete_failure_still_closes_client(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        mock_client.delete.side_effect = Exception("delete failed")
        deployer._client = mock_client
        deployer._sandbox_name = "obelix-abc"

        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result):
            asyncio.get_event_loop().run_until_complete(deployer.destroy())

        mock_client.close.assert_called_once()


class TestDeploy:
    """deploy() chains all steps and returns DeploymentInfo."""

    def test_happy_path_returns_deployment_info(self):
        from obelix.adapters.outbound.openshell.deployer import (
            DeploymentInfo,
            OpenShellDeployer,
        )

        deployer = OpenShellDeployer(_make_factory(), "test_agent", port=8002)
        mock_client = _make_mock_client()

        cli_result = MagicMock(returncode=0, stdout="", stderr="")
        with (
            _patch_sdk(mock_client),
            patch("shutil.which", return_value="/usr/bin/openshell"),
            patch("subprocess.run", return_value=cli_result),
        ):
            info = asyncio.get_event_loop().run_until_complete(deployer.deploy())

        assert isinstance(info, DeploymentInfo)
        assert info.endpoint == "http://localhost:8002"
        assert info.port == 8002
        assert info.sandbox_name.startswith("obelix-")

    def test_custom_endpoint(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(),
            "test_agent",
            port=8002,
            endpoint="https://prod.example.com",
        )
        mock_client = _make_mock_client()

        cli_result = MagicMock(returncode=0, stdout="", stderr="")
        with (
            _patch_sdk(mock_client),
            patch("shutil.which", return_value="/usr/bin/openshell"),
            patch("subprocess.run", return_value=cli_result),
        ):
            info = asyncio.get_event_loop().run_until_complete(deployer.deploy())

        assert info.endpoint == "https://prod.example.com"

    def test_sandbox_error_triggers_cleanup(self):
        """If _create_sandbox fails after validation, destroy is called."""
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()

        call_count = 0

        def subprocess_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "sandbox" in cmd and "create" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="disk full")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            _patch_sdk(mock_client),
            patch("shutil.which", return_value="/usr/bin/openshell"),
            patch("subprocess.run", side_effect=subprocess_side_effect),
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                asyncio.get_event_loop().run_until_complete(deployer.deploy())

        assert deployer._destroyed is True


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
