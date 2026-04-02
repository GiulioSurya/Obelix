# OpenShell Deployer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Obelix agents inside OpenShell sandboxes as A2A servers, transparent to clients.

**Architecture:** `OpenShellDeployer` orchestrates sandbox lifecycle via OpenShell SDK (sandbox CRUD, exec) and CLI (provider, policy, forwarding). `AgentFactory.a2a_openshell_deploy()` is a thin wrapper that creates the deployer, deploys, and blocks until SIGINT. `DeploymentInfo` is the frozen return value.

**Tech Stack:** Python 3.13, `openshell` SDK (v0.0.16), `asyncio`, `subprocess`, `shutil`, `tempfile`, Pydantic (for AgentFactory integration)

**Spec:** `docs/superpowers/specs/2026-03-29-openshell-deployer-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| **Create:** `src/obelix/adapters/outbound/openshell/__init__.py` | Re-export `OpenShellDeployer`, `DeploymentInfo` with guarded import |
| **Create:** `src/obelix/adapters/outbound/openshell/deployer.py` | `OpenShellDeployer` class — full deploy/destroy lifecycle |
| **Modify:** `src/obelix/core/agent/agent_factory.py` | Add `a2a_openshell_deploy()` method |
| **Create:** `tests/adapters/outbound/openshell/__init__.py` | Test package init |
| **Create:** `tests/adapters/outbound/openshell/test_deployer.py` | All deployer tests (mocked SDK + CLI) |

---

### Task 1: DeploymentInfo dataclass + module skeleton

**Files:**
- Create: `src/obelix/adapters/outbound/openshell/__init__.py`
- Create: `src/obelix/adapters/outbound/openshell/deployer.py`
- Create: `tests/adapters/outbound/openshell/__init__.py`
- Create: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/outbound/openshell/test_deployer.py
"""Tests for OpenShellDeployer.

All tests mock the openshell SDK and CLI — no real Gateway or sandbox required.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDeploymentInfo -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/obelix/adapters/outbound/openshell/__init__.py
try:
    from obelix.adapters.outbound.openshell.deployer import (
        DeploymentInfo,
        OpenShellDeployer,
    )
except ImportError:
    pass  # openshell extra not installed

__all__ = ["DeploymentInfo", "OpenShellDeployer"]
```

```python
# src/obelix/adapters/outbound/openshell/deployer.py
"""OpenShell Deployer — deploys Obelix agents inside OpenShell sandboxes as A2A servers.

Uses the ``openshell`` Python SDK for sandbox CRUD and exec, and the ``openshell``
CLI for provider management, policy, and port forwarding.

Requires: ``uv sync --extra serve --extra openshell`` (Linux/macOS only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentInfo:
    """Immutable result of a successful deployment."""

    sandbox_name: str
    endpoint: str  # "http://localhost:{port}"
    port: int
```

```python
# tests/adapters/outbound/openshell/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDeploymentInfo -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/__init__.py src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/__init__.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add DeploymentInfo dataclass and module skeleton"
```

---

### Task 2: OpenShellDeployer constructor + pre-deploy validation

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
import asyncio
import shutil
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

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
        with patch(
            "obelix.adapters.outbound.openshell.deployer.SandboxClient", None
        ):
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
        with _patch_sdk(mock_client), patch("shutil.which", return_value="/usr/bin/openshell"):
            with pytest.raises(RuntimeError, match="gateway"):
                asyncio.get_event_loop().run_until_complete(deployer._validate())

    def test_validation_passes(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        factory = _make_factory()
        deployer = OpenShellDeployer(factory, "test_agent")
        mock_client = _make_mock_client()
        with _patch_sdk(mock_client), patch("shutil.which", return_value="/usr/bin/openshell"):
            # Should not raise
            asyncio.get_event_loop().run_until_complete(deployer._validate())
            assert deployer._client is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestConstructor tests/adapters/outbound/openshell/test_deployer.py::TestValidation -v`
Expected: FAIL — OpenShellDeployer not defined

- [ ] **Step 3: Write minimal implementation**

Add to `src/obelix/adapters/outbound/openshell/deployer.py`:

```python
import asyncio
import os
import shutil

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

try:
    from openshell import SandboxClient, TlsConfig
except ImportError:
    SandboxClient = None
    TlsConfig = None


class OpenShellDeployer:
    """Deploys an Obelix agent inside an OpenShell sandbox as an A2A server."""

    def __init__(
        self,
        agent_factory,
        agent_name: str,
        *,
        port: int = 8002,
        policy: str | None = None,
        providers: list[str] | None = None,
        dockerfile: str | None = None,
        image: str | None = None,
        gateway: str | None = None,
        tls_cert_dir: str | None = None,
        endpoint: str | None = None,
        version: str = "0.1.0",
        description: str | None = None,
        provider_name: str = "Obelix",
        subagents: list[str] | None = None,
        subagent_config: dict | None = None,
    ):
        self._factory = agent_factory
        self._agent_name = agent_name
        self._port = port
        self._policy = policy
        self._providers = providers
        self._dockerfile = dockerfile
        self._image = image
        self._gateway = gateway or os.environ.get("OPENSHELL_GATEWAY")
        self._tls_cert_dir = tls_cert_dir or os.environ.get("OPENSHELL_TLS_CERT_DIR")
        self._endpoint = endpoint
        self._version = version
        self._description = description
        self._provider_name = provider_name
        self._subagents = subagents
        self._subagent_config = subagent_config

        self._client = None
        self._sandbox_name: str | None = None
        self._destroyed = False

    async def _validate(self) -> None:
        """Pre-deploy validation: SDK, CLI, gateway, param conflicts."""
        # 1. Param conflicts
        if self._dockerfile and self._image:
            raise ValueError("dockerfile and image are mutually exclusive")

        # 2. SDK available
        if SandboxClient is None:
            raise ImportError(
                "The 'openshell' package is required for OpenShellDeployer. "
                "Install it with: uv sync --extra openshell\n"
                "Note: openshell is only available on Linux and macOS."
            )

        # 3. CLI in PATH
        if not shutil.which("openshell"):
            raise RuntimeError(
                "The 'openshell' CLI is not in PATH. "
                "Install it from: https://github.com/NVIDIA/OpenShell"
            )

        # 4. Create client
        if self._gateway:
            kwargs: dict = {"endpoint": self._gateway}
            if self._tls_cert_dir:
                from pathlib import Path

                cert_dir = Path(self._tls_cert_dir)
                kwargs["tls"] = TlsConfig(
                    ca_path=cert_dir / "ca.crt",
                    cert_path=cert_dir / "tls.crt",
                    key_path=cert_dir / "tls.key",
                )
            self._client = await asyncio.to_thread(lambda: SandboxClient(**kwargs))
        else:
            self._client = await asyncio.to_thread(SandboxClient.from_active_cluster)

        # 5. Gateway reachable
        try:
            await asyncio.to_thread(self._client.health)
        except Exception as e:
            raise RuntimeError(
                f"OpenShell gateway unreachable: {e}\n"
                "Run: openshell gateway start"
            ) from e

        logger.info("[Deployer] Validation passed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestConstructor tests/adapters/outbound/openshell/test_deployer.py::TestValidation -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add OpenShellDeployer constructor and pre-deploy validation"
```

---

### Task 3: CLI subprocess helper (`_run_cli`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestRunCli -v`
Expected: FAIL — `_run_cli` not defined

- [ ] **Step 3: Write minimal implementation**

Add method to `OpenShellDeployer` in `deployer.py`:

```python
    async def _run_cli(
        self, args: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        """Run an openshell CLI command as async subprocess.

        Args:
            args: CLI arguments after 'openshell' (e.g. ["provider", "get", "anthropic"]).
            check: If True (default), raise RuntimeError on non-zero exit.
            timeout: Subprocess timeout in seconds.

        Returns:
            CompletedProcess result.
        """
        cmd = ["openshell", *args]
        logger.debug(f"[Deployer] CLI: {' '.join(cmd)}")

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if check and result.returncode != 0:
            raise RuntimeError(
                f"openshell CLI failed: {' '.join(cmd)}\n"
                f"exit={result.returncode} stderr={result.stderr.strip()}"
            )

        return result
```

Add `import subprocess` to the top of `deployer.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestRunCli -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add _run_cli async subprocess helper"
```

---

### Task 4: Provider management (`_ensure_providers`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
class TestEnsureProviders:
    """_ensure_providers registers LLM providers in the OpenShell gateway."""

    @pytest.fixture
    def deployer(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        return OpenShellDeployer(
            _make_factory(), "test_agent", providers=["anthropic"]
        )

    def test_provider_already_exists_skips_create(self, deployer):
        """If 'openshell provider get <name>' succeeds, skip create."""
        get_result = MagicMock(returncode=0, stdout="anthropic", stderr="")
        with patch("subprocess.run", return_value=get_result) as mock_run:
            asyncio.get_event_loop().run_until_complete(
                deployer._ensure_providers()
            )
        # Only 'get' called, no 'create'
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
            asyncio.get_event_loop().run_until_complete(
                deployer._ensure_providers()
            )
        assert call_count == 2

    def test_no_providers_is_noop(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent", providers=None)
        with patch("subprocess.run") as mock_run:
            asyncio.get_event_loop().run_until_complete(
                deployer._ensure_providers()
            )
        mock_run.assert_not_called()

    def test_create_failure_raises(self, deployer):
        """If create also fails, RuntimeError is raised."""
        fail = MagicMock(returncode=1, stdout="", stderr="auth error")

        with patch("subprocess.run", return_value=fail):
            with pytest.raises(RuntimeError, match="auth error"):
                asyncio.get_event_loop().run_until_complete(
                    deployer._ensure_providers()
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestEnsureProviders -v`
Expected: FAIL — `_ensure_providers` not defined

- [ ] **Step 3: Write minimal implementation**

Add method to `OpenShellDeployer`:

```python
    async def _ensure_providers(self) -> None:
        """Register LLM providers in the OpenShell gateway.

        For each provider name, checks if it exists (get). If not,
        creates it with --from-existing (auto-detect from local env).

        # TODO: replace with SDK when Provider CRUD is available
        """
        if not self._providers:
            return

        for name in self._providers:
            # Check if already registered
            result = await self._run_cli(
                ["provider", "get", name], check=False
            )
            if result.returncode == 0:
                logger.info(f"[Deployer] Provider '{name}' already registered")
                continue

            # Create from local environment
            logger.info(f"[Deployer] Creating provider '{name}' from local env")
            await self._run_cli([
                "provider", "create",
                "--name", name,
                "--type", "claude",
                "--from-existing",
            ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestEnsureProviders -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add provider management via CLI"
```

---

### Task 5: Dockerfile generation and image handling (`_build_image`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
import tempfile
from pathlib import Path


class TestBuildImage:
    """_build_image generates or uses a Dockerfile, returns the --from arg."""

    def test_prebuilt_image_returns_image_ref(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", image="registry.io/agent:v1"
        )
        result = asyncio.get_event_loop().run_until_complete(
            deployer._build_image()
        )
        assert result == "registry.io/agent:v1"

    def test_custom_dockerfile_returns_path(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", dockerfile="/app/Dockerfile"
        )
        result = asyncio.get_event_loop().run_until_complete(
            deployer._build_image()
        )
        assert result == "/app/Dockerfile"

    def test_auto_generate_creates_temp_dockerfile(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        result = asyncio.get_event_loop().run_until_complete(
            deployer._build_image()
        )
        # Returns path to a generated Dockerfile
        assert Path(result).name == "Dockerfile"
        content = Path(result).read_text()
        assert "python:3.13" in content
        assert "uv" in content
        assert "pyproject.toml" in content

    def test_auto_generate_includes_entrypoint(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", port=9000
        )
        result = asyncio.get_event_loop().run_until_complete(
            deployer._build_image()
        )
        content = Path(result).read_text()
        assert "test_agent" in content
        assert "9000" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestBuildImage -v`
Expected: FAIL — `_build_image` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def _build_image(self) -> str:
        """Prepare the container image source for sandbox creation.

        Returns:
            A string suitable for ``openshell sandbox create --from <value>``:
            - Pre-built image reference (e.g. "registry.io/agent:v1")
            - Path to a user-provided Dockerfile
            - Path to a generated Dockerfile (auto mode)
        """
        if self._image:
            logger.info(f"[Deployer] Using pre-built image: {self._image}")
            return self._image

        if self._dockerfile:
            logger.info(f"[Deployer] Using custom Dockerfile: {self._dockerfile}")
            return self._dockerfile

        # Auto-generate a Dockerfile
        logger.info("[Deployer] Generating Dockerfile")
        return self._generate_dockerfile()

    def _generate_dockerfile(self) -> str:
        """Generate a minimal Dockerfile for the agent.

        The Dockerfile copies the project source, installs dependencies via uv,
        and sets the entrypoint to start the A2A server inside the sandbox.
        """
        import tempfile

        entrypoint_script = (
            "from obelix.core.agent.agent_factory import AgentFactory\n"
            "# The actual agent class must be importable from the copied source.\n"
            "# This entrypoint is a template — the real import is injected by BYOC.\n"
        )

        dockerfile_content = (
            "FROM python:3.13-slim\n"
            "\n"
            "# Install uv\n"
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "# Copy project files\n"
            "COPY pyproject.toml uv.lock ./\n"
            "COPY src/ ./src/\n"
            "\n"
            "# Install dependencies\n"
            "RUN uv sync --no-dev --extra serve --extra openshell --link-mode=copy\n"
            "\n"
            f'EXPOSE {self._port}\n'
            "\n"
            "# Entrypoint: start A2A server for the agent\n"
            f'CMD ["uv", "run", "python", "-c", '
            f'"from obelix.core.agent.agent_factory import AgentFactory; '
            f"# agent={self._agent_name} port={self._port}"
            f'"]\n'
        )

        tmpdir = tempfile.mkdtemp(prefix="obelix-deployer-")
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        logger.info(f"[Deployer] Generated Dockerfile: {dockerfile_path}")
        return dockerfile_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestBuildImage -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add Dockerfile generation and image handling"
```

---

### Task 6: Sandbox creation (`_create_sandbox`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
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
            _make_factory(), "test_agent",
            providers=["anthropic"], policy="policy.yaml"
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
            # Should not raise, just log a warning
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestCreateSandbox -v`
Expected: FAIL — `_create_sandbox` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def _create_sandbox(self, image_source: str) -> None:
        """Create an OpenShell sandbox from an image or Dockerfile.

        Uses CLI for --from (BYOC not in SDK), then SDK wait_ready.

        # TODO: replace with SDK if SandboxSpec adds image/dockerfile support

        Args:
            image_source: Image ref, or path to Dockerfile (from _build_image).
        """
        import uuid

        self._sandbox_name = f"obelix-{uuid.uuid4().hex[:8]}"

        if not self._policy:
            logger.warning(
                "[Deployer] No policy specified — sandbox runs without "
                "security restrictions. Consider passing policy='path/to/policy.yaml'."
            )

        # Build CLI command
        cmd = [
            "sandbox", "create",
            "--name", self._sandbox_name,
            "--from", image_source,
        ]
        if self._providers:
            for p in self._providers:
                cmd.extend(["--provider", p])
        if self._policy:
            cmd.extend(["--policy", self._policy])

        await self._run_cli(cmd)

        # Wait for sandbox to be ready (SDK)
        logger.info(f"[Deployer] Waiting for sandbox '{self._sandbox_name}' to be ready")
        await asyncio.to_thread(
            self._client.wait_ready,
            self._sandbox_name,
            timeout_seconds=120,
        )
        logger.info(f"[Deployer] Sandbox ready: {self._sandbox_name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestCreateSandbox -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add sandbox creation via CLI + SDK wait_ready"
```

---

### Task 7: Server startup (`_start_server`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
class TestStartServer:
    """_start_server starts the A2A server inside the sandbox via SDK exec."""

    def test_exec_called_with_correct_command(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", port=8002
        )
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
        # The command should reference the agent name and port
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestStartServer -v`
Expected: FAIL — `_start_server` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def _start_server(self) -> None:
        """Start the A2A server inside the sandbox via SDK exec().

        Runs the server in background mode (nohup) so exec() returns
        immediately. The server binds to 0.0.0.0 inside the sandbox.
        """
        # Resolve sandbox ID from name
        ref = await asyncio.to_thread(
            self._client.get, sandbox_name=self._sandbox_name
        )

        # Build the command to start the A2A server inside the sandbox
        server_cmd = (
            f"nohup uv run python -c \""
            f"from obelix.core.agent.agent_factory import AgentFactory; "
            f"# Placeholder: real entrypoint is baked into the container image. "
            f"# agent={self._agent_name} port={self._port}"
            f"\" > /tmp/a2a-server.log 2>&1 &"
        )

        logger.info(
            f"[Deployer] Starting A2A server | sandbox={self._sandbox_name} "
            f"port={self._port}"
        )

        try:
            await asyncio.to_thread(
                self._client.exec,
                sandbox_id=ref.id,
                command=["bash", "-c", server_cmd],
                timeout_seconds=30,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to start A2A server in sandbox '{self._sandbox_name}': {e}"
            ) from e

        logger.info(f"[Deployer] A2A server started in sandbox '{self._sandbox_name}'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestStartServer -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add server startup via SDK exec"
```

---

### Task 8: Port forwarding (`_start_forward`, `_stop_forward`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
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
            asyncio.get_event_loop().run_until_complete(
                deployer._start_forward()
            )
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "openshell", "forward", "start",
            "8002", "obelix-abc", "-d",
        ]

    def test_stop_forward(self, deployer):
        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result) as mock_run:
            asyncio.get_event_loop().run_until_complete(
                deployer._stop_forward()
            )
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "openshell", "forward", "stop",
            "8002", "obelix-abc",
        ]

    def test_stop_forward_failure_does_not_raise(self, deployer):
        """Cleanup is best-effort — stop_forward should not raise."""
        fail = MagicMock(returncode=1, stdout="", stderr="no forward found")
        with patch("subprocess.run", return_value=fail):
            # Should not raise
            asyncio.get_event_loop().run_until_complete(
                deployer._stop_forward()
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestPortForwarding -v`
Expected: FAIL — `_start_forward` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def _start_forward(self) -> None:
        """Start port forwarding from localhost to sandbox.

        # TODO: replace with SDK when port forwarding is available
        """
        await self._run_cli([
            "forward", "start",
            str(self._port), self._sandbox_name, "-d",
        ])
        logger.info(
            f"[Deployer] Port forward started | "
            f"localhost:{self._port} -> {self._sandbox_name}"
        )

    async def _stop_forward(self) -> None:
        """Stop port forwarding. Best-effort — does not raise on failure.

        # TODO: replace with SDK when port forwarding is available
        """
        try:
            await self._run_cli([
                "forward", "stop",
                str(self._port), self._sandbox_name,
            ], check=False)
            logger.info(f"[Deployer] Port forward stopped | port={self._port}")
        except Exception as e:
            logger.warning(f"[Deployer] Failed to stop forward: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestPortForwarding -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add port forwarding start/stop via CLI"
```

---

### Task 9: Destroy and cleanup (`destroy`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
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

        # delete called only once
        assert mock_client.delete.call_count == 1

    def test_no_sandbox_created_skips_cleanup(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        deployer._client = None
        deployer._sandbox_name = None

        # Should not raise
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

        # Client still closed despite delete failure
        mock_client.close.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDestroy -v`
Expected: FAIL — `destroy` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def destroy(self) -> None:
        """Stop forward, delete sandbox, close client. Idempotent."""
        if self._destroyed:
            return
        self._destroyed = True

        # 1. Stop port forwarding (best-effort)
        if self._sandbox_name:
            await self._stop_forward()

        # 2. Delete sandbox (SDK)
        if self._sandbox_name and self._client:
            try:
                await asyncio.to_thread(self._client.delete, self._sandbox_name)
                logger.info(f"[Deployer] Sandbox deleted: {self._sandbox_name}")
            except Exception as e:
                logger.warning(
                    f"[Deployer] Failed to delete sandbox "
                    f"'{self._sandbox_name}': {e}"
                )

        # 3. Close SDK client
        if self._client:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception as e:
                logger.warning(f"[Deployer] Failed to close client: {e}")

        logger.info("[Deployer] Cleanup complete")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDestroy -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add destroy with idempotent cleanup"
```

---

### Task 10: Full deploy orchestration (`deploy`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
class TestDeploy:
    """deploy() chains all steps and returns DeploymentInfo."""

    def test_happy_path_returns_deployment_info(self):
        from obelix.adapters.outbound.openshell.deployer import (
            DeploymentInfo,
            OpenShellDeployer,
        )

        deployer = OpenShellDeployer(
            _make_factory(), "test_agent", port=8002
        )
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
            _make_factory(), "test_agent",
            port=8002, endpoint="https://prod.example.com"
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

        # Cleanup should have been attempted
        assert deployer._destroyed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDeploy -v`
Expected: FAIL — `deploy` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def deploy(self) -> DeploymentInfo:
        """Full deploy: validate → providers → image → sandbox → server → forward.

        On failure after sandbox creation, calls destroy() before re-raising.
        """
        await self._validate()

        try:
            await self._ensure_providers()
            image_source = await self._build_image()
            await self._create_sandbox(image_source)
            await self._start_server()
            await self._start_forward()
        except Exception:
            await self.destroy()
            raise

        endpoint = self._endpoint or f"http://localhost:{self._port}"

        info = DeploymentInfo(
            sandbox_name=self._sandbox_name,
            endpoint=endpoint,
            port=self._port,
        )
        logger.info(
            f"[Deployer] Deploy complete | sandbox={info.sandbox_name} "
            f"endpoint={info.endpoint}"
        )
        return info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestDeploy -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add full deploy orchestration with error cleanup"
```

---

### Task 11: Context manager (`__aenter__` / `__aexit__`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
class TestContextManager:
    """OpenShellDeployer works as async context manager."""

    def test_enter_returns_deployment_info(self):
        from obelix.adapters.outbound.openshell.deployer import (
            DeploymentInfo,
            OpenShellDeployer,
        )

        deployer = OpenShellDeployer(_make_factory(), "test_agent", port=8002)
        mock_client = _make_mock_client()
        cli_result = MagicMock(returncode=0, stdout="", stderr="")

        async def run():
            with (
                _patch_sdk(mock_client),
                patch("shutil.which", return_value="/usr/bin/openshell"),
                patch("subprocess.run", return_value=cli_result),
            ):
                async with deployer as info:
                    assert isinstance(info, DeploymentInfo)
                    assert info.port == 8002

        asyncio.get_event_loop().run_until_complete(run())

    def test_exit_calls_destroy(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        cli_result = MagicMock(returncode=0, stdout="", stderr="")

        async def run():
            with (
                _patch_sdk(mock_client),
                patch("shutil.which", return_value="/usr/bin/openshell"),
                patch("subprocess.run", return_value=cli_result),
            ):
                async with deployer:
                    pass
            # After exiting context, deployer is destroyed
            assert deployer._destroyed is True

        asyncio.get_event_loop().run_until_complete(run())

    def test_exit_on_exception_still_destroys(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        mock_client = _make_mock_client()
        cli_result = MagicMock(returncode=0, stdout="", stderr="")

        async def run():
            with (
                _patch_sdk(mock_client),
                patch("shutil.which", return_value="/usr/bin/openshell"),
                patch("subprocess.run", return_value=cli_result),
            ):
                with pytest.raises(ValueError, match="boom"):
                    async with deployer:
                        raise ValueError("boom")
            assert deployer._destroyed is True

        asyncio.get_event_loop().run_until_complete(run())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestContextManager -v`
Expected: FAIL — `__aenter__` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def __aenter__(self) -> DeploymentInfo:
        return await self.deploy()

    async def __aexit__(self, *exc) -> None:
        await self.destroy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestContextManager -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add async context manager support"
```

---

### Task 12: Policy hot-reload (`update_policy`)

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/deployer.py`
- Modify: `tests/adapters/outbound/openshell/test_deployer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/adapters/outbound/openshell/test_deployer.py`:

```python
class TestUpdatePolicy:
    """update_policy() hot-reloads network_policies via CLI."""

    @pytest.fixture
    def deployer(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        d = OpenShellDeployer(_make_factory(), "test_agent")
        d._sandbox_name = "obelix-abc"
        return d

    def test_success_returns_true(self, deployer):
        result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=result) as mock_run:
            ok = asyncio.get_event_loop().run_until_complete(
                deployer.update_policy("new-policy.yaml")
            )
        assert ok is True
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "openshell", "policy", "set", "obelix-abc",
            "--policy", "new-policy.yaml", "--wait",
        ]

    def test_failure_returns_false(self, deployer):
        result = MagicMock(returncode=1, stdout="", stderr="invalid policy")
        with patch("subprocess.run", return_value=result):
            ok = asyncio.get_event_loop().run_until_complete(
                deployer.update_policy("bad-policy.yaml")
            )
        assert ok is False

    def test_no_sandbox_returns_false(self):
        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(_make_factory(), "test_agent")
        deployer._sandbox_name = None
        ok = asyncio.get_event_loop().run_until_complete(
            deployer.update_policy("policy.yaml")
        )
        assert ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestUpdatePolicy -v`
Expected: FAIL — `update_policy` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `OpenShellDeployer`:

```python
    async def update_policy(self, policy_path: str) -> bool:
        """Hot-reload network_policies via CLI. Returns True on success.

        # TODO: replace with SDK when openshell Python SDK exposes policy set
        """
        if not self._sandbox_name:
            logger.warning("[Deployer] Cannot update policy — no sandbox deployed")
            return False

        result = await self._run_cli(
            ["policy", "set", self._sandbox_name,
             "--policy", policy_path, "--wait"],
            check=False,
        )

        if result.returncode == 0:
            logger.info(
                f"[Deployer] Policy updated | sandbox={self._sandbox_name} "
                f"policy={policy_path}"
            )
            return True
        else:
            logger.warning(
                f"[Deployer] Policy update failed | "
                f"stderr={result.stderr.strip()}"
            )
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/outbound/openshell/test_deployer.py::TestUpdatePolicy -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/openshell/deployer.py tests/adapters/outbound/openshell/test_deployer.py
git commit -m "feat(openshell): add policy hot-reload via CLI"
```

---

### Task 13: AgentFactory.a2a_openshell_deploy()

**Files:**
- Modify: `src/obelix/core/agent/agent_factory.py`
- Modify: `tests/core/agent/test_agent_factory.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/agent/test_agent_factory.py`:

```python
class TestA2AOpenShellDeploy:
    """a2a_openshell_deploy() creates deployer and blocks until interrupted."""

    def test_creates_deployer_and_calls_deploy(self):
        """Verify the method creates an OpenShellDeployer and calls deploy()."""
        from unittest.mock import AsyncMock, patch, MagicMock
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
        from unittest.mock import AsyncMock, patch, MagicMock
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/agent/test_agent_factory.py::TestA2AOpenShellDeploy -v`
Expected: FAIL — `a2a_openshell_deploy` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `AgentFactory` in `agent_factory.py`, after the `a2a_serve` section:

```python
    # ─── A2A OpenShell Deploy ─────────────────────────────────────────────────

    def a2a_openshell_deploy(
        self,
        agent: str,
        *,
        port: int = 8002,
        policy: str | None = None,
        providers: list[str] | None = None,
        dockerfile: str | None = None,
        image: str | None = None,
        gateway: str | None = None,
        tls_cert_dir: str | None = None,
        endpoint: str | None = None,
        version: str = "0.1.0",
        description: str | None = None,
        provider_name: str = "Obelix",
        subagents: list[str] | None = None,
        subagent_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Deploy agent in OpenShell sandbox and block until interrupted.

        Like a2a_serve() but runs inside an OpenShell sandbox.
        On Ctrl+C / SIGTERM, destroys the sandbox and exits.
        """
        import asyncio

        from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

        deployer = OpenShellDeployer(
            self,
            agent,
            port=port,
            policy=policy,
            providers=providers,
            dockerfile=dockerfile,
            image=image,
            gateway=gateway,
            tls_cert_dir=tls_cert_dir,
            endpoint=endpoint,
            version=version,
            description=description,
            provider_name=provider_name,
            subagents=subagents,
            subagent_config=subagent_config,
        )

        async def _run() -> None:
            try:
                info = await deployer.deploy()
                logger.info(
                    f"AgentFactory: agent deployed in sandbox | "
                    f"endpoint={info.endpoint} sandbox={info.sandbox_name}"
                )
                logger.info("AgentFactory: press Ctrl+C to stop and destroy sandbox")
                # Block forever until interrupted
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                logger.info("AgentFactory: interrupted, destroying sandbox...")
            finally:
                await deployer.destroy()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/agent/test_agent_factory.py::TestA2AOpenShellDeploy -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/agent/agent_factory.py tests/core/agent/test_agent_factory.py
git commit -m "feat(openshell): add AgentFactory.a2a_openshell_deploy()"
```

---

### Task 14: Run full test suite + lint

**Files:**
- All files from tasks 1-13

- [ ] **Step 1: Run all deployer tests**

Run: `uv run pytest tests/adapters/outbound/openshell/ -v`
Expected: all tests pass

- [ ] **Step 2: Run agent_factory tests**

Run: `uv run pytest tests/core/agent/test_agent_factory.py -v`
Expected: all tests pass (including existing + new)

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: no regressions

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check src/obelix/adapters/outbound/openshell/ src/obelix/core/agent/agent_factory.py tests/adapters/outbound/openshell/ --fix && uv run ruff format src/obelix/adapters/outbound/openshell/ src/obelix/core/agent/agent_factory.py tests/adapters/outbound/openshell/`
Expected: no errors

- [ ] **Step 5: Final commit if lint changed anything**

```bash
git add -u
git commit -m "style: format and lint openshell deployer"
```

---

### Task 15: Update `__init__.py` export in `deployer.py`

**Files:**
- Modify: `src/obelix/adapters/outbound/openshell/__init__.py`

- [ ] **Step 1: Verify exports work**

Run: `uv run python -c "from obelix.adapters.outbound.openshell import OpenShellDeployer, DeploymentInfo; print('OK')"`
Expected: `OK` (on Linux/macOS with openshell installed) or ImportError silently caught (on Windows)

- [ ] **Step 2: Commit if any changes needed**

```bash
git add src/obelix/adapters/outbound/openshell/__init__.py
git commit -m "chore: verify openshell package exports"
```