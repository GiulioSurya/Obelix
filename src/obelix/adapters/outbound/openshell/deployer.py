"""OpenShell Deployer — deploys Obelix agents inside OpenShell sandboxes as A2A servers.

Uses the ``openshell`` Python SDK for sandbox CRUD and exec, and the ``openshell``
CLI for provider management, policy, and port forwarding.

Requires: ``uv sync --extra serve --extra openshell`` (Linux/macOS only).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

try:
    from openshell import SandboxClient, TlsConfig
except ImportError:
    SandboxClient = None
    TlsConfig = None


@dataclass(frozen=True)
class DeploymentInfo:
    """Immutable result of a successful deployment."""

    sandbox_name: str
    endpoint: str  # "http://localhost:{port}"
    port: int


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
                f"OpenShell gateway unreachable: {e}\nRun: openshell gateway start"
            ) from e

        logger.info("[Deployer] Validation passed")

    async def _run_cli(
        self, args: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        """Run an openshell CLI command as async subprocess."""
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
            result = await self._run_cli(["provider", "get", name], check=False)
            if result.returncode == 0:
                logger.info(f"[Deployer] Provider '{name}' already registered")
                continue

            # Create from local environment
            logger.info(f"[Deployer] Creating provider '{name}' from local env")
            await self._run_cli(
                [
                    "provider",
                    "create",
                    "--name",
                    name,
                    "--type",
                    "claude",
                    "--from-existing",
                ]
            )

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
        """Generate a minimal Dockerfile for the agent."""
        import tempfile

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
            f"EXPOSE {self._port}\n"
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

    async def _create_sandbox(self, image_source: str) -> None:
        """Create an OpenShell sandbox from an image or Dockerfile.

        Uses CLI for --from (BYOC not in SDK), then SDK wait_ready.

        # TODO: replace with SDK if SandboxSpec adds image/dockerfile support
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
            "sandbox",
            "create",
            "--name",
            self._sandbox_name,
            "--from",
            image_source,
        ]
        if self._providers:
            for p in self._providers:
                cmd.extend(["--provider", p])
        if self._policy:
            cmd.extend(["--policy", self._policy])

        await self._run_cli(cmd)

        # Wait for sandbox to be ready (SDK)
        logger.info(
            f"[Deployer] Waiting for sandbox '{self._sandbox_name}' to be ready"
        )
        await asyncio.to_thread(
            self._client.wait_ready,
            self._sandbox_name,
            timeout_seconds=120,
        )
        logger.info(f"[Deployer] Sandbox ready: {self._sandbox_name}")

    async def _start_server(self) -> None:
        """Start the A2A server inside the sandbox via SDK exec()."""
        ref = await asyncio.to_thread(self._client.get, sandbox_name=self._sandbox_name)

        server_cmd = (
            f'nohup uv run python -c "'
            f"from obelix.core.agent.agent_factory import AgentFactory; "
            f"# Placeholder: real entrypoint is baked into the container image. "
            f"# agent={self._agent_name} port={self._port}"
            f'" > /tmp/a2a-server.log 2>&1 &'
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

    async def _start_forward(self) -> None:
        """Start port forwarding from localhost to sandbox.

        # TODO: replace with SDK when port forwarding is available
        """
        await self._run_cli(
            [
                "forward",
                "start",
                str(self._port),
                self._sandbox_name,
                "-d",
            ]
        )
        logger.info(
            f"[Deployer] Port forward started | "
            f"localhost:{self._port} -> {self._sandbox_name}"
        )

    async def _stop_forward(self) -> None:
        """Stop port forwarding. Best-effort — does not raise on failure.

        # TODO: replace with SDK when port forwarding is available
        """
        try:
            await self._run_cli(
                [
                    "forward",
                    "stop",
                    str(self._port),
                    self._sandbox_name,
                ],
                check=False,
            )
            logger.info(f"[Deployer] Port forward stopped | port={self._port}")
        except Exception as e:
            logger.warning(f"[Deployer] Failed to stop forward: {e}")
