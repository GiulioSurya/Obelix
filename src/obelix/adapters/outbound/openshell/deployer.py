"""OpenShell Deployer — deploys Obelix agents inside OpenShell sandboxes as A2A servers.

Uses the ``openshell`` Python SDK for sandbox CRUD and exec, and the ``openshell``
CLI for provider management, policy, and port forwarding.

Requires: ``uv sync --extra serve --extra openshell`` (Linux/macOS only).
"""

from __future__ import annotations

import asyncio
import os
import shutil
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
