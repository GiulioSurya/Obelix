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
from pathlib import Path

from obelix.infrastructure.logging import get_logger
from obelix.infrastructure.providers import Providers

logger = get_logger(__name__)

try:
    from openshell import SandboxClient, TlsConfig
except ImportError:
    SandboxClient = None
    TlsConfig = None

# Mapping from provider name to pyproject.toml optional-dependency group.
# Providers in base dependencies (anthropic, openai) map to None.
_PROVIDER_EXTRAS: dict[str, str | None] = {
    Providers.ANTHROPIC.value: None,  # base dependency
    Providers.OPENAI.value: None,  # base dependency
    Providers.OCI_GENERATIVE_AI.value: "oci",
    Providers.IBM_WATSON.value: "ibm",
    Providers.OLLAMA.value: "ollama",
    Providers.VLLM.value: "vllm",
    Providers.LITELLM.value: "litellm",
}


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
        extras: list[str] | None = None,
        dockerfile: str | None = None,
        image: str | None = None,
        gateway: str | None = None,
        tls_cert_dir: str | None = None,
        endpoint: str | None = None,
        entrypoint: str | None = None,
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
        self._extras = extras
        self._dockerfile = dockerfile
        self._image = image
        self._gateway = gateway or os.environ.get("OPENSHELL_GATEWAY")
        self._tls_cert_dir = tls_cert_dir or os.environ.get("OPENSHELL_TLS_CERT_DIR")
        self._endpoint = endpoint
        self._entrypoint = entrypoint
        self._version = version
        self._description = description
        self._provider_name = provider_name
        self._subagents = subagents
        self._subagent_config = subagent_config

        self._client = None
        self._sandbox_name: str | None = None
        self._generated_dockerfile: Path | None = None
        self._forward_proc: subprocess.Popen | None = None
        self._destroyed = False

    async def _validate(self) -> None:
        """Pre-deploy validation: SDK, CLI, gateway, param conflicts."""
        # 1. Param conflicts
        if self._dockerfile and self._image:
            raise ValueError("dockerfile and image are mutually exclusive")

        if not self._image and not self._dockerfile and not self._entrypoint:
            raise ValueError(
                "entrypoint is required when using auto-generated Dockerfile. "
                "Pass the Python module that sets up the factory and calls a2a_serve()."
            )

        # 2. Validate provider names
        if self._providers:
            valid = set(_PROVIDER_EXTRAS)
            invalid = [p for p in self._providers if p not in valid]
            if invalid:
                raise ValueError(
                    f"Unknown provider(s): {invalid}. Valid providers: {sorted(valid)}"
                )

        # 3. SDK available
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
        """Verify that required providers exist in the OpenShell gateway.

        Fails fast if any provider is missing — providers must be created
        beforehand via the CLI (e.g. ``openshell provider create``).

        # TODO: replace with SDK when Provider CRUD is available
        """
        if not self._providers:
            return

        for name in self._providers:
            result = await self._run_cli(["provider", "get", name], check=False)
            if result.returncode == 0:
                logger.info(f"[Deployer] Provider '{name}' found in gateway")
                continue

            raise RuntimeError(
                f"Provider '{name}' not found in the OpenShell gateway.\n"
                f"Create it first:\n"
                f"  openshell provider create --name {name} --type {name} --from-existing\n"
                f"\n"
                f"Or with explicit credentials:\n"
                f"  openshell provider create --name {name} --type {name} "
                f"--credential API_KEY=sk-..."
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

    def _build_extras_flags(self) -> str:
        """Build ``--extra X`` flags for ``uv sync``.

        Always includes ``serve`` and ``openshell``.  If ``extras`` was
        provided explicitly, those are added as-is.  Otherwise, extras
        are derived from ``providers`` via ``_PROVIDER_EXTRAS``.
        """
        result = {"serve", "openshell"}
        if self._extras:
            result.update(self._extras)
        else:
            for p in self._providers or []:
                extra = _PROVIDER_EXTRAS.get(p)
                if extra is not None:
                    result.add(extra)
        return " ".join(f"--extra {e}" for e in sorted(result))

    def _generate_dockerfile(self) -> str:
        """Generate a minimal Dockerfile in the project root.

        The Dockerfile is placed in cwd so that ``openshell sandbox create
        --from <path>`` uses the project directory as Docker build context,
        giving access to all project files (pyproject.toml, src/, etc.).
        The file is cleaned up in :meth:`destroy`.
        """
        dockerfile_content = (
            "FROM python:3.13-slim\n"
            "\n"
            "# Install uv\n"
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv\n"
            "\n"
            "# OpenShell sandbox requirements:\n"
            "# - iproute2: supervisor creates network namespaces via `ip`\n"
            "# - sandbox user: policy run_as_user needs this user to exist\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "RUN useradd -m sandbox\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "# Copy project files\n"
            "COPY . .\n"
            "\n"
            "# Install dependencies\n"
            f"RUN uv sync --no-dev {self._build_extras_flags()} --link-mode=copy\n"
            "\n"
            f"EXPOSE {self._port}\n"
            "\n"
            "# Run with the pre-built venv directly — no uv at runtime.\n"
            "# The sandbox may not have network access to PyPI.\n"
            f'CMD ["/app/.venv/bin/python", "-m", "{self._entrypoint}"]\n'
        )

        project_root = Path.cwd()
        dockerfile_path = project_root / ".Dockerfile.openshell"
        dockerfile_path.write_text(dockerfile_content)
        self._generated_dockerfile = dockerfile_path

        # Ensure .dockerignore excludes heavy/unnecessary dirs
        dockerignore_path = project_root / ".dockerignore"
        if not dockerignore_path.exists():
            dockerignore_path.write_text(
                ".venv\n.git\n__pycache__\n*.pyc\n.pytest_cache\n.ruff_cache\n"
            )
            self._generated_dockerignore = dockerignore_path
        else:
            self._generated_dockerignore = None

        logger.info(f"[Deployer] Generated Dockerfile: {dockerfile_path}")
        return str(dockerfile_path)

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

        # Without "-- true", the CLI opens an interactive SSH shell that
        # blocks forever under subprocess.run(capture_output=True).
        cmd.extend(["--", "true"])

        await self._run_cli(cmd, timeout=600)

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
            f"nohup /app/.venv/bin/python -m {self._entrypoint} "
            f"> /tmp/a2a-server.log 2>&1 &"
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

        Uses Popen instead of _run_cli because the forward process is
        long-running.  The ``-d`` flag makes the CLI daemonize, but the
        parent may still block when stdout/stderr pipes are inherited by
        the child.  Using DEVNULL avoids this entirely.

        # TODO: replace with SDK when port forwarding is available
        """
        cmd = [
            "openshell",
            "forward",
            "start",
            "-d",
            str(self._port),
            self._sandbox_name,
        ]
        logger.debug(f"[Deployer] CLI: {' '.join(cmd)}")

        self._forward_proc = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

        # Give the forward a moment to establish
        await asyncio.sleep(2)

        if self._forward_proc.poll() not in (None, 0):
            stderr = (
                self._forward_proc.stderr.read().decode()
                if self._forward_proc.stderr
                else ""
            )
            raise RuntimeError(
                f"Port forward exited with code {self._forward_proc.returncode}: "
                f"{stderr}"
            )

        logger.info(
            f"[Deployer] Port forward started | "
            f"localhost:{self._port} -> {self._sandbox_name}"
        )

    async def _stop_forward(self) -> None:
        """Stop port forwarding via CLI. Best-effort — does not raise on failure.

        Uses ``openshell forward stop`` instead of terminating the Popen
        process directly, because ``-d`` spawns a separate daemon whose PID
        is tracked by the CLI, not by our Popen handle.

        # TODO: replace with SDK when port forwarding is available
        """
        if self._sandbox_name:
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
        self._forward_proc = None

    async def deploy(self) -> DeploymentInfo:
        """Full deploy: validate -> providers -> image -> sandbox -> server -> forward.

        On failure after sandbox creation, calls destroy() before re-raising.
        """
        await self._validate()
        await self._ensure_providers()
        image_source = await self._build_image()

        # From here on, cleanup on failure (sandbox exists or is being created)
        try:
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

    async def __aenter__(self) -> DeploymentInfo:
        return await self.deploy()

    async def __aexit__(self, *exc) -> None:
        await self.destroy()

    async def update_policy(self, policy_path: str) -> bool:
        """Hot-reload network_policies via CLI. Returns True on success.

        # TODO: replace with SDK when openshell Python SDK exposes policy set
        """
        if not self._sandbox_name:
            logger.warning("[Deployer] Cannot update policy — no sandbox deployed")
            return False

        result = await self._run_cli(
            ["policy", "set", self._sandbox_name, "--policy", policy_path, "--wait"],
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
                f"[Deployer] Policy update failed | stderr={result.stderr.strip()}"
            )
            return False

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
                await asyncio.to_thread(self._client.wait_deleted, self._sandbox_name)
                logger.info(f"[Deployer] Sandbox deleted: {self._sandbox_name}")
            except Exception as e:
                logger.warning(
                    f"[Deployer] Failed to delete sandbox '{self._sandbox_name}': {e}"
                )

        # 3. Close SDK client
        if self._client:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception as e:
                logger.warning(f"[Deployer] Failed to close client: {e}")

        # 4. Clean up generated files
        for attr in ("_generated_dockerfile", "_generated_dockerignore"):
            f = getattr(self, attr, None)
            if f:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

        logger.info("[Deployer] Cleanup complete")
