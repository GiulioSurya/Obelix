# src/obelix/adapters/outbound/shell/openshell_executor.py
"""OpenShell sandbox executor — runs commands inside an NVIDIA OpenShell sandbox.

Uses the ``openshell`` Python SDK to communicate with the Gateway (gRPC).
The sandbox provides kernel-level policy enforcement (filesystem, network,
process controls) without any changes to the command being executed.

Requires: ``uv sync --extra openshell`` (Linux/macOS only, no Windows wheel).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid

from obelix.infrastructure.logging import get_logger
from obelix.ports.outbound.shell_executor import AbstractShellExecutor

logger = get_logger(__name__)

try:
    from openshell import SandboxClient, TlsConfig
except ImportError:
    SandboxClient = None
    TlsConfig = None

_MAX_OUTPUT_CHARS = 50_000
_TRUNCATION_MSG = "\n\n... [output truncated at {limit} chars — {total} total]"


def _truncate(text: str) -> str:
    """Truncate output that would bloat the context window."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + _TRUNCATION_MSG.format(
        limit=_MAX_OUTPUT_CHARS, total=len(text)
    )


async def _run_policy_set(sandbox_name: str, policy_path: str) -> bool:
    """Apply a policy YAML to a sandbox via the openshell CLI.

    Returns True on success, False on failure. Never raises.
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "openshell",
                "policy",
                "set",
                sandbox_name,
                "--policy",
                policy_path,
                "--wait",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info(
                f"[OpenShell] Policy applied | sandbox={sandbox_name} "
                f"policy={policy_path}"
            )
            return True
        else:
            logger.warning(
                f"[OpenShell] Policy apply failed | sandbox={sandbox_name} "
                f"returncode={result.returncode} stderr={result.stderr.strip()}"
            )
            return False
    except Exception as e:
        logger.warning(
            f"[OpenShell] Policy apply error | sandbox={sandbox_name} error={e}"
        )
        return False


class OpenShellExecutor(AbstractShellExecutor):
    """Executes shell commands inside an NVIDIA OpenShell sandbox.

    Two modes of operation:

    **Pre-existing sandbox** — connect to a sandbox already created via CLI::

        executor = OpenShellExecutor(sandbox_name="my-sandbox")

    **Auto-created sandbox** — creates a temporary sandbox on first use,
    destroys it on ``close()``::

        async with OpenShellExecutor() as executor:
            result = await executor.execute("ls -la")
        # sandbox destroyed automatically

    The sandbox is initialized lazily on the first ``execute()`` call.

    Policy is not passed via the SDK — it is either baked into the sandbox
    container image (``/etc/openshell/policy.yaml``) or applied after creation
    via ``openshell policy set <sandbox> --policy <file>``.

    Args:
        sandbox_name: Name of an existing sandbox. If ``None``, a temporary
            sandbox is auto-created on first use.
        gateway: Gateway endpoint (``host:port``). If ``None``, uses the
            active cluster from ``~/.config/openshell/``.
        tls_cert_dir: Path to directory containing ``ca.crt``, ``tls.crt``,
            ``tls.key`` for mTLS. Required when ``gateway`` is set and
            the endpoint differs from the locally registered gateway.
    """

    def __init__(
        self,
        policy: str | None = None,
        sandbox_name: str | None = None,
        gateway: str | None = None,
        tls_cert_dir: str | None = None,
    ):
        if SandboxClient is None:
            raise ImportError(
                "The 'openshell' package is required for OpenShellExecutor. "
                "Install it with: uv sync --extra openshell\n"
                "Note: openshell is only available on Linux and macOS."
            )

        self._policy_path = policy
        self._sandbox_name = sandbox_name
        self._gateway = gateway or os.environ.get("OPENSHELL_GATEWAY")
        self._tls_cert_dir = tls_cert_dir or os.environ.get("OPENSHELL_TLS_CERT_DIR")

        self._client: SandboxClient | None = None
        self._sandbox_id: str | None = None
        self._auto_created: bool = False
        self._initialized: bool = False
        self._closed: bool = False
        self._shell_info: dict = {}
        self._policy_watcher_task: asyncio.Task | None = None

    async def _ensure_sandbox(self) -> None:
        """Lazy initialization: create client and connect/create sandbox.

        Called automatically on the first ``execute()``. Subsequent calls
        are no-ops. All SDK calls are wrapped in ``asyncio.to_thread()``
        because the openshell SDK is synchronous (gRPC).
        """
        if self._initialized:
            return

        # 1. Create client
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

        # 2. Connect to existing sandbox or create a new one
        if self._sandbox_name:
            ref = await asyncio.to_thread(
                self._client.get, sandbox_name=self._sandbox_name
            )
            self._sandbox_id = ref.id
            logger.info(
                f"[OpenShell] Connected to sandbox | "
                f"name={self._sandbox_name} id={self._sandbox_id}"
            )
        else:
            self._sandbox_name = f"obelix-{uuid.uuid4().hex[:8]}"
            ref = await asyncio.to_thread(self._client.create)
            self._sandbox_id = ref.id
            self._sandbox_name = ref.name
            await asyncio.to_thread(
                self._client.wait_ready,
                self._sandbox_name,
                timeout_seconds=120,
            )
            self._auto_created = True
            logger.info(
                f"[OpenShell] Auto-created sandbox | "
                f"name={self._sandbox_name} id={self._sandbox_id}"
            )

        # 3. Probe sandbox environment
        await self._probe()

        # 4. Apply policy if provided
        if self._policy_path:
            await _run_policy_set(self._sandbox_name, self._policy_path)

        self._initialized = True

    async def _probe(self) -> None:
        """Probe the sandbox to discover its environment.

        Runs a one-shot command to get cwd, home directory, and OS info.
        Populates ``_shell_info`` for ``system_prompt_fragment()``.
        """
        try:
            result = await asyncio.to_thread(
                self._client.exec,
                sandbox_id=self._sandbox_id,
                command=["bash", "-c", 'echo "CWD=$(pwd)" && echo "HOME=$HOME"'],
                timeout_seconds=10,
            )
            cwd = None
            home = None
            for line in result.stdout.splitlines():
                if line.startswith("CWD="):
                    cwd = line[4:]
                elif line.startswith("HOME="):
                    home = line[5:]

            self._shell_info = {
                "platform": "Linux",
                "shell": "/bin/bash",
                "shell_name": "bash",
            }
            if cwd:
                self._shell_info["cwd"] = cwd
            if home:
                self._shell_info["home"] = home
            self._shell_info["sandbox"] = self._sandbox_name

            logger.info(f"[OpenShell] Probe complete | cwd={cwd} home={home}")
        except Exception as e:
            logger.warning(f"[OpenShell] Probe failed ({e}) — shell_info will be basic")
            self._shell_info = {
                "platform": "Linux",
                "shell_name": "bash",
                "sandbox": self._sandbox_name,
            }

    @property
    def shell_info(self) -> dict:
        """Platform and shell details for system message injection.

        Before the first ``execute()``, returns static defaults (Linux/bash).
        After initialization, includes cwd, home, and sandbox name from probe.
        """
        if self._shell_info:
            return self._shell_info
        return {"platform": "Linux", "shell_name": "bash"}

    @property
    def is_remote(self) -> bool:
        """Always ``False`` — commands execute in the sandbox, not deferred."""
        return False

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        working_directory: str | None = None,
    ) -> dict:
        """Execute a shell command inside the OpenShell sandbox.

        Returns dict with stdout, stderr, exit_code.
        On error (gateway unreachable, sandbox not found), returns exit_code=-1.
        """
        if self._closed:
            return {
                "stdout": "",
                "stderr": "Executor is closed.",
                "exit_code": -1,
            }

        try:
            await self._ensure_sandbox()
        except Exception as e:
            logger.error(f"[OpenShell] Sandbox init failed | error={e}")
            return {
                "stdout": "",
                "stderr": f"OpenShell initialization error: {e}",
                "exit_code": -1,
            }

        # Prepend cd if working_directory is specified
        full_command = command
        if working_directory:
            full_command = f"cd {working_directory} && {command}"

        logger.info(
            f"[OpenShell] Executing | sandbox={self._sandbox_name} "
            f"command={command!r} timeout={timeout}s"
        )

        try:
            result = await asyncio.to_thread(
                self._client.exec,
                sandbox_id=self._sandbox_id,
                command=["bash", "-c", full_command],
                timeout_seconds=timeout,
            )

            stdout = result.stdout if result.stdout else ""
            stderr = result.stderr if result.stderr else ""

            logger.info(
                f"[OpenShell] Completed | exit_code={result.exit_code} "
                f"stdout_len={len(stdout)} stderr_len={len(stderr)}"
            )

            return {
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "exit_code": result.exit_code,
            }

        except TimeoutError:
            logger.warning(
                f"[OpenShell] Timeout | command={command!r} timeout={timeout}s"
            )
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s.",
                "exit_code": -1,
            }
        except Exception as e:
            logger.error(f"[OpenShell] Error | command={command!r} error={e}")
            return {
                "stdout": "",
                "stderr": f"OpenShell execution error: {e}",
                "exit_code": -1,
            }

    async def close(self) -> None:
        """Clean up resources.

        If the sandbox was auto-created, it is destroyed.
        Pre-existing sandboxes are left untouched.
        """
        if self._closed:
            return
        self._closed = True

        if self._auto_created and self._sandbox_name and self._client:
            try:
                await asyncio.to_thread(self._client.delete, self._sandbox_name)
                logger.info(
                    f"[OpenShell] Destroyed auto-created sandbox | "
                    f"name={self._sandbox_name}"
                )
            except Exception as e:
                logger.warning(
                    f"[OpenShell] Failed to destroy sandbox | "
                    f"name={self._sandbox_name} error={e}"
                )

    async def __aenter__(self) -> OpenShellExecutor:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
