# examples/shell-demo/bash_server_sandboxed.py
"""A2A server with BashTool running inside an OpenShell sandbox.

The BashTool uses OpenShellExecutor — commands execute in a policy-governed
sandbox (filesystem, network, process controls enforced by the kernel).

Requirements (Linux/macOS only):
    uv sync --extra litellm --extra serve --extra openshell

Usage:
    API_KEY=sk-... uv run python examples/shell-demo/bash_server_sandboxed.py

    # Then connect with the CLI client:
    uv run python examples/cli_client.py --url http://localhost:8002
"""

import os

from dotenv import load_dotenv

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.adapters.outbound.shell import OpenShellExecutor
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import ConsoleExporter, Tracer
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin import BashTool

load_dotenv()
setup_logging(console_level="INFO")

LITELLM_MODEL = os.getenv("LITELLM_MODEL", "anthropic/claude-haiku-4-5-20251001")

tracer = Tracer(
    exporter=ConsoleExporter(),
    service_name="shell_demo",
)

# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        max_tokens=4000,
        temperature=1,
    )


# -- Executor ----------------------------------------------------------------

# Gateway and TLS certs are read from env vars:
#   OPENSHELL_GATEWAY=host.docker.internal:8080
#   OPENSHELL_TLS_CERT_DIR=/app/certs
_executor = OpenShellExecutor(policy="./policy.yaml")

# -- Agent -------------------------------------------------------------------

_SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to a bash tool. "
    "Your commands execute inside a sandboxed environment with restricted "
    "filesystem and network access. You can read files, list directories, "
    "and perform system operations within the sandbox boundaries. "
    "Always provide a clear description of what each command does."
)


class SandboxedBashAgent(BaseAgent):
    """Agent with shell access via BashTool + OpenShell sandbox."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_SYSTEM_MESSAGE,
            provider=make_provider(),
            **kwargs,
        )
        self.register_tool(BashTool(executor=_executor))


# -- Serve -------------------------------------------------------------------

if __name__ == "__main__":
    print("BashTool mode: OpenShell sandbox")

    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.register(name="sandbox_bash", cls=SandboxedBashAgent)

    factory.a2a_serve(
        "sandbox_bash",
        port=8002,
        description="Agent with sandboxed shell execution via OpenShell",
    )
