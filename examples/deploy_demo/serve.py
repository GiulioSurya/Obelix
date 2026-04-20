# examples/deploy_demo/serve.py
"""Entrypoint that runs INSIDE the OpenShell sandbox.

This module is specified as `entrypoint` in deploy.py. The deployer
executes `uv run python -m examples.deploy_demo.serve` inside the
sandbox container. It registers the agent and starts the A2A server.

Do NOT run this directly — use deploy.py instead.
"""

from obelix.adapters.outbound.llm.litellm import LiteLLMProvider
from obelix.adapters.outbound.shell import LocalShellExecutor
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import HTTPExporter, Tracer
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin import BashTool

setup_logging(console_level="INFO", log_dir="/tmp/logs")

# -- Tracer ------------------------------------------------------------------

tracer = Tracer(
    exporter=HTTPExporter(
        endpoint="http://host.docker.internal:8100/api/v1/ingest",
    ),
    service_name="deploy_demo",
)

# -- Provider ----------------------------------------------------------------

# LLM calls go through the gateway's managed inference proxy at
# https://inference.local. The gateway injects the real API key at the
# network level — the sandbox never sees it.
# Setup:
#   openshell provider create --name anthropic --type anthropic --from-existing
#   openshell inference set --provider anthropic --model claude-haiku-4-5-20251001


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id="anthropic/claude-haiku-4-5-20251001",
        api_key="placeholder",  # real key injected by gateway at network level
        base_url="https://inference.local",  # gateway-managed inference proxy
        max_tokens=4000,
        temperature=1,
    )


# -- Agent -------------------------------------------------------------------

_SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to a bash tool. "
    "Your commands execute inside a sandboxed environment with restricted "
    "filesystem and network access. You can read files, list directories, "
    "and perform system operations within the sandbox boundaries. "
    "Always provide a clear description of what each command does."
)


class SandboxedBashAgent(BaseAgent):
    """Agent with shell access via BashTool.

    Inside the sandbox, BashTool uses the default LocalShellExecutor —
    commands run as normal subprocesses, but the OpenShell Policy Engine
    enforces filesystem/network/process restrictions at the kernel level.
    """

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_SYSTEM_MESSAGE,
            provider=make_provider(),
            **kwargs,
        )
        # LocalShellExecutor runs commands as subprocesses.
        # The sandbox policy protects us at the kernel level.
        self.register_tool(BashTool(executor=LocalShellExecutor()))


# -- Serve -------------------------------------------------------------------

if __name__ == "__main__":
    factory = AgentFactory()
    factory.register(name="sandbox_bash", cls=SandboxedBashAgent)

    factory.with_tracer(tracer)

    factory.a2a_serve(
        "sandbox_bash",
        port=8002,
        host="0.0.0.0",
        description="Agent with sandboxed shell execution via OpenShell Deployer",
    )
