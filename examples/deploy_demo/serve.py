# examples/deploy_demo/serve.py
"""Entrypoint that runs INSIDE the OpenShell sandbox.

This module is specified as `entrypoint` in deploy.py. The deployer
executes `uv run python -m examples.deploy_demo.serve` inside the
sandbox container. It registers the agent and starts the A2A server.

Do NOT run this directly — use deploy.py instead.
"""

import os

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.adapters.outbound.shell import LocalShellExecutor
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin import BashTool

setup_logging(console_level="INFO", log_dir="/tmp/logs")

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
if not LITELLM_MODEL:
    raise RuntimeError(
        "LITELLM_MODEL env var is required. "
        "Set it via OpenShell provider credentials: "
        "openshell provider create --name anthropic --type anthropic "
        "--from-existing --credential LITELLM_MODEL=anthropic/claude-haiku-4-5-20251001"
    )

# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
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

    factory.a2a_serve(
        "sandbox_bash",
        port=8002,
        host="0.0.0.0",
        description="Agent with sandboxed shell execution via OpenShell Deployer",
    )
