# demo_bash_server.py — A2A server with BashTool
"""
Minimal A2A server that exposes an agent with BashTool.

Two modes available (toggle LOCAL_EXECUTOR below):

- **Local executor** (default): BashTool executes commands directly on the
  server via LocalShellExecutor. No client interaction needed for bash.

- **Deferred**: BashTool returns None, the A2A client must execute the
  command and send back the result. Set LOCAL_EXECUTOR = False.

Usage:
    uv run python demo_bash_server.py
    # Server starts on http://localhost:8002
"""

import os

from dotenv import load_dotenv

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.adapters.outbound.shell import LocalShellExecutor
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import HTTPExporter
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin import BashTool

load_dotenv()
setup_logging(console_level="INFO")

# Toggle: True = server executes commands, False = client executes (deferred)
LOCAL_EXECUTOR = True

tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="demo_bash",
)


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        reasoning_effort="medium",
        max_tokens=4000,
        temperature=1,
    )


_executor = LocalShellExecutor() if LOCAL_EXECUTOR else None


def _build_system_message() -> str:
    """Build system message with shell environment info."""
    base = (
        "You are a helpful assistant with access to a bash tool. "
        "You can execute shell commands to read files, list directories, "
        "and perform system operations. Always provide a clear description "
        "of what each command does before executing it."
    )
    if _executor:
        info = _executor.shell_info
        base += (
            f"\n\nEnvironment: {info['platform']}, shell: {info['shell_name']}. "
            f"Use {info['shell_name']}-compatible syntax (heredoc, pipes, etc. are supported)."
        )
    return base


class BashAgent(BaseAgent):
    """Agent with shell access via BashTool."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_build_system_message(),
            provider=make_provider(),
            **kwargs,
        )
        self.register_tool(BashTool(executor=_executor) if _executor else BashTool())


if __name__ == "__main__":
    mode = "local executor" if LOCAL_EXECUTOR else "deferred (client executes)"
    print(f"BashTool mode: {mode}")

    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.register(name="bash_agent", cls=BashAgent)

    factory.a2a_serve(
        "bash_agent",
        port=8002,
        description="Agent with shell command execution via BashTool",
    )
