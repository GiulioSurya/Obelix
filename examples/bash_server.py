# examples/bash_server.py -- A2A server with BashTool
"""
Minimal A2A server that exposes an agent with BashTool.

Two modes available (toggle LOCAL_EXECUTOR below):

- **Local executor** (default): BashTool executes commands directly on the
  server via LocalShellExecutor. No client interaction needed for bash.

- **Deferred**: BashTool returns None, the A2A client must execute the
  command and send back the result. Set LOCAL_EXECUTOR = False.

Requirements:
    uv sync --extra litellm --extra serve

Usage:
    API_KEY=sk-... uv run python examples/bash_server.py
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
LOCAL_EXECUTOR = False

LITELLM_MODEL = "anthropic/claude-haiku-4-5-20251001"

# tracer = Tracer(exporter=ConsoleExporter(verbosity=3))

tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="demo_bash",
)


# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        reasoning_effort="medium",
        max_tokens=4000,
        temperature=1,
    )


# -- Agent -------------------------------------------------------------------

_executor = LocalShellExecutor() if LOCAL_EXECUTOR else None

_SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to a bash tool. "
    "You can execute shell commands to read files, list directories, "
    "and perform system operations. Always provide a clear description "
    "of what each command does before executing it."
)


class BashAgent(BaseAgent):
    """Agent with shell access via BashTool."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_SYSTEM_MESSAGE,
            provider=make_provider(),
            **kwargs,
        )
        self.register_tool(BashTool(executor=_executor) if _executor else BashTool())


# -- Serve -------------------------------------------------------------------

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
