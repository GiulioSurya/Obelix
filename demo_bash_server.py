# demo_bash_server.py — A2A server with BashTool
"""
Minimal A2A server that exposes an agent with BashTool.
The agent can request shell commands; the A2A client executes them.

Usage:
    uv run python demo_bash_server.py
    # Server starts on http://localhost:8002
"""

import os

from dotenv import load_dotenv

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import HTTPExporter
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin import BashTool

load_dotenv()
setup_logging(console_level="INFO")

tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="demo_bash",
)


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000,
        temperature=1,
    )


class BashAgent(BaseAgent):
    """Agent with shell access via deferred BashTool."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are a helpful assistant with access to a bash tool. "
                "You can execute shell commands on the client's machine. "
                "Use the bash tool to read files, list directories, and perform "
                "system operations. Always provide a clear description of what "
                "each command does before executing it."
            ),
            provider=make_provider(),
            **kwargs,
        )
        self.register_tool(BashTool())


if __name__ == "__main__":
    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.register(name="bash_agent", cls=BashAgent)

    factory.a2a_serve(
        "bash_agent",
        port=8002,
        description="Agent with shell command execution via BashTool",
    )
