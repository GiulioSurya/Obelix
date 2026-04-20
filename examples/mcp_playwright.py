# examples/mcp_playwright.py -- A2A server with Playwright browser via MCP
"""
A2A server that exposes a browser agent powered by Playwright MCP.
No Playwright installation needed — npx downloads it on demand.

The agent connects to the Playwright MCP server via stdio,
discovers browser tools (navigate, click, type, screenshot, etc.)
and uses them to fulfill user queries.

Requirements:
    Node.js >= 18 (for npx)
    uv sync --extra litellm --extra serve

Usage:
    API_KEY=sk-... uv run python examples/mcp_playwright.py
    # Server starts on http://localhost:8004
"""

import os

from dotenv import load_dotenv

from obelix.adapters.outbound.llm.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.llm.anthropic.provider import AnthropicProvider
from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import HTTPExporter, Tracer
from obelix.infrastructure.logging import setup_logging

load_dotenv()
setup_logging(console_level="INFO")

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# tracer = Tracer(exporter=ConsoleExporter(verbosity=3))

tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="playwright",
)

_SYSTEM_MESSAGE = (
    "You are a web browsing assistant. You have access to a real browser "
    "via Playwright MCP tools."
    "You can navigate to URLs, click elements, fill forms, take screenshots, "
    "and interact with any web page."
    "When the user asks you to do something on the web, use the browser tools "
    "to accomplish it step by step."
    "Always describe what you see on the page after navigating."
)

# MCP server config — Playwright via npx stdio
_PLAYWRIGHT_MCP = MCPServerConfig(
    name="playwright",
    transport="stdio",
    command="npx",
    args=["@playwright/mcp@latest"],
)


def make_provider() -> AnthropicProvider:
    return AnthropicProvider(
        connection=AnthropicConnection(api_key=os.getenv("API_KEY")),
        model_id=ANTHROPIC_MODEL,
        max_tokens=9000,
        temperature=1,
    )


class BrowserAgent(BaseAgent):
    """Agent with browser access via Playwright MCP."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_SYSTEM_MESSAGE,
            provider=make_provider(),
            mcp_config=_PLAYWRIGHT_MCP,
            max_iterations=20,
            **kwargs,
        )


# -- Serve -------------------------------------------------------------------

if __name__ == "__main__":
    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.register(name="browser_agent", cls=BrowserAgent)

    factory.a2a_serve(
        "browser_agent",
        port=8004,
        description="Web browsing agent with Playwright browser control via MCP",
    )
