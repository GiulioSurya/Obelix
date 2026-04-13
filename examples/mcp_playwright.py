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

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.adapters.outbound.mcp.config import MCPServerConfig
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import ConsoleExporter, Tracer
from obelix.infrastructure.logging import setup_logging

load_dotenv()
setup_logging(console_level="INFO")

LITELLM_MODEL = "anthropic/claude-sonnet-4-20250514"

tracer = Tracer(exporter=ConsoleExporter(verbosity=3))

_SYSTEM_MESSAGE = (
    "You are a web browsing assistant. You have access to a real browser "
    "via Playwright MCP tools.\n"
    "You can navigate to URLs, click elements, fill forms, take screenshots, "
    "and interact with any web page.\n"
    "When the user asks you to do something on the web, use the browser tools "
    "to accomplish it step by step.\n"
    "Always describe what you see on the page after navigating."
)

# MCP server config — Playwright via npx stdio
_PLAYWRIGHT_MCP = MCPServerConfig(
    name="playwright",
    transport="stdio",
    command="npx",
    args=["@playwright/mcp@latest"],
)


# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        max_tokens=8000,
        temperature=1,
    )


# -- Agent -------------------------------------------------------------------


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
