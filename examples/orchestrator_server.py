# examples/orchestrator_server.py -- A2A orchestrator that delegates to the
# other 4 example agents via outbound A2A.
"""
A2A server for an orchestrator agent that controls the other 4 example
agents via outbound A2A delegation.

Topology:

    user ──> CLI client ──> orchestrator (:8005)
                                │
                                ├─ dispatch_agent ──> dev_workflow (:8001)
                                ├─ dispatch_agent ──> deploy_demo   (:8002)
                                ├─ dispatch_agent ──> bash_agent    (:8003)
                                └─ dispatch_agent ──> browser_agent (:8004)

The orchestrator's LLM sees a system-prompt block listing all 4 remote
agents (auto-generated from each remote's AgentCard at startup). It uses
the `dispatch_agent` tool to delegate work; each call returns immediately
with a task_id. Completion notifications come back via webhook (mounted
on :8005) and are drained into the conversation at the next turn as
``<remote_task_update>`` user-role messages.

Requirements:
    uv sync --extra litellm --extra serve

Pre-flight (start each remote you want to test with — at least one):
    API_KEY=sk-... uv run python examples/dev_workflow_server.py   # :8001
    API_KEY=sk-... uv run python examples/bash_server.py           # :8003
    API_KEY=sk-... uv run python examples/mcp_playwright.py        # :8004
    # deploy_demo requires an OpenShell gateway — see deploy_demo/README.md.

Usage:
    API_KEY=sk-... uv run python examples/orchestrator_server.py   # :8005
    # In another terminal:
    uv run python examples/cli_client.py http://localhost:8005

Notes:
    - Each remote is optional. If a server isn't running, the orchestrator
      logs a WARNING at startup ("AgentCard fetch failed") and skips that
      agent — the orchestrator still starts with the remotes that ARE up.
    - bash_server.py defaults to LOCAL_EXECUTOR=False (deferred). When
      delegating to bash_agent through the orchestrator, the deferred
      protocol surfaces an `input_required` notification carrying the
      shell command — the orchestrator's LLM can choose to ignore, reply,
      or escalate. For a clean smoke test set LOCAL_EXECUTOR=True in
      bash_server.py so commands self-execute on the bash server.
"""

import os

from dotenv import load_dotenv

from obelix.adapters.outbound.llm.litellm import LiteLLMProvider
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import (
    ConsoleExporter,  # noqa: F401
    HTTPExporter,  # noqa: F401
)
from obelix.infrastructure.logging import setup_logging

load_dotenv()
setup_logging(console_level="INFO")

LITELLM_MODEL = "anthropic/claude-haiku-4-5-20251001"

# tracer = Tracer(exporter=ConsoleExporter(verbosity=2), service_name="ORCHESTRATOR")

# HTTP tracer — swap in to see end-to-end spans across all 5 servers
# alongside the other examples in the Obelix tracer backend.
tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="ORCHESTRATOR",
)


# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        max_tokens=8000,
        reasoning_effort="medium",
        temperature=1,
    )


# -- Agent -------------------------------------------------------------------

_SYSTEM_MESSAGE = (
    "You are a task orchestrator. You delegate work to specialized remote "
    "agents instead of doing it yourself. The system prompt below lists "
    "the agents available to you and the skills each one offers.\n\n"
    "Your job:\n"
    "1. Understand the user's intent.\n"
    "2. Pick the most appropriate remote agent for the task.\n"
    "3. Call `dispatch_agent(agent_name, query)` with a clear, complete "
    "   description of what you want done.\n"
    "4. End your turn after dispatching — DO NOT wait for the result.\n"
    "5. When a `<remote_task_update>` notification appears in a later turn, "
    "   read the result and report back to the user.\n\n"
    "If a task naturally splits across multiple agents, dispatch them in "
    "parallel (multiple `dispatch_agent` calls in the same turn). Each one "
    "returns a separate task_id; their notifications drain independently."
)


class OrchestratorAgent(BaseAgent):
    """Pure-orchestrator agent — no local tools, only outbound A2A delegation.

    The 5 outbound A2A tools (``dispatch_agent``, ``respond_to_remote``,
    ``task_list``, ``task_get``, ``task_stop``) are auto-injected by
    AgentFactory.a2a_serve when ``remote_agents=[...]`` is non-empty.
    """

    def __init__(self, **kwargs):
        super().__init__(
            system_message=_SYSTEM_MESSAGE,
            provider=make_provider(),
            max_iterations=20,
            **kwargs,
        )


# -- Serve -------------------------------------------------------------------


# All 4 example agents. Comment out any you don't have running — resolve_all
# will log a WARNING and skip unreachable URLs without aborting startup.
REMOTE_AGENTS = [
    "http://localhost:8001",  # dev_workflow_server  (review/commit/summary)
    "http://localhost:8002",  # deploy_demo          (sandboxed bash)
    "http://localhost:8003",  # bash_server          (shell)
    "http://localhost:8004",  # mcp_playwright       (browser)
]


if __name__ == "__main__":
    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.register(name="orchestrator", cls=OrchestratorAgent)

    factory.a2a_serve(
        "orchestrator",
        port=8005,
        description=(
            "Orchestrator that delegates to specialized remote A2A agents. "
            "Use it as a single entry point for tasks that span multiple "
            "skills (code review, shell, browser)."
        ),
        remote_agents=REMOTE_AGENTS,
    )
