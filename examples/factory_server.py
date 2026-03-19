# examples/factory_server.py -- A2A server with multi-agent orchestration
"""
A2A server with a coordinator agent that orchestrates a math agent
and a report agent via shared memory.

Flow:
  math_agent  -->  report_agent
      |                 |
      +--------+--------+
          coordinator (orchestrator)

Requirements:
    uv sync --extra litellm --extra serve

Usage:
    API_KEY=sk-... uv run python examples/factory_server.py
    # Server starts on http://localhost:8001
"""

import os
import re

from dotenv import load_dotenv
from pydantic import Field

from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.core.agent import BaseAgent, SharedMemoryGraph
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.hooks import AgentEvent
from obelix.core.agent.shared_memory import PropagationPolicy
from obelix.core.model.tool_message import ToolRequirement
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import ConsoleExporter
from obelix.infrastructure.logging import setup_logging

load_dotenv()
setup_logging(console_level="INFO")

LITELLM_MODEL = "anthropic/claude-haiku-4-5-20251001"

tracer = Tracer(exporter=ConsoleExporter(verbosity=3))


# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        max_tokens=8000,
        reasoning_effort="low",
        temperature=1,
    )


# -- Tools -------------------------------------------------------------------


@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(Tool):
    operation: str = Field(
        ..., description="Operation: add, subtract, multiply, divide"
    )
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    async def execute(self) -> dict:
        if self.operation == "add":
            result = self.a + self.b
        elif self.operation == "subtract":
            result = self.a - self.b
        elif self.operation == "multiply":
            result = self.a * self.b
        elif self.operation == "divide":
            result = self.a / self.b if self.b != 0 else "Error: division by zero"
        else:
            result = f"Unknown operation: {self.operation}"
        return {"result": result}


# -- Agents ------------------------------------------------------------------


# Patterns that indicate unsupported math operations
_UNSUPPORTED_OPS = re.compile(
    r"\b(sqrt|square root|radice|power|potenza|exponent|esponente|"
    r"factorial|fattoriale|modulo|logarithm|log|ln|"
    r"sin|cos|tan|integral|derivative|derivata|matrix|matrice)\b",
    re.IGNORECASE,
)


class MathAgent(BaseAgent):
    """Math expert agent."""

    context: str = Field(
        default="", description="Use this field to provide more specific instructions"
    )

    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a math expert equipped with a calculator tool. Use it to solve equations.",
            provider=make_provider(),
            tool_policy=[
                ToolRequirement(
                    tool_name="calculator",
                    require_success=True,
                    min_calls=1,
                    error_message="You must use the calculator tool to solve the equation",
                )
            ],
            **kwargs,
        )
        self.register_tool(CalculatorTool())

        # Reject unsupported operations at the sub-agent level.
        # The coordinator receives this as a clear rejection in ToolResult,
        # not a generic error — so the LLM knows the agent refused on purpose.
        self.on(AgentEvent.BEFORE_LLM_CALL).when(
            lambda s: (
                _UNSUPPORTED_OPS.search(
                    getattr(s.agent.conversation_history[-1], "content", None) or ""
                )
                is not None
            )
        ).reject(
            "This agent only supports basic arithmetic (add, subtract, multiply, divide). "
            "Operations like power, sqrt, log, trigonometry are not available."
        )


class ReportAgent(BaseAgent):
    """Report agent that formats math results into a structured summary."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are an agent specialized in creating reports.\n"
                "You will receive math calculation results as shared context.\n"
                "Your task is:\n"
                "1. Analyze the received results\n"
                "2. Produce a formatted report with:\n"
                "   - The original expressions\n"
                "   - The calculated results\n"
                "   - A logical verification (do the numbers make sense?)\n"
                "   - A brief summary comment\n"
                "ALWAYS respond in a structured format."
            ),
            provider=make_provider(),
            **kwargs,
        )


class CoordinatorAgent(BaseAgent):
    """Coordinator agent."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are an orchestrator agent with a Math Agent and a Report Agent.\n"
                "MANDATORY RULES:\n"
                "- You MUST use the request_user_input tool to confirm or collect input from the user before proceeding.\n"
                "- You CANNOT call the Math Agent without first using request_user_input to validate the request.\n"
                "- If information is missing or ambiguous, use request_user_input to ask for clarification.\n"
                "- Only after the user responds can you call the Math Agent.\n"
                "- After the Math Agent responds, call the Report Agent to format the results.\n"
                "- If a sub-agent REJECTS a request, do NOT try to work around it or retry with a different phrasing. "
                "Report the rejection reason directly to the user as your final answer.\n"
                "Always use request_user_input at least once before doing any calculation."
            ),
            provider=LiteLLMProvider(
                model_id=LITELLM_MODEL,
                api_key=os.getenv("API_KEY"),
                reasoning_effort="medium",
                max_tokens=10_000,
                temperature=1,
            ),
            **kwargs,
            planning=True,
        )

        # Reject queries requesting operations the calculator cannot handle.
        # The coordinator is the A2A entry point, so REJECT here produces
        # TaskState.rejected for the client — no LLM call wasted.
        self.on(AgentEvent.BEFORE_LLM_CALL).when(
            lambda s: (
                _UNSUPPORTED_OPS.search(
                    getattr(s.agent.conversation_history[-1], "content", None) or ""
                )
                is not None
            )
        ).reject(
            "This agent only supports basic arithmetic (add, subtract, multiply, divide). "
            "Operations like power, sqrt, log, trigonometry are not available."
        )


# -- Factory -----------------------------------------------------------------


def create_factory() -> AgentFactory:
    memory_graph = SharedMemoryGraph()
    memory_graph.add_agent("math_agent")
    memory_graph.add_agent("report_agent")
    memory_graph.add_edge(
        "math_agent", "report_agent", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )

    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.with_memory_graph(memory_graph)

    factory.register(
        name="math_agent",
        cls=MathAgent,
        subagent_description="A math expert that can perform calculations",
        stateless=True,
    )
    factory.register(
        name="report_agent",
        cls=ReportAgent,
        subagent_description=(
            "A report writer that formats calculation results into a structured summary. "
            "Call this AFTER the math agent has produced results."
        ),
        stateless=True,
    )
    factory.register(name="coordinator", cls=CoordinatorAgent)

    return factory


# -- Serve -------------------------------------------------------------------

if __name__ == "__main__":
    factory = create_factory()
    factory.a2a_serve(
        "coordinator",
        subagents=["math_agent", "report_agent"],
        port=8001,
        description="Math coordinator with calculator and report sub-agents",
    )
