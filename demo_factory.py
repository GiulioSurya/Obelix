# demo_factory.py - Demo of Agent Factory + Shared Memory
"""
This demo shows how to use the AgentFactory with SharedMemoryGraph.

Flow:
  math_agent  ──▶  report_agent
      │                 │
      └────────┬────────┘
          coordinator (orchestrator)

The coordinator calls math_agent first, which publishes its result
to the SharedMemoryGraph. When report_agent runs, it pulls the math
result via shared memory and produces a formatted report.
"""
import os

from dotenv import load_dotenv
from pydantic import Field

from obelix.core.agent import BaseAgent, SharedMemoryGraph
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.shared_memory import PropagationPolicy
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool
from obelix.adapters.outbound.openai.connection import OpenAIConnection
from obelix.adapters.outbound.openai.provider import OpenAIProvider
from obelix.core.model.tool_message import ToolRequirement
from obelix.infrastructure.logging import setup_logging
from obelix.core.tracer import Tracer, HTTPExporter
from obelix.plugins.builtin.ask_user_question_tool import AskUserQuestionTool

load_dotenv()

tracer = Tracer(exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"))

setup_logging(console_level="TRACE")


anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

openai_connection = OpenAIConnection(
    api_key=anthropic_api_key,
    base_url="https://api.anthropic.com/v1/",
)



@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(Tool):
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
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



class MathAgent(BaseAgent):
    """Math expert agent."""

    context: str = Field(default="", description="Use this field to provide more specific instructions")

    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a math expert equipped with a calculator tool. Use it to solve equations.",
            provider=OpenAIProvider(connection=openai_connection, model_id="claude-3-5-haiku-20241022"),
            tool_policy=[
                ToolRequirement(
                    tool_name="calculator",
                    require_success=True,
                    min_calls=1,
                    error_message="You must use the calculator tool to solve the equation"
                )
            ],
            **kwargs,
        )
        self.register_tool(CalculatorTool())


class ReportAgent(BaseAgent):
    """Report agent that formats math results into a structured summary.

    Depends on math_agent via SharedMemoryGraph: it receives calculation
    results as injected context and produces a formatted report.
    """

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
            provider=OpenAIProvider(connection=openai_connection, model_id="claude-3-5-haiku-20241022"),
            **kwargs,
        )


class CoordinatorAgent(BaseAgent):
    """Coordinator agent."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are an orchestrator agent with a Math Agent and a Report Agent.\n"
                "MANDATORY RULES:\n"
                "- You MUST ALWAYS use ask_user_question to collect or confirm input.\n"
                "- You CANNOT call the Math Agent without first using ask_user_question.\n"
                "- If information is missing or ambiguous, stop and ask for clarification.\n"
                "- Only after the user responds can you call the Math Agent.\n"
                "- After the Math Agent responds, call the Report Agent to format the results.\n"
                "Use the ask_user_question tool at least once."
            ),
            provider=OpenAIProvider(connection=openai_connection, model_id="claude-3-5-haiku-20241022"),
            tools=AskUserQuestionTool,
            **kwargs,
        )



def create_memory_graph() -> SharedMemoryGraph:
    """Create the dependency graph: math_agent -> report_agent."""
    graph = SharedMemoryGraph()
    graph.add_agent("math_agent")
    graph.add_agent("report_agent")
    graph.add_edge("math_agent", "report_agent", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    return graph


def create_factory() -> AgentFactory:
    """Create and configure the agent factory with shared memory."""
    memory_graph = create_memory_graph()

    factory = AgentFactory()

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

    factory.register(
        name="coordinator",
        cls=CoordinatorAgent,
    )

    return factory


if __name__ == "__main__":
    factory = create_factory()

    # Create orchestrator with both subagents.
    # The factory injects the dependency awareness message so the coordinator
    # knows to call math_agent before report_agent.
    coordinator = factory.create(
        "coordinator",
        subagents=["math_agent", "report_agent"]
    )

    # Execute query
    response = coordinator.execute_query(
        "What is ((18 + 6) * (14 - 8)) and also solve ((48-5)+25/8*(35+9))? "
        "After the calculations, generate a formatted report of the results."
    )

    # Print conversation history
    print("\n" + "=" * 50)
    print("CONVERSATION HISTORY")
    print("=" * 50)
    for element in coordinator.conversation_history:
        print(element.model_dump_json(indent=4))