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
from dotenv import load_dotenv
from pydantic import Field

from obelix.core.agent import BaseAgent, SharedMemoryGraph
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.shared_memory import PropagationPolicy
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool
from obelix.adapters.outbound.openai.connection import OpenAIConnection
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.oci.connection import OCIConnection
from obelix.adapters.outbound.openai.provider import OpenAIProvider
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider
from obelix.adapters.outbound.oci.provider import OCILLm
from obelix.infrastructure.k8s import YamlConfig
from obelix.core.model.tool_message import ToolRequirement
from obelix.infrastructure.logging import setup_logging
from obelix.plugins.builtin.ask_user_question_tool import AskUserQuestionTool
import os

load_dotenv()

setup_logging(console_level="TRACE")


# ========== CONFIGURAZIONE PROVIDER ==========
api_key = os.getenv("ANTHROPIC_API_KEY")

#open ai
openai_connection = OpenAIConnection(
    api_key=api_key,
    base_url="https://api.anthropic.com/v1/"
)

#anthrtopic
anthropic_connection = AnthropicConnection(api_key=api_key)

#oci
infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
oci_provider_config = infra_config.get("llm_providers.oci")

oci_config = {
    'user': oci_provider_config["user_id"],
    'fingerprint': oci_provider_config["fingerprint"],
    'key_content': oci_provider_config["private_key_content"],
    'tenancy': oci_provider_config["tenancy"],
    'region': oci_provider_config["region"]
}

oci_connection = OCIConnection(oci_config)


# ========== TOOL ==========
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


# ========== AGENT CLASSES ==========
class MathAgent(BaseAgent):
    """Math expert agent."""

    context: str = Field(default="", description="usa questo campo per dare istruzioni più specifiche")

    def __init__(self):
        super().__init__(
            system_message="sei un esperto di matematica dotato di tool per fare calcoli, usalo per risolverli.",
            provider=OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b"),
            tool_policy=[
                ToolRequirement(
                    tool_name="calculator",
                    require_success=True,
                    min_calls=1,
                    error_message="you have to use the calculator tool to solve the equation"
                )
            ],
        )
        self.register_tool(CalculatorTool())


class ReportAgent(BaseAgent):
    """Report agent that formats math results into a structured summary.

    Depends on math_agent via SharedMemoryGraph: it receives calculation
    results as injected context and produces a formatted report.
    """

    def __init__(self):
        super().__init__(
            system_message=(
                "Sei un agente specializzato nella creazione di report.\n"
                "Riceverai i risultati di calcoli matematici come contesto condiviso.\n"
                "Il tuo compito è:\n"
                "1. Analizzare i risultati ricevuti\n"
                "2. Produrre un report formattato con:\n"
                "   - Le espressioni originali\n"
                "   - I risultati calcolati\n"
                "   - Una verifica logica (i numeri hanno senso?)\n"
                "   - Un breve commento riassuntivo\n"
                "Rispondi SEMPRE in formato strutturato."
            ),
            provider=OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b"),
        )


class CoordinatorAgent(BaseAgent):
    """Coordinator agent."""

    def __init__(self):
        super().__init__(
            system_message=(
                "Sei un agente orchestratore con un Math Agent e un Report Agent.\n"
                "REGOLE OBBLIGATORIE:\n"
                "- Devi SEMPRE usare ask_user_question per raccogliere o confermare input.\n"
                "- NON puoi chiamare il Math Agent senza aver prima usato ask_user_question.\n"
                "- Se mancano o sono ambigue informazioni, fermati e chiedi chiarimenti.\n"
                "- Solo dopo la risposta dell'utente puoi chiamare il Math Agent.\n"
                "- Dopo che il Math Agent ha risposto, chiama il Report Agent per formattare i risultati.\n"
                "utilizza almeno una volta il tool ask user question"
            ),
            provider=OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b"),
            tools=AskUserQuestionTool,
        )


# ========== SHARED MEMORY GRAPH ==========
def create_memory_graph() -> SharedMemoryGraph:
    """Create the dependency graph: math_agent -> report_agent."""
    graph = SharedMemoryGraph()
    graph.add_agent("math_agent")
    graph.add_agent("report_agent")
    graph.add_edge("math_agent", "report_agent", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    return graph


# ========== FACTORY SETUP ==========
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


# ========== TEST ==========
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
        "mi dici quanto fa ((18 + 6) * (14 - 8)) e risolvi anche ((48-5)+25/8*(35+9))? "
        "Dopo i calcoli, genera un report formattato dei risultati."
    )

    # Print conversation history
    print("\n" + "=" * 50)
    print("CONVERSATION HISTORY")
    print("=" * 50)
    for element in coordinator.conversation_history:
        print(element.model_dump_json(indent=4))