# demo_factory.py - Demo of Agent Factory
"""
This demo shows how to use the AgentFactory to create and compose agents.
"""
from dotenv import load_dotenv
from pydantic import Field

from src.domain.agent import BaseAgent
from src.domain.agent.agent_factory import AgentFactory
from src.domain.tool.tool_base import ToolBase
from src.domain.tool.tool_decorator import tool
from src.infrastructure.config import GlobalConfig
from src.infrastructure.providers import Providers
from src.adapters.outbound.openai.connection import OpenAIConnection
from src.adapters.outbound.anthropic.connection import AnthropicConnection
from src.adapters.outbound.ollama.provider import OllamaProvider
from src.adapters.outbound.oci.connection import OCIConnection
from src.adapters.outbound.openai.provider import OpenAIProvider
from src.adapters.outbound.anthropic.provider import AnthropicProvider
from src.adapters.outbound.oci.provider import OCILLm
from src.infrastructure.k8s import YamlConfig
from src.domain.model.tool_message import ToolRequirement
from src.infrastructure.logging import setup_logging
from src.plugins.builtin.ask_user_question_tool import AskUserQuestionTool
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
#ollama

ollama = OllamaProvider(model_id="MFDoom/deepseek-r1-tool-calling:8b")


GlobalConfig().set_provider(provider=Providers.OCI_GENERATIVE_AI, connection=oci_connection)


# ========== TOOL ==========
@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(ToolBase):
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
            provider=ollama,
            #OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b"),
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


class CoordinatorAgent(BaseAgent):
    """Coordinator agent."""

    def __init__(self):
        super().__init__(
            system_message=(
                "Sei un agente orchestratore con un Math Agent.\n"
                "REGOLE OBBLIGATORIE:\n"
                "- Devi SEMPRE usare ask_user_question per raccogliere o confermare input.\n"
                "- NON puoi chiamare il Math Agent senza aver prima usato ask_user_question.\n"
                "- Se mancano o sono ambigue informazioni, fermati e chiedi chiarimenti.\n"
                "- Solo dopo la risposta dell'utente puoi chiamare il Math Agent.\n"
                "utilizza almeno una volta il tool ask user question"
            ),
            provider=OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b"),
            tools=AskUserQuestionTool,
        )


# ========== FACTORY SETUP ==========
def create_factory() -> AgentFactory:
    """Create and configure the agent factory."""
    factory = AgentFactory()

    factory.register(
        name="math_agent",
        cls=MathAgent,
        subagent_description="A math expert that can perform calculations",
        stateless=True,
    )

    factory.register(
        name="coordinator",
        cls=CoordinatorAgent,
    )

    return factory


# ========== TEST ==========
if __name__ == "__main__":
    # Create factory
    factory = create_factory()

    # Create orchestrator with math_agent as subagent
    # This is the key difference: ONE line creates everything!
    coordinator = factory.create(
        "coordinator",
        subagents=["math_agent"]
    )

    # Execute query
    response = coordinator.execute_query(
        "mi dici quanto fa ((18 + 6) * (14 - 8) e risolvi anche ((48-5)+25/8*(35+9)) "
        "risolverli chiamando i due sub agent parallelamente?"
    )

    # Print conversation history
    print("\n" + "=" * 50)
    print("CONVERSATION HISTORY")
    print("=" * 50)
    for element in coordinator.conversation_history:
        print(element.model_dump_json(indent=4))