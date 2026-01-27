# demo_factory.py - Demo of Agent Factory
"""
This demo shows how to use the AgentFactory to create agents.
Compare with demo_subagent.py which uses decorators directly.
"""
from dotenv import load_dotenv
from pydantic import Field

from src.base_agent import BaseAgent
from src.base_agent.agent_factory import AgentFactory
from src.tools import ToolBase, tool
from src.config import GlobalConfig
from src.providers import Providers
from src.connections.llm_connection import OpenAIConnection, AnthropicConnection, OCIConnection
from src.llm_providers.openai_provider import OpenAIProvider
from src.llm_providers.anthropic_provider import AnthropicProvider
from src.llm_providers.oci_provider import OCILLm
from src.k8s_config import YamlConfig
from src.messages.tool_message import ToolRequirement
from src.logging_config import setup_logging
from src.tools.tool.ask_user_question_tool import AskUserQuestionTool
import os

load_dotenv()

setup_logging(console_level="TRACE")


# ========== CONFIGURAZIONE PROVIDER ==========
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY non trovata in .env")

openai_connection = OpenAIConnection(
    api_key=api_key,
    base_url="https://api.anthropic.com/v1/"
)

anthropic_connection = AnthropicConnection(api_key=api_key)

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


# ========== AGENT CLASSES (no decorators!) ==========
class MathAgent(BaseAgent):
    """Math expert agent. Note: NO @subagent decorator needed with factory."""

    # Extra field for subagent input (factory will extract this)
    context: str = Field(default="", description="usa questo campo per dare istruzioni più specifiche")

    def __init__(self, **kwargs):
        # Allow provider override via kwargs, with default
        if 'provider' not in kwargs:
            kwargs['provider'] = OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b")

        # Allow system_message override, with default
        if 'system_message' not in kwargs:
            kwargs['system_message'] = "sei un esperto di matematica dotato di tool per fare calcoli, usalo per risolverli."

        # Set tool_policy if not provided
        if 'tool_policy' not in kwargs:
            kwargs['tool_policy'] = [
                ToolRequirement(
                    tool_name="calculator",
                    require_success=True,
                    min_calls=1,
                    error_message="you have to use the calculator tool to solve the equation"
                )
            ]

        super().__init__(**kwargs)
        self.register_tool(CalculatorTool())


class CoordinatorAgent(BaseAgent):
    """Coordinator agent. Note: NO @orchestrator decorator needed with factory."""

    def __init__(self, **kwargs):
        # Allow provider override via kwargs, with default
        if 'provider' not in kwargs:
            kwargs['provider'] = OCILLm(connection=oci_connection, model_id="openai.gpt-oss-120b")

        # Allow system_message override, with default
        if 'system_message' not in kwargs:
            kwargs['system_message'] = """Sei un agente orchestratore con un Math Agent.
REGOLE OBBLIGATORIE:
- Devi SEMPRE usare ask_user_question per raccogliere o confermare input.
- NON puoi chiamare il Math Agent senza aver prima usato ask_user_question.
- Se mancano o sono ambigue informazioni, fermati e chiedi chiarimenti.
- Solo dopo la risposta dell'utente puoi chiamare il Math Agent.
"""

        # Allow tools override, with default
        if 'tools' not in kwargs:
            kwargs['tools'] = AskUserQuestionTool

        super().__init__(**kwargs)


# ========== FACTORY SETUP ==========
def create_factory() -> AgentFactory:
    """Create and configure the agent factory."""
    factory = AgentFactory()

    # Register MathAgent as subagent
    factory.register(
        name="math_agent",
        cls=MathAgent,
        expose_as_subagent=True,
        subagent_description="A math expert that can perform calculations",
        stateless=True,  # Each call gets fresh history
    )

    # Register CoordinatorAgent (not as subagent, it's the orchestrator)
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