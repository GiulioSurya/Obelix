# demo_subagent.py - Test file for subagent/orchestrator (DELETE AFTER TEST)
from dotenv import load_dotenv
from pydantic import Field
from src.base_agent import BaseAgent, subagent, orchestrator
from src.tools import ToolBase, tool

from src.config import GlobalConfig
from src.providers import Providers

from src.k8s_config import YamlConfig

from src.logging_config import setup_logging
import os

load_dotenv()


# Configura logging all'avvio - chiamare UNA VOLTA prima di tutto
setup_logging(console_level="TRACE")

# Leggi configurazione OCI da infrastructure.yaml
infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
oci_provider_config = infra_config.get("llm_providers.oci")

# Configurazione OCI
oci_config = {
    'user': oci_provider_config["user_id"],
    'fingerprint': oci_provider_config["fingerprint"],
    'key_content': oci_provider_config["private_key_content"],
    'tenancy': oci_provider_config["tenancy"],
    'region': oci_provider_config["region"]
}

# Inizializza connection e registra in GlobalConfig
from src.connections.llm_connection import OCIConnection
oci_connection = OCIConnection(oci_config)
GlobalConfig().set_provider(provider=Providers.OCI_GENERATIVE_AI, connection=oci_connection)



# 1. Tool calcolatrice
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


# 2. SubAgent con il tool
@subagent(name="math_agent", description="A math expert that can perform calculations")
class MathAgent(BaseAgent):
    context: str = Field(default="", description="usa questo campo per dare istruzioni più specifiche")

    def __init__(self):
        super().__init__(system_message="sei un esperto di matematica dotato di tool per fare calcoli, usalo per risolverli. sei obbligato ad usare il tool per rif")
        self.register_tool(CalculatorTool())


# 3. Orchestrator
@orchestrator
class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="""sei un agente orchestratore, sei dotato di altri sub agent a cui puoi delegare i compiti
                                        al momento hai un math agent, se l'utente ti fa utente sulla matematica delega a questo""")


# Test
if __name__ == "__main__":
    print("Creating agents...")

    coordinator = CoordinatorAgent()
    math_agent = MathAgent()

    print(f"Coordinator class: {coordinator.__class__.__name__}")
    print(f"MathAgent subagent_name: {math_agent.subagent_name}")
    print(f"MathAgent schema: {math_agent.create_subagent_schema()}")

    print("\nRegistering sub-agent...")
    coordinator.register_agent(math_agent)

    response = coordinator.execute_query("mi dici quanto fa ((18 + 6) * (14 - 8) / 3 - 5) * (7 + 21 / (3 + 4)) - (40 / 5 + 6) * 2?")
    print(response)
