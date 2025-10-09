# src/agents/notion_agent.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.base_agent.agent_schema import AgentSchema
from src.base_agent.base_agent import BaseAgent
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.tools.tool.notion_tool import NotionPageTool


class NotionInput(BaseModel):
    query: str = Field(..., description="Operazione da eseguire su Notion")


class NotionSchema(AgentSchema):
    name: str = Field(default="notion-agent", description="Nome dell'agent")
    description: str = Field(
        default="Agent specializzato in operazioni Notion: creazione pagine, database, gestione contenuti",
        description="Descrizione delle capacità dell'agent")
    input_schema: Dict[str, Any] = Field(default_factory=lambda: NotionInput.model_json_schema())


class NotionAgent(BaseAgent):
    def __init__(self, provider: AbstractLLMProvider):
        super().__init__(
            system_message="""Sei un esperto di Notion con accesso a tool specializzati.
            Puoi creare e modificare pagine, gestire database, organizzare contenuti.
            Usa sempre i tool disponibili e fornisci link alle risorse create.
            Rispondi in italiano con risultati chiari e strutturati.""",
            provider=provider
        )

        # Registra automaticamente il tool Notion
        notion_tool = NotionPageTool()
        self.register_tool(notion_tool)

    def get_agent_schema(self) -> NotionSchema:
        base_schema = super().get_agent_schema()
        return NotionSchema(
            capabilities=base_schema.capabilities,
            output_schema=base_schema.output_schema
        )

if __name__ == "__main__":
    from src.llm_providers.ibm_provider import IBMWatsonXLLm

    from dotenv import load_dotenv

    load_dotenv()

    ibm_provider = IBMWatsonXLLm(max_tokens=2000)

    agent = NotionAgent(provider=ibm_provider)

    print(agent.get_agent_schema().model_dump_json(indent=2))


    result = agent.execute_query("scrivi una pagina con scritto 'ciao' su notion")



