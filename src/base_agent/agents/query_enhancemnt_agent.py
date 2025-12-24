from typing import Union, Dict, Optional
import os
from src.base_agent.base_agent import BaseAgent
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.k8s_config import YamlConfig

class QueryEnhancementAgent(BaseAgent):
    """
    Agent specializzato nel miglioramento delle query SQL.
    Riceve lo schema database come oggetto nel costruttore.
    """
    def __init__(
        self,
        database_schema: Union[str, Dict],
        examples: Union[str, Dict],
        provider: Optional[AbstractLLMProvider] = None
    ):
        """
        Inizializza l'agent con lo schema database.

        Args:
            system_prompt: System prompt per l'agent (obbligatorio, già formattato con placeholder da K8sConfig).
            database_schema: Schema database come stringa SQL DDL o Dict JSON
            query: Query utente (opzionale, richiesta per semantic search)
            semantic_query: Query semplificata per semantic search (se None, non usa esempi)
            num_examples: Numero di esempi da recuperare (default: 2, usato solo se semantic_query è fornito)
            provider: LLM provider opzionale (se None, usa GlobalConfig)
        """
        if not database_schema or (isinstance(database_schema, str) and not database_schema.strip()):
            raise ValueError("database_schema non può essere vuoto")

        self.database_schema = database_schema
        self.examples = examples

        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_prompt = agents_config.get("prompts.query_enhancement").format(
            database_schema=self.database_schema
        )

        super().__init__(system_message=system_prompt, provider=provider)
