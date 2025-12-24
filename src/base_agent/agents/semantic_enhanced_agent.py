from src.base_agent.base_agent import BaseAgent
from src.k8s_config import YamlConfig
import os


class SemanticEnhancedAgent(BaseAgent):
    """
    Agent specializzato nella generazione di query SQL.
    Riceve la query aumentata dal QueryEnhancementAgent e genera SQL Oracle ottimizzato.
    """
    def __init__(self, provider = None):
        """
        Inizializza l'agent con le informazioni dello schema.

        Args:
        """

        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_prompt = agents_config.get("prompts.semantic_agent")
        super().__init__(system_message=system_prompt,
                         provider=provider)






