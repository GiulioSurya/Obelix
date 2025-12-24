"""
Table Router Agent
==================

Agent specializzato nella selezione delle tabelle database tramite LLM.
Sostituisce il router basato su semantic search + fuzzy matching con un
approccio basato su reasoning LLM.
"""

from typing import Optional
import os

from src.base_agent.base_agent import BaseAgent
from src.tools.table_selection_tool import SelectTablesTool
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.k8s_config import YamlConfig


class TableRouterAgent(BaseAgent):
    """
    Agent per la selezione intelligente delle tabelle database.

    Utilizza un LLM per ragionare su quale/quali tabelle Oracle usare
    basandosi sulla query dell'utente e sulle descrizioni delle tabelle.

    Il processo:
    1. Riceve la query utente in linguaggio naturale
    2. L'LLM analizza la query e ragiona su quali tabelle servono
    3. L'LLM chiama il tool select_tables con reasoning + selected_tables
    4. Il tool valida i nomi e ritorna la lista delle tabelle

    Example:
        >>> from src.base_agent.agents import TableRouterAgent
        >>> from src.llm_providers.oci_provider import OCILLm
        >>> from src.k8s_config import YamlConfig
        >>> import os
        >>>
        >>> agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        >>> router_cfg = agents_config.get("agents.table_router")
        >>> provider = OCILLm(
        ...     model_id=router_cfg["model_id"],
        ...     temperature=router_cfg["temperature"],
        ...     max_tokens=router_cfg["max_tokens"],
        ...     tool_choice=router_cfg.get("tool_choice")
        ... )
        >>>
        >>> agent = TableRouterAgent(provider=provider)
        >>> response = agent.execute_query("mostrami le entrate del 2024")
        >>> tables = response.tool_results[0]["result"]["tables"]
        >>> print(tables)  # ['VISTA_BILANCIO_ENTRATA_AI']
    """

    def __init__(
        self,
        provider: Optional[AbstractLLMProvider] = None,
        agent_name: str = "TableRouterAgent",
        description: str = "Agent per la selezione delle tabelle database Oracle",
        agent_comment: bool = False
    ):
        """
        Inizializza l'agent con il tool di selezione tabelle.

        Args:
            provider: Provider LLM (se None usa GlobalConfig)
            agent_name: Nome dell'agent
            description: Descrizione dell'agent
        """
        # Crea il tool (carica schema cache)
        self._select_tool = SelectTablesTool()

        # Costruisci system prompt con info tabelle
        system_prompt = self._build_system_prompt()

        # Inizializza BaseAgent
        super().__init__(
            system_message=system_prompt,
            provider=provider,
            agent_name=agent_name,
            description=description,
            agent_comment = agent_comment
        )

        # Registra il tool
        self.register_tool(self._select_tool)

    def _build_system_prompt(self) -> str:
        """
        Costruisce il system prompt con le informazioni delle tabelle.

        Usa K8sConfig per caricare il template del prompt e inietta
        le informazioni sulle tabelle dal JSON cache.

        Returns:
            System prompt completo con tabelle disponibili
        """
        # Ottieni info tabelle formattate dal tool
        tables_info = self._select_tool.get_tables_info_formatted()

        # Carica template prompt da config
        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        prompt = agents_config.get("prompts.table_router").format(
            tables_info=tables_info
        )

        return prompt