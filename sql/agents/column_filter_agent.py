# src/base_agent/agents/column_filter_agent.py
"""
Agent per selezionare colonne rilevanti in base alla domanda utente.
"""
from pathlib import Path
from typing import Optional, Dict, List

from src.core.agent import BaseAgent
from sql.sql_tools.column_filter_tool import ColumnFilterTool
from sql.sql_tools.ask_user_question_tool import AskUserQuestionTool
from sql.utils.schema_utils import generate_sql_schema
from src.ports.outbound import AbstractLLMProvider
from src.core.model import ToolRequirement
from src.infrastructure.k8s import YamlConfig
import os

class ColumnFilterAgent(BaseAgent):
    def __init__(
        self,
        cache_path: Path,
        provider: Optional[AbstractLLMProvider] = None,
        max_iterations: int = 5,
    ):
        schema_sql = generate_sql_schema(cache_path)
        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_message = agents_config.get("prompts.table_router").format(schema=schema_sql)

        super().__init__(
            system_message=system_message,
            provider=provider,
            max_iterations=max_iterations,
            tool_policy=[
                ToolRequirement(tool_name="column_filter", min_calls=1, require_success=True)
            ],
        )

        self.register_tool(ColumnFilterTool())
        self.register_tool(AskUserQuestionTool())

    def get_selected_columns(self, query: str) -> Dict[str, List[str]]:
        """
        Esegue l'agent e restituisce le colonne selezionate.

        Args:
            query: Domanda utente in linguaggio naturale

        Returns:
            Dict con tabelle come chiavi e liste di colonne come valori
        """
        response = self.execute_query(query)

        if response.tool_results:
            for result in response.tool_results:
                if result.tool_name == "column_filter" and result.result:
                    return result.result

        return {}
