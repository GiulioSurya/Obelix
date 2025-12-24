"""
Chart Creator Agent
===================

Agent specializzato nella creazione di visualizzazioni grafiche da dati.
Riceve metadata sui dati (non i dati raw) e decide quali grafici creare.
"""

from src.base_agent.base_agent import BaseAgent
from src.tools.chart_creation_tool import ChartCreationTool
import pandas as pd
import os
from typing import Dict, Any, List, Optional
from src.k8s_config import YamlConfig
from src.messages import AssistantMessage, AssistantResponse


class ChartCreatorAgent(BaseAgent):
    """
    Agent specializzato nella creazione di visualizzazioni grafiche.

    Riceve:
    - DataFrame pandas (nascosto all'LLM, usato solo dal tool)
    - Metadata ricchi sui dati (visibili all'LLM nel system prompt)

    Il processo:
    1. Analizza i metadata
    2. Decide quali grafici creare
    3. Chiama il tool create_chart con parametri appropriati
    4. Puo' creare multiple visualizzazioni chiamando il tool piu' volte
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
        output_dir: str = "output/charts",
        provider = None,
        agent_name: str = None,
        description: str = None
    ):
        """
        Inizializza l'agent con DataFrame e metadata.

        Args:
            dataframe: DataFrame pandas con i dati (nascosto all'LLM)
            metadata: Metadata ricchi sui dati (visibili all'LLM)
            output_dir: Directory dove salvare i grafici generati
            provider: Provider LLM opzionale (se None usa GlobalConfig)
            agent_name: Nome dell'agent (opzionale)
            description: Descrizione dell'agent (opzionale)
        """
        self.dataframe = dataframe
        self.metadata = metadata
        self.output_dir = output_dir

        # System prompt già formattato con metadata dal chiamante (main.py)
        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_message = agents_config.get("prompts.chart_creator").format(metadata_json=self.metadata)

        # Inizializza BaseAgent con il system prompt e parametri opzionali
        super().__init__(
            system_message=system_message,
            provider=provider,
            agent_name=agent_name,
            description=description
        )

        chart_tool = ChartCreationTool(
            dataframe=self.dataframe,
            output_dir=self.output_dir
        )
        self.register_tool(chart_tool)

    def get_metadata_summary(self) -> str:
        """
        Restituisce un riepilogo testuale dei metadata per logging/debug.

        Returns:
            Stringa con riepilogo metadata
        """
        summary = []
        summary.append(f"Dataset: {self.metadata['n_rows']} rows x {self.metadata['n_columns']} columns")
        summary.append(f"Size category: {self.metadata.get('size_category', 'unknown')}")

        # Riepilogo colonne per tipo
        insights = self.metadata.get('insights', {})
        col_summary = insights.get('column_types_summary', {})

        if col_summary:
            summary.append(f"Numeric columns: {col_summary.get('total_numeric', 0)}")
            summary.append(f"Temporal columns: {col_summary.get('total_temporal', 0)}")
            summary.append(f"Categorical columns: {col_summary.get('total_categorical', 0)}")

        # Warnings
        warnings = insights.get('warnings', [])
        if warnings:
            summary.append(f"\nWarnings: {len(warnings)}")
            for warn in warnings[:3]:  # Mostra solo primi 3
                summary.append(f"  - {warn.get('message', 'Unknown warning')}")

        return "\n".join(summary)

    def _build_final_response(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[Dict[str, Any]],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        """
        Override per garantire che la tabella dati venga sempre restituita,
        anche se l'LLM non chiama il tool create_chart.
        """
        has_table = any(
            isinstance(result.get("result"), dict) and result["result"].get("data_table_html")
            for result in collected_tool_results
        )

        # Se non c'è la tabella, aggiungi manualmente
        if not has_table:
            # Recupera il ChartCreationTool dai tool registrati (è una lista)
            chart_tool = None
            for tool in self.registered_tools:
                if hasattr(tool, 'schema_class') and tool.schema_class.get_tool_name() == 'create_chart':
                    chart_tool = tool
                    break

            if chart_tool and hasattr(chart_tool, '_data_table_html') and chart_tool._data_table_html:
                table_result = {
                    "tool_name": "data_table",
                    "result": {
                        "data_table_html": chart_tool._data_table_html,
                        "message": "Dataset table (auto-generated)"
                    }
                }
                collected_tool_results.append(table_result)
                print("[INFO] Tabella dati aggiunta automaticamente (LLM non ha chiamato create_chart)")

        return super()._build_final_response(
            assistant_msg, collected_tool_results, execution_error
        )