from typing import Union, Dict, Optional, Any, List
import json
import os
import re
import oracledb
import httpx
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import Field
from obelix.core.agent import BaseAgent, AgentEvent, HookDecision, AgentStatus
from sql.sql_tools.sql_query_executor_tool import SqlQueryExecutorTool
from sql.database.schema.semantic_builder import SemanticSchemaBuilder
from sql.utils.dataframe.analyzer import DataFrameAnalyzer
from obelix.core.model import ToolResult, ToolStatus, ToolRequirement, SystemMessage
from obelix.ports.outbound import AbstractLLMProvider
from obelix.infrastructure.k8s import YamlConfig
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SqlAgent(BaseAgent):
    """
    Agent specializzato nella generazione e esecuzione di query SQL Oracle.

    Costruisce lo schema database arricchito a runtime tramite SemanticSchemaBuilder,
    usando la enhanced_query (dal coordinator) e il filter delle colonne (via SharedMemoryGraph
    dal ColumnFilterAgent).
    """

    enhanced_query: str = Field(
        default="",
        description=("""
        Query riformulata e ottimizzata per ricerca semantica.
        Espandi la domanda utente con sinonimi, termini affini e concetti correlati
        in una frase densa e scorrevole, eliminando parole superflue.
        questo campo sfrutta la ricerca semnantica + fuzzy per arricchire il database, più la espandi più ci sono probabilità
        che le parole chiave verranno mostrare all'agente
        esempio:
        Input: “quanti finanziamenti abbiamo preso per la sostenibilità”
        Output: “finanziamenti e contributi per iniziative legate alla sostenibilità ambientale,
         mobilità sostenibile, tutela del territorio, energia rinnovabile e progetti ecologici”
        """
        ),
    )

    def __init__(
        self,
        postgres_conn,
        oracle_pool: oracledb.ConnectionPool,
        provider: Optional[AbstractLLMProvider] = None,
    ):
        """
        Args:
            postgres_conn: Connessione PostgreSQL per SemanticSchemaBuilder (vector DB)
            oracle_pool: Pool connessioni Oracle async (iniettato via DI)
            provider: LLM provider opzionale (se None, usa GlobalConfig)
        """
        self._postgres_conn = postgres_conn

        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        self._system_prompt_template = agents_config.get("prompts.sql_agent")
        self._search_config = agents_config.get("semantic_search")

        # System prompt placeholder - verrà sostituito dal hook con lo schema arricchito
        super().__init__(
            system_message="Schema in fase di costruzione...",
            provider=provider,
            tool_policy=[
                ToolRequirement(tool_name="sql_query_executor", min_calls=1, require_success=True,
                                error_message="devi utilizzare il tool sql query executor per interrogare il database")
            ],
        )

        sql_tool = SqlQueryExecutorTool(oracle_pool)
        self.register_tool(sql_tool)
        from sql.sql_tools.ask_user_question_tool import AskUserQuestionTool
        self.register_tool(AskUserQuestionTool())

        # Hook: costruisce schema arricchito a runtime (iteration 1, dopo shared memory injection)
        self.on(AgentEvent.BEFORE_LLM_CALL) \
            .when(self._should_build_schema) \
            .handle(
                decision=HookDecision.CONTINUE,
                effects=[self._build_and_inject_schema],
            )

        # Hook: trasforma dati grezzi SQL in metadata prima della reiniezione nel modello
        self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(self._is_successful_sql_result) \
            .handle(
                decision=HookDecision.CONTINUE,
                value=self._transform_to_metadata,
            )

        # Hook: arricchisce errori Oracle con documentazione ufficiale
        self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(self._is_oracle_error_with_docs) \
            .handle(
                decision=HookDecision.CONTINUE,
                value=self._enrich_with_oracle_docs,
            )

    # ─── Schema Building (runtime) ──────────────────────────────────────────

    def _should_build_schema(self, ctx: AgentStatus) -> bool:
        return ctx.iteration == 1

    def _build_and_inject_schema(self, ctx: AgentStatus) -> None:
        """
        Costruisce lo schema arricchito a runtime:
        1. Estrae enhanced_query dalla HumanMessage (iniettata da SubAgentWrapper)
        2. Estrae schema_filter dal contesto SharedMemoryGraph (ColumnFilterAgent)
        3. Chiama SemanticSchemaBuilder.enrich_schema()
        4. Sostituisce il system message con lo schema arricchito
        """
        enhanced_query = self._extract_enhanced_query(ctx)
        schema_filter = self._extract_schema_filter(ctx)

        builder = SemanticSchemaBuilder(postgres_conn=self._postgres_conn)
        database_schema = builder.enrich_schema(
            semantic_query=enhanced_query,
            fuzzy_query=enhanced_query,
            config=self._search_config,
            filter=schema_filter,
        )
        print(database_schema)

        system_prompt = self._system_prompt_template.format(database_schema=database_schema)
        ctx.agent.conversation_history[0] = SystemMessage(content=system_prompt)

        logger.info(
            f"Schema arricchito costruito a runtime "
            f"(enhanced_query={len(enhanced_query)} chars, "
            f"filter={'yes' if schema_filter else 'no'})"
        )

    def _extract_enhanced_query(self, ctx: AgentStatus) -> str:
        """Estrae enhanced_query dalla HumanMessage (formato SubAgentWrapper: 'enhanced_query: ...')."""
        from obelix.core.model.human_message import HumanMessage

        for msg in reversed(ctx.agent.conversation_history):
            if isinstance(msg, HumanMessage) and msg.content:
                for line in msg.content.split("\n"):
                    if line.startswith("enhanced_query:"):
                        return line[len("enhanced_query:"):].strip()
                # fallback: usa l'intero contenuto come query semantica
                return msg.content
        return ""

    def _extract_schema_filter(self, ctx: AgentStatus) -> Optional[Dict[str, List[str]]]:
        """
        Estrae il filter (tabelle/colonne) dal contesto SharedMemoryGraph del ColumnFilterAgent.
        Il ColumnFilterAgent pubblica la sua final response; se contiene JSON strutturato lo parsa.
        """
        for msg in ctx.agent.conversation_history:
            if (
                isinstance(msg, SystemMessage)
                and msg.metadata.get("shared_memory_source") == "column_filter"
            ):
                content = msg.content
                # Rimuovi prefisso "Shared context from column_filter:\n"
                if ":\n" in content:
                    content = content.split(":\n", 1)[1]
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    logger.warning("SharedMemory da column_filter non è JSON, filter=None")
        return None

    # ─── SQL Result → Metadata ────────────────────────────────────────────────

    def _is_successful_sql_result(self, ctx: AgentStatus) -> bool:
        """Condizione hook: tool è sql_query_executor con status SUCCESS e dati tabellari."""
        result = ctx.tool_result
        return (
            result is not None
            and result.tool_name == "sql_query_executor"
            and result.status == ToolStatus.SUCCESS
            and isinstance(result.result, dict)
            and "data" in result.result
            and "columns" in result.result
        )

    def _transform_to_metadata(self, ctx: AgentStatus, result: ToolResult) -> ToolResult:
        """Trasforma i dati grezzi SQL in metadata analitici per il modello."""
        try:
            df = pd.DataFrame(result.result["data"], columns=result.result["columns"])
            metadata = DataFrameAnalyzer.analyze(df)
            logger.info(
                f"SQL result trasformato in metadata: "
                f"{metadata['n_rows']} righe, {metadata['n_columns']} colonne"
            )
            return result.model_copy(update={"result": metadata})
        except Exception as e:
            logger.warning(f"Errore nella trasformazione metadata, dati grezzi mantenuti: {e}")
            return result

    # ─── Oracle Error Docs ───────────────────────────────────────────────────

    async def _fetch_oracle_error_docs(self, error_message: str) -> Optional[Dict[str, str]]:
        """
        Estrae informazioni dalla documentazione Oracle ufficiale per un errore specifico.

        Args:
            error_message: Messaggio di errore Oracle completo (contiene URL help)

        Returns:
            Dict con chiavi 'cause', 'action', 'url' se parsing ha successo, None altrimenti
        """
        url_pattern = r'https://docs\.oracle\.com/error-help/db/[^\s\n]+'
        url_match = re.search(url_pattern, error_message)

        if not url_match:
            return None

        url = url_match.group(0).rstrip('/')

        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.warning(f"Oracle docs fetch failed: status {response.status_code}")
                    return None

                soup = BeautifulSoup(response.text, 'html.parser')

                cause_text = None
                action_text = None

                for h3_tag in soup.find_all('h3'):
                    heading = h3_tag.get_text(strip=True).lower()

                    ca_div = h3_tag.find_next_sibling('div', class_='ca')
                    if not ca_div:
                        continue

                    content = ca_div.get_text(strip=True)

                    if 'cause' in heading:
                        cause_text = content
                    elif 'action' in heading:
                        action_text = content

                if cause_text or action_text:
                    return {
                        'cause': cause_text or 'N/A',
                        'action': action_text or 'N/A',
                        'url': url
                    }

                return None

        except Exception as e:
            logger.warning(f"Failed to fetch Oracle error docs: {e}")
            return None

    def _is_oracle_error_with_docs(self, ctx: AgentStatus) -> bool:
        """Condizione hook: verifica se l'errore Oracle contiene URL documentazione."""
        result = ctx.tool_result
        return (
            result is not None and
            result.status == ToolStatus.ERROR and
            result.tool_name == "sql_query_executor" and
            result.error is not None and
            "https://docs.oracle.com" in result.error
        )

    async def _enrich_with_oracle_docs(self, ctx: AgentStatus, result: ToolResult) -> ToolResult:
        """Trasformazione hook: arricchisce errore con documentazione Oracle."""
        oracle_docs = await self._fetch_oracle_error_docs(result.error)

        if oracle_docs:
            enriched_error = (
                f"{result.error} | Oracle Docs - "
                f"Cause: {oracle_docs['cause']} Action: {oracle_docs['action']}"
            )
            return result.model_copy(update={"error": enriched_error})

        return result
