from typing import Union, Dict, Optional, List, Any
import json
import re
import httpx
from bs4 import BeautifulSoup
from src.base_agent.base_agent import BaseAgent
from src.tools.sql_query_executor_tool import SqlQueryExecutorTool
from src.connections.db_connection import get_oracle_connection
from src.messages import HumanMessage, ToolResult, ToolStatus, AssistantMessage
from src.base_agent.middleware import AgentEvent, MiddlewareContext
from src.k8s_config import YamlConfig
import os


class SQLGeneratorAgent(BaseAgent):
    """
    Agent specializzato nella generazione di query SQL.
    Riceve la query aumentata dal QueryEnhancementAgent e genera SQL Oracle ottimizzato.
    Riceve lo schema database come oggetto nel costruttore.

    Feature: Schema injection intelligente su errori "invalid identifier"
    - Quando Oracle ritorna ORA-00904 "invalid identifier", inietta schema completo
    - Schema iniettato una sola volta per sessione (evita spam)
    """
    def __init__(
        self,
        database_schema: Union[str, Dict],
        plan: str,
        provider=None,
        agent_comment: bool = False,
    ):
        """
        Inizializza l'agent con lo schema database.

        Args:
            system_prompt: System prompt per l'agent (obbligatorio, caricato da K8sConfig).
            database_schema: Schema database come stringa SQL DDL o Dict JSON
            provider: LLM provider (opzionale, usa GlobalConfig se non specificato)
        """
        if not database_schema:
            raise ValueError("database_schema non può essere vuoto")

        self.database_schema = database_schema
        self._schema_injected_in_session = False  # Previene iniezioni multiple per sessione
        self.plan = plan
        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_prompt = agents_config.get("prompts.sql_generator")

        super().__init__(
            system_message=system_prompt,
            provider=provider,
            agent_comment=agent_comment
        )
        #injection del plan per la query sql datto dal precedente agent
        self.on(AgentEvent.ON_QUERY_START).inject_at(2, lambda ctx: AssistantMessage(content=self.plan))

        # Middleware 1: Arricchisce errori Oracle con documentazione ufficiale
        self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(self._is_oracle_error_with_docs) \
            .transform(self._enrich_with_oracle_docs)

        # Middleware 2: Inietta schema database su errori "invalid identifier"
        self.on(AgentEvent.ON_TOOL_ERROR) \
            .when(self._is_invalid_identifier_error) \
            .inject(self._create_schema_injection_message)

        sql_tool = SqlQueryExecutorTool(get_oracle_connection())
        self.register_tool(sql_tool)

    async def _fetch_oracle_error_docs(self, error_message: str) -> Optional[Dict[str, str]]:
        """
        Estrae informazioni dalla documentazione Oracle ufficiale per un errore specifico.

        Args:
            error_message: Messaggio di errore Oracle completo (contiene URL help)

        Returns:
            Dict con chiavi 'cause', 'action', 'url' se parsing ha successo, None altrimenti
        """
        # Estrai URL dalla stringa di errore
        url_pattern = r'https://docs\.oracle\.com/error-help/db/[^\s\n]+'
        url_match = re.search(url_pattern, error_message)

        if not url_match:
            return None

        url = url_match.group(0).rstrip('/')

        try:
            # HTTP GET con timeout e follow redirects
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    print(f"Oracle docs fetch failed: status {response.status_code}")
                    return None

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Estrai Cause e Action
                # Le pagine Oracle usano <h3>Cause</h3> e <h3>Action</h3>
                # seguiti da <div class="ca"><p>...</p></div>
                cause_text = None
                action_text = None

                # Cerca tutti gli h3
                for h3_tag in soup.find_all('h3'):
                    heading = h3_tag.get_text(strip=True).lower()

                    # Trova il <div class="ca"> successivo
                    ca_div = h3_tag.find_next_sibling('div', class_='ca')
                    if not ca_div:
                        continue

                    # Estrai tutto il testo dentro il div (inclusi eventuali <p>)
                    content = ca_div.get_text(strip=True)

                    if 'cause' in heading:
                        cause_text = content
                    elif 'action' in heading:
                        action_text = content

                # Return solo se abbiamo almeno uno dei due
                if cause_text or action_text:
                    return {
                        'cause': cause_text or 'N/A',
                        'action': action_text or 'N/A',
                        'url': url
                    }

                return None

        except Exception as e:
            # Fallback graceful: se fetch fallisce, non bloccare l'esecuzione
            print(f"Failed to fetch Oracle error docs: {e}")
            return None

    def _is_oracle_error_with_docs(self, ctx: MiddlewareContext) -> bool:
        """
        Condizione middleware: verifica se l'errore Oracle contiene URL documentazione.

        Args:
            ctx: Contesto middleware con tool_result

        Returns:
            True se l'errore contiene URL docs.oracle.com
        """
        result = ctx.tool_result
        return (
            result is not None and
            result.status == ToolStatus.ERROR and
            result.tool_name == "sql_query_executor" and
            result.error is not None and
            "https://docs.oracle.com" in result.error
        )

    async def _enrich_with_oracle_docs(self, result: ToolResult, ctx: MiddlewareContext) -> ToolResult:
        """
        Trasformazione middleware: arricchisce errore con documentazione Oracle.

        Args:
            result: ToolResult da arricchire
            ctx: Contesto middleware

        Returns:
            ToolResult con errore arricchito o originale
        """
        oracle_docs = await self._fetch_oracle_error_docs(result.error)

        if oracle_docs:
            enriched_error = (
                f"{result.error} | Oracle Docs - "
                f"Cause: {oracle_docs['cause']} Action: {oracle_docs['action']}"
            )
            return result.model_copy(update={"error": enriched_error})

        return result

    def _is_invalid_identifier_error(self, ctx: MiddlewareContext) -> bool:
        """
        Condizione middleware: verifica se l'errore è "invalid identifier".

        Args:
            ctx: Contesto middleware con error

        Returns:
            True se l'errore indica colonne/tabelle inesistenti e non già iniettato
        """
        return (
            ctx.error is not None and
            ("invalid identifier" in ctx.error or
             "table or view does not exist" in ctx.error) and
            not self._schema_injected_in_session
        )

    def _create_schema_injection_message(self, ctx: MiddlewareContext) -> HumanMessage:
        """
        Factory middleware: crea messaggio con schema database.

        Chiamato quando Oracle ritorna errore "invalid identifier".
        Marca la sessione per evitare iniezioni multiple.

        Args:
            ctx: Contesto middleware

        Returns:
            HumanMessage con schema database
        """
        print("[Schema Injection] Rilevato 'invalid identifier', inietto schema database")
        self._schema_injected_in_session = True

        # Formatta lo schema in base al tipo
        if isinstance(self.database_schema, dict):
            schema_content = json.dumps(self.database_schema, indent=2)
        else:
            schema_content = str(self.database_schema)

        return HumanMessage(
            content=f"""La query SQL ha generato un errore "invalid identifier".
Stai usando nomi di colonne o tabelle che non esistono nel database.
Ecco lo schema del database completo per aiutarti a correggere:

{schema_content}

Verifica attentamente i nomi delle colonne e riesegui la query con i nomi corretti."""
        )


