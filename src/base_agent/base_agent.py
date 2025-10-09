# src/agents/base_agent.py
import asyncio
from typing import List, Optional, Dict, Any


from src.base_agent.agent_schema import AgentSchema, ToolDescription
from src.messages.system_message import SystemMessage
from src.messages.human_message import HumanMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage, ToolCall, ToolResult
from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantResponse  # Nuovo import
from src.tools.tool_base import ToolBase
from src.tools.mcp.mcp_tool import MCPTool
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.config import GlobalConfig
from src.providers import Providers
from src.mapping.provider_mapping import ProviderRegistry
from pydantic import BaseModel

from typing import Optional


class BaseAgent:
    def __init__(self,
                 system_message: str,
                 provider: Optional[AbstractLLMProvider] = None,
                 agent_name: str = None,
                 description: str = None):
        self.system_message = SystemMessage(content=system_message)

        # Se non viene passato un provider, usa quello dal GlobalConfig
        self.provider = provider or GlobalConfig().get_current_provider().create_instance()

        self.agent_name = agent_name or self.__class__.__name__
        self.description = description if description is not None else "AI assistant for specialized task execution"
        self.registered_tools: List[ToolBase] = []
        self.conversation_history: List[StandardMessage] = [self.system_message]

    def register_tool(self, tool: ToolBase):
        """
        Registra un tool per l'agent

        Args:
            tool: Tool da registrare (ToolBase o MCPTool)

        Raises:
            RuntimeError: Se il tool MCP non è connesso
        """
        # Se è un MCPTool, verifica che il manager sia connesso
        if isinstance(tool, MCPTool):
            if not tool.manager.is_connected():
                raise RuntimeError(f"MCP Tool {tool.tool_name} non è connesso. "
                                   f"Il manager deve essere connesso prima della registrazione.")

        # Registra il tool
        if tool not in self.registered_tools:
            self.registered_tools.append(tool)

    def get_agent_schema(self) -> AgentSchema:
        tool_descriptions = []

        for tool in self.registered_tools:
            try:
                schema = tool.create_schema()
                parameter_keys = []
                if hasattr(schema, 'inputSchema') and schema.inputSchema:
                    properties = schema.inputSchema.get('properties', {})
                    parameter_keys = list(properties.keys())

                tool_descriptions.append(ToolDescription(
                    name=schema.name,
                    description=schema.description or f"Tool {schema.name}",
                    parameters=parameter_keys
                ))
            except Exception as e:
                tool_name = getattr(tool, 'tool_name', tool.__class__.__name__)
                tool_descriptions.append(ToolDescription(
                    name=tool_name,
                    description=f"Tool {tool_name} (schema non disponibile: {e})",
                    parameters=[]
                ))

        # Genera output schema dinamico dai tool
        dynamic_output_schema = self.generate_unified_output_schema()

        return AgentSchema(
            name=self.agent_name,
            description=self.description,
            capabilities={
                "tools": len(tool_descriptions) > 0,
                "available_tools": [tool.model_dump() for tool in tool_descriptions]
            },
            output_schema=dynamic_output_schema
        )

    def generate_unified_output_schema(self) -> Dict[str, Any]:
        """Schema generico che include outputs di tutti i tool"""
        tool_outputs = {}

        for tool in self.registered_tools:
            try:
                tool_schema = tool.create_schema()
                if hasattr(tool_schema, 'outputSchema') and tool_schema.outputSchema:
                    tool_outputs[tool_schema.name] = tool_schema.outputSchema
            except Exception as e:
                # Ignora tool che non riescono a creare lo schema
                continue

        properties = {
            "result": {"type": "string", "description": "Risultato principale"}
        }

        # Aggiungi tool_outputs solo se ci sono tool
        if tool_outputs:
            properties["tool_outputs"] = {
                "type": "object",
                "properties": tool_outputs,
                "description": "Output specifici dei tool utilizzati"
            }

        return {
            "type": "object",
            "properties": properties
        }

    def execute_query(self, query: str, max_attempts: int = 5) -> AssistantResponse:
        """
        Esegue una query con retry automatico (metodo esposto)

        Args:
            query: Query da eseguire
            max_attempts: Numero massimo di tentativi

        Returns:
            AssistantResponse: Risposta strutturata dell'agent

        Raises:
            RuntimeError: Se tutti i tentativi falliscono
        """
        for attempt in range(max_attempts):
            try:
                return asyncio.run(self._async_execute_query(query))
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Esecuzione fallita dopo {max_attempts} tentativi. Ultimo errore: {e}")
                print(f"Tentativo {attempt + 1} fallito: {e}. Riprovo...")

        raise RuntimeError("Esecuzione fallita per motivi sconosciuti")

    async def _async_execute_query(self, query: str) -> AssistantResponse:
        """
        Esecuzione asincrona della query (cuore dell'esecuzione)

        Args:
            query: Query da eseguire

        Returns:
            AssistantResponse: Risposta strutturata dell'agent
        """
        # Lista per raccogliere i risultati dei tool eseguiti e variabile per errori
        collected_tool_results = []
        execution_error = None

        # Aggiungi la query dell'utente alla conversazione
        user_message = HumanMessage(content=query)
        self.conversation_history.append(user_message)

        # Loop multi-turn per gestire tool calls
        while True:
            # 1. Invoca il provider LLM
            assistant_msg = self.provider.invoke(self.conversation_history, self.registered_tools)

            # 2. Se ci sono tool calls → esegui (priorità alle tool calls)
            if assistant_msg.tool_calls:
                print(f"{assistant_msg.tool_calls}")
                tool_results = []
                for call in assistant_msg.tool_calls:
                    result = await self._async_execute_tool(call)
                    if result:
                        tool_results.append(result)
                        print(f"Executed {call.name}, result: {result.model_dump_json(indent=2)}")

                        # Raccogli i dati essenziali per la risposta finale
                        tool_summary = {"tool_name": result.tool_name}
                        if result.status == "success" and result.result is not None:
                            tool_summary["result"] = result.result
                        if result.error:
                            tool_summary["error"] = result.error
                            # Se c'è un errore nel tool, lo tracciamo (solo il primo)
                            if not execution_error:
                                execution_error = f"Errore in {result.tool_name}: {result.error}"
                        collected_tool_results.append(tool_summary)

                # Aggiungi assistant e tool results alla history e ripeti
                tool_message = ToolMessage(tool_results=tool_results)
                self.conversation_history.extend([assistant_msg, tool_message])
                continue

            # 3. Se non ci sono tool calls ma c'è contenuto testuale → esce dal loop
            if assistant_msg.content:
                print(f"Assistant response: {assistant_msg.content}")
                self.conversation_history.append(assistant_msg)

                # Crea e restituisce AssistantResponse strutturata
                return AssistantResponse(
                    agent_name=self.agent_name,
                    content=assistant_msg.content,
                    tool_results=collected_tool_results if collected_tool_results else None,
                    error=execution_error
                )

            # 4. Se non c'è né contenuto né tool calls, considera errore
            raise RuntimeError("L'assistant ha restituito una risposta vuota")

    async def _async_execute_tool(self, tool_call: ToolCall) -> Optional[ToolResult]:
        """
        Helper per l'esecuzione asincrona dei tool

        Args:
            tool_call: Chiamata al tool da eseguire

        Returns:
            ToolResult: Risultato dell'esecuzione del tool, None se tool non trovato
        """
        for tool in self.registered_tools:
            if tool.schema_class.get_tool_name() == tool_call.name:
                try:
                    return await tool.execute(tool_call)
                except Exception as e:
                    # Crea un ToolResult con errore invece di fallire completamente
                    return ToolResult(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        result=None,
                        status="error",
                        error=f"Errore esecuzione tool: {e}"
                    )

        print(f"Tool {tool_call.name} non trovato tra i tool registrati")
        return None

    def get_conversation_history(self) -> List[StandardMessage]:
        """
        Restituisce la cronologia della conversazione

        Returns:
            List[StandardMessage]: Lista dei messaggi della conversazione
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self, keep_system_message: bool = True):
        """
        Pulisce la cronologia della conversazione

        Args:
            keep_system_message: Se mantenere il messaggio di sistema
        """
        if keep_system_message:
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []

if __name__ == "__main__":
    import os
    import asyncio
    from dotenv import load_dotenv

    from src.tools.tool.calculator_tool import CalculatorTool
    from src.tools.tool.notion_tool import NotionPageTool
    from src.providers import Providers
    from src.config import GlobalConfig
    from src.messages.human_message import HumanMessage
    from src.tools.mcp.mcp_tool import MCPTool

    # Inizializza tool locali
    from src.tools.mcp.run_time_manager import MCPConfig, create_runtime_manager
    import os
    from src.tools.mcp.mcp_tool import MCPTool

    GlobalConfig().set_provider(Providers.OCI_GENERATIVE_AI)

    load_dotenv()



    db_path = os.path.abspath(r"C:\Users\GLoverde\PycharmProjects\Obelix\spesa.db")

    sqlite_config = MCPConfig(
        name="sqlite",
        transport="stdio",
        command="uvx",
        args=["mcp-server-sqlite", "--db-path", db_path]
    )



    sqlite_manager = create_runtime_manager(sqlite_config)

    # 4. Crea tool multipli dallo stesso manager
    # Prima vedi che tool sono disponibili:
    tools = sqlite_manager.get_tools()

    for tool in tools:
        print(f"Tool disponibile: {tool.name}")

    # 5. Crea MCPTool per ogni tool che vuoi usare
    read_query = MCPTool("read_query", sqlite_manager)
    write_query = MCPTool("write_query", sqlite_manager)
    create_table = MCPTool("create_table", sqlite_manager)
    list_tables = MCPTool("list_tables", sqlite_manager)
    describe_table = MCPTool("describe_table", sqlite_manager)

    calculator = CalculatorTool()
    notion = NotionPageTool()


    # Inizializza provider
    #ibm_provider = IBMWatsonXLLm(max_tokens=2000)
    #oci_provider = OCILLm(max_tokens=2000)

    # Create agent (sync)
    agent = BaseAgent(
        system_message="""You are an expert SQLite database analyst with specialized tools for querying databases.

                        CRITICAL: You have ZERO knowledge about this specific database structure. You must discover it using your tools.

                        MANDATORY WORKFLOW - Follow these steps in EXACT order:

                        STEP 1 (REQUIRED): ALWAYS call 'list_tables' first - you cannot skip this step
                        STEP 2: Call 'describe_table' on relevant tables to understand their structure  
                        STEP 3: Only then write and execute your SQL query with 'read_query'
                        STEP 4: Present results in readable format to the user

                        IMPORTANT RULES:
                        - NEVER attempt to write SQL without first using list_tables and describe_table
                        - Do NOT assume table names, column names, or database structure
                        - The user's question may contain incorrect table/column names - ignore them and discover the real structure
                        - Always respond in Italian in your final answer
                        - Present data clearly, not raw JSON or SQL code

                        You have no prior knowledge of this database. Start every query by exploring the database structure first.""",
        agent_name="SQLite Database Analyst",
        description="intelligent guy",
    )

    # Register tools (sync)
    # agent.register_tool(calculator)
    agent.register_tool(notion)

    agent.register_tool(read_query)
    agent.register_tool(write_query)
    agent.register_tool(list_tables)
    agent.register_tool(describe_table)

    print(agent.get_agent_schema().model_dump_json(indent=2))


    messages = "ho bisogno di sapere Volume di pagamenti per esercizio e titolo e per gli anni dal 2025 al 2027"
    #messages = """controlla nel database e dopo crea un tabella su notion con una riga per ogni prodotto e come colonne: prima colonna stock attuale, saldo in e out  degli ordini e colonna 3 lo stock di ogni prodotto dopo che gli ordini saranno stati evasi"""

    response = agent.execute_query(messages)

    aggiungi = agent.execute_query("puoi aggiungere la descrizione dei titoli? poi crea un report su notion")

    # print(f"Agent Name: {response.agent_name}")
    # print(f"Final Content: {response.content}")
    #
    # # Controlla prima se ci sono errori
    # if response.error:
    #     print(f"\nError durante l'esecuzione: {response.error}")
    #
    # # Poi mostra i tool utilizzati
    # if response.tool_results:
    #     print(f"\nTools utilizzati: {len(response.tool_results)}")
    #     for tool_result in response.tool_results:
    #         status = "Success" if 'result' in tool_result else "Error"
    #         print(f"- {tool_result['tool_name']}: {status}")
    # else:
    #     print("\nNessun tool utilizzato")
    #
    # # Cleanup automatico alla fine del processo - no manual disconnect needed

    import cx_Oracle
    import json


    class OracleInspector:
        def __init__(self, connection):
            self.connection = connection

        def execute_query(self, query, params=None):
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or {})
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        # === LIST TABLES ===
        def list_tables(self):
            """
            Returns all user tables and views with their type and optional comments.
            """
            query = """
                    SELECT object_name AS name, \
                           object_type AS type, \
                           comments    AS description
                    FROM (SELECT table_name AS object_name, 'TABLE' AS object_type \
                          FROM user_tables \
                          UNION ALL \
                          SELECT view_name AS object_name, 'VIEW' AS object_type \
                          FROM user_views) objs
                             LEFT JOIN user_tab_comments com
                                       ON objs.object_name = com.table_name
                    ORDER BY object_name \
                    """
            return self.execute_query(query)

        # === DESCRIBE TABLE ===
        def describe_table(self, table_name, sample_analysis=True):
            """
            Returns detailed column metadata for a table or view.
            If sample_analysis=True, performs lightweight data profiling for each column.
            """
            table_name = table_name.upper()

            # Base column metadata query
            base_query = """
                         SELECT c.column_name, \
                                c.data_type, \
                                c.data_length, \
                                c.data_precision, \
                                c.data_scale, \
                                c.nullable, \
                                c.data_default, \
                                cc.comments                                                   AS column_comment, \
                                CASE WHEN pk.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END AS is_primary_key
                         FROM user_tab_columns c
                                  LEFT JOIN user_col_comments cc
                                            ON c.table_name = cc.table_name
                                                AND c.column_name = cc.column_name
                                  LEFT JOIN (SELECT cols.column_name \
                                             FROM user_constraints cons \
                                                      JOIN user_cons_columns cols ON cons.constraint_name = cols.constraint_name \
                                             WHERE cons.constraint_type = 'P' \
                                               AND cons.table_name = :table_name) pk ON c.column_name = pk.column_name
                         WHERE c.table_name = :table_name
                         ORDER BY c.column_id \
                         """
            columns = self.execute_query(base_query, {"table_name": table_name})

            # Optional: enrich with profiling info
            if sample_analysis:
                for col in columns:
                    col_name = col["COLUMN_NAME"]
                    dtype = col["DATA_TYPE"].upper()
                    profile = self._profile_column(table_name, col_name, dtype)
                    col.update(profile)

            return columns

        # === COLUMN PROFILING ===
        def _profile_column(self, table_name, column_name, data_type):
            """
            Performs simple semantic profiling for a column:
            - detects categorical (few distinct values)
            - detects continuous numeric columns (has wide range)
            - computes min, max, and sample of categories
            """
            cursor = self.connection.cursor()
            profile = {}

            if "CHAR" in data_type or "CLOB" in data_type or "TEXT" in data_type:
                # Text / categorical detection
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT {column_name}) AS distinct_count,
                           COUNT(*) AS total_count
                    FROM {table_name}
                """)
                row = cursor.fetchone()
                distinct, total = row
                ratio = distinct / total if total else 0
                if distinct < 30 or ratio < 0.1:
                    # Likely categorical
                    cursor.execute(f"""
                        SELECT DISTINCT {column_name} FROM {table_name}
                        WHERE {column_name} IS NOT NULL
                        FETCH FIRST 10 ROWS ONLY
                    """)
                    categories = [r[0] for r in cursor.fetchall()]
                    profile.update({
                        "semantic_type": "categorical",
                        "categories_sample": categories,
                        "distinct_count": distinct
                    })
                else:
                    profile.update({"semantic_type": "text", "distinct_count": distinct})

            elif "NUMBER" in data_type or "FLOAT" in data_type or "INT" in data_type or "DECIMAL" in data_type:
                # Numeric profiling
                cursor.execute(f"""
                    SELECT MIN({column_name}), MAX({column_name}), COUNT(DISTINCT {column_name})
                    FROM {table_name}
                """)
                min_val, max_val, distinct = cursor.fetchone()
                continuous = (distinct > 30)
                profile.update({
                    "semantic_type": "continuous" if continuous else "discrete_numeric",
                    "min": min_val,
                    "max": max_val,
                    "distinct_count": distinct
                })

            elif "DATE" in data_type or "TIMESTAMP" in data_type:
                cursor.execute(f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name}")
                min_val, max_val = cursor.fetchone()
                profile.update({
                    "semantic_type": "temporal",
                    "min_date": str(min_val),
                    "max_date": str(max_val)
                })

            cursor.close()
            return profile

        def close(self):
            self.connection.close()
