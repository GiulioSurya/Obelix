from typing import Union, Dict, Optional
import json
import re
import httpx
from bs4 import BeautifulSoup
from src.base_agent.base_agent import BaseAgent
from src.tools.tool.sql_query_executor_tool import SqlQueryExecutorTool
from src.connections.db_connection import get_oracle_connection
from src.messages import HumanMessage, ToolResult, ToolStatus, AssistantMessage
from src.base_agent.hooks import AgentEvent, AgentStatus
from src.k8s_config import YamlConfig
import os


class SQLGeneratorAgent(BaseAgent):
    """
    Agent specialized in SQL query generation.
    Receives the query enhanced by QueryEnhancementAgent and generates optimized Oracle SQL.
    Receives the database schema as an object in the constructor.

    Feature: Intelligent schema injection on "invalid identifier" errors
    - When Oracle returns ORA-00904 "invalid identifier", injects complete schema
    - Schema injected only once per session (avoids spam)
    """
    def __init__(
        self,
        database_schema: Union[str, Dict],
        plan: str,
        provider=None,
        agent_comment: bool = False,
    ):
        """
        Initialize the agent with the database schema.

        Args:
            system_prompt: System prompt for the agent (required, loaded from K8sConfig).
            database_schema: Database schema as SQL DDL string or JSON Dict
            provider: LLM provider (optional, uses GlobalConfig if not specified)
        """
        if not database_schema:
            raise ValueError("database_schema cannot be empty")

        self.database_schema = database_schema
        self._schema_injected_in_session = False  # Prevents multiple injections per session
        self.plan = plan
        agents_config = YamlConfig(os.getenv("CONFIG_PATH"))
        system_prompt = agents_config.get("prompts.sql_generator")

        super().__init__(
            system_message=system_prompt,
            provider=provider,
            agent_comment=agent_comment
        )
        # Inject the plan for the SQL query from the previous agent
        self.on(AgentEvent.ON_QUERY_START).inject_at(2, lambda agent_status: AssistantMessage(content=self.plan))

        # Hook 1: Enriches Oracle errors with official documentation
        self.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(self._is_oracle_error_with_docs) \
            .transform(self._enrich_with_oracle_docs)

        # Hook 2: Injects database schema on "invalid identifier" errors
        self.on(AgentEvent.ON_TOOL_ERROR) \
            .when(self._is_invalid_identifier_error) \
            .inject(self._create_schema_injection_message)

        sql_tool = SqlQueryExecutorTool(get_oracle_connection())
        self.register_tool(sql_tool)

    async def _fetch_oracle_error_docs(self, error_message: str) -> Optional[Dict[str, str]]:
        """
        Extracts information from official Oracle documentation for a specific error.

        Args:
            error_message: Complete Oracle error message (contains help URL)

        Returns:
            Dict with keys 'cause', 'action', 'url' if parsing succeeds, None otherwise
        """
        # Extract URL from error string
        url_pattern = r'https://docs\.oracle\.com/error-help/db/[^\s\n]+'
        url_match = re.search(url_pattern, error_message)

        if not url_match:
            return None

        url = url_match.group(0).rstrip('/')

        try:
            # HTTP GET with timeout and follow redirects
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    print(f"Oracle docs fetch failed: status {response.status_code}")
                    return None

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract Cause and Action
                # Oracle pages use <h3>Cause</h3> and <h3>Action</h3>
                # followed by <div class="ca"><p>...</p></div>
                cause_text = None
                action_text = None

                # Search all h3 tags
                for h3_tag in soup.find_all('h3'):
                    heading = h3_tag.get_text(strip=True).lower()

                    # Find the next <div class="ca">
                    ca_div = h3_tag.find_next_sibling('div', class_='ca')
                    if not ca_div:
                        continue

                    # Extract all text inside the div (including any <p> tags)
                    content = ca_div.get_text(strip=True)

                    if 'cause' in heading:
                        cause_text = content
                    elif 'action' in heading:
                        action_text = content

                # Return only if we have at least one of the two
                if cause_text or action_text:
                    return {
                        'cause': cause_text or 'N/A',
                        'action': action_text or 'N/A',
                        'url': url
                    }

                return None

        except Exception as e:
            # Graceful fallback: if fetch fails, don't block execution
            print(f"Failed to fetch Oracle error docs: {e}")
            return None

    def _is_oracle_error_with_docs(self, agent_status: AgentStatus) -> bool:
        """
        Hook condition: checks if the Oracle error contains documentation URL.

        Args:
            agent_status: Agent status with tool_result

        Returns:
            True if the error contains docs.oracle.com URL
        """
        result = agent_status.tool_result
        return (
            result is not None and
            result.status == ToolStatus.ERROR and
            result.tool_name == "sql_query_executor" and
            result.error is not None and
            "https://docs.oracle.com" in result.error
        )

    async def _enrich_with_oracle_docs(self, result: ToolResult, agent_status: AgentStatus) -> ToolResult:
        """
        Hook transformation: enriches error with Oracle documentation.

        Args:
            result: ToolResult to enrich
            agent_status: Agent status

        Returns:
            ToolResult with enriched error or original
        """
        oracle_docs = await self._fetch_oracle_error_docs(result.error)

        if oracle_docs:
            enriched_error = (
                f"{result.error} | Oracle Docs - "
                f"Cause: {oracle_docs['cause']} Action: {oracle_docs['action']}"
            )
            return result.model_copy(update={"error": enriched_error})

        return result

    def _is_invalid_identifier_error(self, agent_status: AgentStatus) -> bool:
        """
        Hook condition: checks if the error is "invalid identifier".

        Args:
            agent_status: Agent status with error

        Returns:
            True if the error indicates non-existent columns/tables and not already injected
        """
        return (
            agent_status.error is not None and
            ("invalid identifier" in agent_status.error or
             "table or view does not exist" in agent_status.error) and
            not self._schema_injected_in_session
        )

    def _create_schema_injection_message(self, agent_status: AgentStatus) -> HumanMessage:
        """
        Factory hook: creates message with database schema.

        Called when Oracle returns "invalid identifier" error.
        Marks the session to prevent multiple injections.

        Args:
            agent_status: Agent status

        Returns:
            HumanMessage with database schema
        """
        print("[Schema Injection] Detected 'invalid identifier', injecting database schema")
        self._schema_injected_in_session = True

        # Format the schema based on type
        if isinstance(self.database_schema, dict):
            schema_content = json.dumps(self.database_schema, indent=2)
        else:
            schema_content = str(self.database_schema)

        return HumanMessage(
            content=f"""The SQL query generated an "invalid identifier" error.
You are using column or table names that do not exist in the database.
Here is the complete database schema to help you correct the issue:

{schema_content}

Carefully verify the column names and re-execute the query with the correct names."""
        )


