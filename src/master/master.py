# src/agents/master_agent.py
import json
import re
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.system_message import SystemMessage
from src.messages.human_message import HumanMessage
from src.messages.assistant_message import AssistantMessage, AssistantResponse
from src.messages.standard_message import StandardMessage
from src.base_agent.base_agent import BaseAgent


@dataclass
class AgentCall:
    """Rappresenta una chiamata ad un agent specifico"""
    id: str
    agent_name: str
    query: str


class MasterAgent:
    """
    Master Agent che coordina altri agent specializzati.

    Il Master analizza query complesse, le scompone in task specifici,
    ottimizza le query per ogni agent e coordina l'esecuzione sequenziale.
    """

    def __init__(self, provider: AbstractLLMProvider):
        self.provider = provider
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.conversation_history: List[StandardMessage] = []
        self._update_system_message()

    def register_agent(self, agent: BaseAgent):
        """Registra un agent nel master e aggiorna il system prompt"""
        agent_schema = agent.get_agent_schema()
        self.registered_agents[agent_schema.name] = agent
        self._update_system_message()
        print(f"Agent {agent_schema.name} registered successfully")

    def _update_system_message(self):
        """Aggiorna il system message con gli schema degli agent correnti"""
        system_prompt = self._build_system_prompt()

        # Se esiste già un system message, lo sostituisce
        if self.conversation_history and isinstance(self.conversation_history[0], SystemMessage):
            self.conversation_history[0] = SystemMessage(content=system_prompt)
        else:
            # Altrimenti lo inserisce all'inizio
            self.conversation_history.insert(0, SystemMessage(content=system_prompt))

    def _build_system_prompt(self) -> str:
        """Costruisce il system prompt dinamico con schema degli agent"""
        base_prompt = """You are a Master Agent that coordinates specialized agents to solve complex tasks.

Your responsibilities:
1. Analyze user queries and break them into specific sub-tasks
2. Identify which agents are needed and in what order
3. Rewrite queries to be optimized for each specific agent
4. Coordinate sequential execution, passing results between agents

AVAILABLE AGENTS:
"""

        # Aggiungi schema di ogni agent registrato
        if self.registered_agents:
            for agent_name, agent in self.registered_agents.items():
                try:
                    schema = agent.get_agent_schema()
                    tools_list = [tool['name'] for tool in schema.capabilities.get('available_tools', [])]

                    base_prompt += f"""
Agent: {schema.name}
Description: {schema.description}
Available Tools: {', '.join(tools_list) if tools_list else 'None'}
Capabilities: {schema.capabilities}
---"""
                except Exception as e:
                    base_prompt += f"""
Agent: {agent_name}
Description: Schema unavailable ({e})
---"""
        else:
            base_prompt += "\nNo agents currently registered."

        base_prompt += """

AGENT CALL FORMAT:
When you need to delegate to an agent, respond with this exact JSON structure:
{
    "agent_name": "exact_agent_name_from_above",
    "query": "optimized_specific_query_for_this_agent"
}

IMPORTANT RULES:
1. Use exact agent names from the list above
2. Rewrite user queries to be specific and optimized for each agent
3. For multi-step tasks, call agents sequentially and pass results
4. If you can answer directly without agents, respond with normal text
5. Each agent call should have a clear, specific purpose

EXAMPLES:
User: "Get sales data and create a Notion report"
Your response: {"agent_name": "database-agent", "query": "extract all sales data with totals, trends and key metrics for reporting"}

Then after getting results:
{"agent_name": "notion-agent", "query": "create a comprehensive sales report page with this data: [previous results]"}
"""
        return base_prompt

    def execute_query(self, query: str, max_attempts: int = 5) -> AssistantResponse:
        """Esegue una query gestendo agent calls e conversazioni multi-turn"""
        # Aggiungi la query dell'utente alla conversazione
        user_message = HumanMessage(content=query)
        self.conversation_history.append(user_message)

        executed_agents = []

        # Loop per gestire chiamate sequenziali agli agent
        while True:
            # Chiama il provider LLM
            assistant_response = self.provider.invoke(self.conversation_history, [])

            # Aggiungi la risposta alla history
            self.conversation_history.append(assistant_response)

            # Cerca agent calls nella risposta
            agent_calls = self._parse_agent_calls(assistant_response.content)

            if agent_calls:
                # Esegui il primo agent call trovato
                agent_call = agent_calls[0]

                if agent_call.agent_name not in self.registered_agents:
                    error_msg = f"Agent '{agent_call.agent_name}' not found. Available: {list(self.registered_agents.keys())}"
                    return AssistantResponse(
                        agent_name="MasterAgent",
                        content=error_msg,
                        error=error_msg
                    )

                # Esegui l'agent
                try:
                    agent = self.registered_agents[agent_call.agent_name]
                    agent_response = agent.execute_query(agent_call.query)

                    executed_agents.append({
                        "agent_name": agent_call.agent_name,
                        "query": agent_call.query,
                        "result": agent_response.content
                    })

                    # Aggiungi il risultato alla conversazione per il prossimo step
                    result_message = HumanMessage(
                        content=f"Agent {agent_call.agent_name} completed the task. Result: {agent_response.content}"
                    )
                    self.conversation_history.append(result_message)

                    # Se l'agent ha avuto errori, propagali
                    if agent_response.error:
                        return AssistantResponse(
                            agent_name="MasterAgent",
                            content=f"Error in {agent_call.agent_name}: {agent_response.error}",
                            tool_results=executed_agents,
                            error=agent_response.error
                        )

                except Exception as e:
                    return AssistantResponse(
                        agent_name="MasterAgent",
                        content=f"Failed to execute agent {agent_call.agent_name}: {str(e)}",
                        tool_results=executed_agents,
                        error=str(e)
                    )
            else:
                # Nessun agent call trovato - questa è la risposta finale
                return AssistantResponse(
                    agent_name="MasterAgent",
                    content=assistant_response.content,
                    tool_results=executed_agents if executed_agents else None
                )

    def _parse_agent_calls(self, content: str) -> List[AgentCall]:
        """Estrae agent calls dal content usando pattern JSON"""
        if not content:
            return []

        agent_calls = []

        # Cerca pattern JSON nel content
        json_pattern = r'\{\s*"agent_name"\s*:\s*"[^"]+"\s*,\s*"query"\s*:\s*"[^"]+"\s*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                agent_data = json.loads(match)
                if "agent_name" in agent_data and "query" in agent_data:
                    agent_calls.append(AgentCall(
                        id=str(uuid.uuid4()),
                        agent_name=agent_data["agent_name"],
                        query=agent_data["query"]
                    ))
            except json.JSONDecodeError:
                continue

        # Fallback: cerca JSON più flessibile
        if not agent_calls:
            json_pattern_flexible = r'\{[^{}]*"agent_name"[^{}]*"query"[^{}]*\}'
            matches = re.findall(json_pattern_flexible, content, re.DOTALL)

            for match in matches:
                try:
                    agent_data = json.loads(match)
                    if "agent_name" in agent_data and "query" in agent_data:
                        agent_calls.append(AgentCall(
                            id=str(uuid.uuid4()),
                            agent_name=agent_data["agent_name"],
                            query=agent_data["query"]
                        ))
                except json.JSONDecodeError:
                    continue

        return agent_calls

    def get_registered_agents(self) -> Dict[str, str]:
        """Restituisce lista degli agent registrati con le loro descrizioni"""
        return {
            name: agent.get_agent_schema().description
            for name, agent in self.registered_agents.items()
        }

    def clear_conversation_history(self):
        """Pulisce la conversation history mantenendo solo il system message"""
        if self.conversation_history and isinstance(self.conversation_history[0], SystemMessage):
            self.conversation_history = [self.conversation_history[0]]
        else:
            self._update_system_message()


# Test del Master Agent
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.llm_providers.ibm_provider import IBMWatsonXLLm
    from src.llm_providers.oci_provider import OCILLm
    from src.tools.tool.calculator_tool import CalculatorTool
    from src.tools.tool.notion_tool import NotionPageTool

    load_dotenv()

    # Setup provider
    #provider = IBMWatsonXLLm(max_tokens=1000)
    oci = OCILLm(max_tokens=2000)

    # Crea Master Agent
    master = MasterAgent(oci)

    # Crea alcuni agent di test
    notion_agent = BaseAgent(
        system_message="You are a Notion specialist. Create pages and organize content.",
        provider=oci,
        agent_name="notion-agent"
    )
    notion_agent.register_tool(NotionPageTool())

    calc_agent = BaseAgent(
        system_message="You are a calculator specialist. Perform mathematical operations.",
        provider=oci,
        agent_name="calculator-agent"
    )
    calc_agent.register_tool(CalculatorTool())

    # Registra agent nel master
    master.register_agent(notion_agent)
    master.register_agent(calc_agent)

    # Test query
    test_query = "Calculate 125 * 87 and then create a Notion page with the result"

    print("=== Master Agent Test ===")
    print(f"Query: {test_query}")
    print(f"Registered agents: {master.get_registered_agents()}")

    try:
        response = master.execute_query(test_query)
        print(f"\nMaster Response:")
        print(f"Content: {response.content}")
        if response.tool_results:
            print(f"Executed agents: {len(response.tool_results)}")
            for i, agent_result in enumerate(response.tool_results):
                print(f"  {i + 1}. {agent_result['agent_name']}: {agent_result['result'][:100]}...")
        if response.error:
            print(f"Error: {response.error}")

    except Exception as e:
        print(f"Test failed: {e}")