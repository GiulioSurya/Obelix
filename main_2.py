import os
import asyncio
from dotenv import load_dotenv

from src.tools.tool.calculator_tool import CalculatorTool
from src.tools.tool.notion_tool import NotionPageTool
from src.providers import Providers
from src.config import GlobalConfig
from src.llm_providers.ibm_provider import IBMWatsonXLLm
from src.llm_providers.oci_provider import OCILLm
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.standard_message import StandardMessage
from src.tools.mcp.mcp_client_manager import MCPConfig
from src.messages.tool_message import ToolMessage
from src.tools.mcp.mcp_tool import MCPTool

load_dotenv()


def run_agent(provider, messages: list[StandardMessage], tools):
    """Esegue invocazione su provider con messaggi standardizzati e multi-turn"""

    async def _internal_run():
        # Configura provider
        GlobalConfig().set_provider(Providers.OCI_GENERATIVE_AI)

        conversation_history = list(messages)

        while True:
            # 1. Invoca provider
            assistant_msg = provider.invoke(conversation_history, tools)
            print("=== PROVIDER RESPONSE ===")
            print(f"Content: {assistant_msg.content}")
            if assistant_msg.tool_calls:
                print(f"Tool calls: {assistant_msg.tool_calls}")

            # 2. Se ci sono tool calls → esegui (priorità alle tool calls)
            if assistant_msg.tool_calls:
                tool_results = []
                for call in assistant_msg.tool_calls:
                    result = await execute_tool(call, tools)
                    if result:
                        tool_results.append(result)
                        print(f"Executed {call.name}, result: {result.model_dump_json(indent=2)}")

                # Aggiungi assistant e tool results alla history e ripeti
                tool_message = ToolMessage(tool_results=tool_results)
                conversation_history.extend([assistant_msg, tool_message])
                continue

            # 3. Se non ci sono tool calls ma c'è contenuto testuale → esce dal loop
            if assistant_msg.content:
                return assistant_msg

    return asyncio.run(_internal_run())


async def execute_tool(call, tools):
    """Helper per trovare ed eseguire un tool"""
    for tool in tools:
        if tool.schema_class.get_tool_name() == call.name:
            return await tool.execute(call)
    print(f"Tool {call.name} not found")
    return None


if __name__ == "__main__":
    # === SETUP COMPLETAMENTE SINCRONO ===

    # 1. Inizializza tool locali
    calculator = CalculatorTool()
    notion = NotionPageTool()

    # 2. Setup MCP config
    tavily_config = MCPConfig(
        name="tavily",
        transport="streamable-http",
        url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
    )

    # 3. Crea tool MCP autonomo (si connette da solo)
    tavily = MCPTool("tavily_search", tavily_config)

    # 4. Inizializza provider
    oci_provider = OCILLm()
    ibm_provider = IBMWatsonXLLm()

    # 5. Definizione messaggi
    messages = [
        SystemMessage(content="""You are a precise assistant that uses available tools efficiently. Follow these guidelines:

        1. Always call the appropriate tool, retrieve data, or take actions
        2. Wait for tool results before providing your final answer
        3. Only use the tools that have been explicitly provided to you
        4. When calling tools, use the exact parameter names and types specified in the tool schema
        5. After receiving tool results, provide a clear and helpful response that incorporates those results
        6. do not summarise the research results but instead try to elaborate them in an appropriate manner
        7. always give you answer in italian
        8. If a tool call fails with an error, analyze the error message and retry the tool call with corrected parameters
        9. when include tavily never include the parameter country
        10. When using the Notion tool, leverage the extended markdown syntax to create professional, interactive content:"""),
        HumanMessage(
            content="conducimi una ricerca sull'andamento del fitmib nell'ultimo anno e scrivi un report su notion")
    ]

    # === ESECUZIONE SINCRONA ===

    print("=== Starting Agent with Tools ===")
    print(f"Available tools: {[tool.__class__.__name__ for tool in [calculator, notion, tavily]]}")

    # Chiamata sincrona - asyncio.run nascosto dentro run_agent
    result = run_agent(oci_provider, messages, [notion, tavily])

    print(f"\n=== Final Result ===")
    print(f"Assistant response: {result.content}")

#
# # === ESEMPI DI USO ALTERNATIVO ===
#
# def example_multiple_mcp_tools():
#     """Esempio con più tool MCP da server diversi"""
#
#     # Tool da Tavily
#     tavily_config = MCPConfig(
#         name="tavily",
#         transport="streamable-http",
#         url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
#     )
#     search_tool = MCPTool("tavily_search", tavily_config)
#
#     # Tool locale (esempio)
#     everything_config = MCPConfig(
#         name="everything",
#         transport="stdio",
#         command="npx",
#         args=["-y", "@modelcontextprotocol/server-everything"]
#     )
#     echo_tool = MCPTool("echo", everything_config)
#
#     # Tutti i tool sono pronti immediatamente
#     all_tools = [search_tool, echo_tool]
#     print(f"Created {len(all_tools)} MCP tools")
#
#     return all_tools
#
#
# def example_with_error_handling():
#     """Esempio con gestione errori"""
#
#     try:
#         # Config potenzialmente errata
#         bad_config = MCPConfig(
#             name="nonexistent",
#             transport="streamable-http",
#             url="https://nonexistent-server.com/mcp/"
#         )
#
#         # Questo fallirà nel __init__ se server non raggiungibile
#         bad_tool = MCPTool("some_tool", bad_config)
#
#     except Exception as e:
#         print(f"Tool creation failed: {e}")
#
#         # Fallback a tool funzionanti
#         working_config = MCPConfig(
#             name="everything",
#             transport="stdio",
#             command="npx",
#             args=["-y", "@modelcontextprotocol/server-everything"]
#         )
#         working_tool = MCPTool("echo", working_config)
#         return working_tool
#
#
# def example_config_variations():
#     """Esempi di diverse configurazioni MCP"""
#
#     configs = []
#
#     # STDIO - Server locale
#     configs.append(MCPConfig(
#         name="local_everything",
#         transport="stdio",
#         command="npx",
#         args=["-y", "@modelcontextprotocol/server-everything"]
#     ))
#
#     # HTTP - Server remoto con API key in URL
#     if os.getenv('TAVILY_API_KEY'):
#         configs.append(MCPConfig(
#             name="tavily_remote",
#             transport="streamable-http",
#             url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
#         ))
#
#     # HTTP - Server con Bearer token (esempio)
#     # configs.append(MCPConfig(
#     #     name="authenticated_server",
#     #     transport="streamable-http",
#     #     url="https://api.example.com/mcp/",
#     #     headers={"Authorization": f"Bearer {os.getenv('API_TOKEN')}"}
#     # ))
#
#     # STDIO Bridge - Server remoto via proxy locale
#     if os.getenv('TAVILY_API_KEY'):
#         configs.append(MCPConfig(
#             name="tavily_bridge",
#             transport="stdio",
#             command="npx",
#             args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"]
#         ))
#
#     print(f"Example configs: {len(configs)} different server setups")
#     return configs
#
# # Uncomment per testare esempi
# # if __name__ == "__main__":
# #     print("=== Testing Multiple Tools ===")
# #     example_multiple_mcp_tools()
# #
# #     print("\n=== Testing Error Handling ===")
# #     example_with_error_handling()
# #
# #     print("\n=== Testing Config Variations ===")
# #     example_config_variations()