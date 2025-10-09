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
from src.tools.mcp.run_time_manager import MCPConfig, MCPRuntimeManager
from src.messages.tool_message import ToolMessage
from src.tools.mcp.mcp_tool import MCPTool

load_dotenv()

async def run_agent(provider, messages: list[StandardMessage], tools):
    """Esegue invocazione su provider con messaggi standardizzati e multi-turn"""

    # Configura provider
    GlobalConfig().set_provider(Providers.IBM_WATSON)

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

async def execute_tool(call, tools):
    """Helper per trovare ed eseguire un tool"""
    for tool in tools:
        if tool.schema_class.get_tool_name() == call.name:
            return await tool.execute(call)
    print(f"Tool {call.name} not found")
    return None

if __name__ == "__main__":
    # Inizializza tool locali
    from src.tools.mcp.run_time_manager import MCPConfig, create_runtime_manager
    import os
    from src.tools.mcp.mcp_tool import MCPTool

    config = MCPConfig(
        name="tavily",
        transport="streamable-http",
        url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
    )
    manager = create_runtime_manager(config)
    tavily = MCPTool("tavily_search", manager)
    calculator = CalculatorTool()
    notion = NotionPageTool()

    # Inizializza provider
    oci_provider = OCILLm()
    ibm_provider = IBMWatsonXLLm()

    # Test standard con Calculator, Notion e Tavily
    async def test_with_tools():
        # Definizione messaggi
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
            10. When using the Notion tool, leverage the extended markdown syntax to create professional, interactive content:""") ,
            HumanMessage(content="ho bisogno di un report dettagliato sulla situazizone del manazzino: abbiamo qualcosa sotto il livello minimo? fai un report su notion")]

        try:
            await run_agent(ibm_provider, messages, [notion, tavily])
        finally:
            # Shutdown DOPO il test, non prima!
            manager.shutdown()

    asyncio.run(test_with_tools())