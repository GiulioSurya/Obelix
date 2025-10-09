# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Obelix** is a multi-provider LLM agent framework built in Python. It provides a unified abstraction layer for orchestrating AI agents across different LLM providers (IBM Watson X, OCI Generative AI) with standardized tool execution and message handling.

## Key Architecture Components

### 1. Provider Abstraction Layer

The framework uses a singleton-based provider system that allows runtime provider switching without code changes:

- **AbstractLLMProvider** (src/llm_providers/llm_abstraction.py): Base class using SingletonMeta for thread-safe singleton instances
- **Providers enum** (src/providers.py): Factory pattern for creating provider instances (IBM_WATSON, OCI_GENERATIVE_AI)
- **ProviderRegistry** (src/providers.py): Maps provider-specific formats for tools and messages
- **GlobalConfig** (src/config.py): Singleton managing current provider selection

### 2. Message System

All communication uses **StandardMessage** union type (src/messages/standard_message.py):

- **HumanMessage**: User inputs
- **SystemMessage**: Agent instructions
- **AssistantMessage**: LLM responses (includes content + optional tool_calls)
- **ToolMessage**: Tool execution results (contains list of ToolResult objects)

Messages flow through provider-specific mappings (src/mapping/provider_mapping.py) for format conversion.

### 3. Tool System

Tools follow an async-first architecture:

- **ToolBase** (src/tools/tool_base.py): Abstract base with `async def execute(tool_call: ToolCall) -> ToolResult`
- **ToolSchema** (src/tools/tool_schema.py): Pydantic-based schema generation
- **MCPTool** (src/tools/mcp/mcp_tool.py): Wrapper for MCP (Model Context Protocol) external tools
- **MCPRuntimeManager** (src/tools/mcp/run_time_manager.py): Manages MCP tool lifecycle and validation

Tool execution flow:
```
ToolCall → ToolBase.execute() → ToolResult → ToolMessage → conversation_history
```

### 4. Agent System

**BaseAgent** (src/base_agent/base_agent.py):
- Manages conversation history (List[StandardMessage])
- Registers tools and generates dynamic agent schemas
- Core method: `execute_query(query: str) -> AssistantResponse`
- Implements multi-turn conversation loop with automatic tool execution
- Handles retry logic (default max_attempts=5)

**MasterAgent** (src/master/master.py):
- Coordinates multiple specialized BaseAgent instances
- Parses JSON-formatted agent calls from LLM responses: `{"agent_name": "...", "query": "..."}`
- Executes agents sequentially, passing results between them
- Dynamically builds system prompts from registered agent schemas

### 5. MCP Integration

The framework supports external MCP tools via stdio and HTTP transports:

- **MCPConfig**: Transport configuration (stdio with command/args or streamable-http with URL)
- **MCPRuntimeManager**: Synchronous wrapper around async MCP client operations
- **create_runtime_manager()**: Helper function for creating configured managers
- MCP tools auto-validate parameters using Pydantic before execution

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Load environment variables from .env
python-dotenv  # loaded automatically via load_dotenv()
```

### Running the Framework

**BaseAgent example** (from main.py or base_agent.py):
```python
from src.base_agent.base_agent import BaseAgent
from src.providers import Providers
from src.config import GlobalConfig

# Set provider
GlobalConfig().set_provider(Providers.OCI_GENERATIVE_AI)

# Create agent
agent = BaseAgent(
    system_message="Your system instructions here",
    agent_name="agent-name",
    description="Agent description"
)

# Register tools
agent.register_tool(your_tool)

# Execute query
response = agent.execute_query("your query")
```

**MCP Tool setup** (from base_agent.py example):
```python
from src.tools.mcp.run_time_manager import MCPConfig, create_runtime_manager
from src.tools.mcp.mcp_tool import MCPTool

# SQLite MCP example
db_path = os.path.abspath("path/to/database.db")
sqlite_config = MCPConfig(
    name="sqlite",
    transport="stdio",
    command="uvx",
    args=["mcp-server-sqlite", "--db-path", db_path]
)

manager = create_runtime_manager(sqlite_config)
read_query_tool = MCPTool("read_query", manager)
agent.register_tool(read_query_tool)
```

**HTTP MCP example** (from main.py):
```python
config = MCPConfig(
    name="tavily",
    transport="streamable-http",
    url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
)
manager = create_runtime_manager(config)
tavily_tool = MCPTool("tavily_search", manager)
```

### Testing

Run main scripts directly:
```bash
# Test BaseAgent with MCP tools
python -m src.base_agent.base_agent

# Test provider integration
python main.py

# Test MasterAgent coordination
python -m src.master.master
```

## Critical Design Patterns

### Provider Mapping
Each provider requires mappings in `src/mapping/provider_mapping.py`:
- `tool_input.tool_schema`: Converts ToolSchema to provider format
- `tool_output.tool_calls`: Extracts tool calls from provider response
- `message_input`: Converts each StandardMessage type to provider format

### Async Execution
All tool execution is async. BaseAgent uses `asyncio.run()` internally:
```python
# In execute_query (sync)
return asyncio.run(self._async_execute_query(query))

# In _async_execute_query
result = await self._async_execute_tool(tool_call)
```

### Multi-Turn Conversation
BaseAgent implements automatic multi-turn loops:
1. Invoke provider with conversation_history + tools
2. If AssistantMessage has tool_calls → execute all tools
3. Append AssistantMessage + ToolMessage to history
4. Loop back to step 1
5. Exit when AssistantMessage has content but no tool_calls

### Dynamic Schema Generation
Agents generate schemas dynamically from registered tools:
```python
agent.get_agent_schema()  # Returns AgentSchema with capabilities and tool descriptions
```

## Environment Variables

Required in `.env`:
- Provider-specific credentials (IBM Watson X, OCI)
- API keys for external services (TAVILY_API_KEY for Tavily MCP)
- Notion credentials (for NotionPageTool)

## Project Status

Based on README.md roadmap:
- ✅ Standardized message system
- ✅ Unified tool system with async execution
- ✅ Multi-provider LLM abstraction
- ✅ BaseAgent with multi-turn conversation
- ✅ MasterAgent for agent coordination
- ✅ MCP integration with validation
- 🎯 Current phase: Testing and integration

## Common Patterns

### Adding a new LLM provider:
1. Create provider class in `src/llm_providers/` extending AbstractLLMProvider
2. Implement: `invoke()`, `_convert_messages_to_provider_format()`, `_convert_tools_to_provider_format()`, `_convert_response_to_assistant_message()`
3. Add enum value to Providers in `src/providers.py`
4. Create mapping dictionary in `src/mapping/provider_mapping.py`
5. Register with `ProviderRegistry.register(Providers.YOUR_PROVIDER, YOUR_MAPPING)`

### Adding a custom tool:
1. Create tool class extending ToolBase
2. Define schema_class as Type[ToolSchema] with tool_name and tool_description
3. Implement `async def execute(self, tool_call: ToolCall) -> ToolResult`
4. Use Pydantic models in schema_class for automatic validation

### Creating specialized agents:
1. Subclass BaseAgent or create BaseAgent instance with specific system_message
2. Register relevant tools with `agent.register_tool()`
3. Optionally register with MasterAgent for orchestration
