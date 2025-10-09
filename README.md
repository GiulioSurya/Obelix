# Obelix - Multi-Provider LLM Agent Framework

**Obelix** è un framework Python per la creazione di agenti AI multi-provider che possono orchestrare diversi Large Language Models (LLM) con un sistema unificato di tool e messaggi. Il framework supporta provider multipli (IBM Watson X, OCI Generative AI) e integra il protocollo MCP (Model Context Protocol) per tool esterni.

## 🎯 Caratteristiche Principali

- **Multi-Provider**: Supporto per diversi provider LLM con switching runtime
- **Astrazione Tool**: Sistema unificato per tool personalizzati e MCP
- **Message System**: Formato standardizzato per conversation flow multi-turn
- **Agent Orchestration**: Framework per coordinamento di agenti specializzati
- **MCP Integration**: Supporto nativo per Model Context Protocol
- **Async Support**: Esecuzione asincrona di tool per performance ottimali

## 📁 Architettura del Framework

### Struttura delle Directory

```
src/
├── base_agent/                # Sistema agenti base
│   ├── base_agent.py         # BaseAgent con conversation loop
│   ├── agent_schema.py       # Schema per definizione agenti
│   └── agents/               # Agenti specializzati
├── master/                   # Orchestrazione multi-agent
│   └── master.py            # MasterAgent per coordinamento
├── messages/                 # Sistema messaggi standardizzati
│   ├── standard_message.py  # Union type per tutti i messaggi
│   ├── human_message.py     # Input utente
│   ├── system_message.py    # Istruzioni sistema
│   ├── assistant_message.py # Risposte LLM
│   └── tool_message.py      # Risultati tool execution
├── tools/                    # Sistema tool unificato
│   ├── tool_base.py         # Classe base per tool personalizzati
│   ├── tool_schema.py       # Schema Pydantic per validazione
│   ├── mcp/                 # Integrazione MCP
│   │   ├── mcp_tool.py      # Wrapper per tool MCP
│   │   ├── run_time_manager.py # Manager connessioni MCP
│   │   └── mcp_client_manager.py # Client MCP interno
│   └── tool/                # Tool personalizzati
├── llm_providers/           # Astrazione provider LLM
│   ├── llm_abstraction.py  # Classe base provider
│   ├── ibm_provider.py     # Provider IBM Watson X
│   └── oci_provider.py     # Provider OCI Generative AI
├── mapping/                 # Mapping formato provider
│   └── provider_mapping.py # Conversioni provider-specific
├── config.py               # Configurazione globale
└── providers.py            # Factory pattern provider
```

### 🏗️ Componenti Core

#### 1. **Sistema Messaggi (StandardMessage)**

Tutti i messaggi seguono un formato unificato che permette conversation flow consistenti:

```python
from src.messages.standard_message import StandardMessage
from src.messages.human_message import HumanMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage

# Esempi di messaggi
user_input = HumanMessage(content="Calcola 25 + 17")
assistant_response = AssistantMessage(content="Il risultato è 42", tool_calls=[...])
tool_result = ToolMessage(tool_results=[ToolResult(...)])
```

#### 2. **Sistema Tool Unificato**

I tool implementano un'interfaccia standardizzata con validazione automatica:

```python
from src.tools.tool_base import ToolBase
from src.tools.tool_schema import ToolSchema
from src.messages.tool_message import ToolCall, ToolResult

class MyToolSchema(ToolSchema):
    parameter1: str
    parameter2: int = 10

    @classmethod
    def get_tool_name(cls) -> str:
        return "my_tool"

    @classmethod
    def get_tool_description(cls) -> str:
        return "Tool personalizzato per operazioni specifiche"

class MyTool(ToolBase):
    schema_class = MyToolSchema

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        # Validazione automatica dei parametri
        params = MyToolSchema.model_validate(tool_call.arguments)

        # Logica del tool
        result = f"Elaborato: {params.parameter1}"

        return ToolResult(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            result=result,
            status="success"
        )
```

#### 3. **BaseAgent**

Il cuore del framework - gestisce conversation loop multi-turn con esecuzione automatica di tool:

```python
from src.base_agent.base_agent import BaseAgent
from src.providers import Providers
from src.config import GlobalConfig

# Configura provider
GlobalConfig().set_provider(Providers.OCI_GENERATIVE_AI)

# Crea agent
agent = BaseAgent(
    system_message="Sei un assistente specializzato in analisi dati",
    agent_name="DataAnalyst",
    description="Analizza e processa dati con tool specializzati"
)

# Registra tool
agent.register_tool(MyTool())

# Esegui query (gestisce automaticamente tool calls)
response = agent.execute_query("Analizza questi dati: [1,2,3,4,5]")
print(response.content)  # Risultato finale
print(response.tool_results)  # Tool utilizzati
```

#### 4. **MCP Integration**

Supporto completo per Model Context Protocol con gestione automatica delle connessioni:

```python
from src.tools.mcp.run_time_manager import MCPConfig, create_runtime_manager
from src.tools.mcp.mcp_tool import MCPTool

# Configura MCP server (esempio SQLite)
config = MCPConfig(
    name="sqlite",
    transport="stdio",
    command="uvx",
    args=["mcp-server-sqlite", "--db-path", "/path/to/database.db"]
)

# Crea manager
manager = create_runtime_manager(config)

# Crea tool MCP
sqlite_query = MCPTool("read_query", manager)
sqlite_tables = MCPTool("list_tables", manager)

# Registra nell'agent
agent.register_tool(sqlite_query)
agent.register_tool(sqlite_tables)
```

## 🚀 Quick Start

### 1. Installazione

```bash
# Clona repository
git clone <repository-url>
cd Obelix

# Installa dipendenze
pip install -r requirements.txt

# Configura variabili ambiente
cp .env.example .env
# Modifica .env con le tue credenziali
```

### 2. Configurazione Environment

Crea file `.env` con le credenziali necessarie:

```env
# IBM Watson X
IBM_WATSON_API_KEY=your_api_key
IBM_WATSON_PROJECT_ID=your_project_id

# OCI Generative AI
OCI_CONFIG_FILE=/path/to/oci/config
OCI_PROFILE=DEFAULT
OCI_COMPARTMENT_ID=your_compartment_id

# Tool esterni (opzionale)
NOTION_TOKEN=your_notion_token
TAVILY_API_KEY=your_tavily_key
```

### 3. Esempio Base

```python
from src.base_agent.base_agent import BaseAgent
from src.tools.tool.calculator_tool import CalculatorTool
from src.providers import Providers
from src.config import GlobalConfig

# Configura provider
GlobalConfig().set_provider(Providers.OCI_GENERATIVE_AI)

# Crea agent con tool
agent = BaseAgent(
    system_message="Sei un assistente matematico esperto",
    agent_name="MathAssistant"
)

# Aggiungi tool
agent.register_tool(CalculatorTool())

# Esegui query
response = agent.execute_query("Quanto fa 15 * 23 + 47?")
print(response.content)
```

## 🔧 Creazione Tool Personalizzati

### Tool Semplice

```python
from src.tools.tool_base import ToolBase
from src.tools.tool_schema import ToolSchema
from src.messages.tool_message import ToolCall, ToolResult
from pydantic import Field

class TextAnalysisSchema(ToolSchema):
    text: str = Field(description="Testo da analizzare")
    analysis_type: str = Field(
        default="sentiment",
        description="Tipo di analisi: sentiment, keywords, summary"
    )

    @classmethod
    def get_tool_name(cls) -> str:
        return "text_analysis"

    @classmethod
    def get_tool_description(cls) -> str:
        return "Analizza testo per sentiment, keyword o riassunto"

class TextAnalysisTool(ToolBase):
    schema_class = TextAnalysisSchema

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            # Validazione automatica parametri
            params = self.schema_class.model_validate(tool_call.arguments)

            # Logica del tool
            if params.analysis_type == "sentiment":
                result = self._analyze_sentiment(params.text)
            elif params.analysis_type == "keywords":
                result = self._extract_keywords(params.text)
            else:
                result = self._summarize(params.text)

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=result,
                status="success"
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status="error",
                error=str(e)
            )

    def _analyze_sentiment(self, text: str) -> dict:
        # Implementa logica sentiment analysis
        return {"sentiment": "positive", "confidence": 0.85}
```

### Tool con API Esterne

```python
import httpx
from typing import Optional

class WeatherSchema(ToolSchema):
    city: str = Field(description="Nome della città")
    country: Optional[str] = Field(default=None, description="Codice paese (opzionale)")

    @classmethod
    def get_tool_name(cls) -> str:
        return "weather_info"

    @classmethod
    def get_tool_description(cls) -> str:
        return "Ottiene informazioni meteo per una città"

class WeatherTool(ToolBase):
    schema_class = WeatherSchema

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            params = self.schema_class.model_validate(tool_call.arguments)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.openweathermap.org/data/2.5/weather",
                    params={
                        "q": f"{params.city},{params.country or ''}",
                        "appid": self.api_key,
                        "units": "metric"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "temperature": data["main"]["temp"],
                        "description": data["weather"][0]["description"],
                        "humidity": data["main"]["humidity"]
                    }
                    status = "success"
                else:
                    result = None
                    status = "error"
                    error = f"API Error: {response.status_code}"

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=result,
                status=status,
                error=error if status == "error" else None
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status="error",
                error=str(e)
            )
```

## 🔌 Configurazione MCP Server e Tool

### MCP Server Locali

#### SQLite MCP Server

```python
import os
from src.tools.mcp.run_time_manager import MCPConfig, create_runtime_manager
from src.tools.mcp.mcp_tool import MCPTool

# Configura database path
db_path = os.path.abspath("path/to/your/database.db")

# Configura MCP server
sqlite_config = MCPConfig(
    name="sqlite",
    transport="stdio",
    command="uvx",  # oppure "npx" se preferisci npm
    args=["mcp-server-sqlite", "--db-path", db_path]
)

# Crea manager
manager = create_runtime_manager(sqlite_config)

# Crea tool specifici
read_query = MCPTool("read_query", manager)
write_query = MCPTool("write_query", manager)
list_tables = MCPTool("list_tables", manager)
describe_table = MCPTool("describe_table", manager)

# Registra nell'agent
agent.register_tool(read_query)
agent.register_tool(write_query)
agent.register_tool(list_tables)
agent.register_tool(describe_table)
```

#### Filesystem MCP Server

```python
# Configura per operazioni filesystem
fs_config = MCPConfig(
    name="filesystem",
    transport="stdio",
    command="uvx",
    args=["@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
)

fs_manager = create_runtime_manager(fs_config)

# Tool filesystem
read_file = MCPTool("read_file", fs_manager)
write_file = MCPTool("write_file", fs_manager)
list_directory = MCPTool("list_directory", fs_manager)

agent.register_tool(read_file)
agent.register_tool(write_file)
agent.register_tool(list_directory)
```

### MCP Server Remoti (HTTP)

```python
# Configura server HTTP remoto (esempio Tavily)
tavily_config = MCPConfig(
    name="tavily",
    transport="streamable-http",
    url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"
)

tavily_manager = create_runtime_manager(tavily_config)
tavily_search = MCPTool("tavily_search", tavily_manager)

agent.register_tool(tavily_search)
```
`
## 🔗 Link Utili

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Server Registry](https://github.com/modelcontextprotocol/servers)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Obelix** - Framework per agenti AI multi-provider con orchestrazione intelligente 🤖