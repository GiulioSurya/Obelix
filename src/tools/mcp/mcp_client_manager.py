# src/mcp/mcp_client_manager.py
import asyncio
import threading
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool, CallToolResult, Resource, Prompt
from .mcp_validator import MCPValidator, MCPValidationError


@dataclass
class MCPConfig:
    """
    Configurazione per connessione MCP.

    Supporta due transport principali:
    - stdio: Per server locali (subprocess)
    - streamable-http: Per server remoti (HTTP)

    Examples:
        # STDIO - Server locale con subprocess
        config = MCPConfig(
            name="local_server",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            env={"API_KEY": "your-key"}  # Variabili ambiente per il processo
        )

        # STREAMABLE-HTTP - Server remoto con API key nell'URL
        config = MCPConfig(
            name="tavily",
            transport="streamable-http",
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        )

        # STREAMABLE-HTTP - Server remoto con autenticazione Bearer
        config = MCPConfig(
            name="github_copilot",
            transport="streamable-http",
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_token}"}
        )

        # STDIO BRIDGE - Server remoto tramite mcp-remote
        config = MCPConfig(
            name="tavily_bridge",
            transport="stdio",
            command="npx",
            args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"]
        )
    """
    name: str
    transport: str = "stdio"  # stdio, streamable-http

    # STDIO params - per server locali
    command: Optional[str] = None  # Comando da eseguire (es. "npx", "python")
    args: Optional[List[str]] = None  # Argomenti comando (es. ["-y", "server.js"])
    env: Optional[Dict[str, str]] = None  # Variabili ambiente per subprocess

    # Streamable HTTP params - per server remoti
    url: Optional[str] = None  # URL endpoint MCP (es. "https://api.example.com/mcp/")
    headers: Optional[Dict[str, str]] = None  # Headers HTTP (es. Authorization)

    def get_key(self) -> tuple:
        """Genera chiave univoca per singleton"""
        if self.transport == "stdio":
            return self.name, self.transport, self.command, tuple(self.args or [])
        else:
            return self.name, self.transport, self.url


class MCPClientManager:
    """
    Manager universale per connessioni MCP con validazione integrata.

    Gestisce connessioni sia stdio che streamable-http con pattern singleton.
    Supporta tutte le funzionalità MCP: tools, resources, prompts, session management.

    Features:
        - Singleton pattern per riutilizzo connessioni
        - Support stdio (subprocess locali) e streamable-http (server remoti)
        - Gestione automatica sessioni HTTP con session ID
        - Loading completo capabilities (tools, resources, prompts)
        - Configuration loading da file mcp.json
        - Environment variables support per API keys
        - NUOVO: Validazione automatica argomenti con MCPValidator

    Usage:
        # Modalità 1: Configurazione diretta
        config = MCPConfig(
            name="tavily",
            transport="streamable-http",
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        )
        manager = MCPClientManager(config)

        # Modalità 2: Da file di configurazione
        manager = MCPClientManager.from_config_file("mcp.json", "server_name")

        # Modalità 3: Backward compatible
        manager = MCPClientManager("everything", "npx", ["-y", "server-everything"])

        # Utilizzo
        async with manager:
            if await manager.connect_server():
                tools = manager.get_available_tools()
                result = await manager.call_tool("tool_name", {"arg": "value"})
                await manager.disconnect()

    Transport Support:
        stdio:
            - Server locali come subprocess
            - Comunicazione via stdin/stdout
            - Supporta environment variables
            - Esempi: server-everything, server locali custom

        streamable-http:
            - Server remoti via HTTP/SSE
            - Supporta session management
            - Headers customizzabili per auth
            - Esempi: Tavily, GitHub Copilot, server cloud
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, config: Union[MCPConfig, str], command: str = None, args: List[str] = None):
        # Backward compatibility
        if isinstance(config, str):
            config = MCPConfig(name=config, command=command, args=args)

        key = config.get_key()
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, config: Union[MCPConfig, str], command: str = None, args: List[str] = None):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Backward compatibility
        if isinstance(config, str):
            config = MCPConfig(name=config, command=command, args=args)

        self.config = config
        self.name = config.name

        # Stato connessione
        self._session = None
        self._client_connection = None
        self._available_tools: List[Tool] = []
        self._available_resources: List[Resource] = []
        self._available_prompts: List[Prompt] = []
        self._connected = False
        self._session_id: Optional[str] = None

        # NUOVO: Validator integrato per conversione automatica tipi
        self._validator = MCPValidator()

    @classmethod
    def from_config_file(cls, config_path: str, server_name: str) -> 'MCPClientManager':
        """
        Carica configurazione da file mcp.json standard.

        Il file deve seguire la struttura standard MCP:
        {
          "mcpServers": {
            "server_name": {
              "command": "npx",
              "args": ["-y", "server-package"],
              "env": {"API_KEY": "value"}
            }
          }
        }

        Per server HTTP:
        {
          "mcpServers": {
            "server_name": {
              "url": "https://api.example.com/mcp/",
              "headers": {"Authorization": "Bearer token"}
            }
          }
        }

        Args:
            config_path: Percorso al file mcp.json
            server_name: Nome del server nel file config

        Returns:
            MCPClientManager configurato

        Raises:
            ValueError: Se server non trovato nel file
            FileNotFoundError: Se file non esiste
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        server_config = config_data.get('mcpServers', {}).get(server_name)
        if not server_config:
            raise ValueError(f"Server '{server_name}' not found in config file")

        # Determina il transport dal config
        if 'url' in server_config:
            transport = "streamable-http"
            return cls(MCPConfig(
                name=server_name,
                transport=transport,
                url=server_config['url'],
                headers=server_config.get('headers'),
                env=server_config.get('env')
            ))
        else:
            transport = "stdio"
            return cls(MCPConfig(
                name=server_name,
                transport=transport,
                command=server_config['command'],
                args=server_config.get('args', []),
                env=server_config.get('env')
            ))

    async def connect_server(self) -> bool:
        """
        Connette al server MCP usando il transport configurato.

        Gestisce automaticamente:
        - Selezione del transport (stdio/streamable-http)
        - Impostazione environment variables (se specificate)
        - Inizializzazione sessione MCP
        - Loading di tools, resources, prompts disponibili
        - Estrazione session ID per connessioni HTTP

        Returns:
            bool: True se connessione riuscita, False altrimenti

        Note:
            - Per stdio: lancia subprocess e comunica via stdin/stdout
            - Per streamable-http: apre connessione HTTP con possibile SSE stream
            - Environment variables vengono settate solo per stdio transport
        """
        try:
            print(f"DEBUG: Connessione {self.config.transport} a {self.name}")

            # Set environment variables if provided
            if self.config.env:
                for key, value in self.config.env.items():
                    os.environ[key] = value

            # Factory per transport
            if self.config.transport == "stdio":
                return await self._connect_stdio()
            elif self.config.transport in ["streamable-http", "http"]:
                return await self._connect_streamable_http()
            else:
                raise ValueError(f"Transport non supportato: {self.config.transport}")

        except Exception as e:
            print(f"Errore connessione: {e}")
            return False

    async def _connect_stdio(self) -> bool:
        """Connessione STDIO"""
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env
        )

        self._client_connection = stdio_client(params)
        read, write = await self._client_connection.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        # Load available capabilities
        await self._load_capabilities()
        self._connected = True
        return True

    async def _connect_streamable_http(self) -> bool:
        """Connessione Streamable HTTP (MCP 2025-06-18)"""
        try:
            from mcp.client.streamable_http import streamablehttp_client

            # Prepara headers con versione protocollo
            headers = self.config.headers or {}
            headers['MCP-Protocol-Version'] = '2025-06-18'
            headers['Accept'] = 'application/json, text/event-stream'

            # Usa il client streamable HTTP corretto
            self._client_connection = streamablehttp_client(self.config.url, headers=headers)
            read, write, _ = await self._client_connection.__aenter__()

            self._session = ClientSession(read, write)
            await self._session.__aenter__()

            # Initialize and potentially get session ID
            init_result = await self._session.initialize()

            # Extract session ID from response headers if available
            if hasattr(init_result, 'headers') and 'Mcp-Session-Id' in init_result.headers:
                self._session_id = init_result.headers['Mcp-Session-Id']

            # Load available capabilities
            await self._load_capabilities()
            self._connected = True
            return True

        except ImportError:
            print("Streamable HTTP client non disponibile. Installa: pip install 'mcp[client]'")
            return False
        except Exception as e:
            print(f"Errore Streamable HTTP: {e}")
            return False

    async def _load_capabilities(self):
        """Carica tools, resources e prompts disponibili"""
        # Load tools
        tools_result = await self._session.list_tools()
        self._available_tools = tools_result.tools

        # Load resources (if supported)
        try:
            resources_result = await self._session.list_resources()
            self._available_resources = resources_result.resources
        except:
            self._available_resources = []

        # Load prompts (if supported)
        try:
            prompts_result = await self._session.list_prompts()
            self._available_prompts = prompts_result.prompts
        except:
            self._available_prompts = []

    # Properties and basic methods
    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_available_tools(self) -> List[Tool]:
        return self._available_tools

    def get_available_resources(self) -> List[Resource]:
        return self._available_resources

    def get_available_prompts(self) -> List[Prompt]:
        return self._available_prompts

    def find_tool(self, tool_name: str) -> Optional[Tool]:
        for tool in self._available_tools:
            if tool.name == tool_name:
                return tool
        return None

    async def get_session_id(self) -> Optional[str]:
        """
        Ritorna session ID per connessioni HTTP, None per stdio.

        Il session ID viene estratto dal header 'Mcp-Session-Id' durante
        l'inizializzazione per server che supportano session management.

        Returns:
            Optional[str]: Session ID se disponibile, None per stdio transport

        Note:
            Utilizzato per:
            - Session management HTTP stateful
            - Debugging connessioni remote
            - Terminazione esplicita sessioni
        """
        return self._session_id

    # Core MCP operations con validazione integrata
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Chiama tool MCP con validazione automatica degli argomenti.

        MODIFICHE FASE 2:
        - Valida e converte argomenti usando MCPValidator prima dell'invio
        - Conversione automatica "10" → 10, "true" → True tramite Pydantic
        - Mantiene signature identica (stesso input/output)
        - Gestisce errori di validazione appropriatamente

        Args:
            tool_name: Nome del tool
            arguments: Argomenti per il tool (possono essere stringhe dal LLM)

        Returns:
            CallToolResult: Risultato del tool

        Raises:
            RuntimeError: Se non connesso
            Exception: Errori di validazione o esecuzione con dettagli migliorati
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        try:
            # NUOVO: Validazione e conversione automatica tramite MCPValidator
            validated_arguments = self._validate_tool_arguments(tool_name, arguments)

            # Chiamata al server con argomenti validati e convertiti
            tool_result = await self._session.call_tool(tool_name, validated_arguments)
            return tool_result

        except MCPValidationError as e:
            # Errori di validazione - informazioni dettagliate per debug
            error_msg = f"Validation error for tool '{tool_name}': {e.validation_errors}"
            raise Exception(error_msg)

        except Exception as e:
            # Altri errori - comportamento invariato con dettagli aggiuntivi
            error_msg = str(e)

            # Estrai dettagli aggiuntivi dall'eccezione
            if hasattr(e, 'data') and e.data:
                error_msg += f" | Data: {e.data}"

            # Controlla codici di errore specifici
            if hasattr(e, 'code'):
                error_msg += f" | Code: {e.code}"

            # Per errori HTTP, aggiungi status code
            if "400" in str(e) or "Bad Request" in str(e):
                error_msg += f" | Params sent: {arguments}"

            raise Exception(error_msg)

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida argomenti del tool usando schema MCP e MCPValidator.

        Args:
            tool_name: Nome del tool
            arguments: Argomenti raw (possono essere stringhe dal LLM)

        Returns:
            Dict[str, Any]: Argomenti validati e convertiti nei tipi corretti

        Raises:
            MCPValidationError: Se validazione fallisce
        """
        # Trova il tool e il suo schema
        tool = self.find_tool(tool_name)
        if not tool:
            raise MCPValidationError(tool_name, [{"error": f"Tool '{tool_name}' not found"}])

        # Se non ha inputSchema, passa argomenti così come sono
        if not hasattr(tool, 'inputSchema') or not tool.inputSchema:
            return arguments

        # Validazione tramite MCPValidator - fa la magia della conversione tipi
        return self._validator.validate_and_convert(tool_name, tool.inputSchema, arguments)

    async def list_resources(self) -> List[Resource]:
        """
        Lista resources disponibili dal server MCP.

        Le resources sono data sources che il server espone per fornire
        contesto ai modelli LLM. Simili a endpoint GET in REST API.

        Returns:
            List[Resource]: Lista resources con uri, name, description

        Raises:
            RuntimeError: Se non connesso al server

        Examples:
            resources = await manager.list_resources()
            for res in resources:
                print(f"{res.name}: {res.uri}")
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        result = await self._session.list_resources()
        return result.resources

    async def read_resource(self, uri: str):
        """
        Leggi contenuto di una risorsa specifica tramite URI.

        Args:
            uri: URI della risorsa (es. "file://document.txt", "config://settings")

        Returns:
            ResourceContent: Contenuto della risorsa

        Raises:
            RuntimeError: Se non connesso al server

        Examples:
            content = await manager.read_resource("file://readme.md")
            if hasattr(content, 'text'):
                print(content.text)
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        result = await self._session.read_resource(uri)
        return result

    async def list_prompts(self) -> List[Prompt]:
        """
        Lista prompts disponibili dal server MCP.

        I prompts sono template riutilizzabili per interazioni con LLM,
        possono avere argomenti parametrizzabili.

        Returns:
            List[Prompt]: Lista prompts con name, description, arguments

        Raises:
            RuntimeError: Se non connesso al server

        Examples:
            prompts = await manager.list_prompts()
            for prompt in prompts:
                print(f"{prompt.name}: {[arg.name for arg in prompt.arguments]}")
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        result = await self._session.list_prompts()
        return result.prompts

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None):
        """
        Esegui un prompt con argomenti specificati.

        Args:
            name: Nome del prompt da eseguire
            arguments: Dizionario argomenti per il prompt

        Returns:
            PromptResult: Risultato con messages formatati per LLM

        Raises:
            RuntimeError: Se non connesso al server

        Examples:
            result = await manager.get_prompt("greet_user", {"name": "Alice"})
            for message in result.messages:
                print(message.content)
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        result = await self._session.get_prompt(name, arguments or {})
        return result

    async def disconnect(self):
        """Disconnette dal server gestendo entrambi i transport"""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if self._client_connection:
            await self._client_connection.__aexit__(None, None, None)

        # Reset state
        self._connected = False
        self._session = None
        self._client_connection = None
        self._available_tools = []
        self._available_resources = []
        self._available_prompts = []
        self._session_id = None

        print(f"Disconnesso da {self.name}")

