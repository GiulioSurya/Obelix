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
    Configuration for MCP connection.

    Supports two main transports:
    - stdio: For local servers (subprocess)
    - streamable-http: For remote servers (HTTP)

    Examples:
        # STDIO - Local server with subprocess
        config = MCPConfig(
            name="local_server",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            env={"API_KEY": "your-key"}  # Environment variables for process
        )

        # STREAMABLE-HTTP - Remote server with API key in URL
        config = MCPConfig(
            name="tavily",
            transport="streamable-http",
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        )

        # STREAMABLE-HTTP - Remote server with Bearer authentication
        config = MCPConfig(
            name="github_copilot",
            transport="streamable-http",
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_token}"}
        )

        # STDIO BRIDGE - Remote server via mcp-remote
        config = MCPConfig(
            name="tavily_bridge",
            transport="stdio",
            command="npx",
            args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"]
        )
    """
    name: str
    transport: str = "stdio"  # stdio, streamable-http

    # STDIO params - for local servers
    command: Optional[str] = None  # Command to execute (e.g. "npx", "python")
    args: Optional[List[str]] = None  # Command arguments (e.g. ["-y", "server.js"])
    env: Optional[Dict[str, str]] = None  # Environment variables for subprocess

    # Streamable HTTP params - for remote servers
    url: Optional[str] = None  # MCP endpoint URL (e.g. "https://api.example.com/mcp/")
    headers: Optional[Dict[str, str]] = None  # HTTP headers (e.g. Authorization)

    def get_key(self) -> tuple:
        """Generate unique key for singleton"""
        if self.transport == "stdio":
            return self.name, self.transport, self.command, tuple(self.args or [])
        else:
            return self.name, self.transport, self.url


class MCPClientManager:
    """
    Universal manager for MCP connections with integrated validation.

    Manages both stdio and streamable-http connections with singleton pattern.
    Supports all MCP features: tools, resources, prompts, session management.

    Features:
        - Singleton pattern for connection reuse
        - Support stdio (local subprocess) and streamable-http (remote servers)
        - Automatic HTTP session management with session ID
        - Complete loading of capabilities (tools, resources, prompts)
        - Configuration loading from mcp.json file
        - Environment variables support for API keys
        - NEW: Automatic argument validation with MCPValidator

    Usage:
        # Mode 1: Direct configuration
        config = MCPConfig(
            name="tavily",
            transport="streamable-http",
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        )
        manager = MCPClientManager(config)

        # Mode 2: From configuration file
        manager = MCPClientManager.from_config_file("mcp.json", "server_name")

        # Mode 3: Backward compatible
        manager = MCPClientManager("everything", "npx", ["-y", "server-everything"])

        # Usage
        async with manager:
            if await manager.connect_server():
                tools = manager.get_available_tools()
                result = await manager.call_tool("tool_name", {"arg": "value"})
                await manager.disconnect()

    Transport Support:
        stdio:
            - Local servers as subprocess
            - Communication via stdin/stdout
            - Supports environment variables
            - Examples: server-everything, custom local servers

        streamable-http:
            - Remote servers via HTTP/SSE
            - Supports session management
            - Customizable headers for auth
            - Examples: Tavily, GitHub Copilot, cloud servers
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

        # Connection state
        self._session = None
        self._client_connection = None
        self._available_tools: List[Tool] = []
        self._available_resources: List[Resource] = []
        self._available_prompts: List[Prompt] = []
        self._connected = False
        self._session_id: Optional[str] = None

        # NEW: Integrated validator for automatic type conversion
        self._validator = MCPValidator()

    @classmethod
    def from_config_file(cls, config_path: str, server_name: str) -> 'MCPClientManager':
        """
        Load configuration from standard mcp.json file.

        File must follow MCP standard structure:
        {
          "mcpServers": {
            "server_name": {
              "command": "npx",
              "args": ["-y", "server-package"],
              "env": {"API_KEY": "value"}
            }
          }
        }

        For HTTP servers:
        {
          "mcpServers": {
            "server_name": {
              "url": "https://api.example.com/mcp/",
              "headers": {"Authorization": "Bearer token"}
            }
          }
        }

        Args:
            config_path: Path to mcp.json file
            server_name: Name of server in config file

        Returns:
            Configured MCPClientManager

        Raises:
            ValueError: If server not found in file
            FileNotFoundError: If file does not exist
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        server_config = config_data.get('mcpServers', {}).get(server_name)
        if not server_config:
            raise ValueError(f"Server '{server_name}' not found in config file")

        # Determine transport from config
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
        Connect to MCP server using configured transport.

        Automatically handles:
        - Transport selection (stdio/streamable-http)
        - Environment variables setup (if specified)
        - MCP session initialization
        - Loading available tools, resources, prompts
        - Session ID extraction for HTTP connections

        Returns:
            bool: True if connection successful, False otherwise

        Note:
            - For stdio: launches subprocess and communicates via stdin/stdout
            - For streamable-http: opens HTTP connection with possible SSE stream
            - Environment variables are set only for stdio transport
        """
        try:
            print(f"DEBUG: Connecting {self.config.transport} to {self.name}")

            # Set environment variables if provided
            if self.config.env:
                for key, value in self.config.env.items():
                    os.environ[key] = value

            # Transport factory
            if self.config.transport == "stdio":
                return await self._connect_stdio()
            elif self.config.transport in ["streamable-http", "http"]:
                return await self._connect_streamable_http()
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    async def _connect_stdio(self) -> bool:
        """STDIO connection"""
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
        """Streamable HTTP connection (MCP 2025-06-18)"""
        try:
            from mcp.client.streamable_http import streamablehttp_client

            # Prepare headers with protocol version
            headers = self.config.headers or {}
            headers['MCP-Protocol-Version'] = '2025-06-18'
            headers['Accept'] = 'application/json, text/event-stream'

            # Use correct streamable HTTP client
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
            print("Streamable HTTP client not available. Install: pip install 'mcp[client]'")
            return False
        except Exception as e:
            print(f"Streamable HTTP error: {e}")
            return False

    async def _load_capabilities(self):
        """Load available tools, resources and prompts"""
        # Load tools
        tools_result = await self._session.list_tools()
        self._available_tools = tools_result.tools

        # Load resources (if supported)
        try:
            resources_result = await self._session.list_resources()
            self._available_resources = resources_result.resources
        except Exception:
            self._available_resources = []

        # Load prompts (if supported)
        try:
            prompts_result = await self._session.list_prompts()
            self._available_prompts = prompts_result.prompts
        except Exception:
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
        Return session ID for HTTP connections, None for stdio.

        Session ID is extracted from 'Mcp-Session-Id' header during
        initialization for servers that support session management.

        Returns:
            Optional[str]: Session ID if available, None for stdio transport

        Note:
            Used for:
            - Stateful HTTP session management
            - Debugging remote connections
            - Explicit session termination
        """
        return self._session_id

    # Core MCP operations with integrated validation
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Call MCP tool with automatic argument validation.

        PHASE 2 CHANGES:
        - Validates and converts arguments using MCPValidator before sending
        - Automatic conversion "10" → 10, "true" → True via Pydantic
        - Maintains identical signature (same input/output)
        - Appropriately handles validation errors

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool (can be strings from LLM)

        Returns:
            CallToolResult: Tool result

        Raises:
            RuntimeError: If not connected
            Exception: Validation or execution errors with enhanced details
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        try:
            # NEW: Automatic validation and conversion via MCPValidator
            validated_arguments = self._validate_tool_arguments(tool_name, arguments)

            # Call server with validated and converted arguments
            tool_result = await self._session.call_tool(tool_name, validated_arguments)
            return tool_result

        except MCPValidationError as e:
            # Validation errors - detailed information for debugging
            error_msg = f"Validation error for tool '{tool_name}': {e.validation_errors}"
            raise Exception(error_msg)

        except Exception as e:
            # Other errors - unchanged behavior with additional details
            error_msg = str(e)

            # Extract additional details from exception
            if hasattr(e, 'data') and e.data:
                error_msg += f" | Data: {e.data}"

            # Check for specific error codes
            if hasattr(e, 'code'):
                error_msg += f" | Code: {e.code}"

            # For HTTP errors, add status code
            if "400" in str(e) or "Bad Request" in str(e):
                error_msg += f" | Params sent: {arguments}"

            raise Exception(error_msg)

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool arguments using MCP schema and MCPValidator.

        Args:
            tool_name: Name of the tool
            arguments: Raw arguments (can be strings from LLM)

        Returns:
            Dict[str, Any]: Validated and converted arguments with correct types

        Raises:
            MCPValidationError: If validation fails
        """
        # Find tool and its schema
        tool = self.find_tool(tool_name)
        if not tool:
            raise MCPValidationError(tool_name, [{"error": f"Tool '{tool_name}' not found"}])

        # If no inputSchema, pass arguments as-is
        if not hasattr(tool, 'inputSchema') or not tool.inputSchema:
            return arguments

        # Validation via MCPValidator - performs type conversion magic
        return self._validator.validate_and_convert(tool_name, tool.inputSchema, arguments)

    async def list_resources(self) -> List[Resource]:
        """
        List available resources from MCP server.

        Resources are data sources that the server exposes to provide
        context to LLM models. Similar to GET endpoints in REST API.

        Returns:
            List[Resource]: List of resources with uri, name, description

        Raises:
            RuntimeError: If not connected to server

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
        Read content of a specific resource via URI.

        Args:
            uri: Resource URI (e.g. "file://document.txt", "config://settings")

        Returns:
            ResourceContent: Resource content

        Raises:
            RuntimeError: If not connected to server

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
        List available prompts from MCP server.

        Prompts are reusable templates for interactions with LLM,
        can have parameterizable arguments.

        Returns:
            List[Prompt]: List of prompts with name, description, arguments

        Raises:
            RuntimeError: If not connected to server

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
        Execute a prompt with specified arguments.

        Args:
            name: Name of the prompt to execute
            arguments: Dictionary of arguments for the prompt

        Returns:
            PromptResult: Result with obelix_types formatted for LLM

        Raises:
            RuntimeError: If not connected to server

        Examples:
            result = await manager.get_prompt("greet_user", {"name": "Alice"})
            for message in result.obelix_types:
                print(message.content)
        """
        if not self._connected:
            raise RuntimeError("Not connected to server")

        result = await self._session.get_prompt(name, arguments or {})
        return result

    async def disconnect(self):
        """Disconnect from server handling both transports"""
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

        print(f"Disconnected from {self.name}")

