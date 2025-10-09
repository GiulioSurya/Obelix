# src/mcp/mcp_runtime_manager.py
import asyncio
import threading
import queue
import atexit
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from src.tools.mcp.mcp_client_manager import MCPClientManager, MCPConfig


@dataclass
class RuntimeCommand:
    """Comando per comunicazione thread-safe"""
    action: str
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class MCPRuntimeManager:
    """
    Runtime Manager per connessioni MCP persistenti con interfaccia sincrona.

    Mantiene una connessione MCP attiva per tutto il ciclo di vita dell'applicazione,
    esponendo un'interfaccia sincrona semplice che nasconde la complessità asincrona.

    Features:
    - Connessione automatica al primo utilizzo
    - Interfaccia completamente sincrona
    - Riconnessione automatica su errori
    - Cleanup automatico a fine programma
    - Thread-safe con queue interno

    Usage:
        # Inizializzazione
        config = MCPConfig(name="server", command="npx", args=["server"])
        runtime_manager = MCPRuntimeManager(config)

        # Uso semplice
        tools = runtime_manager.get_tools()
        result = runtime_manager.call_tool("search", {"query": "test"})

        # Si disconnette automaticamente
    """

    def __init__(self, config: Union[MCPConfig, str], command: str = None, args: List[str] = None):
        # Backward compatibility
        if isinstance(config, str):
            config = MCPConfig(name=config, command=command, args=args)

        self.config = config
        self._mcp_manager = None

        # Threading setup
        self._thread = None
        self._loop = None
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        # State
        self._connected = False
        self._tools_cache = []

        # Auto cleanup
        atexit.register(self.shutdown)

    def _ensure_running(self):
        """Assicura che il thread async sia attivo"""
        if self._thread is None or not self._thread.is_alive():
            self._start_async_thread()

    def _start_async_thread(self):
        """Avvia il thread con event loop async"""
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

    def _run_async_loop(self):
        """Event loop principale del thread async"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_worker())
        finally:
            self._loop.close()

    async def _async_worker(self):
        """Worker asincrono che processa i comandi dalla queue"""
        self._mcp_manager = MCPClientManager(self.config)

        while not self._shutdown_event.is_set():
            try:
                # Controlla per nuovi comandi (non-blocking)
                try:
                    cmd = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Processa il comando
                try:
                    result = await self._execute_command(cmd)
                    self._response_queue.put(('success', result))
                except Exception as e:
                    self._response_queue.put(('error', e))

            except Exception as e:
                print(f"Error in async worker: {e}")

    async def _execute_command(self, cmd: RuntimeCommand):
        """Esegue un comando specifico"""
        # Auto-connect se necessario
        if not self._connected:
            connected = await self._mcp_manager.connect_server()
            if not connected:
                raise ConnectionError("Failed to connect to MCP server")
            self._connected = True
            self._tools_cache = self._mcp_manager.get_available_tools()

        # Esegui comando
        if cmd.action == 'get_tools':
            return self._tools_cache

        elif cmd.action == 'call_tool':
            tool_name, arguments = cmd.args
            return await self._mcp_manager.call_tool(tool_name, arguments)

        elif cmd.action == 'get_resources':
            return self._mcp_manager.get_available_resources()

        elif cmd.action == 'list_resources':
            return await self._mcp_manager.list_resources()

        elif cmd.action == 'read_resource':
            uri = cmd.args[0]
            return await self._mcp_manager.read_resource(uri)

        elif cmd.action == 'get_prompts':
            return self._mcp_manager.get_available_prompts()

        elif cmd.action == 'get_prompt':
            name, arguments = cmd.args
            return await self._mcp_manager.get_prompt(name, arguments)

        elif cmd.action == 'find_tool':
            tool_name = cmd.args[0]
            return self._mcp_manager.find_tool(tool_name)

        elif cmd.action == 'is_connected':
            return self._connected

        else:
            raise ValueError(f"Unknown command: {cmd.action}")

    def _send_command(self, action: str, *args, **kwargs):
        """Invia comando e attende risposta"""
        self._ensure_running()

        cmd = RuntimeCommand(action, args, kwargs)
        self._request_queue.put(cmd)

        # Attendi risposta
        try:
            result_type, result = self._response_queue.get(timeout=30)
            if result_type == 'error':
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("Command timeout")

    # Public interface - tutti metodi sincroni
    def get_tools(self) -> List:
        """Ottieni lista tools disponibili"""
        return self._send_command('get_tools')

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Chiama un tool con argomenti specificati"""
        return self._send_command('call_tool', tool_name, arguments)

    def get_resources(self) -> List:
        """Ottieni lista resources cache"""
        return self._send_command('get_resources')

    def list_resources(self) -> List:
        """Lista resources dal server (fresh)"""
        return self._send_command('list_resources')

    def read_resource(self, uri: str):
        """Leggi contenuto di una risorsa"""
        return self._send_command('read_resource', uri)

    def get_prompts(self) -> List:
        """Ottieni lista prompts disponibili"""
        return self._send_command('get_prompts')

    def get_prompt(self, name: str, arguments: Dict[str, Any] = None):
        """Esegui un prompt"""
        return self._send_command('get_prompt', name, arguments or {})

    def find_tool(self, tool_name: str):
        """Trova tool specifico per nome"""
        return self._send_command('find_tool', tool_name)

    def is_connected(self) -> bool:
        """Controlla se connesso"""
        try:
            return self._send_command('is_connected')
        except:
            return False

    def shutdown(self):
        """Shutdown pulito del manager"""
        if self._thread and self._thread.is_alive():
            self._shutdown_event.set()
            self._thread.join(timeout=5)

        # Cleanup finale
        if self._loop and not self._loop.is_closed():
            try:
                # Disconnetti il manager MCP se ancora attivo
                if self._mcp_manager and self._connected:
                    asyncio.run_coroutine_threadsafe(
                        self._mcp_manager.disconnect(),
                        self._loop
                    ).result(timeout=5)
            except:
                pass


# Utility per creazione rapida
def create_runtime_manager(config: Union[MCPConfig, str], **kwargs) -> MCPRuntimeManager:
    """
    Factory function per creazione rapida di runtime manager.

    Args:
        config: MCPConfig object o nome server (backward compatible)
        **kwargs: Argomenti aggiuntivi per backward compatibility

    Returns:
        MCPRuntimeManager configurato e pronto all'uso

    Examples:
        # Modo nuovo
        config = MCPConfig(name="tavily", transport="streamable-http", url="...")
        manager = create_runtime_manager(config)

        # Backward compatible
        manager = create_runtime_manager("everything", command="npx", args=["-y", "server"])
    """
    return MCPRuntimeManager(config, **kwargs)


if __name__ == "__main__":
    # Test del runtime manager
    config = MCPConfig(
        name="everything",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"]
    )

    manager = create_runtime_manager(config)

    try:
        print("Testing sync interface...")

        # Test connessione
        print(f"Connected: {manager.is_connected()}")

        # Test tools
        tools = manager.get_tools()
        print(f"Available tools: {[tool.name for tool in tools[:3]]}")

        # Test tool call se disponibile
        if tools:
            result = manager.call_tool(tools[0].name, {})
            print(f"Tool result type: {type(result)}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        manager.shutdown()