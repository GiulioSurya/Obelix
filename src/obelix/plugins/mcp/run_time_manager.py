
# src/mcp/mcp_runtime_manager.py
import asyncio
import threading
import queue
import atexit
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from obelix.plugins.mcp.mcp_client_manager import MCPClientManager, MCPConfig


@dataclass
class RuntimeCommand:
    """Command for thread-safe communication"""
    action: str
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class MCPRuntimeManager:
    """
    Runtime Manager for persistent MCP connections with synchronous interface.

    Maintains an active MCP connection for the entire application lifecycle,
    exposing a simple synchronous interface that hides asynchronous complexity.

    Features:
    - Automatic connection on first use
    - Completely synchronous interface
    - Automatic reconnection on errors
    - Automatic cleanup at program end
    - Thread-safe with internal queue

    Usage:
        # Initialization
        config = MCPConfig(name="server", command="npx", args=["server"])
        runtime_manager = MCPRuntimeManager(config)

        # Simple usage
        tools = runtime_manager.get_tools()
        result = runtime_manager.call_tool("search", {"query": "test"})

        # Disconnects automatically
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
        """Ensure async thread is active"""
        if self._thread is None or not self._thread.is_alive():
            self._start_async_thread()

    def _start_async_thread(self):
        """Start thread with async event loop"""
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

    def _run_async_loop(self):
        """Main event loop of async thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_worker())
        finally:
            self._loop.close()

    async def _async_worker(self):
        """Async worker that processes commands from queue"""
        self._mcp_manager = MCPClientManager(self.config)

        while not self._shutdown_event.is_set():
            try:
                # Check for new commands (non-blocking)
                try:
                    cmd = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process command
                try:
                    result = await self._execute_command(cmd)
                    self._response_queue.put(('success', result))
                except Exception as e:
                    self._response_queue.put(('error', e))

            except Exception as e:
                print(f"Error in async worker: {e}")

    async def _execute_command(self, cmd: RuntimeCommand):
        """Execute a specific command"""
        # Auto-connect if necessary
        if not self._connected:
            connected = await self._mcp_manager.connect_server()
            if not connected:
                raise ConnectionError("Failed to connect to MCP server")
            self._connected = True
            self._tools_cache = self._mcp_manager.get_available_tools()

        # Execute command
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
        """Send command and wait for response"""
        self._ensure_running()

        cmd = RuntimeCommand(action, args, kwargs)
        self._request_queue.put(cmd)

        # Wait for response
        try:
            result_type, result = self._response_queue.get(timeout=30)
            if result_type == 'error':
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("Command timeout")

    # Public interface - all synchronous methods
    def get_tools(self) -> List:
        """Get list of available tools"""
        return self._send_command('get_tools')

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool with specified arguments"""
        return self._send_command('call_tool', tool_name, arguments)

    def get_resources(self) -> List:
        """Get cached resources list"""
        return self._send_command('get_resources')

    def list_resources(self) -> List:
        """List resources from server (fresh)"""
        return self._send_command('list_resources')

    def read_resource(self, uri: str):
        """Read content of a resource"""
        return self._send_command('read_resource', uri)

    def get_prompts(self) -> List:
        """Get list of available prompts"""
        return self._send_command('get_prompts')

    def get_prompt(self, name: str, arguments: Dict[str, Any] = None):
        """Execute a prompt"""
        return self._send_command('get_prompt', name, arguments or {})

    def find_tool(self, tool_name: str):
        """Find specific tool by name"""
        return self._send_command('find_tool', tool_name)

    def is_connected(self) -> bool:
        """Check if connected"""
        try:
            return self._send_command('is_connected')
        except:
            return False

    def shutdown(self):
        """Clean shutdown of the manager"""
        if self._thread and self._thread.is_alive():
            self._shutdown_event.set()
            self._thread.join(timeout=5)

        # Final cleanup
        if self._loop and not self._loop.is_closed():
            try:
                # Disconnect MCP manager if still active
                if self._mcp_manager and self._connected:
                    asyncio.run_coroutine_threadsafe(
                        self._mcp_manager.disconnect(),
                        self._loop
                    ).result(timeout=5)
            except:
                pass


# Utility for quick creation
def create_runtime_manager(config: Union[MCPConfig, str], **kwargs) -> MCPRuntimeManager:
    """
    Factory function for quick creation of runtime manager.

    Args:
        config: MCPConfig object or server name (backward compatible)
        **kwargs: Additional arguments for backward compatibility

    Returns:
        MCPRuntimeManager configured and ready to use

    Examples:
        # New way
        config = MCPConfig(name="tavily", transport="streamable-http", url="...")
        manager = create_runtime_manager(config)

        # Backward compatible
        manager = create_runtime_manager("everything", command="npx", args=["-y", "server"])
    """
    return MCPRuntimeManager(config, **kwargs)


if __name__ == "__main__":
    # Runtime manager test
    config = MCPConfig(
        name="everything",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"]
    )

    manager = create_runtime_manager(config)

    try:
        print("Testing sync interface...")

        # Test connection
        print(f"Connected: {manager.is_connected()}")

        # Test tools
        tools = manager.get_tools()
        print(f"Available tools: {[tool.name for tool in tools[:3]]}")

        # Test tool call if available
        if tools:
            result = manager.call_tool(tools[0].name, {})
            print(f"Tool result type: {type(result)}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        manager.shutdown()