# src/client_adapters/llm_abstraction.py
"""
Abstract MCP Provider — contract for MCP client implementations.

Follows the same pattern as AbstractLLMProvider: the port defines
the contract, the adapter (adapters/outbound/mcp/) implements it.
"""

from abc import ABC, abstractmethod
from typing import Any

from obelix.core.tool.tool_base import Tool


class AbstractMCPProvider(ABC):
    """Contract for MCP client implementations."""

    @abstractmethod
    async def connect(self) -> list[Tool]:
        """Connect to all configured MCP servers.

        Discovers tools on each server and returns them converted
        to the Obelix Tool protocol, ready for BaseAgent.register_tool().

        Returns:
            List of Tool-protocol-compatible objects.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close all MCP server connections and release resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the provider currently has active connections."""
        ...

    @abstractmethod
    def get_resources(self) -> dict[str, Any]:
        """Return discovered MCP resources, keyed by server name.

        Placeholder for the future skill bridge. Returns an empty
        dict until resource consumption is implemented.
        """
        ...
