# src/obelix/adapters/outbound/mcp/resource_adapter.py
"""MCPResourceAdapter — placeholder for future MCP resource → skill bridge.

MCP servers can expose resources (files, data) that will be used
by the future skill system. This module will convert mcp.types.Resource
into an internal representation consumable by the skill loader.

Not implemented yet — resources are discovered and stored by MCPManager
but not consumed.
"""

from typing import Any

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MCPResourceAdapter:
    """Placeholder: converts MCP resources for future skill bridge."""

    def __init__(self, server_name: str, mcp_resource: Any):
        self.server_name = server_name
        self.uri = str(getattr(mcp_resource, "uri", ""))
        self.name = getattr(mcp_resource, "name", "")
        self.description = getattr(mcp_resource, "description", "")
        self.mime_type = getattr(mcp_resource, "mimeType", None)
