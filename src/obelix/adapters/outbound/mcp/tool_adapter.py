"""MCPToolAdapter — wraps an MCP SDK Tool to satisfy the Obelix Tool protocol."""

import time
from typing import Any

from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MCPToolAdapter:
    """Wraps an MCP SDK Tool to satisfy the Obelix Tool protocol.

    The adapter bridges the MCP SDK's tool representation to the Obelix
    ``Tool`` protocol (structural typing), handling name-prefixing,
    schema generation, and execution delegation.
    """

    def __init__(self, server_name: str, mcp_tool: Any, group: Any) -> None:
        self.tool_name = f"mcp__{server_name}__{mcp_tool.name}"
        self.tool_description: str = mcp_tool.description or ""
        self.is_deferred: bool = False

        self._mcp_tool = mcp_tool
        self._original_name: str = mcp_tool.name
        self._server_name: str = server_name
        self._group = group

        # Behavioral annotations from MCP ToolAnnotations
        ann = getattr(mcp_tool, "annotations", None)
        self.read_only: bool | None = (
            getattr(ann, "readOnlyHint", None) if ann else None
        )
        self.destructive: bool | None = (
            getattr(ann, "destructiveHint", None) if ann else None
        )
        self.idempotent: bool | None = (
            getattr(ann, "idempotentHint", None) if ann else None
        )
        self.open_world: bool | None = (
            getattr(ann, "openWorldHint", None) if ann else None
        )

    # ------------------------------------------------------------------
    # Tool protocol
    # ------------------------------------------------------------------

    def create_schema(self) -> MCPToolSchema:
        """Return an ``MCPToolSchema`` derived from the underlying MCP tool."""
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema=self._mcp_tool.inputSchema,
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute the MCP tool via the server group and return a ToolResult."""
        start = time.time()
        try:
            result = await self._group.call_tool(
                self._original_name, tool_call.arguments
            )
            elapsed = time.time() - start

            # Extract text from content blocks
            text_parts: list[str] = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            text = "\n".join(text_parts)

            # Determine status
            status = ToolStatus.ERROR if result.isError else ToolStatus.SUCCESS

            # Build result payload
            if result.structuredContent is not None:
                payload: Any = {
                    "text": text,
                    "structuredContent": result.structuredContent,
                }
            else:
                payload = text

            return ToolResult(
                tool_name=self.tool_name,
                tool_call_id=tool_call.id,
                result=payload,
                status=status,
                error=text if status == ToolStatus.ERROR else None,
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "MCP tool %s execution failed: %s",
                self.tool_name,
                exc,
                exc_info=True,
            )
            return ToolResult(
                tool_name=self.tool_name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(exc),
                execution_time=elapsed,
            )
