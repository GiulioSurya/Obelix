# src/mcp/mcp_tool.py
import time
from typing import Any

from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from obelix.plugins.mcp.run_time_manager import MCPRuntimeManager


class MCPTool:
    """
    Tool that wraps an external MCP tool using Runtime Manager.

    PHASE 3 CHANGES - Centralized Validation:
    - REMOVED manual validation (_convert_args_using_mcp_schema)
    - REMOVED custom type conversion (_convert_value_by_json_schema)
    - execute() now passes arguments directly (already validated by manager)
    - Public interface UNCHANGED (execute, create_schema)
    - Much simpler code: from 150+ to 80 lines
    - Single responsibility: orchestration and output formatting

    Validation and type conversion now happens in MCPClientManager via
    MCPValidator, leveraging Pydantic for automatic conversion.

    Note:
        MCPTool does NOT use @tool decorator because the schema is dynamic
        (comes from MCP server). It implements tool_name,
        tool_description and create_schema() directly.
    """

    def __init__(self, tool_name: str, manager: MCPRuntimeManager):
        """
        Initialize MCPTool for a specific tool.

        Args:
            tool_name: Name of the MCP tool to wrap
            manager: Runtime manager for synchronous communication
        """
        self.manager = manager
        self._init_tool_metadata(tool_name)

    def _init_tool_metadata(self, tool_name: str) -> None:
        """
        Initialize tool_name and tool_description from MCP server.

        Args:
            tool_name: Name of the tool to find

        Raises:
            ValueError: If tool does not exist on MCP server
        """
        tool = self.manager.find_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Populate attributes directly (compatible with new API)
        self.tool_name = tool.name
        self.tool_description = tool.description or f"MCP tool: {tool.name}"

    def create_schema(self) -> MCPToolSchema:
        """
        Create schema using directly MCP definition from server.

        Interface UNCHANGED - same signature and behavior.

        Returns:
            MCPToolSchema: Complete schema of MCP tool

        Raises:
            ValueError: If tool not found
        """
        tool = self.manager.find_tool(self.tool_name)
        if not tool:
            raise ValueError(f"Tool {self.tool_name} not found")

        return MCPToolSchema(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.inputSchema,
            title=getattr(tool, "title", None),
            outputSchema=getattr(tool, "outputSchema", None),
            annotations=getattr(tool, "annotations", None),
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute MCP tool passing arguments directly to manager.

        PHASE 3 CHANGES:
        - REMOVED _convert_args_using_mcp_schema() completely
        - REMOVED manual type conversion logic
        - Arguments passed directly (already validated in MCPClientManager)
        - Interface UNCHANGED (same input/output)
        - Focus on orchestration and result handling

        Flow is now:
        1. LLM → MCPTool.execute() with raw arguments
        2. MCPTool → MCPRuntimeManager.call_tool() (synchronous)
        3. MCPRuntimeManager → MCPClientManager.call_tool() (internal async)
        4. MCPClientManager validates with MCPValidator (automatic Pydantic)
        5. Send to MCP server with validated arguments

        Args:
            tool_call: ToolCall with tool name and arguments

        Returns:
            ToolResult: Execution result with status, timing, content
        """
        start_time = time.time()

        try:
            # SIMPLIFIED: No argument conversion, already validated by manager
            # Validation/conversion happens in:
            # MCPRuntimeManager → MCPClientManager → MCPValidator (Pydantic)
            mcp_result = self.manager.call_tool(tool_call.name, tool_call.arguments)
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=self._extract_content(mcp_result),
                status=ToolStatus.SUCCESS,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=execution_time,
            )

    def _extract_content(self, mcp_result) -> Any:
        """
        Extract content from MCP result.

        Only responsibility left for the tool: format MCP output
        in a format understandable for the calling system.

        Args:
            mcp_result: Raw result from MCP server

        Returns:
            Any: Extracted and formatted content
        """
        if hasattr(mcp_result, "content"):
            if isinstance(mcp_result.content, list) and len(mcp_result.content) > 0:
                first_item = mcp_result.content[0]
                if hasattr(first_item, "text"):
                    content = first_item.text

                    # Error enrichment for debugging - behavior unchanged
                    if "Internal error" in content and "400" in content:
                        return f"{content} | Check parameters format and values"

                    return content
        return str(mcp_result)
