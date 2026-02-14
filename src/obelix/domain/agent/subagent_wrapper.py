"""
SubAgentWrapper: bridges BaseAgent to ToolBase interface.

Allows any BaseAgent to be registered as a sub-agent (tool) on another agent.
All field extraction, schema generation, and execution logic is self-contained.

Usage:
    wrapper = SubAgentWrapper(
        agent=my_agent,
        name="sql_analyzer",
        description="Analyzes SQL errors",
        stateless=True,
    )
    orchestrator.registered_tools.append(wrapper)
"""
import asyncio
import copy
import time
from typing import Dict, Type, TYPE_CHECKING, get_type_hints

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from obelix.domain.tool.tool_base import ToolBase
from obelix.domain.model.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.domain.agent.base_agent import BaseAgent

logger = get_logger(__name__)


class SubAgentWrapper(ToolBase):
    """
    Wraps a BaseAgent as a ToolBase so it can be registered as a tool.

    Handles:
    - Field extraction from the agent class (Pydantic Fields become input parameters)
    - Dynamic Pydantic schema generation for input validation
    - Stateless (parallel-safe, copy-based) and stateful (serialized, shared history) execution
    - MCP schema generation
    """

    def __init__(
        self,
        agent: 'BaseAgent',
        *,
        name: str,
        description: str,
        stateless: bool = False,
    ):
        self._agent = agent
        self.tool_name = name
        self.tool_description = description
        self._stateless = stateless
        self._execution_lock = asyncio.Lock()

        # Extract fields and build schema
        self._fields = self._extract_fields(agent.__class__)
        self._input_schema = self._build_input_schema(name)

    # ─── Execution ────────────────────────────────────────────────────────────

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        if self._stateless:
            return await self._execute_stateless(tool_call)
        else:
            async with self._execution_lock:
                return await self._execute_stateful(tool_call)

    async def _execute_stateless(self, tool_call: ToolCall) -> ToolResult:
        start_time = time.time()
        try:
            agent_copy = copy.copy(self._agent)
            agent_copy.conversation_history = self._agent.conversation_history.copy()

            full_query = self._build_query(tool_call)
            response = await agent_copy.execute_query_async(query=full_query)

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=response.content,
                status=ToolStatus.SUCCESS,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_stateful(self, tool_call: ToolCall) -> ToolResult:
        start_time = time.time()
        try:
            full_query = self._build_query(tool_call)
            response = await self._agent.execute_query_async(query=full_query)

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=response.content,
                status=ToolStatus.SUCCESS,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=time.time() - start_time
            )

    # ─── Query Building ───────────────────────────────────────────────────────

    def _build_query(self, tool_call: ToolCall) -> str:
        validated = self._input_schema(**tool_call.arguments)

        query_parts = []
        for field_name in self._fields.keys():
            if field_name == 'query':
                continue
            field_value = getattr(validated, field_name, None)
            if field_value:
                query_parts.append(f"{field_name}: {field_value}")

        query_parts.append(validated.query)
        return "\n\n".join(query_parts)

    # ─── Schema ───────────────────────────────────────────────────────────────

    def create_schema(self) -> MCPToolSchema:
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema=self._input_schema.model_json_schema(),
            outputSchema={"type": "object", "additionalProperties": True}
        )

    # ─── Field Extraction (from agent class) ──────────────────────────────────

    @staticmethod
    def _extract_fields(cls: Type) -> Dict[str, tuple]:
        """
        Extract Pydantic Field definitions from the agent class.

        Always adds 'query' as the first parameter.
        Other fields with FieldInfo become optional context parameters.
        """
        other_fields = {}

        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = getattr(cls, '__annotations__', {})

        for attr_name, attr_type in hints.items():
            if attr_name.startswith('_') or attr_name == 'query':
                continue

            default = getattr(cls, attr_name, ...)
            if isinstance(default, FieldInfo):
                other_fields[attr_name] = (attr_type, default)

        return {
            'query': (str, Field(..., description="Query for the sub-agent")),
            **other_fields
        }

    def _build_input_schema(self, name: str) -> Type:
        """Create dynamic Pydantic model from extracted fields."""
        schema_class_name = f"{name.title().replace('_', '')}SubAgentSchema"
        return create_model(schema_class_name, **self._fields)