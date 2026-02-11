# src/base_agent/orchestrator_decorator.py
"""
@orchestrator decorator for enabling agent coordination capabilities.

Usage:
    @orchestrator
    class CoordinatorAgent(BaseAgent):
        def __init__(self):
            super().__init__(system_message="You coordinate tasks...")

    @subagent(name="analyzer", description="Analyzes data")
    class AnalyzerAgent(BaseAgent):
        ...

    # Only orchestrators can register sub-agents
    coordinator = CoordinatorAgent()
    coordinator.register_agent(AnalyzerAgent())
"""
import asyncio
import copy
import time
from typing import Type, TYPE_CHECKING

from src.domain.tool.tool_base import ToolBase
from src.domain.model.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema
from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from src.domain.agent.base_agent import BaseAgent

logger = get_logger(__name__)


class _SubAgentWrapper(ToolBase):
    """
    Wrapper that exposes a BaseAgent as a ToolBase.

    This class bridges the interface gap between BaseAgent and ToolBase,
    allowing sub-agents to be registered and executed as tools.
    """

    def __init__(self, agent: 'BaseAgent'):
        """
        Initialize wrapper with a sub-agent instance.

        Args:
            agent: BaseAgent instance decorated with @subagent
        """
        self._agent = agent
        self.tool_name = agent.subagent_name
        self.tool_description = agent.subagent_description
        self._input_schema = agent._subagent_input_schema
        self._fields = agent._subagent_fields
        self._stateless = getattr(agent, 'subagent_stateless', False)
        # Lock for serializing calls when stateless=False
        self._execution_lock = asyncio.Lock()

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute the sub-agent with arguments from the tool call.

        Behavior depends on stateless flag:
        - stateless=True: Creates isolated copy of agent (fork of conversation_history),
          allows parallel execution, copy is discarded after execution.
        - stateless=False: Serializes calls with lock, preserves conversation_history.

        Args:
            tool_call: ToolCall containing query and optional context fields

        Returns:
            ToolResult with the sub-agent's response
        """
        if self._stateless:
            return await self._execute_stateless(tool_call)
        else:
            async with self._execution_lock:
                return await self._execute_stateful(tool_call)

    async def _execute_stateless(self, tool_call: ToolCall) -> ToolResult:
        """Execute in stateless mode: isolated copy, parallel-safe."""
        start_time = time.time()
        try:
            # Create isolated copy of agent for this execution
            agent_copy = copy.copy(self._agent)
            # Fork conversation_history (copy current state, will be discarded after)
            agent_copy.conversation_history = self._agent.conversation_history.copy()

            # Build query from tool_call arguments
            full_query = self._build_query(tool_call)

            # Execute on isolated copy
            response = await agent_copy.execute_query_async(query=full_query)

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=response.content,
                status=ToolStatus.SUCCESS,
                execution_time=time.time() - start_time
            )
        #todo use validation pydantic? evaluate
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
        """Execute in stateful mode: serialized, preserves conversation_history."""
        start_time = time.time()
        try:
            # Build query from tool_call arguments
            full_query = self._build_query(tool_call)

            # Execute on original agent (lock ensures serialization)
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

    def _build_query(self, tool_call: ToolCall) -> str:
        """Build full query string from tool_call arguments."""
        # Validate arguments and get values
        validated = self._input_schema(**tool_call.arguments)

        # Build query with optional context fields
        query_parts = []
        for field_name in self._fields.keys():
            if field_name == 'query':
                continue
            field_value = getattr(validated, field_name, None)
            if field_value:
                query_parts.append(f"{field_name}: {field_value}")

        # Add main query
        query_parts.append(validated.query)
        return "\n\n".join(query_parts)

    @classmethod
    def create_schema(cls) -> MCPToolSchema:
        """This is overridden per-instance, see __init__"""
        raise NotImplementedError("create_schema must be called on instance")

    def create_schema(self) -> MCPToolSchema:
        """Generate MCP schema for the wrapped sub-agent"""
        return MCPToolSchema(
            name=self.tool_name,
            description=self.tool_description,
            inputSchema=self._input_schema.model_json_schema(),
            outputSchema={"type": "object", "additionalProperties": True}
        )


def orchestrator(cls: Type = None, *, name: str = None, description: str = None):
    """
    Decorator that enables agent coordination capabilities.

    Adds the register_agent() method to the decorated class,
    allowing it to register and coordinate sub-agents.

    Can be used with or without parameters:
        @orchestrator
        @orchestrator()
        @orchestrator(name="coordinator", description="Coordinates tasks")

    Args:
        cls: BaseAgent subclass to decorate (when used without parentheses)
        name: Optional name for the orchestrator (defaults to class name)
        description: Optional description of the orchestrator

    Returns:
        Decorated class with orchestration capabilities
    """
    def decorator(cls_inner: Type) -> Type:
        # Mark as orchestrator
        cls_inner._is_orchestrator = True

        # Add optional metadata with fallback to class name
        cls_inner.orchestrator_name = name
        cls_inner.orchestrator_description = description

        # Add register_agent method
        def register_agent(self, agent: 'BaseAgent'):
            """
            Register a sub-agent as a tool.

            The agent must be decorated with @subagent.

            Args:
                agent: BaseAgent instance decorated with @subagent

            Raises:
                ValueError: If agent is not decorated with @subagent
            """
            # Verify agent is decorated with @subagent
            if not hasattr(agent, 'subagent_name'):
                raise ValueError(
                    f"Agent {agent.__class__.__name__} must be decorated with @subagent. "
                    f"Usage: @subagent(name='...', description='...')"
                )

            # Create wrapper and register
            wrapper = _SubAgentWrapper(agent)
            self.registered_tools.append(wrapper)
            # Use orchestrator_name if set, otherwise fallback to class name
            display_name = self.orchestrator_name or self.__class__.__name__
            logger.info(f"Agent {display_name}: sub-agent '{agent.subagent_name}' registered")

        cls_inner.register_agent = register_agent

        return cls_inner

    # Handle both @orchestrator and @orchestrator() and @orchestrator(name=..., description=...)
    if cls is not None:
        # Called without parentheses: @orchestrator
        return decorator(cls)
    else:
        # Called with parentheses: @orchestrator() or @orchestrator(name=..., description=...)
        return decorator
