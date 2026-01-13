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
import copy
import time
from typing import Type, TYPE_CHECKING

from src.tools.tool_base import ToolBase
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent

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

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute the sub-agent with arguments from the tool call.

        Args:
            tool_call: ToolCall containing query and optional context fields

        Returns:
            ToolResult with the sub-agent's response
        """
        start_time = time.time()
        try:
            # Create isolated copy for parallel execution
            instance = copy.copy(self)

            # Validate arguments and populate attributes
            validated = self._input_schema(**tool_call.arguments)
            for field_name in validated.model_fields:
                setattr(instance, field_name, getattr(validated, field_name))

            # Build query with optional context fields
            query_parts = []
            for field_name in instance._fields.keys():
                if field_name == 'query':
                    continue
                field_value = getattr(instance, field_name, None)
                if field_value:
                    query_parts.append(f"{field_name}: {field_value}")

            # Add main query
            query_parts.append(instance.query)
            full_query = "\n\n".join(query_parts)

            # Execute sub-agent
            response = await instance._agent.execute_query_async(query=full_query)

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=response.model_dump(),
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
