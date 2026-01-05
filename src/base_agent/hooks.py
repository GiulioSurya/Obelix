"""
Hook system for BaseAgent.

Provides a fluent API to intercept agent lifecycle events,
inject messages into conversation history, and transform results.

Example:
    agent.on(AgentEvent.ON_TOOL_ERROR) \
        .when(lambda agent_status: "invalid identifier" in agent_status.error) \
        .inject(lambda agent_status: HumanMessage(content="Schema: ..."))
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any, List, Union, TYPE_CHECKING
import asyncio

from src.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent
    from src.messages.tool_message import ToolCall, ToolResult
    from src.messages.assistant_message import AssistantMessage
    from src.messages.standard_message import StandardMessage


class AgentEvent(str, Enum):
    """Agent lifecycle events"""

    # === Lifecycle ===
    ON_QUERY_START = "on_query_start"
    """Before starting query execution"""

    ON_QUERY_END = "on_query_end"
    """End of execution (success or error)"""

    # === LLM ===
    BEFORE_LLM_CALL = "before_llm_call"
    """Before calling provider.invoke()"""

    AFTER_LLM_CALL = "after_llm_call"
    """After LLM response (can transform AssistantMessage)"""

    # === Tool Calls ===
    BEFORE_TOOL_EXECUTION = "before_tool_execution"
    """Before executing tool.execute() (can transform ToolCall)"""

    AFTER_TOOL_EXECUTION = "after_tool_execution"
    """After tool execution (can transform ToolResult)"""

    # === Conditionals ===
    ON_TOOL_ERROR = "on_tool_error"
    """When a tool returns an error (status == ERROR)"""

    ON_MAX_ITERATIONS = "on_max_iterations"
    """When iteration limit is reached"""


@dataclass
class AgentStatus:
    """
    Rich context passed to hooks.

    Contains information about the agent's current state
    and allows access to conversation history.
    """
    event: AgentEvent
    agent: 'BaseAgent'
    iteration: int = 0
    tool_call: Optional['ToolCall'] = None
    tool_result: Optional['ToolResult'] = None
    assistant_message: Optional['AssistantMessage'] = None
    error: Optional[str] = None

    @property
    def conversation_history(self) -> List['StandardMessage']:
        """Direct access to agent's conversation history"""
        return self.agent.conversation_history


class Hook:
    """
    Hook with fluent API to intercept agent events.

    Supports:
    - Activation conditions (.when())
    - Message injection (.inject(), .inject_at())
    - Value transformation (.transform())
    - Generic actions (.do())

    Example:
        agent.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(lambda agent_status: agent_status.tool_result.error) \
            .transform(lambda result, agent_status: enrich_error(result))
    """

    def __init__(self, event: AgentEvent):
        self.event = event
        self._condition: Optional[Callable[[AgentStatus], bool]] = None
        self._actions: List[Callable] = []
        logger.debug(f"Hook created for event: {event.value}")

    def when(self, condition: Callable[[AgentStatus], bool]) -> 'Hook':
        """
        Set the condition to activate the hook.

        Args:
            condition: Function that receives AgentStatus and returns bool.
                      Can be sync or async.

        Returns:
            self for method chaining
        """
        self._condition = condition
        condition_name = getattr(condition, '__name__', str(condition))
        logger.debug(f"Hook [{self.event.value}] - condition set: {condition_name}")
        return self

    def inject(
        self,
        message_factory: Callable[[AgentStatus], 'StandardMessage']
    ) -> 'Hook':
        """
        Inject a message at the end of conversation history (append).

        Args:
            message_factory: Function that receives AgentStatus and returns
                           the message to inject

        Returns:
            self for method chaining
        """
        factory_name = getattr(message_factory, '__name__', str(message_factory))
        logger.debug(f"Hook [{self.event.value}] - registered inject action: {factory_name}")

        def action(agent_status: AgentStatus, current: Any) -> Any:
            message = message_factory(agent_status)
            logger.debug(f"Hook [{agent_status.event.value}] - inject executed, message type: {type(message).__name__}")
            agent_status.agent.conversation_history.append(message)
            return current

        self._actions.append(action)
        return self

    def inject_at(
        self,
        position: Union[int, Callable[[AgentStatus], int]],
        message_factory: Callable[[AgentStatus], 'StandardMessage']
    ) -> 'Hook':
        """
        Inject a message at a specific position in conversation history.

        Args:
            position: Index to insert at, or function that calculates index.
                     Negative values are relative to end (-1 = before last)
            message_factory: Function that returns the message to inject

        Returns:
            self for method chaining
        """
        factory_name = getattr(message_factory, '__name__', str(message_factory))
        pos_desc = position if isinstance(position, int) else getattr(position, '__name__', 'dynamic')
        logger.debug(f"Hook [{self.event.value}] - registered inject_at action: {factory_name} @ pos={pos_desc}")

        def action(agent_status: AgentStatus, current: Any) -> Any:
            message = message_factory(agent_status)
            pos = position(agent_status) if callable(position) else position
            logger.debug(f"Hook [{agent_status.event.value}] - inject_at executed @ pos={pos}, message type: {type(message).__name__}")
            agent_status.agent.conversation_history.insert(pos, message)

        self._actions.append(action)
        return self

    def do(self, action: Callable[[AgentStatus], Any]) -> 'Hook':
        """
        Execute a generic action (logging, side effects, etc.).

        The action does not modify the current value in the chain.

        Args:
            action: Function that receives AgentStatus. Can be sync or async.

        Returns:
            self for method chaining
        """
        action_name = getattr(action, '__name__', str(action))
        logger.debug(f"Hook [{self.event.value}] - registered do action: {action_name}")

        def wrapped(agent_status: AgentStatus, current: Any) -> Any:
            logger.debug(f"Hook [{agent_status.event.value}] - do action executed: {action_name}")
            action(agent_status)
            return current

        self._actions.append(wrapped)
        return self

    def transform(
        self,
        transformer: Callable[[Any, AgentStatus], Any]
    ) -> 'Hook':
        """
        Transform the current value (ToolResult, AssistantMessage, ToolCall).

        Useful with events like AFTER_TOOL_EXECUTION or AFTER_LLM_CALL.

        Args:
            transformer: Function (current_value, agent_status) -> new_value.
                        Can be sync or async.

        Returns:
            self for method chaining
        """
        transformer_name = getattr(transformer, '__name__', str(transformer))
        logger.debug(f"Hook [{self.event.value}] - registered transform action: {transformer_name}")

        def action(agent_status: AgentStatus, current: Any) -> Any:
            logger.debug(f"Hook [{agent_status.event.value}] - transform executed: {transformer_name}, input type: {type(current).__name__}")
            result = transformer(current, agent_status)
            logger.debug(f"Hook [{agent_status.event.value}] - transform completed, output type: {type(result).__name__}")
            return result

        self._actions.append(action)
        return self

    async def execute(self, agent_status: AgentStatus, current_value: Any = None) -> Any:
        """
        Execute the hook if condition is satisfied.

        Args:
            agent_status: Current agent state
            current_value: Current value to pass to transformations

        Returns:
            Transformed value or current_value if no transformation
        """
        logger.debug(f"Hook [{self.event.value}] - execute invoked, iteration={agent_status.iteration}")

        # Check condition
        if self._condition is not None:
            condition_name = getattr(self._condition, '__name__', 'anonymous')
            logger.debug(f"Hook [{self.event.value}] - evaluating condition: {condition_name}")
            cond_result = self._condition(agent_status)
            if asyncio.iscoroutine(cond_result):
                cond_result = await cond_result
            if not cond_result:
                logger.debug(f"Hook [{self.event.value}] - condition NOT satisfied, skip")
                return current_value
            logger.debug(f"Hook [{self.event.value}] - condition SATISFIED, executing {len(self._actions)} actions")
        else:
            logger.debug(f"Hook [{self.event.value}] - no condition, executing {len(self._actions)} actions")

        # Execute all actions in sequence
        result = current_value
        for i, action in enumerate(self._actions):
            logger.debug(f"Hook [{self.event.value}] - executing action {i+1}/{len(self._actions)}")
            action_result = action(agent_status, result)
            if asyncio.iscoroutine(action_result):
                action_result = await action_result
            if action_result is not None:
                result = action_result

        logger.debug(f"Hook [{self.event.value}] - execute completed")
        return result
