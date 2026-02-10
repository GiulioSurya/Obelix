"""
Hook system for BaseAgent.

Provides a uniform API for intercepting agent lifecycle events.
The API uses when(...) and handle(...) to define:
- activation conditions
- state effects via context
- value transformations in the pipeline
- flow decisions (CONTINUE/RETRY/FAIL/STOP)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any, List, TYPE_CHECKING
import asyncio

from src.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent
    from src.obelix_types.tool_message import ToolCall, ToolResult
    from src.obelix_types.assistant_message import AssistantMessage
    from src.obelix_types.standard_message import StandardMessage


class AgentEvent(str, Enum):
    """Agent lifecycle events"""

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

    # === Finalization ===
    BEFORE_FINAL_RESPONSE = "before_final_response"
    """Before building the final response (validation/retry)"""

    QUERY_END = "query_end"
    """End of execution (success or error)"""


class HookDecision(str, Enum):
    """Hook decision for flow control"""
    CONTINUE = "continue"
    RETRY = "retry"
    FAIL = "fail"
    STOP = "stop"


@dataclass
class Outcome:
    """Uniform result of a hook execution."""
    decision: HookDecision
    value: Any


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
    Hook with fluent API:
    - when(condition)
    - handle(decision, value=None, effects=None)
    """

    def __init__(self, event: AgentEvent):
        self.event = event
        self._condition: Optional[Callable[[AgentStatus], bool]] = None
        self._decision: HookDecision = HookDecision.CONTINUE
        self._value: Optional[Callable[..., Any] | Any] = None
        self._effects: List[Callable[[AgentStatus], Any]] = []
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

    def handle(
        self,
        decision: HookDecision,
        value: Optional[Callable[..., Any] | Any] = None,
        effects: Optional[List[Callable[[AgentStatus], Any]]] = None,
    ) -> 'Hook':
        """
        Defines decision, value and effects for the hook.

        Args:
            decision: CONTINUE | RETRY | FAIL | STOP
            value: direct value or function (status, current_value) -> new_value
            effects: list of effect(status) -> None

        Returns:
            self for method chaining
        """
        self._decision = decision
        self._value = value
        self._effects = effects or []
        logger.debug(
            f"Hook [{self.event.value}] - handle registered: "
            f"decision={decision.value}, effects={len(self._effects)}"
        )
        return self

    async def execute(self, agent_status: AgentStatus, current_value: Any = None) -> Outcome:
        """
        Execute the hook if condition is satisfied.

        Returns:
            Outcome with decision and value
        """
        logger.debug(f"Hook [{self.event.value}] - execute invoked, iteration={agent_status.iteration}")

        if self._condition is not None:
            condition_name = getattr(self._condition, '__name__', 'anonymous')
            logger.debug(f"Hook [{self.event.value}] - evaluating condition: {condition_name}")
            cond_result = self._condition(agent_status)
            if asyncio.iscoroutine(cond_result):
                cond_result = await cond_result
            if not cond_result:
                logger.debug(f"Hook [{self.event.value}] - condition NOT satisfied, skip")
                return Outcome(HookDecision.CONTINUE, current_value)
            logger.debug(
                f"Hook [{self.event.value}] - condition SATISFIED, effects={len(self._effects)}"
            )
        else:
            logger.debug(
                f"Hook [{self.event.value}] - no condition, effects={len(self._effects)}"
            )

        for i, effect in enumerate(self._effects):
            logger.debug(
                f"Hook [{self.event.value}] - executing effect {i+1}/{len(self._effects)}"
            )
            effect_result = effect(agent_status)
            if asyncio.iscoroutine(effect_result):
                await effect_result

        new_value = current_value
        if self._value is not None:
            if callable(self._value):
                value_result = self._value(agent_status, current_value)
                if asyncio.iscoroutine(value_result):
                    value_result = await value_result
                new_value = value_result
            else:
                new_value = self._value

        logger.debug(
            f"Hook [{self.event.value}] - execute completed, decision={self._decision.value}"
        )
        return Outcome(self._decision, new_value)
