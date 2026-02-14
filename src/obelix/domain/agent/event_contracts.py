from dataclasses import dataclass
from typing import Any, Dict

from obelix.domain.agent.hooks import AgentEvent
from obelix.domain.model.assistant_message import AssistantMessage, AssistantResponse
from obelix.domain.model.tool_message import ToolCall, ToolResult


@dataclass(frozen=True)
class EventContract:
    input_type: Any
    output_type: Any
    retryable: bool
    stop_output: Any


def get_event_contracts() -> Dict[AgentEvent, EventContract]:
    return {
        AgentEvent.BEFORE_LLM_CALL: EventContract(
            input_type=None,
            output_type=None,
            retryable=False,
            stop_output=AssistantMessage,
        ),
        AgentEvent.AFTER_LLM_CALL: EventContract(
            input_type=AssistantMessage,
            output_type=AssistantMessage,
            retryable=True,
            stop_output=AssistantMessage,
        ),
        AgentEvent.BEFORE_TOOL_EXECUTION: EventContract(
            input_type=ToolCall,
            output_type=ToolCall,
            retryable=False,
            stop_output=None,
        ),
        AgentEvent.AFTER_TOOL_EXECUTION: EventContract(
            input_type=ToolResult,
            output_type=ToolResult,
            retryable=False,
            stop_output=None,
        ),
        AgentEvent.ON_TOOL_ERROR: EventContract(
            input_type=ToolResult,
            output_type=ToolResult,
            retryable=False,
            stop_output=None,
        ),
        AgentEvent.BEFORE_FINAL_RESPONSE: EventContract(
            input_type=AssistantMessage,
            output_type=AssistantMessage,
            retryable=True,
            stop_output=AssistantMessage,
        ),
        AgentEvent.QUERY_END: EventContract(
            input_type=(AssistantResponse, type(None)),
            output_type=(AssistantResponse, type(None)),
            retryable=False,
            stop_output=None,
        ),
    }
