"""Tests for obelix.core.agent.event_contracts module.

Covers EventContract frozen dataclass, get_event_contracts() completeness,
retryable flags, stop_output types, and input/output type correctness.
"""

from dataclasses import FrozenInstanceError

import pytest

from obelix.core.agent.event_contracts import EventContract, get_event_contracts
from obelix.core.agent.hooks import AgentEvent
from obelix.core.model.assistant_message import AssistantMessage, AssistantResponse
from obelix.core.model.tool_message import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# EventContract dataclass
# ---------------------------------------------------------------------------


class TestEventContractDataclass:
    """Tests for EventContract frozen dataclass properties."""

    def test_construction(self):
        """EventContract can be created with all fields."""
        ec = EventContract(
            input_type=str,
            output_type=int,
            retryable=True,
            stop_output=None,
        )
        assert ec.input_type is str
        assert ec.output_type is int
        assert ec.retryable is True
        assert ec.stop_output is None

    def test_frozen_immutability(self):
        """EventContract is frozen and does not allow field mutation."""
        ec = EventContract(
            input_type=None,
            output_type=None,
            retryable=False,
            stop_output=None,
        )
        with pytest.raises(FrozenInstanceError):
            ec.retryable = True

    def test_frozen_cannot_set_new_attribute(self):
        """EventContract is frozen and does not allow new attributes."""
        ec = EventContract(
            input_type=None,
            output_type=None,
            retryable=False,
            stop_output=None,
        )
        with pytest.raises(FrozenInstanceError):
            ec.extra_field = "nope"


# ---------------------------------------------------------------------------
# get_event_contracts() completeness
# ---------------------------------------------------------------------------


class TestGetEventContracts:
    """Tests for get_event_contracts() coverage and completeness."""

    def test_returns_dict(self):
        """get_event_contracts() returns a dict."""
        contracts = get_event_contracts()
        assert isinstance(contracts, dict)

    def test_all_agent_events_have_contracts(self):
        """Every AgentEvent member has a corresponding contract."""
        contracts = get_event_contracts()
        for event in AgentEvent:
            assert event in contracts, f"Missing contract for {event.name}"

    def test_no_extra_keys(self):
        """Contract dict keys match AgentEvent members exactly."""
        contracts = get_event_contracts()
        assert set(contracts.keys()) == set(AgentEvent)

    def test_contract_count_matches_event_count(self):
        """Number of contracts equals number of events (7)."""
        contracts = get_event_contracts()
        assert len(contracts) == 7

    def test_all_values_are_event_contracts(self):
        """All values in the dict are EventContract instances."""
        contracts = get_event_contracts()
        for event, contract in contracts.items():
            assert isinstance(contract, EventContract), (
                f"Contract for {event.name} is {type(contract)}, expected EventContract"
            )


# ---------------------------------------------------------------------------
# retryable flag
# ---------------------------------------------------------------------------


class TestRetryableContracts:
    """Tests for retryable flag on each event contract."""

    def test_only_after_llm_call_and_before_final_response_are_retryable(self):
        """Only AFTER_LLM_CALL and BEFORE_FINAL_RESPONSE have retryable=True."""
        contracts = get_event_contracts()
        retryable_events = {event for event, c in contracts.items() if c.retryable}
        assert retryable_events == {
            AgentEvent.AFTER_LLM_CALL,
            AgentEvent.BEFORE_FINAL_RESPONSE,
        }

    def test_non_retryable_events(self):
        """BEFORE_LLM_CALL, tool events, and QUERY_END are not retryable."""
        contracts = get_event_contracts()
        non_retryable = [
            AgentEvent.BEFORE_LLM_CALL,
            AgentEvent.BEFORE_TOOL_EXECUTION,
            AgentEvent.AFTER_TOOL_EXECUTION,
            AgentEvent.ON_TOOL_ERROR,
            AgentEvent.QUERY_END,
        ]
        for event in non_retryable:
            assert contracts[event].retryable is False, (
                f"{event.name} should not be retryable"
            )


# ---------------------------------------------------------------------------
# stop_output
# ---------------------------------------------------------------------------


class TestStopOutputContracts:
    """Tests for stop_output type on each event contract."""

    def test_before_llm_call_stop_output_is_assistant_message(self):
        """BEFORE_LLM_CALL stop_output is AssistantMessage."""
        contracts = get_event_contracts()
        assert contracts[AgentEvent.BEFORE_LLM_CALL].stop_output is AssistantMessage

    def test_after_llm_call_stop_output_is_assistant_message(self):
        """AFTER_LLM_CALL stop_output is AssistantMessage."""
        contracts = get_event_contracts()
        assert contracts[AgentEvent.AFTER_LLM_CALL].stop_output is AssistantMessage

    def test_before_final_response_stop_output_is_assistant_message(self):
        """BEFORE_FINAL_RESPONSE stop_output is AssistantMessage."""
        contracts = get_event_contracts()
        assert (
            contracts[AgentEvent.BEFORE_FINAL_RESPONSE].stop_output is AssistantMessage
        )

    @pytest.mark.parametrize(
        "event",
        [
            AgentEvent.BEFORE_TOOL_EXECUTION,
            AgentEvent.AFTER_TOOL_EXECUTION,
            AgentEvent.ON_TOOL_ERROR,
            AgentEvent.QUERY_END,
        ],
        ids=["before_tool", "after_tool", "on_tool_error", "query_end"],
    )
    def test_tool_and_query_end_stop_output_is_none(self, event):
        """Tool events and QUERY_END have stop_output=None (STOP not allowed)."""
        contracts = get_event_contracts()
        assert contracts[event].stop_output is None


# ---------------------------------------------------------------------------
# input_type / output_type
# ---------------------------------------------------------------------------


class TestInputOutputTypes:
    """Tests for input_type and output_type on each event contract."""

    def test_before_llm_call_types(self):
        """BEFORE_LLM_CALL: input=None, output=None."""
        c = get_event_contracts()[AgentEvent.BEFORE_LLM_CALL]
        assert c.input_type is None
        assert c.output_type is None

    def test_after_llm_call_types(self):
        """AFTER_LLM_CALL: input=AssistantMessage, output=AssistantMessage."""
        c = get_event_contracts()[AgentEvent.AFTER_LLM_CALL]
        assert c.input_type is AssistantMessage
        assert c.output_type is AssistantMessage

    def test_before_tool_execution_types(self):
        """BEFORE_TOOL_EXECUTION: input=ToolCall, output=ToolCall."""
        c = get_event_contracts()[AgentEvent.BEFORE_TOOL_EXECUTION]
        assert c.input_type is ToolCall
        assert c.output_type is ToolCall

    def test_after_tool_execution_types(self):
        """AFTER_TOOL_EXECUTION: input=ToolResult, output=ToolResult."""
        c = get_event_contracts()[AgentEvent.AFTER_TOOL_EXECUTION]
        assert c.input_type is ToolResult
        assert c.output_type is ToolResult

    def test_on_tool_error_types(self):
        """ON_TOOL_ERROR: input=ToolResult, output=ToolResult."""
        c = get_event_contracts()[AgentEvent.ON_TOOL_ERROR]
        assert c.input_type is ToolResult
        assert c.output_type is ToolResult

    def test_before_final_response_types(self):
        """BEFORE_FINAL_RESPONSE: input=AssistantMessage, output=AssistantMessage."""
        c = get_event_contracts()[AgentEvent.BEFORE_FINAL_RESPONSE]
        assert c.input_type is AssistantMessage
        assert c.output_type is AssistantMessage

    def test_query_end_types(self):
        """QUERY_END: input/output are tuple of (AssistantResponse, NoneType)."""
        c = get_event_contracts()[AgentEvent.QUERY_END]
        assert isinstance(c.input_type, tuple)
        assert AssistantResponse in c.input_type
        assert type(None) in c.input_type
        assert isinstance(c.output_type, tuple)
        assert AssistantResponse in c.output_type
        assert type(None) in c.output_type
