"""Tests for obelix.core.model.standard_message — StandardMessage type alias."""

from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.standard_message import StandardMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import ToolMessage, ToolResult


class TestStandardMessageTypeAlias:
    """Verify the Union type alias accepts all four message types."""

    def test_system_message_is_valid(self):
        msg: StandardMessage = SystemMessage(content="sys")
        assert isinstance(msg, SystemMessage)

    def test_human_message_is_valid(self):
        msg: StandardMessage = HumanMessage(content="hi")
        assert isinstance(msg, HumanMessage)

    def test_assistant_message_is_valid(self):
        msg: StandardMessage = AssistantMessage(content="reply")
        assert isinstance(msg, AssistantMessage)

    def test_tool_message_is_valid(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=1)
        msg: StandardMessage = ToolMessage(tool_results=[tr])
        assert isinstance(msg, ToolMessage)
