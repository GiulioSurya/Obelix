"""Tests for obelix.core.model.human_message — HumanMessage model."""

from datetime import datetime

from obelix.core.model.human_message import HumanMessage
from obelix.core.model.roles import MessageRole


class TestHumanMessageConstruction:
    """Construction and default values."""

    def test_default_role(self):
        msg = HumanMessage(content="Hi")
        assert msg.role == MessageRole.HUMAN

    def test_content_stored(self):
        msg = HumanMessage(content="What is 2+2?")
        assert msg.content == "What is 2+2?"

    def test_default_content_is_empty_string(self):
        msg = HumanMessage()
        assert msg.content == ""

    def test_timestamp_auto_generated(self):
        before = datetime.now()
        msg = HumanMessage(content="x")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_default_metadata_is_empty_dict(self):
        msg = HumanMessage()
        assert msg.metadata == {}

    def test_custom_metadata(self):
        msg = HumanMessage(content="x", metadata={"user_id": "abc"})
        assert msg.metadata["user_id"] == "abc"


class TestHumanMessageSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        msg = HumanMessage(content="test", metadata={"a": 1})
        restored = HumanMessage.model_validate(msg.model_dump())
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata

    def test_json_round_trip(self):
        msg = HumanMessage(content="json")
        restored = HumanMessage.model_validate_json(msg.model_dump_json())
        assert restored.content == "json"

    def test_dump_keys(self):
        data = HumanMessage(content="x").model_dump()
        assert set(data.keys()) == {"role", "content", "timestamp", "metadata"}


class TestHumanMessageEdgeCases:
    """Edge cases."""

    def test_empty_content(self):
        assert HumanMessage(content="").content == ""

    def test_special_characters(self):
        msg = HumanMessage(content='He said "hello" & <bye>')
        assert '"hello"' in msg.content
        assert "<bye>" in msg.content
