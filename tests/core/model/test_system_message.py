"""Tests for obelix.core.model.system_message — SystemMessage model."""

from datetime import datetime

from obelix.core.model.roles import MessageRole
from obelix.core.model.system_message import SystemMessage


class TestSystemMessageConstruction:
    """Construction and default values."""

    def test_default_role(self):
        msg = SystemMessage(content="Be helpful.")
        assert msg.role == MessageRole.SYSTEM

    def test_content_stored(self):
        msg = SystemMessage(content="Instructions here.")
        assert msg.content == "Instructions here."

    def test_default_content_is_empty_string(self):
        msg = SystemMessage()
        assert msg.content == ""

    def test_timestamp_auto_generated(self):
        before = datetime.now()
        msg = SystemMessage(content="x")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_default_metadata_is_empty_dict(self):
        msg = SystemMessage()
        assert msg.metadata == {}

    def test_custom_metadata(self):
        meta = {"source": "config", "version": 2}
        msg = SystemMessage(content="x", metadata=meta)
        assert msg.metadata == meta

    def test_role_override_allowed(self):
        """Pydantic allows setting role explicitly (no validator blocks it)."""
        msg = SystemMessage(content="x", role=MessageRole.HUMAN)
        assert msg.role == MessageRole.HUMAN


class TestSystemMessageSerialization:
    """model_dump / model_validate round-trip."""

    def test_round_trip(self):
        msg = SystemMessage(content="Hello", metadata={"k": "v"})
        data = msg.model_dump()
        restored = SystemMessage.model_validate(data)
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata
        assert restored.role == msg.role

    def test_dump_contains_expected_keys(self):
        data = SystemMessage(content="abc").model_dump()
        assert set(data.keys()) == {"role", "content", "timestamp", "metadata"}

    def test_json_round_trip(self):
        msg = SystemMessage(content="json test")
        json_str = msg.model_dump_json()
        restored = SystemMessage.model_validate_json(json_str)
        assert restored.content == "json test"


class TestSystemMessageEdgeCases:
    """Edge cases and boundary values."""

    def test_empty_content(self):
        msg = SystemMessage(content="")
        assert msg.content == ""

    def test_very_long_content(self):
        long_text = "x" * 100_000
        msg = SystemMessage(content=long_text)
        assert len(msg.content) == 100_000

    def test_unicode_content(self):
        msg = SystemMessage(content="Ciao mondo! Ecco caratteri speciali: e e e")
        assert "Ciao" in msg.content

    def test_multiline_content(self):
        text = "Line 1\nLine 2\nLine 3"
        msg = SystemMessage(content=text)
        assert msg.content.count("\n") == 2

    def test_metadata_with_nested_structures(self):
        meta = {"nested": {"a": [1, 2, 3]}, "flag": True}
        msg = SystemMessage(content="x", metadata=meta)
        assert msg.metadata["nested"]["a"] == [1, 2, 3]
