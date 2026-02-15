"""Tests for obelix.core.model.roles — MessageRole enum."""

import pytest

from obelix.core.model.roles import MessageRole


class TestMessageRole:
    """Tests for the MessageRole string enum."""

    def test_enum_values(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.HUMAN == "human"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_enum_is_str_subclass(self):
        """MessageRole inherits from str, so values are usable as plain strings."""
        assert isinstance(MessageRole.SYSTEM, str)

    def test_enum_members_count(self):
        assert len(MessageRole) == 4

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("system", MessageRole.SYSTEM),
            ("human", MessageRole.HUMAN),
            ("assistant", MessageRole.ASSISTANT),
            ("tool", MessageRole.TOOL),
        ],
        ids=["system", "human", "assistant", "tool"],
    )
    def test_construct_from_value(self, value: str, expected: MessageRole):
        assert MessageRole(value) is expected

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            MessageRole("invalid_role")

    def test_string_comparison(self):
        """Enum values can be compared directly with plain strings."""
        assert MessageRole.SYSTEM == "system"
        assert "human" == MessageRole.HUMAN
