"""Tests for obelix.core.model.usage — Usage and AgentUsage models."""

import pytest
from pydantic import ValidationError

from obelix.core.model.usage import AgentUsage, Usage

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class TestUsageConstruction:
    """Construction and validation."""

    def test_construction(self):
        u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        assert u.input_tokens == 10
        assert u.output_tokens == 20
        assert u.total_tokens == 30

    def test_zero_tokens(self):
        u = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        assert u.total_tokens == 0

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            Usage(input_tokens=10, output_tokens=20)  # missing total_tokens

    @pytest.mark.parametrize(
        "field",
        ["input_tokens", "output_tokens", "total_tokens"],
        ids=["input", "output", "total"],
    )
    def test_negative_tokens_rejected(self, field: str):
        """ge=0 constraint rejects negative values."""
        kwargs = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        kwargs[field] = -1
        with pytest.raises(ValidationError):
            Usage(**kwargs)

    def test_validate_assignment(self):
        """validate_assignment=True means setting a negative value post-init raises."""
        u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        with pytest.raises(ValidationError):
            u.input_tokens = -5


class TestUsageSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        u = Usage(input_tokens=5, output_tokens=10, total_tokens=15)
        restored = Usage.model_validate(u.model_dump())
        assert restored.input_tokens == 5
        assert restored.total_tokens == 15

    def test_json_round_trip(self):
        u = Usage(input_tokens=1, output_tokens=2, total_tokens=3)
        restored = Usage.model_validate_json(u.model_dump_json())
        assert restored.output_tokens == 2

    def test_dump_keys(self):
        data = Usage(input_tokens=0, output_tokens=0, total_tokens=0).model_dump()
        assert set(data.keys()) == {"input_tokens", "output_tokens", "total_tokens"}


# ---------------------------------------------------------------------------
# AgentUsage
# ---------------------------------------------------------------------------


class TestAgentUsageConstruction:
    """Construction and default values."""

    def test_construction_defaults(self):
        au = AgentUsage(model_id="gpt-4")
        assert au.model_id == "gpt-4"
        assert au.total_input_tokens == 0
        assert au.total_output_tokens == 0
        assert au.total_tokens == 0
        assert au.call_count == 0

    def test_model_id_required(self):
        with pytest.raises(ValidationError):
            AgentUsage()

    def test_custom_initial_values(self):
        au = AgentUsage(
            model_id="m",
            total_input_tokens=100,
            total_output_tokens=200,
            total_tokens=300,
            call_count=5,
        )
        assert au.call_count == 5
        assert au.total_tokens == 300

    @pytest.mark.parametrize(
        "field",
        [
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "call_count",
        ],
    )
    def test_negative_values_rejected(self, field: str):
        kwargs = {"model_id": "m", field: -1}
        with pytest.raises(ValidationError):
            AgentUsage(**kwargs)


class TestAgentUsageAddUsage:
    """Tests for the add_usage method."""

    def test_add_single_usage(self):
        au = AgentUsage(model_id="m")
        u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        au.add_usage(u)
        assert au.total_input_tokens == 10
        assert au.total_output_tokens == 20
        assert au.total_tokens == 30
        assert au.call_count == 1

    def test_add_multiple_usages_accumulates(self):
        au = AgentUsage(model_id="m")
        u1 = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        u2 = Usage(input_tokens=5, output_tokens=15, total_tokens=20)
        au.add_usage(u1)
        au.add_usage(u2)
        assert au.total_input_tokens == 15
        assert au.total_output_tokens == 35
        assert au.total_tokens == 50
        assert au.call_count == 2

    def test_add_usage_with_zero_tokens(self):
        au = AgentUsage(model_id="m")
        u = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        au.add_usage(u)
        assert au.total_tokens == 0
        assert au.call_count == 1


class TestAgentUsageReset:
    """Tests for the reset method."""

    def test_reset_clears_all_counters(self):
        au = AgentUsage(model_id="m")
        u = Usage(input_tokens=100, output_tokens=200, total_tokens=300)
        au.add_usage(u)
        au.add_usage(u)
        au.reset()
        assert au.total_input_tokens == 0
        assert au.total_output_tokens == 0
        assert au.total_tokens == 0
        assert au.call_count == 0

    def test_reset_preserves_model_id(self):
        au = AgentUsage(model_id="my-model")
        au.add_usage(Usage(input_tokens=1, output_tokens=1, total_tokens=2))
        au.reset()
        assert au.model_id == "my-model"

    def test_reset_on_fresh_instance_is_noop(self):
        au = AgentUsage(model_id="m")
        au.reset()
        assert au.total_tokens == 0
        assert au.call_count == 0


class TestAgentUsageSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        au = AgentUsage(model_id="m")
        au.add_usage(Usage(input_tokens=10, output_tokens=20, total_tokens=30))
        restored = AgentUsage.model_validate(au.model_dump())
        assert restored.model_id == "m"
        assert restored.call_count == 1
        assert restored.total_tokens == 30

    def test_json_round_trip(self):
        au = AgentUsage(model_id="m", total_input_tokens=5, call_count=2)
        restored = AgentUsage.model_validate_json(au.model_dump_json())
        assert restored.total_input_tokens == 5
        assert restored.call_count == 2
