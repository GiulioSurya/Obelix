"""Tests for obelix.infrastructure.providers -- Providers enum."""

import pytest

from obelix.infrastructure.providers import Providers


class TestProviders:
    """Tests for the Providers enum."""

    def test_enum_values(self):
        assert Providers.ANTHROPIC.value == "anthropic"
        assert Providers.IBM_WATSON.value == "ibm_watson"
        assert Providers.OCI_GENERATIVE_AI.value == "oci"
        assert Providers.OLLAMA.value == "ollama"
        assert Providers.VLLM.value == "vllm"
        assert Providers.OPENAI.value == "openai"
        assert Providers.LITELLM.value == "litellm"

    def test_enum_members_count(self):
        assert len(Providers) == 7

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("anthropic", Providers.ANTHROPIC),
            ("ibm_watson", Providers.IBM_WATSON),
            ("oci", Providers.OCI_GENERATIVE_AI),
            ("ollama", Providers.OLLAMA),
            ("vllm", Providers.VLLM),
            ("openai", Providers.OPENAI),
            ("litellm", Providers.LITELLM),
        ],
        ids=[
            "anthropic",
            "ibm_watson",
            "oci",
            "ollama",
            "vllm",
            "openai",
            "litellm",
        ],
    )
    def test_construct_from_value(self, value: str, expected: Providers):
        assert Providers(value) is expected

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Providers("nonexistent_provider")

    def test_all_members_are_strings(self):
        """Every Providers member should have a string value."""
        for member in Providers:
            assert isinstance(member.value, str)

    def test_unique_values(self):
        """No two enum members should share the same value."""
        values = [p.value for p in Providers]
        assert len(values) == len(set(values))

    def test_name_and_value_differ(self):
        """Enum names (e.g. OCI_GENERATIVE_AI) differ from their values (e.g. 'oci')."""
        # Specific check for the most obvious case
        assert Providers.OCI_GENERATIVE_AI.name == "OCI_GENERATIVE_AI"
        assert Providers.OCI_GENERATIVE_AI.value == "oci"

    def test_membership_check(self):
        """Can use 'in' to check if a value is a valid provider."""
        valid_values = {p.value for p in Providers}
        assert "anthropic" in valid_values
        assert "unknown" not in valid_values

    def test_iteration(self):
        """Enum is iterable and yields all members."""
        members = list(Providers)
        assert len(members) == 7
        assert Providers.ANTHROPIC in members
        assert Providers.LITELLM in members
