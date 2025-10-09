"""
Unit tests for ProviderRegistry class.

Tests the ProviderRegistry registration and retrieval of provider mappings.
"""

import pytest
from src.providers import Providers, ProviderRegistry


class TestProviderRegistryRegistration:
    """Test provider mapping registration functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry._mappings.clear()

    def teardown_method(self):
        """Re-register real mappings after tests."""
        # Re-import to restore real mappings
        import src.mapping.provider_mapping

    def test_register_provider_mapping(self):
        """Test registering a provider mapping."""
        # Arrange
        test_mapping = {"test_key": "test_value"}

        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, test_mapping)

        # Assert
        assert Providers.IBM_WATSON.value in ProviderRegistry._mappings
        assert ProviderRegistry._mappings[Providers.IBM_WATSON.value] == test_mapping

    def test_register_multiple_providers(self):
        """Test registering multiple provider mappings."""
        # Arrange
        ibm_mapping = {"provider": "ibm", "config": "ibm_config"}
        oci_mapping = {"provider": "oci", "config": "oci_config"}

        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, ibm_mapping)
        ProviderRegistry.register(Providers.OCI_GENERATIVE_AI, oci_mapping)

        # Assert
        assert len(ProviderRegistry._mappings) == 2
        assert ProviderRegistry._mappings[Providers.IBM_WATSON.value] == ibm_mapping
        assert ProviderRegistry._mappings[Providers.OCI_GENERATIVE_AI.value] == oci_mapping

    def test_register_overwrites_existing_mapping(self):
        """Test that registering a provider again overwrites the previous mapping."""
        # Arrange
        first_mapping = {"version": 1}
        second_mapping = {"version": 2}

        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, first_mapping)
        ProviderRegistry.register(Providers.IBM_WATSON, second_mapping)

        # Assert
        retrieved = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        assert retrieved == second_mapping
        assert retrieved != first_mapping


class TestProviderRegistryRetrieval:
    """Test provider mapping retrieval functionality."""

    def setup_method(self):
        """Setup test mappings before each test."""
        ProviderRegistry._mappings.clear()
        self.test_ibm_mapping = {
            "tool_input": {"test": "ibm_tool"},
            "message_input": {"test": "ibm_message"}
        }
        self.test_oci_mapping = {
            "tool_input": {"test": "oci_tool"},
            "message_input": {"test": "oci_message"}
        }
        ProviderRegistry.register(Providers.IBM_WATSON, self.test_ibm_mapping)
        ProviderRegistry.register(Providers.OCI_GENERATIVE_AI, self.test_oci_mapping)

    def teardown_method(self):
        """Re-register real mappings after tests."""
        import src.mapping.provider_mapping

    def test_get_mapping_returns_correct_mapping(self):
        """Test that get_mapping returns the correct mapping for a provider."""
        # Act
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        assert ibm_mapping == self.test_ibm_mapping
        assert oci_mapping == self.test_oci_mapping

    def test_get_mapping_returns_same_reference(self):
        """Test that get_mapping returns the same reference each time."""
        # Act
        mapping1 = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        mapping2 = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert mapping1 is mapping2

    def test_get_mapping_different_providers_different_mappings(self):
        """Test that different providers return different mappings."""
        # Act
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        assert ibm_mapping is not oci_mapping
        assert ibm_mapping != oci_mapping


class TestProviderRegistryRealMappings:
    """Test that real provider mappings are registered correctly."""

    def setup_method(self):
        """Reload real mappings before each test."""
        ProviderRegistry._mappings.clear()
        # Force reload of provider_mapping module to re-register real mappings
        import importlib
        import src.mapping.provider_mapping
        importlib.reload(src.mapping.provider_mapping)

    def test_ibm_watson_mapping_is_registered(self):
        """Test that IBM Watson mapping is registered in the real system."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert mapping is not None
        assert "tool_input" in mapping
        assert "tool_output" in mapping
        assert "message_input" in mapping

    def test_oci_mapping_is_registered(self):
        """Test that OCI mapping is registered in the real system."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        assert mapping is not None
        assert "tool_input" in mapping
        assert "tool_output" in mapping
        assert "message_input" in mapping

    def test_ibm_watson_mapping_has_tool_schema_converter(self):
        """Test that IBM Watson mapping has tool schema converter."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert "tool_schema" in mapping["tool_input"]
        assert callable(mapping["tool_input"]["tool_schema"])

    def test_oci_mapping_has_tool_schema_converter(self):
        """Test that OCI mapping has tool schema converter."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        assert "tool_schema" in mapping["tool_input"]
        assert callable(mapping["tool_input"]["tool_schema"])

    def test_ibm_watson_mapping_has_message_converters(self):
        """Test that IBM Watson mapping has message converters."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        message_input = mapping["message_input"]
        assert "human_message" in message_input
        assert "system_message" in message_input
        assert "assistant_message" in message_input
        assert "tool_message" in message_input
        assert all(callable(message_input[key]) for key in message_input)

    def test_oci_mapping_has_message_converters(self):
        """Test that OCI mapping has message converters."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        message_input = mapping["message_input"]
        assert "human_message" in message_input
        assert "system_message" in message_input
        assert "assistant_message" in message_input
        assert "tool_message" in message_input
        assert all(callable(message_input[key]) for key in message_input)

    def test_ibm_watson_mapping_has_tool_calls_extractor(self):
        """Test that IBM Watson mapping has tool calls extractor."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert "tool_calls" in mapping["tool_output"]
        assert callable(mapping["tool_output"]["tool_calls"])

    def test_oci_mapping_has_tool_calls_extractor(self):
        """Test that OCI mapping has tool calls extractor."""
        # Act
        mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        # Assert
        assert "tool_calls" in mapping["tool_output"]
        assert callable(mapping["tool_output"]["tool_calls"])


class TestProviderRegistryEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry._mappings.clear()

    def teardown_method(self):
        """Re-register real mappings after tests."""
        import src.mapping.provider_mapping

    def test_get_mapping_for_unregistered_provider_raises_keyerror(self):
        """Test that getting mapping for unregistered provider raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError):
            ProviderRegistry.get_mapping(Providers.IBM_WATSON)

    def test_register_with_empty_mapping(self):
        """Test registering a provider with an empty mapping."""
        # Arrange
        empty_mapping = {}

        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, empty_mapping)
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert mapping == {}

    def test_register_with_none_mapping(self):
        """Test registering a provider with None mapping."""
        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, None)
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        # Assert
        assert mapping is None

    def test_mappings_are_class_level_not_instance_level(self):
        """Test that _mappings is shared at class level, not instance level."""
        # Arrange
        test_mapping = {"shared": "data"}

        # Act
        ProviderRegistry.register(Providers.IBM_WATSON, test_mapping)

        # Assert - accessing through class directly
        assert ProviderRegistry._mappings[Providers.IBM_WATSON.value] == test_mapping
        assert ProviderRegistry.get_mapping(Providers.IBM_WATSON) == test_mapping