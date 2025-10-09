"""
Unit tests for GlobalConfig singleton class.

Tests the GlobalConfig singleton pattern, provider setting/getting,
and thread-safety guarantees.
"""

import pytest
from src.config import GlobalConfig
from src.providers import Providers


class TestGlobalConfigSingleton:
    """Test GlobalConfig singleton behavior."""

    def test_global_config_is_singleton(self):
        """Test that GlobalConfig returns the same instance."""
        # Arrange & Act
        instance1 = GlobalConfig()
        instance2 = GlobalConfig()

        # Assert
        assert instance1 is instance2

    def test_global_config_multiple_calls_same_instance(self):
        """Test that multiple calls return the same instance."""
        # Arrange & Act
        instances = [GlobalConfig() for _ in range(10)]

        # Assert
        assert all(inst is instances[0] for inst in instances)

    def test_global_config_singleton_preserves_state(self):
        """Test that singleton preserves state across instances."""
        # Arrange
        instance1 = GlobalConfig()
        instance1.set_provider(Providers.IBM_WATSON)

        # Act
        instance2 = GlobalConfig()
        provider = instance2.get_current_provider()

        # Assert
        assert provider == Providers.IBM_WATSON


class TestGlobalConfigProviderManagement:
    """Test provider setting and getting functionality."""

    def setup_method(self):
        """Reset provider state before each test."""
        config = GlobalConfig()
        config._current_provider = None

    def test_set_provider_ibm_watson(self):
        """Test setting IBM Watson provider."""
        # Arrange
        config = GlobalConfig()

        # Act
        config.set_provider(Providers.IBM_WATSON)
        result = config.get_current_provider()

        # Assert
        assert result == Providers.IBM_WATSON

    def test_set_provider_oci(self):
        """Test setting OCI Generative AI provider."""
        # Arrange
        config = GlobalConfig()

        # Act
        config.set_provider(Providers.OCI_GENERATIVE_AI)
        result = config.get_current_provider()

        # Assert
        assert result == Providers.OCI_GENERATIVE_AI

    def test_get_current_provider_raises_when_not_set(self):
        """Test that getting provider before setting raises ValueError."""
        # Arrange
        config = GlobalConfig()
        config._current_provider = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            config.get_current_provider()

        assert "Provider non impostato" in str(exc_info.value)

    def test_set_provider_updates_existing_value(self):
        """Test that setting provider updates the existing value."""
        # Arrange
        config = GlobalConfig()
        config.set_provider(Providers.IBM_WATSON)

        # Act
        config.set_provider(Providers.OCI_GENERATIVE_AI)
        result = config.get_current_provider()

        # Assert
        assert result == Providers.OCI_GENERATIVE_AI

    def test_provider_persists_across_instances(self):
        """Test that provider setting persists across instance calls."""
        # Arrange
        config1 = GlobalConfig()
        config1.set_provider(Providers.IBM_WATSON)

        # Act
        config2 = GlobalConfig()
        result = config2.get_current_provider()

        # Assert
        assert result == Providers.IBM_WATSON
        assert config1 is config2


class TestGlobalConfigEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset provider state before each test."""
        config = GlobalConfig()
        config._current_provider = None

    def test_set_provider_with_none_stores_none(self):
        """Test that setting provider to None is allowed."""
        # Arrange
        config = GlobalConfig()
        config.set_provider(Providers.IBM_WATSON)

        # Act
        config.set_provider(None)

        # Assert
        assert config._current_provider is None

    def test_multiple_set_provider_calls(self):
        """Test multiple provider changes work correctly."""
        # Arrange
        config = GlobalConfig()

        # Act
        config.set_provider(Providers.IBM_WATSON)
        assert config.get_current_provider() == Providers.IBM_WATSON

        config.set_provider(Providers.OCI_GENERATIVE_AI)
        assert config.get_current_provider() == Providers.OCI_GENERATIVE_AI

        config.set_provider(Providers.IBM_WATSON)
        result = config.get_current_provider()

        # Assert
        assert result == Providers.IBM_WATSON

    def test_get_current_provider_after_none_raises_error(self):
        """Test that getting provider after setting to None raises error."""
        # Arrange
        config = GlobalConfig()
        config.set_provider(None)

        # Act & Assert
        with pytest.raises(ValueError):
            config.get_current_provider()