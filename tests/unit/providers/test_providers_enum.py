"""
Unit tests for Providers enum and factory pattern.

Tests the Providers enum values, factory method for creating provider instances,
and error handling for unsupported providers.
"""

import pytest
from src.providers import Providers
from src.llm_providers.llm_abstraction import AbstractLLMProvider, SingletonMeta


@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """Clear singleton instances before each test to avoid test interference."""
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()


class TestProvidersEnum:
    """Test Providers enum values and attributes."""

    def test_providers_enum_has_ibm_watson(self):
        """Test that Providers enum has IBM_WATSON value."""
        # Act & Assert
        assert hasattr(Providers, 'IBM_WATSON')
        assert Providers.IBM_WATSON.value == "ibm_watson"

    def test_providers_enum_has_oci_generative_ai(self):
        """Test that Providers enum has OCI_GENERATIVE_AI value."""
        # Act & Assert
        assert hasattr(Providers, 'OCI_GENERATIVE_AI')
        assert Providers.OCI_GENERATIVE_AI.value == "oci"

    def test_providers_enum_values(self):
        """Test all provider enum values."""
        # Arrange
        expected_values = {"ibm_watson", "oci"}

        # Act
        actual_values = {p.value for p in Providers}

        # Assert
        assert actual_values == expected_values

    def test_providers_enum_count(self):
        """Test that there are exactly 2 providers."""
        # Act
        provider_count = len(list(Providers))

        # Assert
        assert provider_count == 2

    def test_providers_enum_membership(self):
        """Test that enum members are accessible."""
        # Act & Assert
        assert Providers.IBM_WATSON in Providers
        assert Providers.OCI_GENERATIVE_AI in Providers

    def test_providers_enum_equality(self):
        """Test provider enum equality."""
        # Arrange
        provider1 = Providers.IBM_WATSON
        provider2 = Providers.IBM_WATSON
        provider3 = Providers.OCI_GENERATIVE_AI

        # Act & Assert
        assert provider1 == provider2
        assert provider1 != provider3


class TestProvidersFactoryMethod:
    """Test the create_instance factory method."""

    def test_create_instance_returns_abstract_llm_provider(self, mocker):
        """Test that create_instance returns AbstractLLMProvider subclass."""
        # Arrange
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')
        provider_enum = Providers.IBM_WATSON

        # Act
        instance = provider_enum.create_instance()

        # Assert
        assert isinstance(instance, AbstractLLMProvider)

    def test_create_instance_ibm_watson_returns_ibm_provider(self, mocker):
        """Test that IBM_WATSON creates IBMWatsonXLLm instance."""
        # Arrange
        from src.llm_providers.ibm_provider import IBMWatsonXLLm
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')

        # Act
        instance = Providers.IBM_WATSON.create_instance()

        # Assert
        assert isinstance(instance, IBMWatsonXLLm)
        assert isinstance(instance, AbstractLLMProvider)

    def test_create_instance_oci_returns_oci_provider(self, mocker):
        """Test that OCI_GENERATIVE_AI creates OCILLm instance."""
        # Arrange
        from src.llm_providers.oci_provider import OCILLm
        mocker.patch('src.llm_providers.oci_provider.GenerativeAiInferenceClient')
        mocker.patch('src.llm_providers.oci_provider.os.getenv', return_value='mock_value')

        # Act
        instance = Providers.OCI_GENERATIVE_AI.create_instance()

        # Assert
        assert isinstance(instance, OCILLm)
        assert isinstance(instance, AbstractLLMProvider)

    def test_create_instance_returns_singleton(self, mocker):
        """Test that create_instance returns the same singleton instance."""
        # Arrange
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')

        # Act
        instance1 = Providers.IBM_WATSON.create_instance()
        instance2 = Providers.IBM_WATSON.create_instance()

        # Assert
        assert instance1 is instance2

    def test_create_instance_different_providers_different_instances(self, mocker):
        """Test that different providers return different instances."""
        # Arrange
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')
        mocker.patch('src.llm_providers.oci_provider.GenerativeAiInferenceClient')

        # Act
        ibm_instance = Providers.IBM_WATSON.create_instance()
        oci_instance = Providers.OCI_GENERATIVE_AI.create_instance()

        # Assert
        assert ibm_instance is not oci_instance
        assert type(ibm_instance) != type(oci_instance)

    def test_create_instance_multiple_calls_same_singleton(self, mocker):
        """Test that multiple calls return the same singleton for each provider."""
        # Arrange
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')
        mocker.patch('src.llm_providers.oci_provider.GenerativeAiInferenceClient')

        # Act
        ibm1 = Providers.IBM_WATSON.create_instance()
        ibm2 = Providers.IBM_WATSON.create_instance()
        oci1 = Providers.OCI_GENERATIVE_AI.create_instance()
        oci2 = Providers.OCI_GENERATIVE_AI.create_instance()

        # Assert
        assert ibm1 is ibm2
        assert oci1 is oci2
        assert ibm1 is not oci1


class TestProvidersFactoryEdgeCases:
    """Test edge cases and error handling for factory method."""

    def test_create_instance_has_required_methods(self, mocker):
        """Test that created instances have required abstract methods."""
        # Arrange
        mocker.patch('src.llm_providers.ibm_provider.ModelInference')
        mocker.patch('src.llm_providers.ibm_provider.os.getenv', return_value='mock_key')

        # Act
        instance = Providers.IBM_WATSON.create_instance()

        # Assert
        assert hasattr(instance, 'invoke')
        assert callable(instance.invoke)
        assert hasattr(instance, '_convert_messages_to_provider_format')
        assert callable(instance._convert_messages_to_provider_format)
        assert hasattr(instance, '_convert_tools_to_provider_format')
        assert callable(instance._convert_tools_to_provider_format)
        assert hasattr(instance, '_convert_response_to_assistant_message')
        assert callable(instance._convert_response_to_assistant_message)

    def test_providers_enum_iteration(self):
        """Test that we can iterate over all providers."""
        # Act
        providers_list = list(Providers)

        # Assert
        assert len(providers_list) == 2
        assert Providers.IBM_WATSON in providers_list
        assert Providers.OCI_GENERATIVE_AI in providers_list

    def test_providers_enum_by_value(self):
        """Test that we can get provider by value."""
        # Act
        provider_ibm = Providers("ibm_watson")
        provider_oci = Providers("oci")

        # Assert
        assert provider_ibm == Providers.IBM_WATSON
        assert provider_oci == Providers.OCI_GENERATIVE_AI

    def test_providers_enum_invalid_value_raises_error(self):
        """Test that invalid provider value raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            Providers("invalid_provider")