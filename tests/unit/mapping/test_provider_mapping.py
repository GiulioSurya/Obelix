"""
Test suite for provider mapping functionality.

This module tests the conversion functions for different LLM providers:
- IBM Watson mapping functions
- OCI Generative AI mapping functions
- Message conversions (human, system, assistant, tool)
- Tool schema conversions
- Tool call extraction

Author: Testing System
Created: 2025-01-08
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.providers import Providers, ProviderRegistry
from src.messages.standard_message import HumanMessage, SystemMessage, AssistantMessage, ToolMessage
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus
from src.tools.tool_schema import ToolSchema
from pydantic import BaseModel

# Test fixtures for common objects
@pytest.fixture
def sample_tool_schema():
    """Create a sample tool schema for testing."""

    class SampleToolSchema(ToolSchema):
        tool_name = "test_tool"
        tool_description = "Test tool for unit testing"

        param1: str
        param2: int = 42

    return SampleToolSchema

@pytest.fixture
def sample_tool_call():
    """Create a sample tool call for testing."""
    return ToolCall(
        id="call_123",
        name="test_tool",
        arguments={"param1": "test", "param2": 100}
    )

@pytest.fixture
def sample_tool_result_success():
    """Create a successful tool result."""
    return ToolResult(
        tool_call_id="call_123",
        tool_name="test_tool",
        result="Success result",
        status=ToolStatus.SUCCESS,
        execution_time=0.5
    )

@pytest.fixture
def sample_tool_result_error():
    """Create an error tool result."""
    return ToolResult(
        tool_call_id="call_456",
        tool_name="test_tool",
        result=None,
        error="Test error message",
        status=ToolStatus.ERROR,
        execution_time=0.1
    )


class TestIBMWatsonMapping:
    """Test IBM Watson provider mapping functions."""

    def test_tool_schema_conversion(self, sample_tool_schema):
        """Test IBM Watson tool schema conversion."""
        # Get IBM Watson mapping
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        tool_converter = ibm_mapping["tool_input"]["tool_schema"]

        # Create schema instance
        schema_instance = sample_tool_schema()

        # Convert to IBM Watson format
        result = tool_converter(schema_instance)

        # Verify structure
        assert result["type"] == "function"
        assert "function" in result

        function_def = result["function"]
        assert function_def["name"] == "test_tool"
        assert function_def["description"] == "Test tool for unit testing"
        assert "parameters" in function_def

        # Verify parameters structure
        params = function_def["parameters"]
        assert "properties" in params
        assert "param1" in params["properties"]
        assert "param2" in params["properties"]
        assert "required" in params
        assert "param1" in params["required"]
        assert "param2" not in params["required"]  # Has default value

    def test_human_message_conversion(self):
        """Test IBM Watson human message conversion."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        human_converter = ibm_mapping["message_input"]["human_message"]

        # Create human message
        human_msg = HumanMessage(content="Hello, how can I help you?")

        # Convert to IBM Watson format
        result = human_converter(human_msg)

        # Verify structure
        assert result["role"] == "user"
        assert result["content"] == "Hello, how can I help you?"

    def test_system_message_conversion(self):
        """Test IBM Watson system message conversion."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        system_converter = ibm_mapping["message_input"]["system_message"]

        # Create system message
        system_msg = SystemMessage(content="You are a helpful assistant.")

        # Convert to IBM Watson format
        result = system_converter(system_msg)

        # Verify structure
        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_assistant_message_with_content_only(self):
        """Test IBM Watson assistant message with content only."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        assistant_converter = ibm_mapping["message_input"]["assistant_message"]

        # Create assistant message with content only
        assistant_msg = AssistantMessage(content="I can help you with that.")

        # Convert to IBM Watson format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert result["role"] == "assistant"
        assert result["content"] == "I can help you with that."
        assert result.get("tool_calls") is None

    def test_assistant_message_with_tool_calls(self, sample_tool_call):
        """Test IBM Watson assistant message with tool calls."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        assistant_converter = ibm_mapping["message_input"]["assistant_message"]

        # Create assistant message with tool calls
        assistant_msg = AssistantMessage(
            content="I'll help you with that calculation.",
            tool_calls=[sample_tool_call]
        )

        # Convert to IBM Watson format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert result["role"] == "assistant"
        assert result["content"] == "I'll help you with that calculation."
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["id"] == "call_123"
        assert "function" in tool_call

        function_call = tool_call["function"]
        assert function_call["name"] == "test_tool"
        assert json.loads(function_call["arguments"]) == {"param1": "test", "param2": 100}

    def test_assistant_message_tool_calls_only(self, sample_tool_call):
        """Test IBM Watson assistant message with tool calls only (no content)."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        assistant_converter = ibm_mapping["message_input"]["assistant_message"]

        # Create assistant message with tool calls only
        assistant_msg = AssistantMessage(tool_calls=[sample_tool_call])

        # Convert to IBM Watson format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

    def test_assistant_message_empty(self):
        """Test IBM Watson assistant message with no content or tool calls."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        assistant_converter = ibm_mapping["message_input"]["assistant_message"]

        # Create empty assistant message
        assistant_msg = AssistantMessage()

        # Convert to IBM Watson format
        result = assistant_converter(assistant_msg)

        # Verify fallback structure
        assert result["role"] == "assistant"
        assert result["content"] == ""

    def test_tool_message_conversion(self, sample_tool_result_success, sample_tool_result_error):
        """Test IBM Watson tool message conversion."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        tool_converter = ibm_mapping["message_input"]["tool_message"]

        # Create tool message with multiple results
        tool_msg = ToolMessage(tool_results=[sample_tool_result_success, sample_tool_result_error])

        # Convert to IBM Watson format
        result = tool_converter(tool_msg)

        # Verify structure - should return a list
        assert isinstance(result, list)
        assert len(result) == 2

        # Check success result
        success_msg = result[0]
        assert success_msg["role"] == "tool"
        assert success_msg["tool_call_id"] == "call_123"
        assert success_msg["content"] == "Success result"

        # Check error result
        error_msg = result[1]
        assert error_msg["role"] == "tool"
        assert error_msg["tool_call_id"] == "call_456"
        assert error_msg["content"] == "Test error message"

    def test_tool_message_with_none_result(self):
        """Test IBM Watson tool message with None result."""
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        tool_converter = ibm_mapping["message_input"]["tool_message"]

        # Create tool result with None result and no error
        tool_result = ToolResult(
            tool_call_id="call_789",
            tool_name="test_tool",
            result=None,
            error=None,
            status=ToolStatus.SUCCESS,
            execution_time=0.1
        )

        tool_msg = ToolMessage(tool_results=[tool_result])

        # Convert to IBM Watson format
        result = tool_converter(tool_msg)

        # Verify fallback content
        assert len(result) == 1
        assert result[0]["content"] == "No result"


class TestOCIGenerativeAIMapping:
    """Test OCI Generative AI provider mapping functions."""

    def test_tool_schema_conversion(self, sample_tool_schema):
        """Test OCI Generative AI tool schema conversion."""
        # Get OCI mapping
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        tool_converter = oci_mapping["tool_input"]["tool_schema"]

        # Create schema instance
        schema_instance = sample_tool_schema()

        # Convert to OCI format
        result = tool_converter(schema_instance)

        # Verify it's a FunctionDefinition object (from OCI SDK)
        assert hasattr(result, 'type')
        assert hasattr(result, 'name')
        assert hasattr(result, 'description')
        assert hasattr(result, 'parameters')

        assert result.type == "FUNCTION"
        assert result.name == "test_tool"
        assert result.description == "Test tool for unit testing"

    def test_human_message_conversion(self):
        """Test OCI Generative AI human message conversion."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        human_converter = oci_mapping["message_input"]["human_message"]

        # Create human message
        human_msg = HumanMessage(content="Hello, how can I help you?")

        # Convert to OCI format
        result = human_converter(human_msg)

        # Verify it's a UserMessage object with TextContent
        assert hasattr(result, 'content')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert hasattr(result.content[0], 'text')
        assert result.content[0].text == "Hello, how can I help you?"

    def test_system_message_conversion(self):
        """Test OCI Generative AI system message conversion."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        system_converter = oci_mapping["message_input"]["system_message"]

        # Create system message
        system_msg = SystemMessage(content="You are a helpful assistant.")

        # Convert to OCI format
        result = system_converter(system_msg)

        # Verify it's a SystemMessage object with TextContent
        assert hasattr(result, 'content')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert result.content[0].text == "You are a helpful assistant."

    def test_assistant_message_with_content_only(self):
        """Test OCI Generative AI assistant message with content only."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        assistant_converter = oci_mapping["message_input"]["assistant_message"]

        # Create assistant message with content only
        assistant_msg = AssistantMessage(content="I can help you with that.")

        # Convert to OCI format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert hasattr(result, 'content')
        assert hasattr(result, 'tool_calls')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert result.content[0].text == "I can help you with that."
        assert result.tool_calls is None

    def test_assistant_message_with_tool_calls(self, sample_tool_call):
        """Test OCI Generative AI assistant message with tool calls."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        assistant_converter = oci_mapping["message_input"]["assistant_message"]

        # Create assistant message with tool calls
        assistant_msg = AssistantMessage(
            content="I'll help you with that calculation.",
            tool_calls=[sample_tool_call]
        )

        # Convert to OCI format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert hasattr(result, 'content')
        assert hasattr(result, 'tool_calls')
        assert isinstance(result.content, list)
        assert result.content[0].text == "I'll help you with that calculation."

        assert isinstance(result.tool_calls, list)
        assert len(result.tool_calls) == 1

        function_call = result.tool_calls[0]
        assert hasattr(function_call, 'id')
        assert hasattr(function_call, 'name')
        assert hasattr(function_call, 'arguments')
        assert function_call.id == "call_123"
        assert function_call.name == "test_tool"
        assert json.loads(function_call.arguments) == {"param1": "test", "param2": 100}

    def test_assistant_message_no_content(self, sample_tool_call):
        """Test OCI Generative AI assistant message with no content."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        assistant_converter = oci_mapping["message_input"]["assistant_message"]

        # Create assistant message with no content
        assistant_msg = AssistantMessage(tool_calls=[sample_tool_call])

        # Convert to OCI format
        result = assistant_converter(assistant_msg)

        # Verify structure
        assert result.content is None
        assert isinstance(result.tool_calls, list)
        assert len(result.tool_calls) == 1

    def test_tool_message_conversion(self, sample_tool_result_success, sample_tool_result_error):
        """Test OCI Generative AI tool message conversion."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        tool_converter = oci_mapping["message_input"]["tool_message"]

        # Create tool message with multiple results
        tool_msg = ToolMessage(tool_results=[sample_tool_result_success, sample_tool_result_error])

        # Convert to OCI format
        result = tool_converter(tool_msg)

        # Verify structure - should return a list of OCIToolMessage objects
        assert isinstance(result, list)
        assert len(result) == 2

        # Check success result
        success_msg = result[0]
        assert hasattr(success_msg, 'content')
        assert hasattr(success_msg, 'tool_call_id')
        assert isinstance(success_msg.content, list)
        assert success_msg.content[0].text == "Success result"
        assert success_msg.tool_call_id == "call_123"

        # Check error result
        error_msg = result[1]
        assert isinstance(error_msg.content, list)
        assert error_msg.content[0].text == "Test error message"
        assert error_msg.tool_call_id == "call_456"

    def test_tool_message_with_none_result(self):
        """Test OCI Generative AI tool message with None result."""
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        tool_converter = oci_mapping["message_input"]["tool_message"]

        # Create tool result with None result but has error
        tool_result = ToolResult(
            tool_call_id="call_789",
            tool_name="test_tool",
            result=None,
            error="No result available",
            status=ToolStatus.ERROR,
            execution_time=0.1
        )

        tool_msg = ToolMessage(tool_results=[tool_result])

        # Convert to OCI format
        result = tool_converter(tool_msg)

        # Verify fallback to error message
        assert len(result) == 1
        assert result[0].content[0].text == "No result available"


class TestProviderRegistryIntegration:
    """Test provider registry integration with mappings."""

    def test_ibm_watson_mapping_registered(self):
        """Test IBM Watson mapping is properly registered."""
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)

        assert mapping is not None
        assert "tool_input" in mapping
        assert "tool_output" in mapping
        assert "message_input" in mapping

        # Verify all required converters exist
        assert "tool_schema" in mapping["tool_input"]
        assert "tool_calls" in mapping["tool_output"]
        assert "human_message" in mapping["message_input"]
        assert "system_message" in mapping["message_input"]
        assert "assistant_message" in mapping["message_input"]
        assert "tool_message" in mapping["message_input"]

    def test_oci_generative_ai_mapping_registered(self):
        """Test OCI Generative AI mapping is properly registered."""
        mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)

        assert mapping is not None
        assert "tool_input" in mapping
        assert "tool_output" in mapping
        assert "message_input" in mapping

        # Verify all required converters exist
        assert "tool_schema" in mapping["tool_input"]
        assert "tool_calls" in mapping["tool_output"]
        assert "human_message" in mapping["message_input"]
        assert "system_message" in mapping["message_input"]
        assert "assistant_message" in mapping["message_input"]
        assert "tool_message" in mapping["message_input"]

    def test_tool_output_functions_exist(self):
        """Test tool output extraction functions are callable."""
        # Test IBM Watson
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        ibm_extractor = ibm_mapping["tool_output"]["tool_calls"]
        assert callable(ibm_extractor)

        # Test OCI Generative AI
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        oci_extractor = oci_mapping["tool_output"]["tool_calls"]
        assert callable(oci_extractor)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tool_message(self):
        """Test tool message conversion with empty tool results."""
        # Test IBM Watson
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        ibm_converter = ibm_mapping["message_input"]["tool_message"]

        empty_tool_msg = ToolMessage(tool_results=[])
        result = ibm_converter(empty_tool_msg)
        assert isinstance(result, list)
        assert len(result) == 0

        # Test OCI Generative AI
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        oci_converter = oci_mapping["message_input"]["tool_message"]

        result = oci_converter(empty_tool_msg)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_assistant_message_multiple_tool_calls(self):
        """Test assistant message with multiple tool calls."""
        tool_call_1 = ToolCall(id="call_1", name="tool_1", arguments={"arg1": "value1"})
        tool_call_2 = ToolCall(id="call_2", name="tool_2", arguments={"arg2": "value2"})

        assistant_msg = AssistantMessage(
            content="Using multiple tools",
            tool_calls=[tool_call_1, tool_call_2]
        )

        # Test IBM Watson
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        ibm_converter = ibm_mapping["message_input"]["assistant_message"]
        ibm_result = ibm_converter(assistant_msg)

        assert len(ibm_result["tool_calls"]) == 2
        assert ibm_result["tool_calls"][0]["id"] == "call_1"
        assert ibm_result["tool_calls"][1]["id"] == "call_2"

        # Test OCI Generative AI
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        oci_converter = oci_mapping["message_input"]["assistant_message"]
        oci_result = oci_converter(assistant_msg)

        assert len(oci_result.tool_calls) == 2
        assert oci_result.tool_calls[0].id == "call_1"
        assert oci_result.tool_calls[1].id == "call_2"

    def test_tool_schema_with_complex_parameters(self):
        """Test tool schema conversion with complex parameter types."""

        class ComplexToolSchema(ToolSchema):
            tool_name = "complex_tool"
            tool_description = "Tool with complex parameters"

            string_param: str
            optional_int: int = None
            list_param: list = []
            dict_param: dict = {}

        schema_instance = ComplexToolSchema()

        # Test IBM Watson conversion
        ibm_mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        ibm_converter = ibm_mapping["tool_input"]["tool_schema"]
        ibm_result = ibm_converter(schema_instance)

        assert ibm_result["function"]["name"] == "complex_tool"
        params = ibm_result["function"]["parameters"]
        assert "string_param" in params["properties"]
        assert "optional_int" in params["properties"]
        assert "list_param" in params["properties"]
        assert "dict_param" in params["properties"]

        # Test OCI Generative AI conversion
        oci_mapping = ProviderRegistry.get_mapping(Providers.OCI_GENERATIVE_AI)
        oci_converter = oci_mapping["tool_input"]["tool_schema"]
        oci_result = oci_converter(schema_instance)

        assert oci_result.name == "complex_tool"
        assert oci_result.description == "Tool with complex parameters"
        assert oci_result.parameters is not None


if __name__ == "__main__":
    pytest.main([__file__])