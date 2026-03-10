"""Tests for OCI request strategies (Generic and Cohere).

Covers message conversion, tool conversion, tool call extraction,
request building, strategy metadata, and the _inline_schema_refs helper.
"""

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import Field

pytest.importorskip("oci", reason="OCI SDK not installed")

from obelix.adapters.outbound.oci.strategies.base_strategy import OCIRequestStrategy
from obelix.adapters.outbound.oci.strategies.cohere_strategy import (
    CohereRequestStrategy,
)
from obelix.adapters.outbound.oci.strategies.generic_strategy import (
    GenericRequestStrategy,
    ToolCallExtractionError,
    _inline_schema_refs,
)
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import ToolCall, ToolMessage, ToolResult, ToolStatus
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool

# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generic_strategy():
    return GenericRequestStrategy()


@pytest.fixture
def cohere_strategy():
    return CohereRequestStrategy()


@pytest.fixture
def sample_tool():
    @tool(name="calculator", description="Adds two numbers")
    class CalcTool(Tool):
        a: float = Field(..., description="First operand")
        b: float = Field(..., description="Second operand")

        def execute(self) -> dict:
            return {"result": self.a + self.b}

    return CalcTool()


@pytest.fixture
def sample_tool_with_refs():
    """A tool whose schema produces $defs/$ref (nested model)."""

    @tool(name="complex_tool", description="Tool with nested schema")
    class ComplexTool(Tool):
        query: str = Field(..., description="The query")

        def execute(self) -> dict:
            return {"ok": True}

    return ComplexTool()


def _make_generic_response(
    content_text: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> MagicMock:
    """Build a mock OCI response in the Generic format (choices[0].message)."""
    from obelix.adapters.outbound.oci.connection import OCIResponse

    message_data: dict[str, Any] = {}
    if content_text is not None:
        message_data["content"] = [{"type": "TEXT", "text": content_text}]
    if tool_calls is not None:
        message_data["toolCalls"] = tool_calls

    choice = {"message": message_data}
    response_data: dict[str, Any] = {
        "chatResponse": {
            "choices": [choice],
        }
    }
    if usage is not None:
        response_data["chatResponse"]["usage"] = usage

    return OCIResponse(response_data)


def _make_cohere_response(
    text: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a mock OCI response in the Cohere format."""
    from obelix.adapters.outbound.oci.connection import OCIResponse

    chat_data: dict[str, Any] = {}
    if text is not None:
        chat_data["text"] = text
    if tool_calls is not None:
        chat_data["toolCalls"] = tool_calls

    return OCIResponse({"chatResponse": chat_data})


# ---------------------------------------------------------------------------
# _inline_schema_refs
# ---------------------------------------------------------------------------


class TestInlineSchemaRefs:
    """Tests for _inline_schema_refs helper."""

    def test_no_refs_passthrough(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = _inline_schema_refs(schema)
        assert result == schema

    def test_resolves_ref(self):
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
            "type": "object",
            "properties": {"addr": {"$ref": "#/$defs/Address"}},
        }
        result = _inline_schema_refs(schema)
        assert "$defs" not in result
        assert "$ref" not in result["properties"]["addr"]
        assert result["properties"]["addr"]["type"] == "object"

    def test_nested_refs(self):
        schema = {
            "$defs": {
                "Inner": {"type": "string"},
                "Outer": {
                    "type": "object",
                    "properties": {"val": {"$ref": "#/$defs/Inner"}},
                },
            },
            "properties": {"wrapper": {"$ref": "#/$defs/Outer"}},
        }
        result = _inline_schema_refs(schema)
        assert result["properties"]["wrapper"]["properties"]["val"]["type"] == "string"

    def test_non_local_ref_preserved(self):
        schema = {
            "properties": {"ext": {"$ref": "https://example.com/schema.json"}},
        }
        result = _inline_schema_refs(schema)
        assert result["properties"]["ext"]["$ref"] == "https://example.com/schema.json"

    def test_list_in_schema(self):
        schema = {
            "$defs": {"Item": {"type": "string"}},
            "items": [{"$ref": "#/$defs/Item"}, {"type": "integer"}],
        }
        result = _inline_schema_refs(schema)
        assert result["items"][0]["type"] == "string"
        assert result["items"][1]["type"] == "integer"


# ---------------------------------------------------------------------------
# GenericRequestStrategy - Message Conversion
# ---------------------------------------------------------------------------


class TestGenericMessageConversion:
    """Tests for GenericRequestStrategy.convert_messages."""

    def test_human_message(self, generic_strategy):
        msgs = [HumanMessage(content="Hello")]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 1
        # OCI UserMessage
        assert result[0].content[0].text == "Hello"

    def test_system_message(self, generic_strategy):
        msgs = [SystemMessage(content="You are helpful")]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 1
        assert result[0].content[0].text == "You are helpful"

    def test_assistant_message_with_content(self, generic_strategy):
        msgs = [AssistantMessage(content="I can help")]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 1
        assert result[0].content[0].text == "I can help"

    def test_assistant_message_with_tool_calls(self, generic_strategy):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1})
        msgs = [AssistantMessage(content="", tool_calls=[tc])]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 1
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "calc"
        assert result[0].tool_calls[0].id == "tc1"

    def test_assistant_message_empty_content(self, generic_strategy):
        msgs = [AssistantMessage(content="")]
        result = generic_strategy.convert_messages(msgs)
        assert result[0].content == []

    def test_tool_message(self, generic_strategy):
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc1",
            result={"sum": 5},
            status=ToolStatus.SUCCESS,
        )
        msgs = [ToolMessage(tool_results=[tr])]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 1
        assert result[0].tool_call_id == "tc1"
        assert "5" in result[0].content[0].text

    def test_tool_message_error(self, generic_strategy):
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc1",
            result=None,
            status=ToolStatus.ERROR,
            error="Division by zero",
        )
        msgs = [ToolMessage(tool_results=[tr])]
        result = generic_strategy.convert_messages(msgs)
        assert "Division by zero" in result[0].content[0].text

    def test_tool_message_multiple_results(self, generic_strategy):
        tr1 = ToolResult(
            tool_name="t1", tool_call_id="tc1", result="r1", status=ToolStatus.SUCCESS
        )
        tr2 = ToolResult(
            tool_name="t2", tool_call_id="tc2", result="r2", status=ToolStatus.SUCCESS
        )
        msgs = [ToolMessage(tool_results=[tr1, tr2])]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 2

    def test_mixed_messages(self, generic_strategy):
        msgs = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AssistantMessage(content="Assistant"),
        ]
        result = generic_strategy.convert_messages(msgs)
        assert len(result) == 3

    def test_empty_messages(self, generic_strategy):
        result = generic_strategy.convert_messages([])
        assert result == []

    def test_assistant_tool_call_arguments_serialized_as_json(self, generic_strategy):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1, "b": 2})
        msgs = [AssistantMessage(content="", tool_calls=[tc])]
        result = generic_strategy.convert_messages(msgs)
        args_str = result[0].tool_calls[0].arguments
        parsed = json.loads(args_str)
        assert parsed == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# GenericRequestStrategy - Tool Conversion
# ---------------------------------------------------------------------------


class TestGenericToolConversion:
    """Tests for GenericRequestStrategy.convert_tools."""

    def test_converts_tool(self, generic_strategy, sample_tool):
        result = generic_strategy.convert_tools([sample_tool])
        assert len(result) == 1
        assert result[0].name == "calculator"
        assert result[0].description == "Adds two numbers"
        assert result[0].type == "FUNCTION"

    def test_empty_tools(self, generic_strategy):
        result = generic_strategy.convert_tools([])
        assert result == []

    def test_multiple_tools(self, generic_strategy, sample_tool):
        @tool(name="search", description="Searches the web")
        class SearchTool(Tool):
            query: str = Field(..., description="Search query")

            def execute(self) -> dict:
                return {"results": []}

        result = generic_strategy.convert_tools([sample_tool, SearchTool()])
        assert len(result) == 2
        names = [t.name for t in result]
        assert "calculator" in names
        assert "search" in names

    def test_tool_parameters_extracted(self, generic_strategy, sample_tool):
        result = generic_strategy.convert_tools([sample_tool])
        params = result[0].parameters
        assert "properties" in params
        assert "a" in params["properties"]
        assert "b" in params["properties"]


# ---------------------------------------------------------------------------
# GenericRequestStrategy - Tool Call Extraction
# ---------------------------------------------------------------------------


class TestGenericToolCallExtraction:
    """Tests for GenericRequestStrategy.extract_tool_calls."""

    def test_no_tool_calls(self, generic_strategy):
        resp = _make_generic_response(content_text="Just text")
        result = generic_strategy.extract_tool_calls(resp)
        assert result == []

    def test_empty_response_data(self, generic_strategy):
        from obelix.adapters.outbound.oci.connection import OCIResponse

        resp = OCIResponse({})
        result = generic_strategy.extract_tool_calls(resp)
        assert result == []

    def test_extract_single_tool_call(self, generic_strategy):

        # Build response with tool calls in the raw dict format
        # but the extract method accesses via DotDict attributes
        call_mock = MagicMock()
        call_mock.type = "FUNCTION"
        call_mock.id = "tc_123"
        call_mock.name = "calculator"
        call_mock.arguments = '{"a": 1, "b": 2}'

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert len(result) == 1
        assert result[0].name == "calculator"
        assert result[0].arguments == {"a": 1, "b": 2}
        assert result[0].id == "tc_123"

    def test_extract_double_encoded_arguments(self, generic_strategy):
        """Some models double-encode JSON arguments as a string."""
        call_mock = MagicMock()
        call_mock.type = "FUNCTION"
        call_mock.id = "tc_1"
        call_mock.name = "calc"
        call_mock.arguments = '"{\\"a\\": 1}"'

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert len(result) == 1
        assert result[0].arguments == {"a": 1}

    def test_extract_dict_arguments(self, generic_strategy):
        """Arguments already parsed as dict."""
        call_mock = MagicMock()
        call_mock.type = "FUNCTION"
        call_mock.id = "tc_1"
        call_mock.name = "calc"
        call_mock.arguments = {"a": 1, "b": 2}

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert result[0].arguments == {"a": 1, "b": 2}

    def test_invalid_json_raises_extraction_error(self, generic_strategy):
        call_mock = MagicMock()
        call_mock.type = "FUNCTION"
        call_mock.id = "tc_1"
        call_mock.name = "broken_tool"
        call_mock.arguments = "{invalid json"

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        with pytest.raises(ToolCallExtractionError, match="broken_tool"):
            generic_strategy.extract_tool_calls(response_mock)

    def test_non_function_type_skipped(self, generic_strategy):
        call_mock = MagicMock()
        call_mock.type = "OTHER"
        call_mock.id = "tc_1"
        call_mock.name = "tool"
        call_mock.arguments = "{}"

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert result == []

    def test_missing_id_generates_uuid(self, generic_strategy):
        call_mock = MagicMock()
        call_mock.type = "FUNCTION"
        call_mock.id = None
        call_mock.name = "calc"
        call_mock.arguments = '{"a": 1}'

        message_mock = MagicMock()
        message_mock.tool_calls = [call_mock]

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert len(result) == 1
        # Should be a valid UUID
        uuid.UUID(result[0].id)

    def test_no_choices_returns_empty(self, generic_strategy):
        chat_response_mock = MagicMock()
        chat_response_mock.choices = []

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert result == []

    def test_no_message_tool_calls_returns_empty(self, generic_strategy):
        message_mock = MagicMock()
        message_mock.tool_calls = None

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        chat_response_mock = MagicMock()
        chat_response_mock.choices = [choice_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = generic_strategy.extract_tool_calls(response_mock)
        assert result == []


# ---------------------------------------------------------------------------
# GenericRequestStrategy - Build Request
# ---------------------------------------------------------------------------


class TestGenericBuildRequest:
    """Tests for GenericRequestStrategy.build_request."""

    def test_basic_request(self, generic_strategy):
        from oci.generative_ai_inference.models import GenericChatRequest

        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=1000,
            temperature=0.5,
        )
        assert isinstance(result, GenericChatRequest)

    def test_optional_params(self, generic_strategy):
        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_sequences=["STOP"],
        )
        assert result.top_p == 0.9
        assert result.top_k == 50
        assert result.frequency_penalty == 0.5
        assert result.presence_penalty == 0.3
        assert result.stop == ["STOP"]

    def test_stream_flag(self, generic_strategy):
        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            is_stream=True,
        )
        assert result.is_stream is True

    def test_tool_choice_auto(self, generic_strategy):
        from oci.generative_ai_inference.models import ToolChoiceAuto

        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            tool_choice="AUTO",
        )
        assert isinstance(result.tool_choice, ToolChoiceAuto)

    def test_tool_choice_required(self, generic_strategy):
        from oci.generative_ai_inference.models import ToolChoiceRequired

        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            tool_choice="REQUIRED",
        )
        assert isinstance(result.tool_choice, ToolChoiceRequired)

    def test_tool_choice_none(self, generic_strategy):
        from oci.generative_ai_inference.models import ToolChoiceNone

        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            tool_choice="NONE",
        )
        assert isinstance(result.tool_choice, ToolChoiceNone)

    def test_tool_choice_case_insensitive(self, generic_strategy):
        from oci.generative_ai_inference.models import ToolChoiceAuto

        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            tool_choice="auto",
        )
        assert isinstance(result.tool_choice, ToolChoiceAuto)

    def test_invalid_tool_choice_raises(self, generic_strategy):
        with pytest.raises(ValueError, match="Invalid tool_choice"):
            generic_strategy.build_request(
                converted_messages=[],
                converted_tools=[],
                max_tokens=100,
                temperature=0.1,
                tool_choice="INVALID",
            )

    def test_tool_choice_object_passthrough(self, generic_strategy):
        from oci.generative_ai_inference.models import ToolChoiceAuto

        tc_obj = ToolChoiceAuto()
        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            tool_choice=tc_obj,
        )
        assert result.tool_choice is tc_obj

    def test_generic_specific_kwargs(self, generic_strategy):
        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            seed=42,
        )
        assert result.seed == 42

    def test_none_optional_params_not_set(self, generic_strategy):
        result = generic_strategy.build_request(
            converted_messages=[],
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            top_p=None,
            top_k=None,
        )
        assert result.top_p is None
        assert result.top_k is None


# ---------------------------------------------------------------------------
# GenericRequestStrategy - Metadata
# ---------------------------------------------------------------------------


class TestGenericMetadata:
    """Tests for GenericRequestStrategy metadata methods."""

    def test_api_format(self, generic_strategy):
        from oci.generative_ai_inference.models import BaseChatRequest

        assert generic_strategy.get_api_format() == BaseChatRequest.API_FORMAT_GENERIC

    def test_supported_prefixes(self, generic_strategy):
        prefixes = generic_strategy.get_supported_model_prefixes()
        assert "meta." in prefixes
        assert "google." in prefixes
        assert "xai." in prefixes
        assert "openai." in prefixes

    def test_is_abstract_subclass(self, generic_strategy):
        assert isinstance(generic_strategy, OCIRequestStrategy)


# ---------------------------------------------------------------------------
# CohereRequestStrategy - Message Conversion
# ---------------------------------------------------------------------------


class TestCohereMessageConversion:
    """Tests for CohereRequestStrategy.convert_messages."""

    def test_single_human_message(self, cohere_strategy):
        msgs = [HumanMessage(content="Hello")]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "Hello"
        assert result["chat_history"] is None
        assert result["tool_results"] is None

    def test_system_message(self, cohere_strategy):
        msgs = [SystemMessage(content="Be helpful"), HumanMessage(content="Hi")]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "Hi"
        assert len(result["chat_history"]) == 1

    def test_multiple_human_messages_last_is_message(self, cohere_strategy):
        msgs = [
            HumanMessage(content="First"),
            AssistantMessage(content="Reply"),
            HumanMessage(content="Second"),
        ]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "Second"
        assert len(result["chat_history"]) == 2  # First user + assistant

    def test_assistant_message_in_history(self, cohere_strategy):
        msgs = [
            HumanMessage(content="Q"),
            AssistantMessage(content="A"),
            HumanMessage(content="Follow-up"),
        ]
        result = cohere_strategy.convert_messages(msgs)
        assert len(result["chat_history"]) == 2

    def test_assistant_empty_content(self, cohere_strategy):
        msgs = [
            HumanMessage(content="Q"),
            AssistantMessage(content=""),
            HumanMessage(content="Again"),
        ]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "Again"

    def test_tool_results_set_message_to_none(self, cohere_strategy):
        """When tool_results are present, message must be None (Cohere multistep)."""
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1})
        msgs = [
            HumanMessage(content="Compute"),
            AssistantMessage(content="", tool_calls=[tc]),
            ToolMessage(
                tool_results=[
                    ToolResult(
                        tool_name="calc",
                        tool_call_id="tc1",
                        result={"sum": 5},
                        status=ToolStatus.SUCCESS,
                    )
                ]
            ),
        ]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] is None
        assert result["tool_results"] is not None
        assert len(result["tool_results"]) == 1

    def test_tool_result_error_uses_error_field(self, cohere_strategy):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1})
        msgs = [
            HumanMessage(content="Compute"),
            AssistantMessage(content="", tool_calls=[tc]),
            ToolMessage(
                tool_results=[
                    ToolResult(
                        tool_name="calc",
                        tool_call_id="tc1",
                        result=None,
                        status=ToolStatus.ERROR,
                        error="Division by zero",
                    )
                ]
            ),
        ]
        result = cohere_strategy.convert_messages(msgs)
        assert result["tool_results"] is not None
        outputs = result["tool_results"][0].outputs
        assert "Division by zero" in str(outputs)

    def test_empty_messages_returns_continue(self, cohere_strategy):
        result = cohere_strategy.convert_messages([])
        assert result["message"] == "continue"
        assert result["chat_history"] is None
        assert result["tool_results"] is None

    def test_only_system_message_returns_continue(self, cohere_strategy):
        msgs = [SystemMessage(content="System")]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "continue"

    def test_human_message_whitespace_only(self, cohere_strategy):
        msgs = [HumanMessage(content="   ")]
        result = cohere_strategy.convert_messages(msgs)
        assert result["message"] == "continue"


# ---------------------------------------------------------------------------
# CohereRequestStrategy - Tool Conversion
# ---------------------------------------------------------------------------


class TestCohereToolConversion:
    """Tests for CohereRequestStrategy.convert_tools."""

    def test_converts_tool(self, cohere_strategy, sample_tool):
        from oci.generative_ai_inference.models import CohereTool

        result = cohere_strategy.convert_tools([sample_tool])
        assert len(result) == 1
        assert isinstance(result[0], CohereTool)
        assert result[0].name == "calculator"

    def test_empty_tools(self, cohere_strategy):
        result = cohere_strategy.convert_tools([])
        assert result == []

    def test_parameter_definitions(self, cohere_strategy, sample_tool):
        result = cohere_strategy.convert_tools([sample_tool])
        params = result[0].parameter_definitions
        assert "a" in params
        assert "b" in params
        assert params["a"].is_required is True


# ---------------------------------------------------------------------------
# CohereRequestStrategy - Tool Call Extraction
# ---------------------------------------------------------------------------


class TestCohereToolCallExtraction:
    """Tests for CohereRequestStrategy.extract_tool_calls."""

    def test_no_tool_calls(self, cohere_strategy):
        resp = _make_cohere_response(text="Just text")
        result = cohere_strategy.extract_tool_calls(resp)
        assert result == []

    def test_empty_data(self, cohere_strategy):
        from obelix.adapters.outbound.oci.connection import OCIResponse

        resp = OCIResponse({})
        result = cohere_strategy.extract_tool_calls(resp)
        assert result == []

    def test_extract_tool_call(self, cohere_strategy):
        call_mock = MagicMock()
        call_mock.name = "calculator"
        call_mock.parameters = {"a": 1, "b": 2}

        chat_response_mock = MagicMock()
        chat_response_mock.tool_calls = [call_mock]

        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock

        result = cohere_strategy.extract_tool_calls(response_mock)
        assert len(result) == 1
        assert result[0].name == "calculator"
        assert result[0].arguments == {"a": 1, "b": 2}
        # Cohere generates UUID for id
        uuid.UUID(result[0].id)

    def test_no_tool_calls_attr(self, cohere_strategy):
        chat_response_mock = MagicMock(spec=[])
        response_mock = MagicMock()
        response_mock.data.chat_response = chat_response_mock
        result = cohere_strategy.extract_tool_calls(response_mock)
        assert result == []


# ---------------------------------------------------------------------------
# CohereRequestStrategy - Build Request
# ---------------------------------------------------------------------------


class TestCohereBuildRequest:
    """Tests for CohereRequestStrategy.build_request."""

    def test_basic_request(self, cohere_strategy):
        from oci.generative_ai_inference.models import CohereChatRequest

        converted_msgs = {
            "message": "Hello",
            "chat_history": None,
            "tool_results": None,
        }
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=1000,
            temperature=0.5,
        )
        assert isinstance(result, CohereChatRequest)
        assert result.message == "Hello"

    def test_none_message_becomes_empty_string(self, cohere_strategy):
        converted_msgs = {"message": None, "chat_history": None, "tool_results": None}
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
        )
        assert result.message == ""

    def test_with_chat_history(self, cohere_strategy):
        from oci.generative_ai_inference.models import CohereUserMessage

        history = [CohereUserMessage(message="Previous")]
        converted_msgs = {
            "message": "Current",
            "chat_history": history,
            "tool_results": None,
        }
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
        )
        assert result.chat_history == history

    def test_optional_params(self, cohere_strategy):
        converted_msgs = {"message": "Hi", "chat_history": None, "tool_results": None}
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_sequences=["STOP"],
        )
        assert result.top_p == 0.9
        assert result.top_k == 50

    def test_cohere_specific_params(self, cohere_strategy):
        converted_msgs = {"message": "Hi", "chat_history": None, "tool_results": None}
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            preamble_override="Custom preamble",
            safety_mode="STRICT",
        )
        assert result.preamble_override == "Custom preamble"
        assert result.safety_mode == "STRICT"

    def test_stream_flag(self, cohere_strategy):
        converted_msgs = {"message": "Hi", "chat_history": None, "tool_results": None}
        result = cohere_strategy.build_request(
            converted_messages=converted_msgs,
            converted_tools=[],
            max_tokens=100,
            temperature=0.1,
            is_stream=True,
        )
        assert result.is_stream is True


# ---------------------------------------------------------------------------
# CohereRequestStrategy - Metadata
# ---------------------------------------------------------------------------


class TestCohereMetadata:
    """Tests for CohereRequestStrategy metadata methods."""

    def test_api_format(self, cohere_strategy):
        from oci.generative_ai_inference.models import BaseChatRequest

        assert cohere_strategy.get_api_format() == BaseChatRequest.API_FORMAT_COHERE

    def test_supported_prefixes(self, cohere_strategy):
        prefixes = cohere_strategy.get_supported_model_prefixes()
        assert prefixes == ["cohere."]

    def test_is_abstract_subclass(self, cohere_strategy):
        assert isinstance(cohere_strategy, OCIRequestStrategy)


# ---------------------------------------------------------------------------
# ToolCallExtractionError
# ---------------------------------------------------------------------------


class TestToolCallExtractionError:
    """Tests for ToolCallExtractionError."""

    def test_is_exception(self):
        err = ToolCallExtractionError("bad json")
        assert isinstance(err, Exception)
        assert str(err) == "bad json"
