"""Tests for obelix.adapters.outbound.llm.litellm.provider.LiteLLMProvider.

Covers constructor, provider_type, message conversion, tool conversion,
invoke flow (happy path, tool calls, usage, response_schema), tool call
extraction (valid, malformed JSON, double-encoding, retry loop),
drop_params, extra kwargs, and HTTP error propagation.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.core.model import AssistantMessage, HumanMessage, SystemMessage, ToolMessage
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult
from obelix.core.model.usage import Usage
from obelix.infrastructure.providers import Providers

# ---------------------------------------------------------------------------
# Helpers — mock LiteLLM ModelResponse structure
# ---------------------------------------------------------------------------


def _make_function(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


def _make_tool_call(
    call_id: str, name: str, arguments: str, call_type: str = "function"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=_make_function(name, arguments),
    )


def _make_message(
    content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _make_response(
    content: str | None = "Hello",
    tool_calls: list | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a mock LiteLLM ModelResponse."""
    message = _make_message(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=usage if usage is not None else _make_usage(),
    )


def _make_mock_tool(
    name: str = "calculator", description: str = "Does math"
) -> MagicMock:
    """Create a mock Tool that satisfies the Tool Protocol."""
    mock_tool = MagicMock()
    mock_tool.tool_name = name
    mock_tool.tool_description = description
    mock_tool.create_schema.return_value = MCPToolSchema(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    return mock_tool


# ---------------------------------------------------------------------------
# Fixture — patch litellm lazy imports so tests work without litellm installed
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_litellm():
    """Patch _get_litellm and _get_litellm_exceptions for all tests."""
    mock_module = MagicMock()
    mock_module.acompletion = AsyncMock(return_value=_make_response())

    # Build exception classes that behave like real exceptions
    mock_module.exceptions = SimpleNamespace(
        RateLimitError=type("RateLimitError", (Exception,), {"message": ""}),
        Timeout=type("Timeout", (Exception,), {"message": ""}),
        InternalServerError=type("InternalServerError", (Exception,), {"message": ""}),
        ServiceUnavailableError=type(
            "ServiceUnavailableError", (Exception,), {"message": ""}
        ),
        APIConnectionError=type("APIConnectionError", (Exception,), {"message": ""}),
        AuthenticationError=type(
            "AuthenticationError", (Exception,), {"message": "rate limit"}
        ),
        BadRequestError=type(
            "BadRequestError", (Exception,), {"message": "bad request"}
        ),
        NotFoundError=type("NotFoundError", (Exception,), {"message": "not found"}),
        PermissionDeniedError=type(
            "PermissionDeniedError", (Exception,), {"message": "denied"}
        ),
        ContextWindowExceededError=type(
            "ContextWindowExceededError", (Exception,), {"message": "ctx"}
        ),
        ContentPolicyViolationError=type(
            "ContentPolicyViolationError", (Exception,), {"message": "policy"}
        ),
        APIError=type("APIError", (Exception,), {"message": "api error"}),
    )

    retryable = (
        mock_module.exceptions.RateLimitError,
        mock_module.exceptions.Timeout,
        mock_module.exceptions.InternalServerError,
        mock_module.exceptions.ServiceUnavailableError,
        mock_module.exceptions.APIConnectionError,
    )

    with (
        patch(
            "obelix.adapters.outbound.llm.litellm.provider._get_litellm",
            return_value=mock_module,
        ),
        patch(
            "obelix.adapters.outbound.llm.litellm.provider._get_litellm_exceptions",
            return_value=retryable,
        ),
    ):
        yield mock_module


@pytest.fixture
def provider(mock_litellm):
    """A LiteLLMProvider with default params, litellm mocked."""
    from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

    return LiteLLMProvider(model_id="openai/gpt-4o", api_key="sk-test")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLiteLLMProviderConstructor:
    """Tests for LiteLLMProvider.__init__."""

    def test_default_parameters(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(model_id="anthropic/claude-sonnet-4-20250514")
        assert p.model_id == "anthropic/claude-sonnet-4-20250514"
        assert p.api_key is None
        assert p.base_url is None
        assert p.max_tokens == 4096
        assert p.temperature == 0.1
        assert p.top_p is None
        assert p.frequency_penalty is None
        assert p.presence_penalty is None
        assert p.seed is None
        assert p.stop is None
        assert p.tool_choice is None
        assert p.reasoning_effort is None
        assert p.timeout is None
        assert p.extra_headers is None
        assert p.drop_params is True
        assert p._extra_kwargs == {}

    def test_custom_parameters(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="ollama/llama3",
            api_key="sk-custom",
            base_url="http://localhost:11434",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            stop=["STOP", "END"],
            tool_choice="auto",
            reasoning_effort="high",
            timeout=30.0,
            extra_headers={"X-Custom": "value"},
            drop_params=False,
        )
        assert p.model_id == "ollama/llama3"
        assert p.api_key == "sk-custom"
        assert p.base_url == "http://localhost:11434"
        assert p.max_tokens == 2048
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.frequency_penalty == 0.5
        assert p.presence_penalty == 0.3
        assert p.seed == 42
        assert p.stop == ["STOP", "END"]
        assert p.tool_choice == "auto"
        assert p.reasoning_effort == "high"
        assert p.timeout == 30.0
        assert p.extra_headers == {"X-Custom": "value"}
        assert p.drop_params is False

    def test_extra_kwargs_stored(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="openai/gpt-4o",
            custom_llm_provider="my_provider",
            api_version="2024-01",
            num_retries=5,
        )
        assert p._extra_kwargs == {
            "custom_llm_provider": "my_provider",
            "api_version": "2024-01",
            "num_retries": 5,
        }

    def test_litellm_not_installed_raises_import_error(self):
        """When _get_litellm raises ImportError, constructor propagates it."""
        with patch(
            "obelix.adapters.outbound.llm.litellm.provider._get_litellm",
            side_effect=ImportError("litellm is not installed"),
        ):
            from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

            with pytest.raises(ImportError, match="litellm is not installed"):
                LiteLLMProvider(model_id="openai/gpt-4o")


# ---------------------------------------------------------------------------
# Provider type
# ---------------------------------------------------------------------------


class TestProviderType:
    """Tests for LiteLLMProvider.provider_type."""

    def test_returns_litellm(self, provider):
        assert provider.provider_type == Providers.LITELLM


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestConvertMessages:
    """Tests for LiteLLMProvider._convert_messages."""

    def test_system_message(self, provider):
        msgs = [SystemMessage(content="You are helpful.")]
        result = provider._convert_messages(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_human_message(self, provider):
        msgs = [HumanMessage(content="Hello")]
        result = provider._convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_with_content(self, provider):
        msgs = [AssistantMessage(content="Hi there")]
        result = provider._convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"

    def test_assistant_message_with_tool_calls(self, provider):
        tc = ToolCall(id="call_1", name="calc", arguments={"a": 1, "b": 2})
        msgs = [AssistantMessage(content="Let me calculate", tool_calls=[tc])]
        result = provider._convert_messages(msgs)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me calculate"
        assert len(msg["tool_calls"]) == 1

        tool_call = msg["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["id"] == "call_1"
        assert tool_call["function"]["name"] == "calc"
        assert json.loads(tool_call["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_assistant_message_no_content_no_tool_calls(self, provider):
        """Empty assistant message gets content=''."""
        msgs = [AssistantMessage()]
        result = provider._convert_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""

    def test_tool_message_with_results(self, provider):
        tool_msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="calc",
                    tool_call_id="call_1",
                    result={"sum": 3},
                ),
                ToolResult(
                    tool_name="search",
                    tool_call_id="call_2",
                    result=None,
                    error="Not found",
                ),
            ]
        )
        result = provider._convert_messages([tool_msg])

        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert "3" in result[0]["content"]

        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_2"
        assert result[1]["content"] == "Not found"

    def test_tool_message_none_result_no_error(self, provider):
        """When result is None and error is None, content is 'No result'."""
        tool_msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="noop",
                    tool_call_id="call_x",
                    result=None,
                    error=None,
                ),
            ]
        )
        result = provider._convert_messages([tool_msg])
        assert result[0]["content"] == "No result"

    def test_mixed_conversation(self, provider):
        """Full conversation with multiple message types."""
        tc = ToolCall(id="call_1", name="calc", arguments={"expr": "2+2"})
        msgs = [
            SystemMessage(content="You are a calculator."),
            HumanMessage(content="What is 2+2?"),
            AssistantMessage(content="", tool_calls=[tc]),
            ToolMessage(
                tool_results=[
                    ToolResult(tool_name="calc", tool_call_id="call_1", result="4")
                ]
            ),
            AssistantMessage(content="The answer is 4."),
        ]
        result = provider._convert_messages(msgs)

        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"

    def test_empty_message_list(self, provider):
        result = provider._convert_messages([])
        assert result == []


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


class TestConvertTools:
    """Tests for LiteLLMProvider._convert_tools."""

    def test_empty_tools(self, provider):
        result = provider._convert_tools([])
        assert result == []

    def test_single_tool(self, provider):
        tool = _make_mock_tool("calculator", "Does math")
        result = provider._convert_tools([tool])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "Does math"
        assert "properties" in result[0]["function"]["parameters"]

    def test_multiple_tools(self, provider):
        tools = [
            _make_mock_tool("calculator", "Does math"),
            _make_mock_tool("search", "Searches the web"),
        ]
        result = provider._convert_tools(tools)

        assert len(result) == 2
        names = {t["function"]["name"] for t in result}
        assert names == {"calculator", "search"}

    def test_openai_format_structure(self, provider):
        """Verify the exact OpenAI function calling format."""
        tool = _make_mock_tool()
        result = provider._convert_tools([tool])

        entry = result[0]
        assert set(entry.keys()) == {"type", "function"}
        assert set(entry["function"].keys()) == {"name", "description", "parameters"}


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    """Tests for LiteLLMProvider._extract_tool_calls."""

    def test_valid_tool_call(self, provider):
        response = _make_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "calc", json.dumps({"a": 1, "b": 2})),
            ],
        )
        result = provider._extract_tool_calls(response)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].id == "call_1"
        assert result[0].name == "calc"
        assert result[0].arguments == {"a": 1, "b": 2}

    def test_multiple_tool_calls(self, provider):
        response = _make_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "calc", json.dumps({"a": 1})),
                _make_tool_call("call_2", "search", json.dumps({"q": "test"})),
            ],
        )
        result = provider._extract_tool_calls(response)
        assert len(result) == 2
        assert result[0].name == "calc"
        assert result[1].name == "search"

    def test_no_tool_calls_returns_empty(self, provider):
        response = _make_response(content="Hello", tool_calls=None)
        result = provider._extract_tool_calls(response)
        assert result == []

    def test_empty_choices_returns_empty(self, provider):
        response = SimpleNamespace(choices=[], usage=_make_usage())
        result = provider._extract_tool_calls(response)
        assert result == []

    def test_no_choices_attr_returns_empty(self, provider):
        response = SimpleNamespace(usage=_make_usage())
        result = provider._extract_tool_calls(response)
        assert result == []

    def test_malformed_json_raises_extraction_error(self, provider):
        from obelix.adapters.outbound.llm.litellm.provider import (
            ToolCallExtractionError,
        )

        response = _make_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "calc", "{bad json}"),
            ],
        )
        with pytest.raises(ToolCallExtractionError, match="Invalid JSON"):
            provider._extract_tool_calls(response)

    def test_double_encoded_json(self, provider):
        """JSON string within JSON string should be decoded correctly."""
        inner = json.dumps({"a": 1, "b": 2})
        double_encoded = json.dumps(inner)  # '"{\\"a\\": 1, \\"b\\": 2}"'
        response = _make_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "calc", double_encoded),
            ],
        )
        result = provider._extract_tool_calls(response)
        assert len(result) == 1
        assert result[0].arguments == {"a": 1, "b": 2}

    def test_arguments_already_dict(self, provider):
        """When arguments is already a dict (not str), it should work."""
        tc = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="calc", arguments={"a": 1}),
        )
        response = _make_response(content=None, tool_calls=[tc])
        result = provider._extract_tool_calls(response)
        assert len(result) == 1
        assert result[0].arguments == {"a": 1}

    def test_non_function_type_skipped(self, provider):
        """Tool calls with type != 'function' are ignored."""
        response = _make_response(
            content=None,
            tool_calls=[
                _make_tool_call(
                    "call_1", "calc", json.dumps({"a": 1}), call_type="other"
                ),
            ],
        )
        result = provider._extract_tool_calls(response)
        assert result == []


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


class TestExtractContent:
    """Tests for LiteLLMProvider._extract_content."""

    def test_content_present(self, provider):
        response = _make_response(content="Hello world")
        assert provider._extract_content(response) == "Hello world"

    def test_content_none(self, provider):
        response = _make_response(content=None)
        assert provider._extract_content(response) == ""

    def test_no_choices(self, provider):
        response = SimpleNamespace(choices=[], usage=_make_usage())
        assert provider._extract_content(response) == ""

    def test_no_choices_attr(self, provider):
        response = SimpleNamespace(usage=_make_usage())
        assert provider._extract_content(response) == ""


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class TestExtractUsage:
    """Tests for LiteLLMProvider._extract_usage."""

    def test_usage_present(self, provider):
        response = _make_response()
        result = provider._extract_usage(response)
        assert isinstance(result, Usage)
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30

    def test_usage_custom_values(self, provider):
        response = _make_response(
            usage=_make_usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        )
        result = provider._extract_usage(response)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    def test_usage_none(self, provider):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=_make_message())], usage=None
        )
        result = provider._extract_usage(response)
        assert result is None

    def test_no_usage_attr(self, provider):
        response = SimpleNamespace(choices=[SimpleNamespace(message=_make_message())])
        result = provider._extract_usage(response)
        assert result is None


# ---------------------------------------------------------------------------
# Response to AssistantMessage conversion
# ---------------------------------------------------------------------------


class TestConvertResponseToAssistantMessage:
    """Tests for LiteLLMProvider._convert_response_to_assistant_message."""

    def test_text_response(self, provider):
        response = _make_response(content="Hello world")
        result = provider._convert_response_to_assistant_message(response)
        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello world"
        assert result.tool_calls == []

    def test_tool_call_response(self, provider):
        response = _make_response(
            content="",
            tool_calls=[
                _make_tool_call("call_1", "calc", json.dumps({"a": 1})),
            ],
        )
        result = provider._convert_response_to_assistant_message(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"

    def test_usage_included(self, provider):
        response = _make_response(
            usage=_make_usage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        )
        result = provider._convert_response_to_assistant_message(response)
        assert result.usage is not None
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 15

    def test_malformed_tool_call_raises(self, provider):
        from obelix.adapters.outbound.llm.litellm.provider import (
            ToolCallExtractionError,
        )

        response = _make_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "calc", "not-json")],
        )
        with pytest.raises(ToolCallExtractionError):
            provider._convert_response_to_assistant_message(response)


# ---------------------------------------------------------------------------
# invoke() flow
# ---------------------------------------------------------------------------


class TestInvoke:
    """Tests for LiteLLMProvider.invoke (async, mocked)."""

    @pytest.mark.asyncio
    async def test_invoke_happy_path(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(content="Answer")

        result = await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[],
        )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Answer"
        mock_litellm.acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(
            content="",
            tool_calls=[
                _make_tool_call("call_1", "calculator", json.dumps({"a": 2, "b": 3})),
            ],
        )

        result = await provider.invoke(
            messages=[HumanMessage(content="Add 2+3")],
            tools=[_make_mock_tool()],
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"a": 2, "b": 3}

    @pytest.mark.asyncio
    async def test_invoke_with_usage(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(
            content="Hi",
            usage=_make_usage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )

        result = await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[],
        )

        assert result.usage is not None
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 25
        assert result.usage.total_tokens == 75

    @pytest.mark.asyncio
    async def test_invoke_with_response_schema(self, provider, mock_litellm):
        from pydantic import BaseModel

        class MySchema(BaseModel):
            answer: str
            score: float

        mock_litellm.acompletion.return_value = _make_response(
            content='{"answer": "yes", "score": 0.9}'
        )

        result = await provider.invoke(
            messages=[HumanMessage(content="Question")],
            tools=[],
            response_schema=MySchema,
        )

        assert isinstance(result, AssistantMessage)
        # Verify response_format was passed to acompletion
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["response_format"] is MySchema

    @pytest.mark.asyncio
    async def test_invoke_passes_tools_to_acompletion(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        tool = _make_mock_tool("search", "Searches")
        await provider.invoke(
            messages=[HumanMessage(content="Search")],
            tools=[tool],
        )

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_invoke_no_tools_omits_tools_param(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[],
        )

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_invoke_tool_choice_passed_when_tools_present(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="openai/gpt-4o", api_key="sk-test", tool_choice="required"
        )
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[_make_mock_tool()],
        )

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_invoke_tool_choice_not_passed_without_tools(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="openai/gpt-4o", api_key="sk-test", tool_choice="auto"
        )
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[],
        )

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# Tool call extraction retry loop
# ---------------------------------------------------------------------------


class TestInvokeRetryLoop:
    """Tests for the tool call extraction retry logic in invoke()."""

    @pytest.mark.asyncio
    async def test_retry_on_extraction_error_then_success(self, provider, mock_litellm):
        """First response has bad tool call, second is clean text."""
        bad_response = _make_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "calc", "{bad json}")],
        )
        good_response = _make_response(content="Fixed answer")

        mock_litellm.acompletion.side_effect = [bad_response, good_response]

        result = await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[_make_mock_tool()],
        )

        assert result.content == "Fixed answer"
        assert mock_litellm.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_appends_error_feedback(self, provider, mock_litellm):
        """Error feedback message is appended to working_messages on retry."""
        bad_response = _make_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "calc", "{bad}")],
        )
        good_response = _make_response(content="ok")

        mock_litellm.acompletion.side_effect = [bad_response, good_response]

        await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[_make_mock_tool()],
        )

        # Second call should have more messages (original + error feedback)
        second_call_kwargs = mock_litellm.acompletion.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        # Original had 1 message, retry adds error feedback
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "ERROR" in messages[1]["content"]
        assert "malformed" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_raises(self, provider, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import (
            ToolCallExtractionError,
        )

        bad_response = _make_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "calc", "{bad}")],
        )
        mock_litellm.acompletion.return_value = bad_response

        with pytest.raises(ToolCallExtractionError, match="Invalid JSON"):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[_make_mock_tool()],
            )

        assert mock_litellm.acompletion.await_count == provider.MAX_EXTRACTION_RETRIES


# ---------------------------------------------------------------------------
# drop_params and extra kwargs
# ---------------------------------------------------------------------------


class TestApiParams:
    """Tests for api_params construction passed to litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_drop_params_passed(self, provider, mock_litellm):
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await provider.invoke(
            messages=[HumanMessage(content="Hello")],
            tools=[],
        )

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["drop_params"] is True

    @pytest.mark.asyncio
    async def test_drop_params_false(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(model_id="openai/gpt-4o", drop_params=False)
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(messages=[HumanMessage(content="Hello")], tools=[])

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["drop_params"] is False

    @pytest.mark.asyncio
    async def test_extra_kwargs_merged(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="openai/gpt-4o",
            api_key="sk-test",
            custom_llm_provider="my_proxy",
            num_retries=5,
        )
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(messages=[HumanMessage(content="Hello")], tools=[])

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "my_proxy"
        assert call_kwargs["num_retries"] == 5

    @pytest.mark.asyncio
    async def test_optional_params_included_when_set(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(
            model_id="openai/gpt-4o",
            api_key="sk-key",
            base_url="http://proxy",
            top_p=0.95,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            seed=123,
            stop=["END"],
            reasoning_effort="high",
            timeout=60.0,
            extra_headers={"X-Foo": "bar"},
        )
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(messages=[HumanMessage(content="Hello")], tools=[])

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-key"
        assert call_kwargs["base_url"] == "http://proxy"
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["frequency_penalty"] == 0.2
        assert call_kwargs["presence_penalty"] == 0.1
        assert call_kwargs["seed"] == 123
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["reasoning_effort"] == "high"
        assert call_kwargs["timeout"] == 60.0
        assert call_kwargs["extra_headers"] == {"X-Foo": "bar"}

    @pytest.mark.asyncio
    async def test_optional_params_omitted_when_none(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        p = LiteLLMProvider(model_id="openai/gpt-4o")
        mock_litellm.acompletion.return_value = _make_response(content="ok")

        await p.invoke(messages=[HumanMessage(content="Hello")], tools=[])

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        for param in [
            "api_key",
            "base_url",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "reasoning_effort",
            "timeout",
            "extra_headers",
        ]:
            assert param not in call_kwargs, f"{param} should not be in call kwargs"


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestHttpErrorHandling:
    """Tests for HTTP error logging and propagation via _acompletion_with_retry."""

    @pytest.mark.asyncio
    async def test_authentication_error_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.AuthenticationError
        mock_litellm.acompletion.side_effect = exc_cls("auth failed")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_bad_request_error_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.BadRequestError
        mock_litellm.acompletion.side_effect = exc_cls("bad request")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_not_found_error_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.NotFoundError
        mock_litellm.acompletion.side_effect = exc_cls("not found")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_permission_denied_error_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.PermissionDeniedError
        mock_litellm.acompletion.side_effect = exc_cls("denied")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_context_window_exceeded_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.ContextWindowExceededError
        mock_litellm.acompletion.side_effect = exc_cls("too long")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_content_policy_violation_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.ContentPolicyViolationError
        mock_litellm.acompletion.side_effect = exc_cls("policy violation")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_generic_api_error_propagates(self, provider, mock_litellm):
        exc_cls = mock_litellm.exceptions.APIError
        mock_litellm.acompletion.side_effect = exc_cls("generic error")

        with pytest.raises(exc_cls):
            await provider.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )


# ---------------------------------------------------------------------------
# __init__.py re-export
# ---------------------------------------------------------------------------


class TestModuleReExport:
    """Tests for obelix.adapters.outbound.llm.litellm.__init__."""

    def test_litellm_provider_importable(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm import LiteLLMProvider

        assert LiteLLMProvider is not None

    def test_all_exports(self):
        import obelix.adapters.outbound.llm.litellm as mod

        assert "LiteLLMProvider" in mod.__all__


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for provider-level constants."""

    def test_max_extraction_retries(self, mock_litellm):
        from obelix.adapters.outbound.llm.litellm.provider import LiteLLMProvider

        assert LiteLLMProvider.MAX_EXTRACTION_RETRIES == 3
