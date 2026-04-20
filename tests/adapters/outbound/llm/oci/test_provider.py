"""Tests for obelix.adapters.outbound.llm.oci.provider.OCILLm.

Covers constructor, strategy auto-detection, provider_type, invoke flow,
response conversion, content extraction, usage extraction, reasoning extraction,
and the retry loop for ToolCallExtractionError.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("oci", reason="OCI SDK not installed")

from obelix.adapters.outbound.llm.oci.connection import OCIConnection, OCIResponse
from obelix.adapters.outbound.llm.oci.strategies.cohere_strategy import (
    CohereRequestStrategy,
)
from obelix.adapters.outbound.llm.oci.strategies.generic_strategy import (
    GenericRequestStrategy,
    ToolCallExtractionError,
)
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.usage import Usage
from obelix.infrastructure.providers import Providers

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_connection():
    """OCIConnection mock with a mock client."""
    conn = MagicMock(spec=OCIConnection)
    conn.get_client.return_value = MagicMock()
    return conn


@pytest.fixture
def mock_oci_modules():
    """Patch OCI SDK modules to avoid import errors."""
    mock_oci = MagicMock()
    mock_oci.base_client.is_http_log_enabled = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "oci": mock_oci,
            "oci.signer": MagicMock(),
            "oci.config": MagicMock(),
            "oci.generative_ai_inference": MagicMock(),
            "oci.generative_ai_inference.models": MagicMock(),
        },
    ):
        yield mock_oci


# ---------------------------------------------------------------------------
# Constructor and initialization
# ---------------------------------------------------------------------------


class TestOCILLmConstructor:
    """Tests for OCILLm.__init__ and configuration."""

    def test_default_parameters(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )
        assert provider.model_id == "meta.llama-3.3-70b"
        assert provider.max_tokens == 3500
        assert provider.temperature == 0.1
        assert provider.top_p is None
        assert provider.top_k is None
        assert provider.frequency_penalty is None
        assert provider.presence_penalty is None
        assert provider.stop_sequences is None
        assert provider.is_stream is False

    def test_custom_parameters(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_sequences=["STOP"],
            is_stream=True,
        )
        assert provider.max_tokens == 2000
        assert provider.temperature == 0.7
        assert provider.top_p == 0.9
        assert provider.top_k == 40
        assert provider.frequency_penalty == 0.5
        assert provider.presence_penalty == 0.3
        assert provider.stop_sequences == ["STOP"]
        assert provider.is_stream is True

    def test_strategy_kwargs_stored(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
            reasoning_effort="HIGH",
        )
        assert provider.strategy_kwargs == {"reasoning_effort": "HIGH"}

    def test_explicit_strategy(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = CohereRequestStrategy()
        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="custom.model",
            strategy=strategy,
        )
        assert provider.strategy is strategy

    def test_connection_stored(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )
        assert provider.connection is mock_connection


# ---------------------------------------------------------------------------
# Strategy auto-detection
# ---------------------------------------------------------------------------


class TestStrategyDetection:
    """Tests for OCILLm._detect_strategy."""

    def test_meta_model_uses_generic(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = OCILLm._detect_strategy("meta.llama-3.3-70b-instruct")
        assert isinstance(strategy, GenericRequestStrategy)

    def test_google_model_uses_generic(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = OCILLm._detect_strategy("google.gemini-2.0-flash")
        assert isinstance(strategy, GenericRequestStrategy)

    def test_xai_model_uses_generic(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = OCILLm._detect_strategy("xai.grok-2")
        assert isinstance(strategy, GenericRequestStrategy)

    def test_openai_model_uses_generic(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = OCILLm._detect_strategy("openai.gpt-oss-120b")
        assert isinstance(strategy, GenericRequestStrategy)

    def test_cohere_model_uses_cohere(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        strategy = OCILLm._detect_strategy("cohere.command-r-plus")
        assert isinstance(strategy, CohereRequestStrategy)

    def test_unknown_model_raises(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        with pytest.raises(ValueError, match="No strategy found"):
            OCILLm._detect_strategy("unknown.model-v1")

    def test_error_shows_supported_prefixes(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        with pytest.raises(ValueError, match="meta."):
            OCILLm._detect_strategy("bad.model")


# ---------------------------------------------------------------------------
# Provider type
# ---------------------------------------------------------------------------


class TestProviderType:
    """Tests for OCILLm.provider_type."""

    def test_returns_oci_generative_ai(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )
        assert provider.provider_type == Providers.OCI_GENERATIVE_AI


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestConvertResponseToAssistantMessage:
    """Tests for OCILLm._convert_response_to_assistant_message."""

    @pytest.fixture
    def provider(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        return OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )

    def test_text_response(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "TEXT", "text": "Hello world"}]
                            }
                        }
                    ],
                    "usage": {
                        "promptTokens": 10,
                        "completionTokens": 5,
                        "totalTokens": 15,
                    },
                }
            }
        )
        result = provider._convert_response_to_assistant_message(response)
        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello world"

    def test_usage_extracted(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": ""}]}}
                    ],
                    "usage": {
                        "promptTokens": 100,
                        "completionTokens": 50,
                        "totalTokens": 150,
                    },
                }
            }
        )
        result = provider._convert_response_to_assistant_message(response)
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_no_usage_returns_none(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": "hi"}]}}
                    ]
                }
            }
        )
        result = provider._convert_response_to_assistant_message(response)
        assert result.usage is None

    def test_reasoning_extracted(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "TEXT", "text": "result"}],
                                "reasoningContent": "I thought about this carefully",
                            }
                        }
                    ]
                }
            }
        )
        result = provider._convert_response_to_assistant_message(response)
        assert result.metadata.get("reasoning") == "I thought about this carefully"

    def test_no_reasoning(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": "hi"}]}}
                    ]
                }
            }
        )
        result = provider._convert_response_to_assistant_message(response)
        assert "reasoning" not in result.metadata


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


class TestExtractContent:
    """Tests for OCILLm._extract_content."""

    @pytest.fixture
    def provider(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        return OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )

    def test_generic_format_text(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "TEXT", "text": "Part 1"},
                                    {"type": "TEXT", "text": "Part 2"},
                                ]
                            }
                        }
                    ]
                }
            }
        )
        result = provider._extract_content(response)
        assert result == "Part 1Part 2"

    def test_cohere_format_text(self, provider):
        """Falls back to chat_response.text for Cohere format."""
        response = OCIResponse({"chatResponse": {"text": "Cohere response"}})
        result = provider._extract_content(response)
        assert result == "Cohere response"

    def test_empty_response(self, provider):
        response = OCIResponse({})
        result = provider._extract_content(response)
        assert result == ""

    def test_no_choices(self, provider):
        response = OCIResponse({"chatResponse": {}})
        result = provider._extract_content(response)
        assert result == ""

    def test_no_content_in_message(self, provider):
        response = OCIResponse({"chatResponse": {"choices": [{"message": {}}]}})
        result = provider._extract_content(response)
        assert result == ""

    def test_non_text_content_skipped(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "IMAGE", "url": "http://img.png"},
                                    {"type": "TEXT", "text": "Caption"},
                                ]
                            }
                        }
                    ]
                }
            }
        )
        result = provider._extract_content(response)
        assert result == "Caption"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class TestExtractUsage:
    """Tests for OCILLm._extract_usage."""

    @pytest.fixture
    def provider(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        return OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )

    def test_usage_present(self, provider):
        response = OCIResponse(
            {
                "chatResponse": {
                    "usage": {
                        "promptTokens": 10,
                        "completionTokens": 20,
                        "totalTokens": 30,
                    }
                }
            }
        )
        result = provider._extract_usage(response)
        assert isinstance(result, Usage)
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30

    def test_usage_missing(self, provider):
        response = OCIResponse({"chatResponse": {}})
        result = provider._extract_usage(response)
        assert result is None

    def test_usage_empty_response(self, provider):
        response = OCIResponse({})
        result = provider._extract_usage(response)
        assert result is None


# ---------------------------------------------------------------------------
# Invoke flow (mocked)
# ---------------------------------------------------------------------------


class TestOCILLmInvoke:
    """Tests for OCILLm.invoke method with mocked dependencies."""

    @pytest.fixture
    def provider_with_mocks(self, mock_connection):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        provider = OCILLm(
            connection=mock_connection,
            compartment_id="ocid1.test",
            model_id="meta.llama-3.3-70b",
        )

        # Mock the strategy
        provider.strategy = MagicMock(spec=GenericRequestStrategy)
        provider.strategy.convert_tools.return_value = []
        provider.strategy.convert_messages.return_value = []
        provider.strategy.build_request.return_value = MagicMock()
        provider.strategy.extract_tool_calls.return_value = []

        return provider

    @pytest.mark.asyncio
    async def test_invoke_success(self, provider_with_mocks):
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": "Answer"}]}}
                    ],
                    "usage": {
                        "promptTokens": 10,
                        "completionTokens": 5,
                        "totalTokens": 15,
                    },
                },
                "modelId": "meta.llama-3.3-70b",
            }
        )

        mock_client = AsyncMock()
        mock_client.chat.return_value = response
        provider_with_mocks.connection.get_client.return_value = mock_client

        mock_yaml = MagicMock()
        mock_yaml.get.return_value = {"compartment_id": "ocid1.compartment.test"}

        with (
            patch(
                "obelix.infrastructure.k8s.YamlConfig",
                return_value=mock_yaml,
            ),
            patch.dict("os.environ", {"INFRASTRUCTURE_CONFIG_PATH": "/fake"}),
        ):
            result = await provider_with_mocks.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Answer"

    @pytest.mark.asyncio
    async def test_invoke_retries_on_extraction_error(self, provider_with_mocks):
        """When ToolCallExtractionError is raised, invoke retries with error feedback."""
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": "ok"}]}}
                    ],
                    "usage": {
                        "promptTokens": 5,
                        "completionTokens": 5,
                        "totalTokens": 10,
                    },
                },
                "modelId": "meta.llama-3.3-70b",
            }
        )

        mock_client = AsyncMock()
        mock_client.chat.return_value = response
        provider_with_mocks.connection.get_client.return_value = mock_client

        # First call raises, second succeeds
        provider_with_mocks.strategy.extract_tool_calls.side_effect = [
            ToolCallExtractionError("Bad JSON"),
            [],
        ]

        mock_yaml = MagicMock()
        mock_yaml.get.return_value = {"compartment_id": "ocid1.compartment.test"}

        with (
            patch(
                "obelix.infrastructure.k8s.YamlConfig",
                return_value=mock_yaml,
            ),
            patch.dict("os.environ", {"INFRASTRUCTURE_CONFIG_PATH": "/fake"}),
        ):
            result = await provider_with_mocks.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
            )

        assert isinstance(result, AssistantMessage)
        # Strategy was called twice
        assert provider_with_mocks.strategy.extract_tool_calls.call_count == 2

    @pytest.mark.asyncio
    async def test_invoke_max_extraction_retries_raises(self, provider_with_mocks):
        """After MAX_EXTRACTION_RETRIES, the error is raised."""
        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {"message": {"content": [{"type": "TEXT", "text": "x"}]}}
                    ],
                    "usage": {
                        "promptTokens": 5,
                        "completionTokens": 5,
                        "totalTokens": 10,
                    },
                },
                "modelId": "meta.llama-3.3-70b",
            }
        )

        mock_client = AsyncMock()
        mock_client.chat.return_value = response
        provider_with_mocks.connection.get_client.return_value = mock_client

        provider_with_mocks.strategy.extract_tool_calls.side_effect = (
            ToolCallExtractionError("Persistent bad JSON")
        )

        mock_yaml = MagicMock()
        mock_yaml.get.return_value = {"compartment_id": "ocid1.compartment.test"}

        with (
            patch(
                "obelix.infrastructure.k8s.YamlConfig",
                return_value=mock_yaml,
            ),
            patch.dict("os.environ", {"INFRASTRUCTURE_CONFIG_PATH": "/fake"}),
        ):
            with pytest.raises(ToolCallExtractionError, match="Persistent bad JSON"):
                await provider_with_mocks.invoke(
                    messages=[HumanMessage(content="Hello")],
                    tools=[],
                )

        assert (
            provider_with_mocks.strategy.extract_tool_calls.call_count
            == provider_with_mocks.MAX_EXTRACTION_RETRIES
        )

    @pytest.mark.asyncio
    async def test_invoke_with_response_schema(self, provider_with_mocks):
        """When response_schema is provided, it creates JsonSchemaResponseFormat."""
        from pydantic import BaseModel

        class MySchema(BaseModel):
            answer: str
            score: float

        response = OCIResponse(
            {
                "chatResponse": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {
                                        "type": "TEXT",
                                        "text": '{"answer": "yes", "score": 0.9}',
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {
                        "promptTokens": 5,
                        "completionTokens": 5,
                        "totalTokens": 10,
                    },
                },
                "modelId": "meta.llama-3.3-70b",
            }
        )

        mock_client = AsyncMock()
        mock_client.chat.return_value = response
        provider_with_mocks.connection.get_client.return_value = mock_client

        mock_yaml = MagicMock()
        mock_yaml.get.return_value = {"compartment_id": "ocid1.compartment.test"}

        with (
            patch(
                "obelix.infrastructure.k8s.YamlConfig",
                return_value=mock_yaml,
            ),
            patch.dict("os.environ", {"INFRASTRUCTURE_CONFIG_PATH": "/fake"}),
        ):
            result = await provider_with_mocks.invoke(
                messages=[HumanMessage(content="Hello")],
                tools=[],
                response_schema=MySchema,
            )

        assert isinstance(result, AssistantMessage)
        # Verify build_request was called with response_format in kwargs
        build_call_kwargs = provider_with_mocks.strategy.build_request.call_args
        assert "response_format" in build_call_kwargs.kwargs


# ---------------------------------------------------------------------------
# MAX_EXTRACTION_RETRIES constant
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for provider-level constants."""

    def test_max_extraction_retries(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        assert OCILLm.MAX_EXTRACTION_RETRIES == 3

    def test_strategies_list(self):
        from obelix.adapters.outbound.llm.oci.provider import OCILLm

        assert len(OCILLm._STRATEGIES) == 2
        types = [type(s) for s in OCILLm._STRATEGIES]
        assert GenericRequestStrategy in types
        assert CohereRequestStrategy in types
