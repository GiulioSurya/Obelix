"""Tests for obelix.adapters.outbound.llm.oci.connection.

Covers DotDict, OCIResponse, OCIServiceError hierarchy, helper functions,
OCIAsyncHttpClient (signed requests, error handling, guardrails),
and OCIConnection (singleton, thread-safety, config resolution).
"""

import base64
import hashlib
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytest.importorskip("oci", reason="OCI SDK not installed")

from obelix.adapters.outbound.llm.oci.connection import (
    DEFAULT_OCI_CONFIG_PATH,
    DotDict,
    OCIAsyncHttpClient,
    OCIConnection,
    OCIContentModerationError,
    OCIGuardrailsChecker,
    OCIIncorrectStateError,
    OCIRateLimitError,
    OCIResponse,
    OCIServerError,
    OCIServiceError,
    _build_service_endpoint,
    _calculate_content_sha256,
    _extract_text_from_chat_details,
    _serialize_oci_model,
    _snake_to_camel,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCalculateContentSha256:
    """Tests for _calculate_content_sha256."""

    def test_known_hash(self):
        body = b'{"key": "value"}'
        expected = base64.b64encode(hashlib.sha256(body).digest()).decode("utf-8")
        assert _calculate_content_sha256(body) == expected

    def test_empty_body(self):
        body = b""
        expected = base64.b64encode(hashlib.sha256(b"").digest()).decode("utf-8")
        assert _calculate_content_sha256(body) == expected

    def test_returns_string(self):
        result = _calculate_content_sha256(b"test")
        assert isinstance(result, str)


class TestBuildServiceEndpoint:
    """Tests for _build_service_endpoint."""

    def test_default_region(self):
        result = _build_service_endpoint("us-chicago-1")
        assert (
            result == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )

    def test_custom_region(self):
        result = _build_service_endpoint("eu-frankfurt-1")
        assert (
            result
            == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
        )


class TestSnakeToCamel:
    """Tests for _snake_to_camel."""

    def test_single_word(self):
        assert _snake_to_camel("hello") == "hello"

    def test_two_words(self):
        assert _snake_to_camel("hello_world") == "helloWorld"

    def test_three_words(self):
        assert _snake_to_camel("chat_response_data") == "chatResponseData"

    def test_empty_string(self):
        assert _snake_to_camel("") == ""


class TestSerializeOciModel:
    """Tests for _serialize_oci_model."""

    def test_none(self):
        assert _serialize_oci_model(None) is None

    def test_primitives(self):
        assert _serialize_oci_model("hello") == "hello"
        assert _serialize_oci_model(42) == 42
        assert _serialize_oci_model(3.14) == 3.14
        assert _serialize_oci_model(True) is True

    def test_list(self):
        result = _serialize_oci_model([1, None, "hello"])
        assert result == [1, "hello"]

    def test_dict(self):
        result = _serialize_oci_model({"a": 1, "b": None, "c": "ok"})
        assert result == {"a": 1, "c": "ok"}

    def test_oci_model_object(self):
        """Simulates an OCI SDK model with swagger_types and attribute_map."""
        mock_model = MagicMock()
        mock_model.swagger_types = {"model_id": "str", "status": "str"}
        mock_model.attribute_map = {"model_id": "modelId", "status": "status"}
        mock_model.model_id = "test-model"
        mock_model.status = "ACTIVE"
        result = _serialize_oci_model(mock_model)
        assert result == {"modelId": "test-model", "status": "ACTIVE"}

    def test_oci_model_with_none_attribute(self):
        mock_model = MagicMock()
        mock_model.swagger_types = {"model_id": "str", "description": "str"}
        mock_model.attribute_map = {"model_id": "modelId", "description": "description"}
        mock_model.model_id = "test"
        mock_model.description = None
        result = _serialize_oci_model(mock_model)
        assert result == {"modelId": "test"}

    def test_nested_list_with_none(self):
        result = _serialize_oci_model([[1, None], [None, 2]])
        assert result == [[1], [2]]

    def test_passthrough_unknown_type(self):
        """Non-primitive, non-dict, non-list, non-OCI objects pass through."""

        class Custom:
            pass

        obj = Custom()
        assert _serialize_oci_model(obj) is obj


class TestExtractTextFromChatDetails:
    """Tests for _extract_text_from_chat_details."""

    def test_plain_dict_with_messages(self):
        chat_details = {
            "chatRequest": {
                "messages": [
                    {"content": [{"text": "Hello"}, {"text": "World"}]},
                    {"content": [{"text": "How are you?"}]},
                ]
            }
        }
        result = _extract_text_from_chat_details(chat_details)
        assert result == "Hello\nWorld\nHow are you?"

    def test_empty_messages(self):
        chat_details = {"chatRequest": {"messages": []}}
        result = _extract_text_from_chat_details(chat_details)
        assert result == ""

    def test_no_chat_request(self):
        result = _extract_text_from_chat_details({})
        assert result == ""

    def test_content_blocks_without_text(self):
        chat_details = {
            "chatRequest": {
                "messages": [
                    {"content": [{"type": "image", "url": "http://example.com"}]}
                ]
            }
        }
        result = _extract_text_from_chat_details(chat_details)
        assert result == ""

    def test_oci_model_object_serialized(self):
        """When chat_details has swagger_types, it is serialized first."""
        mock_obj = MagicMock()
        mock_obj.swagger_types = {"chat_request": "ChatRequest"}
        mock_obj.attribute_map = {"chat_request": "chatRequest"}
        mock_obj.chat_request = MagicMock()
        mock_obj.chat_request.swagger_types = {"messages": "list[Message]"}
        mock_obj.chat_request.attribute_map = {"messages": "messages"}
        mock_obj.chat_request.messages = []
        result = _extract_text_from_chat_details(mock_obj)
        assert result == ""


# ---------------------------------------------------------------------------
# DotDict
# ---------------------------------------------------------------------------


class TestDotDict:
    """Tests for DotDict wrapper."""

    def test_basic_access(self):
        d = DotDict({"name": "test", "value": 42})
        assert d.name == "test"
        assert d.value == 42

    def test_camel_case_conversion(self):
        d = DotDict({"chatResponse": {"text": "hello"}})
        result = d.chat_response
        assert isinstance(result, DotDict)
        assert result.text == "hello"

    def test_exact_key_preferred(self):
        d = DotDict({"chat_response": "exact", "chatResponse": "camel"})
        assert d.chat_response == "exact"

    def test_missing_key_returns_none(self):
        d = DotDict({"a": 1})
        assert d.nonexistent is None

    def test_nested_dict_wrapped(self):
        d = DotDict({"outer": {"inner": "value"}})
        assert isinstance(d.outer, DotDict)
        assert d.outer.inner == "value"

    def test_list_access_by_index(self):
        d = DotDict({"items": [{"name": "a"}, {"name": "b"}]})
        items = d.items
        assert isinstance(items, list)
        assert isinstance(items[0], DotDict)
        assert items[0].name == "a"
        assert items[1].name == "b"

    def test_getitem_dict(self):
        d = DotDict({"key": "value"})
        assert d["key"] == "value"

    def test_getitem_list(self):
        d = DotDict([10, 20, 30])
        assert d[0] == 10
        assert d[2] == 30

    def test_getitem_non_subscriptable(self):
        d = DotDict("string_value")
        with pytest.raises(TypeError, match="not subscriptable"):
            d[0]

    def test_len(self):
        assert len(DotDict([1, 2, 3])) == 3
        assert len(DotDict({"a": 1, "b": 2})) == 2

    def test_iter_list(self):
        d = DotDict([{"x": 1}, {"x": 2}])
        results = list(d)
        assert len(results) == 2
        assert results[0].x == 1

    def test_iter_dict(self):
        d = DotDict({"a": 1, "b": 2})
        keys = list(d)
        assert set(keys) == {"a", "b"}

    def test_bool_truthy(self):
        assert bool(DotDict({"a": 1})) is True
        assert bool(DotDict([1])) is True

    def test_bool_falsy(self):
        assert bool(DotDict({})) is False
        assert bool(DotDict([])) is False

    def test_repr(self):
        d = DotDict({"a": 1})
        assert repr(d) == "DotDict({'a': 1})"

    def test_get_existing(self):
        d = DotDict({"key": "value"})
        assert d.get("key") == "value"

    def test_get_missing_with_default(self):
        d = DotDict({"key": "value"})
        assert d.get("missing", "fallback") == "fallback"

    def test_get_none_value_returns_default(self):
        d = DotDict({"key": None})
        # None maps to default
        assert d.get("key", "fallback") == "fallback"

    def test_non_dict_data_attribute_error(self):
        d = DotDict(42)
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = d.some_attr

    def test_deep_nesting(self):
        data = {"l1": {"l2": {"l3": {"value": "deep"}}}}
        d = DotDict(data)
        assert d.l1.l2.l3.value == "deep"


# ---------------------------------------------------------------------------
# OCIResponse
# ---------------------------------------------------------------------------


class TestOCIResponse:
    """Tests for OCIResponse wrapper."""

    def test_data_property(self):
        resp = OCIResponse({"modelId": "test-model"})
        assert isinstance(resp.data, DotDict)
        assert resp.data.model_id == "test-model"

    def test_repr(self):
        resp = OCIResponse({"key": "val"})
        assert "OCIResponse" in repr(resp)

    def test_nested_response_access(self):
        data = {
            "chatResponse": {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"totalTokens": 100},
            }
        }
        resp = OCIResponse(data)
        assert resp.data.chat_response.choices[0].message.content == "hello"
        assert resp.data.chat_response.usage.total_tokens == 100


# ---------------------------------------------------------------------------
# OCIServiceError hierarchy
# ---------------------------------------------------------------------------


class TestOCIServiceErrorHierarchy:
    """Tests for error classes."""

    def test_service_error_fields(self):
        err = OCIServiceError(
            status=400,
            code="BadRequest",
            message="Invalid input",
            opc_request_id="req-123",
            headers={"x-test": "1"},
        )
        assert err.status == 400
        assert err.code == "BadRequest"
        assert err.message == "Invalid input"
        assert err.opc_request_id == "req-123"
        assert err.headers == {"x-test": "1"}

    def test_service_error_format_message(self):
        err = OCIServiceError(
            status=500, code="ServerError", message="Oops", opc_request_id="req-999"
        )
        formatted = str(err)
        assert "(500) ServerError: Oops" in formatted
        assert "opc-request-id: req-999" in formatted

    def test_service_error_no_opc_request_id(self):
        err = OCIServiceError(status=400, code="Bad", message="msg")
        formatted = str(err)
        assert "opc-request-id" not in formatted

    def test_default_headers(self):
        err = OCIServiceError(status=400, code="x", message="y")
        assert err.headers == {}

    def test_rate_limit_error_is_service_error(self):
        err = OCIRateLimitError(status=429, code="TooManyRequests", message="slow down")
        assert isinstance(err, OCIServiceError)
        assert err.status == 429

    def test_server_error_is_service_error(self):
        err = OCIServerError(status=503, code="Unavailable", message="retry later")
        assert isinstance(err, OCIServiceError)
        assert err.status == 503

    def test_incorrect_state_error_is_service_error(self):
        err = OCIIncorrectStateError(
            status=409, code="IncorrectState", message="conflict"
        )
        assert isinstance(err, OCIServiceError)
        assert err.status == 409

    def test_content_moderation_error_replaces_message(self):
        err = OCIContentModerationError(
            status=400, code="BadRequest", message="Unsafe Text detected in input"
        )
        assert isinstance(err, OCIServiceError)
        assert err.technical_message == "Unsafe Text detected in input"
        assert "moderation system" in err.message
        assert "Unsafe Text detected" not in err.message

    def test_content_moderation_user_message_constant(self):
        assert "moderation" in OCIContentModerationError.USER_MESSAGE.lower()


# ---------------------------------------------------------------------------
# OCIAsyncHttpClient
# ---------------------------------------------------------------------------


class TestOCIAsyncHttpClientInit:
    """Tests for OCIAsyncHttpClient initialization."""

    @patch("obelix.adapters.outbound.llm.oci.connection.Signer", create=True)
    def test_init_with_key_file(self, mock_signer_cls):
        """Constructor accepts key_file config and creates Signer."""
        # Patch the import inside the class
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock(Signer=mock_signer_cls)},
        ):
            config = {
                "tenancy": "ocid1.tenancy.oc1..test",
                "user": "ocid1.user.oc1..test",
                "fingerprint": "aa:bb:cc",
                "key_file": "/path/to/key.pem",
                "region": "us-phoenix-1",
            }
            client = OCIAsyncHttpClient(config)
            assert client._region == "us-phoenix-1"
            assert (
                client._endpoint
                == "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com"
            )

    @patch("obelix.adapters.outbound.llm.oci.connection.Signer", create=True)
    def test_init_default_region(self, mock_signer_cls):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock(Signer=mock_signer_cls)},
        ):
            config = {
                "tenancy": "t",
                "user": "u",
                "fingerprint": "f",
                "key_file": "/k",
            }
            client = OCIAsyncHttpClient(config)
            assert client._region == "us-chicago-1"

    @patch("obelix.adapters.outbound.llm.oci.connection.Signer", create=True)
    def test_guardrails_checker_attached(self, mock_signer_cls):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock(Signer=mock_signer_cls)},
        ):
            checker = MagicMock(spec=OCIGuardrailsChecker)
            config = {
                "tenancy": "t",
                "user": "u",
                "fingerprint": "f",
                "key_file": "/k",
            }
            client = OCIAsyncHttpClient(config, guardrails_checker=checker)
            checker.attach.assert_called_once_with(client)


class TestOCIAsyncHttpClientRaiseForStatus:
    """Tests for OCIAsyncHttpClient._raise_for_status static method."""

    def _make_response(self, status_code, json_body=None, text="", headers=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.text = text or (json.dumps(json_body) if json_body else "")
        if json_body is not None:
            resp.json.return_value = json_body
        else:
            resp.json.side_effect = json.JSONDecodeError("", "", 0)
        return resp

    def test_429_raises_rate_limit(self):
        resp = self._make_response(
            429, {"code": "TooManyRequests", "message": "Rate limited"}
        )
        with pytest.raises(OCIRateLimitError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.status == 429

    def test_500_raises_server_error(self):
        resp = self._make_response(500, {"code": "InternalError", "message": "Boom"})
        with pytest.raises(OCIServerError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.status == 500

    def test_503_raises_server_error(self):
        resp = self._make_response(503, {"code": "Unavailable", "message": "Down"})
        with pytest.raises(OCIServerError):
            OCIAsyncHttpClient._raise_for_status(resp)

    def test_409_incorrect_state(self):
        resp = self._make_response(
            409, {"code": "IncorrectState", "message": "conflict"}
        )
        with pytest.raises(OCIIncorrectStateError):
            OCIAsyncHttpClient._raise_for_status(resp)

    def test_409_other_code_raises_generic(self):
        resp = self._make_response(
            409, {"code": "OtherConflict", "message": "not incorrect state"}
        )
        with pytest.raises(OCIServiceError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert type(exc_info.value) is OCIServiceError

    def test_400_unsafe_text_raises_content_moderation(self):
        resp = self._make_response(
            400, {"code": "BadRequest", "message": "Unsafe Text detected in response"}
        )
        with pytest.raises(OCIContentModerationError):
            OCIAsyncHttpClient._raise_for_status(resp)

    def test_400_moderation_flagged_raises_content_moderation(self):
        resp = self._make_response(
            400,
            {
                "code": "BadRequest",
                "message": "Content was blocked by moderation system flagged",
            },
        )
        with pytest.raises(OCIContentModerationError):
            OCIAsyncHttpClient._raise_for_status(resp)

    def test_400_other_raises_generic(self):
        resp = self._make_response(
            400, {"code": "BadRequest", "message": "Invalid parameter"}
        )
        with pytest.raises(OCIServiceError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert type(exc_info.value) is OCIServiceError

    def test_json_decode_error_fallback(self):
        resp = self._make_response(400, text="plain text error")
        with pytest.raises(OCIServiceError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.code == "HTTP_400"
        assert exc_info.value.message == "plain text error"

    def test_opc_request_id_extracted(self):
        resp = self._make_response(
            500,
            {"code": "Error", "message": "fail"},
            headers={"opc-request-id": "req-abc"},
        )
        with pytest.raises(OCIServerError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.opc_request_id == "req-abc"

    def test_gemini_nested_error_format(self):
        """Handles Gemini-style nested error format: {"error": {"code": ..., "message": ...}}"""
        resp = self._make_response(
            400,
            {"error": {"code": 400, "message": "Gemini error", "status": "INVALID"}},
        )
        with pytest.raises(OCIServiceError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.message == "Gemini error"
        assert exc_info.value.code == "400"

    def test_content_moderation_with_request_body_preview(self):
        """Content moderation logs request body preview."""
        resp = self._make_response(
            400,
            {"code": "BadRequest", "message": "Unsafe Text detected"},
        )
        body = b'{"messages": [{"content": "test"}]}'
        with pytest.raises(OCIContentModerationError):
            OCIAsyncHttpClient._raise_for_status(resp, body)

    def test_missing_code_uses_http_fallback(self):
        resp = self._make_response(403, {"message": "Forbidden"})
        with pytest.raises(OCIServiceError) as exc_info:
            OCIAsyncHttpClient._raise_for_status(resp)
        assert exc_info.value.code == "HTTP_403"


class TestOCIAsyncHttpClientRequest:
    """Tests for OCIAsyncHttpClient._request (async)."""

    @pytest.fixture
    def mock_client(self):
        """Create an OCIAsyncHttpClient with mocked internals."""
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._endpoint = "https://test.endpoint.com"
                client._timeout = 120.0
                client._signer = MagicMock()
                client._client = None
                client._client_loop = None
                client._guardrails_checker = None
                return client

    @pytest.mark.asyncio
    async def test_request_success(self, mock_client):
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_http.request.return_value = mock_response

        mock_client._sign_request = MagicMock(return_value={"authorization": "sig"})
        mock_client._get_client = AsyncMock(return_value=mock_http)

        result = await mock_client._request("POST", "/actions/chat", {"key": "val"})
        assert result == {"result": "ok"}
        mock_http.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_error_raises(self, mock_client):
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.text = '{"code": "Error", "message": "fail"}'
        mock_response.json.return_value = {"code": "Error", "message": "fail"}
        mock_http.request.return_value = mock_response

        mock_client._sign_request = MagicMock(return_value={"authorization": "sig"})
        mock_client._get_client = AsyncMock(return_value=mock_http)

        with pytest.raises(OCIServerError):
            await mock_client._request("POST", "/actions/chat", {"key": "val"})

    @pytest.mark.asyncio
    async def test_request_none_body(self, mock_client):
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_http.request.return_value = mock_response

        mock_client._sign_request = MagicMock(return_value={"authorization": "sig"})
        mock_client._get_client = AsyncMock(return_value=mock_http)

        result = await mock_client._request("GET", "/health")
        assert result == {"ok": True}
        call_kwargs = mock_http.request.call_args
        assert call_kwargs.kwargs["content"] is None

    @pytest.mark.asyncio
    async def test_request_serializes_oci_model(self, mock_client):
        """When body has swagger_types, it is serialized via _serialize_oci_model."""
        mock_body = MagicMock()
        mock_body.swagger_types = {"field": "str"}
        mock_body.attribute_map = {"field": "field"}
        mock_body.field = "value"

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_http.request.return_value = mock_response

        mock_client._sign_request = MagicMock(return_value={"authorization": "sig"})
        mock_client._get_client = AsyncMock(return_value=mock_http)

        await mock_client._request("POST", "/actions/chat", mock_body)

        call_kwargs = mock_http.request.call_args
        body_bytes = call_kwargs.kwargs["content"]
        parsed = json.loads(body_bytes)
        assert parsed == {"field": "value"}


class TestOCIAsyncHttpClientChat:
    """Tests for OCIAsyncHttpClient.chat method."""

    @pytest.fixture
    def mock_client(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._guardrails_checker = None
                client._request = AsyncMock(
                    return_value={
                        "chatResponse": {"choices": [{"message": {"content": "hello"}}]}
                    }
                )
                return client

    @pytest.mark.asyncio
    async def test_chat_returns_oci_response(self, mock_client):
        result = await mock_client.chat({"messages": []})
        assert isinstance(result, OCIResponse)
        assert result.data.chat_response.choices[0].message.content == "hello"

    @pytest.mark.asyncio
    async def test_chat_with_guardrails(self, mock_client):
        checker = MagicMock(spec=OCIGuardrailsChecker)
        checker.check = AsyncMock()
        mock_client._guardrails_checker = checker

        chat_details = {"chatRequest": {"messages": [{"content": [{"text": "test"}]}]}}
        await mock_client.chat(chat_details)

        checker.check.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_without_guardrails(self, mock_client):
        """No guardrails checker means no pre-check."""
        await mock_client.chat({"messages": []})
        mock_client._request.assert_called_once()


class TestOCIAsyncHttpClientEmbedText:
    """Tests for OCIAsyncHttpClient.embed_text."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_oci_response(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._request = AsyncMock(
                    return_value={"embeddings": [[0.1, 0.2, 0.3]]}
                )
                result = await client.embed_text({"inputs": ["hello"]})
                assert isinstance(result, OCIResponse)
                assert result.data.embeddings[0] == [0.1, 0.2, 0.3]


class TestOCIAsyncHttpClientGetClient:
    """Tests for lazy client creation and event loop handling."""

    @pytest.mark.asyncio
    async def test_get_client_creates_httpx_client(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._timeout = 30.0
                client._client = None
                client._client_loop = None

                result = await client._get_client()
                assert isinstance(result, httpx.AsyncClient)
                assert client._client is result
                assert client._client_loop is not None

                # Cleanup
                await result.aclose()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._timeout = 30.0
                client._client = None
                client._client_loop = None

                first = await client._get_client()
                second = await client._get_client()
                assert first is second

                await first.aclose()


class TestOCIAsyncHttpClientClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                mock_http = AsyncMock()
                client._client = mock_http

                await client.close()
                mock_http.aclose.assert_called_once()
                assert client._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            with patch(
                "obelix.adapters.outbound.llm.oci.connection.OCIAsyncHttpClient.__init__",
                return_value=None,
            ):
                client = OCIAsyncHttpClient.__new__(OCIAsyncHttpClient)
                client._client = None
                # Should not raise
                await client.close()


# ---------------------------------------------------------------------------
# OCIGuardrailsChecker
# ---------------------------------------------------------------------------


class TestOCIGuardrailsChecker:
    """Tests for OCIGuardrailsChecker."""

    def test_init_defaults(self):
        checker = OCIGuardrailsChecker(compartment_id="ocid1.compartment.test")
        assert checker.enabled is False
        assert checker._threshold == 0.5
        assert checker._client is None

    def test_enabled_flag(self):
        checker = OCIGuardrailsChecker(
            compartment_id="test", enabled=True, threshold=0.8
        )
        assert checker.enabled is True
        assert checker._threshold == 0.8

    def test_attach(self):
        checker = OCIGuardrailsChecker(compartment_id="test")
        mock_client = MagicMock()
        checker.attach(mock_client)
        assert checker._client is mock_client

    @pytest.mark.asyncio
    async def test_check_disabled_skips(self):
        checker = OCIGuardrailsChecker(compartment_id="test", enabled=False)
        # Should not raise even without a client
        await checker.check("some text")

    @pytest.mark.asyncio
    async def test_check_no_client_skips(self):
        checker = OCIGuardrailsChecker(compartment_id="test", enabled=True)
        await checker.check("some text")

    @pytest.mark.asyncio
    async def test_check_empty_text_skips(self):
        checker = OCIGuardrailsChecker(compartment_id="test", enabled=True)
        checker._client = MagicMock()
        await checker.check("   ")

    @pytest.mark.asyncio
    async def test_check_calls_apply_guardrails(self):
        checker = OCIGuardrailsChecker(compartment_id="test", enabled=True)
        mock_client = MagicMock()
        mock_client.apply_guardrails = AsyncMock(
            return_value={
                "results": {
                    "contentModeration": {"categories": []},
                    "promptInjection": {"score": 0.0},
                }
            }
        )
        checker.attach(mock_client)
        await checker.check("Hello world")
        mock_client.apply_guardrails.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exception_swallowed(self):
        checker = OCIGuardrailsChecker(compartment_id="test", enabled=True)
        mock_client = MagicMock()
        mock_client.apply_guardrails = AsyncMock(side_effect=RuntimeError("boom"))
        checker.attach(mock_client)
        # Should not raise
        await checker.check("Hello world")

    def test_evaluate_flagged(self):
        checker = OCIGuardrailsChecker(compartment_id="test", threshold=0.5)
        result = {
            "contentModeration": {
                "categories": [
                    {"name": "HATE", "score": 0.9},
                    {"name": "SAFE", "score": 0.1},
                ]
            },
            "promptInjection": {"score": 0.0},
        }
        # Should not raise, just log
        checker._evaluate(result, "test text", "test_label")

    def test_evaluate_prompt_injection_flagged(self):
        checker = OCIGuardrailsChecker(compartment_id="test", threshold=0.3)
        result = {
            "contentModeration": {"categories": []},
            "promptInjection": {"score": 0.8},
        }
        checker._evaluate(result, "injected text", "test_label")

    def test_evaluate_all_ok(self):
        checker = OCIGuardrailsChecker(compartment_id="test", threshold=0.5)
        result = {
            "contentModeration": {"categories": [{"name": "SAFE", "score": 0.05}]},
            "promptInjection": {"score": 0.01},
        }
        checker._evaluate(result, "safe text", "label")

    def test_evaluate_result_at_top_level(self):
        """When 'results' key wraps the actual results."""
        checker = OCIGuardrailsChecker(compartment_id="test", threshold=0.5)
        result = {
            "results": {
                "contentModeration": {"categories": []},
                "promptInjection": {"score": 0.0},
            }
        }
        checker._evaluate(result, "text", "label")


# ---------------------------------------------------------------------------
# OCIConnection (singleton)
# ---------------------------------------------------------------------------


class TestOCIConnection:
    """Tests for OCIConnection singleton and config resolution."""

    def setup_method(self):
        """Reset singleton state before each test."""
        OCIConnection._instance = None
        OCIConnection._client = None
        OCIConnection._initialized = False

    def test_singleton_returns_same_instance(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        c1 = OCIConnection(oci_config=config)
        c2 = OCIConnection(oci_config=config)
        assert c1 is c2

    def test_singleton_thread_safety(self):
        """Multiple threads all get the same instance."""
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        instances = []

        def create():
            instances.append(OCIConnection(oci_config=config))

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_resolve_config_dict(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        conn = OCIConnection(oci_config=config)
        assert conn._oci_config is config

    def test_resolve_config_nonexistent_path_raises(self):
        with pytest.raises(ValueError, match="OCI config file not found"):
            OCIConnection(oci_config="/nonexistent/path/config")

    def test_resolve_config_none_no_default_file_raises(self):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="OCI config file not found"):
                OCIConnection(oci_config=None)

    def test_resolve_config_from_file(self):
        mock_from_file = MagicMock(
            return_value={"tenancy": "t", "user": "u", "fingerprint": "f"}
        )
        with (
            patch("os.path.exists", return_value=True),
            patch.dict(
                "sys.modules",
                {
                    "oci": MagicMock(),
                    "oci.config": MagicMock(from_file=mock_from_file),
                },
            ),
        ):
            OCIConnection(oci_config="/valid/config")
            mock_from_file.assert_called_once_with("/valid/config", "DEFAULT")

    def test_get_client_returns_async_http_client(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        conn = OCIConnection(oci_config=config)

        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            client = conn.get_client()
            assert isinstance(client, OCIAsyncHttpClient)

    def test_get_client_lazy_singleton(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        conn = OCIConnection(oci_config=config)

        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):
            c1 = conn.get_client()
            c2 = conn.get_client()
            assert c1 is c2

    def test_get_client_thread_safety(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        conn = OCIConnection(oci_config=config)
        clients = []

        with patch.dict(
            "sys.modules",
            {"oci": MagicMock(), "oci.signer": MagicMock()},
        ):

            def get():
                clients.append(conn.get_client())

            threads = [threading.Thread(target=get) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(c is clients[0] for c in clients)

    def test_initialized_flag_prevents_reinit(self):
        config1 = {"tenancy": "t1", "user": "u", "fingerprint": "f", "key_file": "/k"}
        config2 = {"tenancy": "t2", "user": "u", "fingerprint": "f", "key_file": "/k"}
        OCIConnection(oci_config=config1)
        conn2 = OCIConnection(oci_config=config2)
        # Second init is skipped, config remains from first
        assert conn2._oci_config["tenancy"] == "t1"

    def test_default_config_path(self):
        import os

        expected = os.path.join(os.path.expanduser("~"), ".oci", "config")
        assert DEFAULT_OCI_CONFIG_PATH == expected

    def test_guardrails_checker_passed_to_client(self):
        config = {"tenancy": "t", "user": "u", "fingerprint": "f", "key_file": "/k"}
        checker = MagicMock(spec=OCIGuardrailsChecker)
        conn = OCIConnection(oci_config=config, guardrails_checker=checker)
        assert conn._guardrails_checker is checker
