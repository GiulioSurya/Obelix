# src/connections/llm_connection/oci_connection.py
"""
Async HTTP client for OCI Generative AI API.

Uses httpx.AsyncClient for native async I/O instead of blocking threads.
Reuses oci.signer.Signer for request authentication.

The response is wrapped in OCIResponse to maintain compatibility with
existing code that expects SDK-style dot notation access.
"""

import asyncio
import base64
import email.utils
import hashlib
import json
import os
import threading
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

from src.connections.llm_connection.base_llm_connection import AbstractLLMConnection
from src.logging_config import get_logger

_logger = get_logger(__name__)

DEFAULT_OCI_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oci", "config")


def _calculate_content_sha256(body: bytes) -> str:
    """SHA256 hash of body, base64 encoded (required for OCI request signing)."""
    m = hashlib.sha256()
    m.update(body)
    return base64.b64encode(m.digest()).decode("utf-8")


def _build_service_endpoint(region: str) -> str:
    """Build the OCI Generative AI service endpoint for a region."""
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com"


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _serialize_oci_model(obj: Any) -> Any:
    """
    Serialize OCI model to JSON-compatible dict with camelCase keys.

    OCI API expects camelCase field names, but Python models use snake_case.
    The SDK stores the mapping in `attribute_map` on each model class.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [_serialize_oci_model(item) for item in obj if item is not None]

    if isinstance(obj, dict):
        return {k: _serialize_oci_model(v) for k, v in obj.items() if v is not None}

    # OCI model objects have swagger_types and attribute_map
    if hasattr(obj, "swagger_types") and hasattr(obj, "attribute_map"):
        result: dict[str, Any] = {}
        for attr in obj.swagger_types:
            value = getattr(obj, attr, None)
            if value is not None:
                key = obj.attribute_map.get(attr, attr)
                result[key] = _serialize_oci_model(value)
        return result

    return obj



class DotDict:
    """
    Wrapper that allows dict access with dot notation.

    Automatically handles camelCase to snake_case conversion for compatibility
    with existing code that expects OCI SDK-style attribute access.

    Example:
        data = {"chatResponse": {"choices": [{"message": {"content": "hi"}}]}}
        wrapped = DotDict(data)
        wrapped.chat_response.choices[0].message.content  # "hi"
    """

    def __init__(self, data: Any):
        self.data = data

    def __getattr__(self, key: str) -> Any:
        if key == 'data':
            return object.__getattribute__(self, 'data')

        raw_data = object.__getattribute__(self, 'data')

        if not isinstance(raw_data, dict):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

        # Try exact key first
        if key in raw_data:
            return self._wrap(raw_data[key])

        # Try camelCase version
        camel_key = _snake_to_camel(key)
        if camel_key in raw_data:
            return self._wrap(raw_data[camel_key])

        return None

    def __getitem__(self, key: Any) -> Any:
        raw_data = object.__getattribute__(self, 'data')
        if isinstance(raw_data, (list, dict)):
            return self._wrap(raw_data[key])
        raise TypeError(f"'{type(self).__name__}' object is not subscriptable")

    def __len__(self) -> int:
        return len(object.__getattribute__(self, 'data'))

    def __iter__(self):
        raw_data = object.__getattribute__(self, 'data')
        if isinstance(raw_data, list):
            for item in raw_data:
                yield self._wrap(item)
        elif isinstance(raw_data, dict):
            for key in raw_data:
                yield key

    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, 'data'))

    def __repr__(self) -> str:
        return f"DotDict({object.__getattribute__(self, 'data')!r})"

    def _wrap(self, value: Any) -> Any:
        """Wrap nested dicts/lists in DotDict."""
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        try:
            value = self.__getattr__(key)
            return value if value is not None else default
        except AttributeError:
            return default


class OCIResponse:
    """
    Wrapper for OCI API response that mimics SDK response structure.

    The OCI SDK returns responses with a `.data` attribute containing
    the actual response data. This wrapper maintains that interface.

    Example:
        # SDK style access still works:
        response.data.chat_response.choices[0].message.content
        response.data.chat_response.usage.total_tokens
        response.data.model_id
    """

    def __init__(self, json_data: dict[str, Any]):
        self._json_data = json_data
        self._data = DotDict(json_data)

    @property
    def data(self) -> DotDict:
        """Return response data with dot notation access."""
        return self._data

    def __repr__(self) -> str:
        return f"OCIResponse({self._json_data!r})"



class OCIServiceError(Exception):
    """
    Base OCI service error with full response details.

    Mirrors the OCI SDK's ServiceError pattern: captures status, code, message,
    opc-request-id, and response headers so no error information is ever lost.
    """

    def __init__(self, status: int, code: str | None, message: str,
                 opc_request_id: str | None = None, headers: dict | None = None):
        self.status = status
        self.code = code
        self.message = message
        self.opc_request_id = opc_request_id
        self.headers = headers or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [f"({self.status}) {self.code}: {self.message}"]
        if self.opc_request_id:
            parts.append(f"opc-request-id: {self.opc_request_id}")
        return " | ".join(parts)


class OCIRateLimitError(OCIServiceError):
    """Rate limit error (429) - retryable."""
    pass


class OCIServerError(OCIServiceError):
    """Server error (5xx) - retryable."""
    pass


class OCIIncorrectStateError(OCIServiceError):
    """Incorrect state error (409 with code=IncorrectState) - retryable."""
    pass


class OCIContentModerationError(OCIServiceError):
    """Content moderation error (400 with 'Unsafe Text detected') - NOT retryable."""

    USER_MESSAGE = (
        "The request contains content that the moderation system flagged as potentially "
        "problematic. Please try rephrasing your message."
    )

    def __init__(self, status: int, code: str | None, message: str, **kwargs):
        self.technical_message = message
        super().__init__(status, code, self.USER_MESSAGE, **kwargs)



class OCIAsyncHttpClient:
    """
    Async HTTP client for OCI Generative AI API.

    Uses httpx.AsyncClient for native async I/O instead of blocking threads.
    Reuses oci.signer.Signer for request authentication.

    Returns OCIResponse objects that maintain compatibility with existing
    code expecting OCI SDK-style response access.
    """

    BASE_PATH = "/20231130"

    def __init__(self, config: dict[str, Any], timeout: float = 120.0) -> None:
        try:
            from oci.signer import Signer
        except ImportError:
            raise ImportError("oci is not installed. Install with: pip install oci")

        self._region = config.get("region", "us-chicago-1")
        self._endpoint = _build_service_endpoint(self._region)
        self._timeout = timeout

        # Handle both key_file (path) and key_content (inline key)
        key_file = config.get("key_file")
        key_content = config.get("key_content")

        self._signer = Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=key_file,
            private_key_content=key_content,
            pass_phrase=config.get("pass_phrase"),
        )
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None  # Track which event loop owns the client

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or lazily create the async HTTP client.

        Handles event loop changes: if the current event loop is different from
        the one where the client was created (e.g., after asyncio.run() completes
        and a new one starts), recreates the client for the new loop.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Check if we need to recreate the client:
        # 1. Client doesn't exist yet
        # 2. Event loop has changed
        # 3. Previous event loop is closed
        need_new_client = (
            self._client is None or
            self._client_loop is None or
            self._client_loop != current_loop or
            self._client_loop.is_closed()
        )

        if need_new_client:
            # Close old client if it exists and its loop is still running
            if self._client is not None:
                try:
                    if self._client_loop and not self._client_loop.is_closed():
                        await self._client.aclose()
                except Exception:
                    pass  # Ignore errors closing old client

            # Create new client for current event loop
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._client_loop = current_loop

        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _sign_request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
    ) -> dict[str, str]:
        """Sign a request using OCI Signer and return headers."""
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"

        headers: dict[str, str] = {
            "date": email.utils.formatdate(usegmt=True),
            "host": host,
            "content-type": "application/json",
        }

        if body and method.upper() in ("POST", "PUT", "PATCH"):
            headers["content-length"] = str(len(body))
            headers["x-content-sha256"] = _calculate_content_sha256(body)

            signed = self._signer._body_signer.sign(
                headers,
                host=host,
                method=method.upper(),
                path=path,
            )
        else:
            signed = self._signer._basic_signer.sign(
                headers,
                host=host,
                method=method.upper(),
                path=path,
            )

        return dict(signed)

    async def _request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
    ) -> dict[str, Any]:
        """Make a signed async request to OCI. Returns raw JSON as dict."""
        url = f"{self._endpoint}{self.BASE_PATH}{path}"

        if body is not None:
            if hasattr(body, "swagger_types"):
                body_dict = _serialize_oci_model(body)
            else:
                body_dict = body
            body_bytes = json.dumps(body_dict).encode("utf-8")
        else:
            body_bytes = None

        headers = self._sign_request(method, url, body_bytes)
        headers["accept"] = "application/json"

        client = await self._get_client()

        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            content=body_bytes,
        )

        if response.status_code >= 400:
            self._raise_for_status(response)

        return response.json()

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """
        Parse error response and raise the appropriate OCIServiceError subclass.

        Follows the OCI SDK pattern: extracts code, message, and opc-request-id
        from every error response so no diagnostic information is lost.
        """
        status = response.status_code
        headers = dict(response.headers)
        opc_request_id = headers.get("opc-request-id")

        # Parse body — JSON when possible, raw text as fallback.
        # Handles both OCI format {"code": ..., "message": ...}
        # and Gemini nested format {"error": {"code": ..., "message": ..., "status": ...}}
        code: str | None = None
        message: str = response.text or f"HTTP {status}"
        try:
            error_data = response.json()
            if "error" in error_data and isinstance(error_data["error"], dict):
                error_data = error_data["error"]
            code = str(error_data["code"]) if "code" in error_data else None
            message = error_data.get("message") or message
        except (json.JSONDecodeError, ValueError):
            pass

        error_kwargs = dict(
            status=status,
            code=code or f"HTTP_{status}",
            message=message,
            opc_request_id=opc_request_id,
            headers=headers,
        )

        _logger.error(
            "OCI API error: status=%s code=%s opc-request-id=%s message=%s",
            status, error_kwargs["code"], opc_request_id, message,
        )

        if status == 429:
            raise OCIRateLimitError(**error_kwargs)

        if status >= 500:
            raise OCIServerError(**error_kwargs)

        if status == 409 and code == "IncorrectState":
            raise OCIIncorrectStateError(**error_kwargs)

        if status == 400 and "Unsafe Text detected" in message:
            raise OCIContentModerationError(**error_kwargs)

        raise OCIServiceError(**error_kwargs)

    async def chat(self, chat_details: Any) -> OCIResponse:
        """Call the chat endpoint. Returns OCIResponse compatible with SDK access."""
        json_response = await self._request("POST", "/actions/chat", chat_details)
        return OCIResponse(json_response)

    async def embed_text(self, embed_details: Any) -> OCIResponse:
        """Call the embed text endpoint. Returns OCIResponse compatible with SDK access."""
        json_response = await self._request("POST", "/actions/embedText", embed_details)
        return OCIResponse(json_response)


class OCIConnection(AbstractLLMConnection):
    """
    Thread-safe singleton to manage connection to OCI Generative AI.

    The connection is shared among all OCI providers using the same credentials.
    Client is lazy initialized on first access.

    Args:
        oci_config: OCI configuration. Can be:
            - None: loads from default path (~/.oci/config)
            - str: path to OCI config file
            - dict: configuration dictionary with keys: user, fingerprint, key_content/key_file, tenancy, region
    """

    _instance: Optional['OCIConnection'] = None
    _lock = threading.Lock()
    _client = None
    _client_lock = threading.Lock()
    _initialized = False

    def __new__(cls, oci_config: Optional[Union[dict, str]] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, oci_config: Optional[Union[dict, str]] = None):
        if self._initialized:
            return
        self._oci_config = self._resolve_config(oci_config)
        self._initialized = True

    def _resolve_config(self, oci_config: Optional[Union[dict, str]]) -> dict:
        """
        Resolves the OCI configuration from various input types.

        Args:
            oci_config: Config dict, file path, or None for default

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If config cannot be resolved
        """
        if isinstance(oci_config, dict):
            return oci_config

        config_path = oci_config if isinstance(oci_config, str) else DEFAULT_OCI_CONFIG_PATH

        if not os.path.exists(config_path):
            raise ValueError(
                f"OCI config file not found at '{config_path}'. "
                "Please provide either:\n"
                "  - A configuration dictionary with keys: user, fingerprint, key_content/key_file, tenancy, region\n"
                "  - A valid path to an OCI config file\n"
                f"  - Or create a config file at the default location: {DEFAULT_OCI_CONFIG_PATH}"
            )

        try:
            from oci.config import from_file
        except ImportError:
            raise ImportError("oci is not installed. Install with: pip install oci")

        return from_file(config_path, "DEFAULT")

    def get_client(self) -> OCIAsyncHttpClient:
        """
        Returns the configured OCI Generative AI async client.
        Lazy initialization: creates client only on first access.

        Returns:
            Configured OCIAsyncHttpClient (native async, no thread blocking)

        Raises:
            ValueError: If OCI credentials are not configured
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self) -> OCIAsyncHttpClient:
        """
        Creates and configures the OCI Generative AI async client.

        Returns:
            Configured OCIAsyncHttpClient

        Raises:
            ImportError: If oci or httpx libraries are not installed
        """
        return OCIAsyncHttpClient(self._oci_config)