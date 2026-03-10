"""JSON-RPC 2.0 handlers for A2A protocol methods."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.task_store import TaskStore
    from obelix.core.agent.base_agent import BaseAgent

logger = get_logger(__name__)


def _jsonrpc_error(
    req_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _jsonrpc_result(req_id: int | str | None, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


# Standard JSON-RPC error codes
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

# A2A-specific error codes
_TASK_NOT_FOUND = -32001
_UNSUPPORTED_OPERATION = -32002
_TASK_NOT_CANCELABLE = -32003


async def handle_send_message(
    params: dict,
    req_id: int | str | None,
    agent: BaseAgent,
    task_store: TaskStore,
) -> dict:
    """Handle the SendMessage JSON-RPC method."""
    message = params.get("message")
    if not message:
        return _jsonrpc_error(req_id, _INVALID_PARAMS, "Missing 'message' in params")

    # Extract text from the first TextPart
    parts = message.get("parts", [])
    text = None
    for part in parts:
        if isinstance(part, dict):
            # Support both {"text": "..."} and {"type": "text", "text": "..."}
            if "text" in part:
                text = part["text"]
                break

    if not text:
        return _jsonrpc_error(req_id, _INVALID_PARAMS, "No text part found in message")

    # Create task
    context_id = params.get("contextId")
    task = await task_store.create(context_id=context_id)
    task_id = task["id"]

    logger.info(f"[A2A] SendMessage | task_id={task_id} text_len={len(text)}")

    try:
        response = await agent.execute_query_async(text)

        # Create artifact with the response
        artifact = {
            "artifactId": str(uuid.uuid4()),
            "parts": [{"type": "text", "text": response.content}],
        }
        await task_store.add_artifact(task_id, artifact)
        task = await task_store.update_status(task_id, "completed")

        logger.info(f"[A2A] SendMessage completed | task_id={task_id}")
    except Exception as e:
        logger.error(f"[A2A] SendMessage failed | task_id={task_id} error={e}")
        task = await task_store.update_status(task_id, "failed", message=str(e))

    return _jsonrpc_result(req_id, task)


async def handle_get_task(
    params: dict,
    req_id: int | str | None,
    task_store: TaskStore,
) -> dict:
    """Handle the GetTask JSON-RPC method."""
    task_id = params.get("id")
    if not task_id:
        return _jsonrpc_error(req_id, _INVALID_PARAMS, "Missing 'id' in params")

    task = await task_store.get(task_id)
    if not task:
        return _jsonrpc_error(req_id, _TASK_NOT_FOUND, f"Task '{task_id}' not found")

    return _jsonrpc_result(req_id, task)


async def handle_list_tasks(
    params: dict,
    req_id: int | str | None,
    task_store: TaskStore,
) -> dict:
    """Handle the ListTasks JSON-RPC method."""
    context_id = params.get("contextId")
    status = params.get("status")
    page_size = params.get("pageSize", 50)
    page_token = params.get("pageToken")

    result = await task_store.list(
        context_id=context_id,
        status=status,
        page_size=page_size,
        page_token=page_token,
    )
    return _jsonrpc_result(req_id, result)


async def handle_cancel_task(
    params: dict,
    req_id: int | str | None,
    task_store: TaskStore,
) -> dict:
    """Handle the CancelTask JSON-RPC method."""
    task_id = params.get("id")
    if not task_id:
        return _jsonrpc_error(req_id, _INVALID_PARAMS, "Missing 'id' in params")

    task, canceled = await task_store.cancel(task_id)
    if task is None:
        return _jsonrpc_error(req_id, _TASK_NOT_FOUND, f"Task '{task_id}' not found")
    if not canceled:
        state = task.get("status", {}).get("state", "unknown")
        return _jsonrpc_error(
            req_id,
            _TASK_NOT_CANCELABLE,
            f"Task '{task_id}' is in state '{state}' and cannot be canceled",
        )

    return _jsonrpc_result(req_id, task)


def handle_unsupported(method: str, req_id: int | str | None) -> dict:
    """Return UnsupportedOperationError for unimplemented methods."""
    return _jsonrpc_error(
        req_id,
        _UNSUPPORTED_OPERATION,
        f"Method '{method}' is not supported",
    )


def handle_method_not_found(method: str, req_id: int | str | None) -> dict:
    """Return MethodNotFound error for unknown methods."""
    return _jsonrpc_error(
        req_id,
        _METHOD_NOT_FOUND,
        f"Method '{method}' not found",
    )
