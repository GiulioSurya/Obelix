"""A2A protocol controller — FastAPI APIRouter for JSON-RPC 2.0 dispatch."""

from typing import TYPE_CHECKING, Any

from obelix.adapters.inbound.a2a.handlers import (
    handle_cancel_task,
    handle_get_task,
    handle_list_tasks,
    handle_method_not_found,
    handle_send_message,
    handle_unsupported,
)
from obelix.adapters.inbound.a2a.task_store import TaskStore
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from fastapi import APIRouter

    from obelix.core.agent.base_agent import BaseAgent

logger = get_logger(__name__)

# Methods that exist in the spec but are not implemented in this MVP
_UNSUPPORTED_METHODS = frozenset(
    {
        "SendStreamingMessage",
        "SubscribeToTask",
        "CreateTaskPushNotificationConfig",
        "GetTaskPushNotificationConfig",
        "ListTaskPushNotificationConfigs",
        "DeleteTaskPushNotificationConfig",
        "GetExtendedAgentCard",
    }
)


class ObelixA2AController:
    """
    A2A-compliant controller exposing an Obelix agent over HTTP.

    Provides:
    - GET  /.well-known/agent-card.json  -> Agent Card
    - POST /                             -> JSON-RPC 2.0 dispatcher
    - GET  /health                       -> health check
    """

    def __init__(
        self,
        agent: "BaseAgent",
        agent_card: dict[str, Any],
        task_store: TaskStore | None = None,
    ) -> None:
        self._agent = agent
        self._agent_card = agent_card
        self._task_store = task_store or TaskStore()

    def build_router(self) -> "APIRouter":
        """Create and return the FastAPI APIRouter with all A2A routes."""
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse

        router = APIRouter()

        @router.get("/.well-known/agent-card.json")
        async def agent_card() -> JSONResponse:
            return JSONResponse(content=self._agent_card)

        @router.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        @router.post("/")
        async def jsonrpc_dispatch(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                        },
                    },
                    status_code=200,
                )

            method = body.get("method")
            params = body.get("params", {})
            req_id = body.get("id")

            if not method:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing 'method'",
                        },
                    },
                    status_code=200,
                )

            logger.debug(f"[A2A] JSON-RPC dispatch | method={method}")

            response = await self._dispatch(method, params, req_id)
            return JSONResponse(content=response, status_code=200)

        return router

    async def _dispatch(
        self,
        method: str,
        params: dict,
        req_id: int | str | None,
    ) -> dict:
        """Route a JSON-RPC method to the appropriate handler."""
        if method == "SendMessage":
            return await handle_send_message(
                params, req_id, self._agent, self._task_store
            )
        elif method == "GetTask":
            return await handle_get_task(params, req_id, self._task_store)
        elif method == "ListTasks":
            return await handle_list_tasks(params, req_id, self._task_store)
        elif method == "CancelTask":
            return await handle_cancel_task(params, req_id, self._task_store)
        elif method in _UNSUPPORTED_METHODS:
            return handle_unsupported(method, req_id)
        else:
            return handle_method_not_found(method, req_id)
