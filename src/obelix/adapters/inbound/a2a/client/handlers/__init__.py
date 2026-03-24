"""Deferred tool handler registry and dispatcher."""

from __future__ import annotations

from obelix.adapters.inbound.a2a.client.handlers.base import (
    BaseDeferredHandler,
    InputCallback,
    PermissionPolicy,
)
from obelix.adapters.inbound.a2a.client.handlers.bash import BashHandler
from obelix.adapters.inbound.a2a.client.handlers.request_input import (
    RequestInputHandler,
)


class HandlerDispatcher:
    """Routes deferred tool calls to the appropriate handler.

    Falls back to ``BaseDeferredHandler`` (raw JSON + free-form input)
    for unrecognised tool names.
    """

    def __init__(self, handlers: list[BaseDeferredHandler] | None = None):
        self._handlers: list[BaseDeferredHandler] = handlers or []
        self._fallback = BaseDeferredHandler()

    def register(self, handler: BaseDeferredHandler) -> None:
        self._handlers.append(handler)

    def get(self, tool_name: str) -> BaseDeferredHandler:
        for h in self._handlers:
            if h.can_handle(tool_name):
                return h
        return self._fallback


def default_dispatcher(
    bash_permission: PermissionPolicy = PermissionPolicy.ALWAYS_ASK,
) -> HandlerDispatcher:
    """Create a dispatcher with the standard built-in handlers."""
    return HandlerDispatcher(
        [
            RequestInputHandler(),
            BashHandler(permission=bash_permission),
        ]
    )


__all__ = [
    "BaseDeferredHandler",
    "BashHandler",
    "HandlerDispatcher",
    "InputCallback",
    "PermissionPolicy",
    "RequestInputHandler",
    "default_dispatcher",
]
