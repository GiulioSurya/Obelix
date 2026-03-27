"""A2A server package — bridges a2a-sdk to Obelix BaseAgent."""

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.inbound.a2a.server.deferred import (
    inject_deferred_response,
    parse_deferred_result,
)
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.adapters.inbound.a2a.server.helpers import (
    DEFAULT_MAX_CONTEXTS,
    agent_message,
)
from obelix.adapters.inbound.a2a.server.middleware import (
    ClientIPMiddleware,
    client_ip_var,
    resolve_client_ip,
)
from obelix.adapters.inbound.a2a.server.push_config_store import (
    SmartPushNotificationConfigStore,
)
from obelix.adapters.inbound.a2a.server.push_sender import (
    SmartPushNotificationSender,
)

__all__ = [
    "ObelixAgentExecutor",
    "ContextEntry",
    "ContextStore",
    "inject_deferred_response",
    "parse_deferred_result",
    "agent_message",
    "DEFAULT_MAX_CONTEXTS",
    "ClientIPMiddleware",
    "client_ip_var",
    "resolve_client_ip",
    "SmartPushNotificationConfigStore",
    "SmartPushNotificationSender",
]
