"""Push notification sender with structured warning logging.

Extends ``BasePushNotificationSender`` to log clear warnings when push
delivery fails, so operators can diagnose connectivity issues between
the A2A server and the client's webhook endpoint.
"""

from __future__ import annotations

import httpx
from a2a.server.tasks.base_push_notification_sender import (
    BasePushNotificationSender,
)
from a2a.server.tasks.push_notification_config_store import (
    PushNotificationConfigStore,
)
from a2a.types import PushNotificationConfig, Task

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SmartPushNotificationSender(BasePushNotificationSender):
    """Push sender with detailed failure logging.

    On failure, emits a ``WARNING`` explaining:
    - Which task and URL failed
    - The error encountered
    - That the client should rely on polling as fallback
    """

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        config_store: PushNotificationConfigStore,
    ) -> None:
        super().__init__(httpx_client=httpx_client, config_store=config_store)

    async def _dispatch_notification(
        self,
        task: Task,
        push_info: PushNotificationConfig,
    ) -> bool:
        """Send a push notification, logging a clear warning on failure."""
        url = push_info.url
        try:
            headers = None
            if push_info.token:
                headers = {"X-A2A-Notification-Token": push_info.token}
            response = await self._client.post(
                url,
                json=task.model_dump(mode="json", exclude_none=True),
                headers=headers,
                timeout=0.3,
            )
            response.raise_for_status()
            logger.info(
                "Push notification sent | task_id={} url={} state={}",
                task.id,
                url,
                task.status.state.value if task.status else "unknown",
            )
            return True
        except httpx.ConnectError as exc:
            logger.warning(
                "Push notification FAILED (connection refused) | "
                "task_id={} url={} error={} — "
                "the webhook endpoint is unreachable from this server. "
                "If the server runs in a container, ensure the webhook URL "
                "resolves to the client host (not 127.0.0.1). "
                "The client can fall back to polling via get_task().",
                task.id,
                url,
                exc,
            )
            return False
        except httpx.TimeoutException as exc:
            logger.warning(
                "Push notification FAILED (timeout) | "
                "task_id={} url={} error={} — "
                "the webhook endpoint did not respond within 0.3s. "
                "The client can fall back to polling via get_task().",
                task.id,
                url,
                exc,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Push notification FAILED | "
                "task_id={} url={} error={} — "
                "the client can fall back to polling via get_task().",
                task.id,
                url,
                exc,
            )
            return False
