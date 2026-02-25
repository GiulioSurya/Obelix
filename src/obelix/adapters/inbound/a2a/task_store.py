"""In-memory task store for A2A protocol."""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskRecord:
    """Internal record wrapping an A2A Task with timestamps."""

    task: dict
    created_at: datetime
    updated_at: datetime


class TaskStore:
    """Thread-safe in-memory task store for A2A tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, context_id: str | None = None) -> dict:
        """Create a new task in 'working' state."""
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "contextId": context_id or str(uuid.uuid4()),
            "status": {"state": "working"},
            "artifacts": [],
            "metadata": {},
        }
        async with self._lock:
            self._tasks[task_id] = TaskRecord(
                task=task,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        logger.debug(f"[TaskStore] Task created | task_id={task_id}")
        return task

    async def get(self, task_id: str) -> dict | None:
        """Retrieve a task by ID."""
        async with self._lock:
            record = self._tasks.get(task_id)
            return record.task if record else None

    async def list(
        self,
        context_id: str | None = None,
        status: str | None = None,
        page_size: int = 50,
        page_token: str | None = None,
    ) -> dict:
        """List tasks with optional filtering and pagination."""
        async with self._lock:
            tasks = list(self._tasks.values())

        # Filter
        if context_id:
            tasks = [r for r in tasks if r.task.get("contextId") == context_id]
        if status:
            tasks = [
                r for r in tasks if r.task.get("status", {}).get("state") == status
            ]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda r: r.created_at, reverse=True)

        # Pagination
        start_idx = 0
        if page_token:
            for i, r in enumerate(tasks):
                if r.task["id"] == page_token:
                    start_idx = i + 1
                    break

        page = tasks[start_idx : start_idx + page_size]
        next_token = (
            page[-1].task["id"]
            if len(page) == page_size and start_idx + page_size < len(tasks)
            else None
        )

        result: dict = {"tasks": [r.task for r in page]}
        if next_token:
            result["nextPageToken"] = next_token
        return result

    async def update_status(
        self,
        task_id: str,
        state: str,
        message: str | None = None,
    ) -> dict | None:
        """Update the status of a task."""
        async with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            record.task["status"] = {"state": state}
            if message:
                record.task["status"]["message"] = {
                    "role": "agent",
                    "parts": [{"type": "text", "text": message}],
                }
            record.updated_at = datetime.now(UTC)
            return record.task

    async def add_artifact(self, task_id: str, artifact: dict) -> dict | None:
        """Add an artifact to a task."""
        async with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None
            record.task.setdefault("artifacts", []).append(artifact)
            record.updated_at = datetime.now(UTC)
            return record.task

    async def cancel(self, task_id: str) -> tuple[dict | None, bool]:
        """Cancel a task by setting its state to 'canceled'.

        Returns:
            (task, canceled) — task is None if not found; canceled is False
            if the task is in a terminal state and cannot be canceled.
        """
        async with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return None, False
            current_state = record.task.get("status", {}).get("state")
            if current_state in ("completed", "failed", "canceled"):
                return record.task, False
            record.task["status"] = {"state": "canceled"}
            record.updated_at = datetime.now(UTC)
            logger.debug(f"[TaskStore] Task canceled | task_id={task_id}")
            return record.task, True
