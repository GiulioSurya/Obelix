"""Verify dispatch_agent appends an entry to T1.metadata.dispatched_peers
the moment it submits the message to the remote agent.

Iron rule: no SDK mocks. We use the FakeA2AServer for the remote and
exercise the tool's execute() against a real registry / store.
"""

from __future__ import annotations

import uuid

import pytest
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.tools.dispatch import DispatchAgentTool
from obelix.core.model.tool_message import ToolCall, ToolStatus
from tests._fakes.fake_a2a_server import FakeA2AServer


class _SubmittedTaskExecutor(AgentExecutor):
    """Emits a single Task in 'working' state then returns.

    Enough to satisfy the polling client's ``send_message`` (blocking=False),
    which returns after the first event creates a Task. The dispatch tool
    then receives a real Task with a real ``id`` and proceeds to patch
    T1.metadata.
    """

    async def execute(  # noqa: D401
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or "remote-ctx"
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
            )
        )

    async def cancel(  # noqa: D401
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        return


def _make_call(args: dict) -> ToolCall:
    return ToolCall(id="c-1", name="dispatch_agent", arguments=args)


@pytest.mark.asyncio
async def test_dispatch_appends_peer_to_t1_metadata():
    async with FakeA2AServer(executor=_SubmittedTaskExecutor()) as remote:
        # Seed T1 in the LOCAL store. The dispatch tool reads/writes the
        # local store via the metadata-patch helper.
        local_store = InMemoryTaskStore()
        await local_store.save(
            Task(
                id="t1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.working),
                metadata=None,
            )
        )

        # Build a registry with one remote keyed under remote's card name.
        # RemoteAgentRegistry has no ``add()`` API: in production it
        # populates _cards/_clients from URL-driven resolution. For this
        # test we splice the FakeA2AServer's already-built client directly.
        import httpx

        registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
        remote_name = remote._card.name
        registry._cards[remote_name] = remote._card
        registry._clients[remote_name] = remote.client

        entry = ContextEntry()
        entry.current_task_id = "t1"
        entry.context_id = "ctx-1"

        tool = DispatchAgentTool(registry=registry)
        tool.set_context_entry(entry, context_id="ctx-1")
        tool.set_webhook_url("http://placeholder/webhook")
        tool.set_task_store(local_store)

        result = await tool.execute(
            _make_call({"agent_name": remote_name, "query": "do something"})
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.result["status"] == "submitted"
        remote_task_id = result.result["task_id"]

        refreshed = await local_store.get("t1")
        assert refreshed is not None
        peers = (refreshed.metadata or {}).get("dispatched_peers", [])
        assert len(peers) == 1
        assert peers[0]["name"] == remote_name
        assert peers[0]["task_id"] == remote_task_id
        assert peers[0]["state"] == "working"


@pytest.mark.asyncio
async def test_dispatch_skips_metadata_write_when_task_store_unset():
    """If set_task_store was never called, the tool falls through cleanly:
    the dispatch still works, just no metadata side-effect."""
    async with FakeA2AServer(executor=_SubmittedTaskExecutor()) as remote:
        import httpx

        registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
        remote_name = remote._card.name
        registry._cards[remote_name] = remote._card
        registry._clients[remote_name] = remote.client

        entry = ContextEntry()
        entry.current_task_id = "t1"
        entry.context_id = "ctx-1"

        tool = DispatchAgentTool(registry=registry)
        tool.set_context_entry(entry, context_id="ctx-1")
        tool.set_webhook_url("http://placeholder/webhook")
        # NOT calling set_task_store

        result = await tool.execute(
            _make_call({"agent_name": remote_name, "query": "do something"})
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.result["status"] == "submitted"  # works regardless


@pytest.mark.asyncio
async def test_dispatch_skips_metadata_write_when_no_current_task_id():
    """If entry.current_task_id is None (no parent task in flight), the
    metadata-patch branch must not fire — even if a task_store is injected.
    Covers the half of the guard not tested by
    test_dispatch_skips_metadata_write_when_task_store_unset."""
    async with FakeA2AServer(executor=_SubmittedTaskExecutor()) as remote:
        local_store = InMemoryTaskStore()
        # Note: no T1 seeded — no parent task at all.

        import httpx

        # FRAGILE: splices private attrs because RemoteAgentRegistry has no
        # public add() API. Same pattern as the other two tests in this file.
        registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
        remote_name = remote._card.name
        registry._cards[remote_name] = remote._card
        registry._clients[remote_name] = remote.client

        entry = ContextEntry()
        entry.context_id = "ctx-1"
        entry.current_task_id = None  # NO parent task

        tool = DispatchAgentTool(registry=registry)
        tool.set_context_entry(entry, context_id="ctx-1")
        tool.set_webhook_url("http://placeholder/webhook")
        tool.set_task_store(local_store)

        result = await tool.execute(
            _make_call({"agent_name": remote_name, "query": "do something"})
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.result["status"] == "submitted"

        # No T1 in store -> no metadata anywhere -> nothing to assert beyond
        # "didn't crash and didn't try to patch a non-existent task". The
        # update_task_metadata helper no-ops on missing tasks (Task 3 contract).
        # Verify no T1 was created accidentally.
        assert await local_store.get("t1") is None
