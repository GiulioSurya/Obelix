"""Shared memory hooks for BaseAgent.

Provides injection of shared context from predecessor agents
and publication of results to the shared memory graph.

These hooks are registered on the agent at construction time
via register_memory_hooks(). They are no-ops when the agent
has no memory_graph or agent_id configured.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from obelix.core.agent.hooks import AgentEvent, AgentStatus, HookDecision
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import ToolMessage, ToolStatus
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model.standard_message import StandardMessage

logger = get_logger(__name__)


def register_memory_hooks(agent: BaseAgent) -> None:
    """Register shared memory injection and publication hooks on the agent.

    Hooks guard at REGISTRATION with .when() on the memory_graph/agent_id
    attributes. If the agent has no shared memory configured they are skipped
    entirely — keeping the tracer ``hook.fired`` stream free of no-op chatter.
    """
    agent.on(AgentEvent.BEFORE_LLM_CALL).when(_has_memory_binding).handle(
        decision=HookDecision.CONTINUE,
        effects=[_inject_shared_memory],
    )
    agent.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(_has_memory_binding).handle(
        decision=HookDecision.CONTINUE,
        effects=[_publish_to_memory],
    )


def _has_memory_binding(status: AgentStatus) -> bool:
    """True iff the agent has both a memory graph and an agent_id attached."""
    agent = status.agent
    return bool(agent.memory_graph and agent.agent_id)


async def _inject_shared_memory(status: AgentStatus) -> None:
    """Pull and inject shared context from predecessor agents."""
    agent = status.agent
    if not agent.memory_graph or not agent.agent_id:
        return

    memories = agent.memory_graph.pull_for(agent.agent_id)
    if not memories:
        return

    history = agent.conversation_history
    tracer = getattr(agent, "_tracer", None)

    for mem in memories:
        existing_idx = None
        for idx, msg in enumerate(history):
            if (
                isinstance(msg, SystemMessage)
                and msg.metadata.get("shared_memory_source") == mem.source_id
            ):
                existing_idx = idx
                break

        if existing_idx is not None:
            existing_ts = history[existing_idx].metadata.get("shared_memory_timestamp")
            if existing_ts == mem.timestamp:
                continue
            history.pop(existing_idx)
            logger.debug(
                f"[SharedMemory] Replacing stale context | agent={agent.agent_id} source={mem.source_id} chars={len(mem.content)}"
            )

        msg = SystemMessage(
            content=f"Shared context from {mem.source_id}:\n{mem.content}",
            metadata={
                "shared_memory": True,
                "shared_memory_source": mem.source_id,
                "shared_memory_timestamp": mem.timestamp,
            },
        )
        history.append(msg)
        logger.debug(
            f"[SharedMemory] Injected context | target_agent={agent.agent_id} source={mem.source_id} chars={len(mem.content)}"
        )

        if tracer is not None:
            await tracer.add_event(
                "memory.pull",
                {
                    "from_agent": mem.source_id,
                    "policy": mem.policy.value
                    if hasattr(mem.policy, "value")
                    else str(mem.policy),
                    "bytes": len(mem.content or ""),
                },
            )


async def _publish_to_memory(status: AgentStatus) -> None:
    """Publish the agent's final response and last tool result to the shared memory graph."""
    agent = status.agent
    if not agent.memory_graph or not agent.agent_id:
        return

    if status.assistant_message and status.assistant_message.content:
        await agent.memory_graph.publish(
            agent.agent_id, status.assistant_message.content
        )
        logger.debug(
            f"[SharedMemory] Published final response | agent={agent.agent_id} chars={len(status.assistant_message.content)}"
        )

    last_tool_content = _extract_last_tool_result(agent.conversation_history)
    if last_tool_content:
        await agent.memory_graph.publish(
            agent.agent_id, last_tool_content, kind="tool_result"
        )
        logger.debug(
            f"[SharedMemory] Published tool result | agent={agent.agent_id} chars={len(last_tool_content)}"
        )


def _extract_last_tool_result(
    conversation_history: list[StandardMessage],
) -> str | None:
    """Extract the last successful tool result from conversation history, serialized as JSON."""
    for msg in reversed(conversation_history):
        if isinstance(msg, ToolMessage):
            for result in reversed(msg.tool_results):
                if result.status == ToolStatus.SUCCESS and result.result is not None:
                    if isinstance(result.result, dict | list):
                        return json.dumps(result.result, ensure_ascii=False)
                    return str(result.result)
    return None
