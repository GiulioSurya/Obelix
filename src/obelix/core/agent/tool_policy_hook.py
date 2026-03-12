"""Tool policy enforcement as a standalone hook.

ToolPolicyHook registers BEFORE_FINAL_RESPONSE hooks on a BaseAgent
to ensure that required tools are called before the agent responds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from obelix.core.agent.hooks import AgentEvent, AgentStatus, HookDecision
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model.standard_message import StandardMessage
    from obelix.core.model.tool_message import ToolRequirement

logger = get_logger(__name__)


class ToolPolicyHook:
    """Enforces tool usage requirements before the agent can finalize a response.

    Usage:
        policy = [ToolRequirement(tool_name="search", min_calls=1)]
        ToolPolicyHook(policy).register(agent)
    """

    def __init__(self, policy: list[ToolRequirement]) -> None:
        self._policy = policy

    def register(self, agent: BaseAgent) -> None:
        """Register BEFORE_FINAL_RESPONSE hooks on the agent."""
        agent.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(self._should_fail).handle(
            decision=HookDecision.FAIL,
            effects=[self._inject_message],
        )
        agent.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(self._should_retry).handle(
            decision=HookDecision.RETRY,
            effects=[self._inject_message],
        )

    def _get_violation(
        self, status: AgentStatus
    ) -> tuple[HookDecision | None, str | None]:
        """Check if any tool policy requirement is violated.

        Returns (decision, message) or (None, None) if all satisfied.
        Uses a cache on AgentStatus to avoid re-computation within the same hook cycle.
        """
        cached = getattr(status, "_tool_policy_cache", None)
        if cached is not None:
            return cached

        invocation_start = getattr(status.agent, "_invocation_start", 0)
        current_history = status.agent.conversation_history[invocation_start:]
        results = _collect_tool_results(current_history)
        logger.debug(
            f"[ToolPolicy] Evaluating | required={[r.tool_name for r in self._policy]} results={[r.tool_name for r in results]}"
        )

        for req in self._policy:
            matching = [r for r in results if r.tool_name == req.tool_name]
            logger.debug(
                f"[ToolPolicy] Check | tool={req.tool_name} calls_found={len(matching)} min_required={req.min_calls} require_success={req.require_success}"
            )
            if len(matching) < req.min_calls:
                msg = req.error_message or (
                    f"You must call tool '{req.tool_name}' at least {req.min_calls} time(s) "
                    "before responding."
                )
                decision = (
                    HookDecision.FAIL
                    if status.iteration >= status.agent.max_iterations
                    else HookDecision.RETRY
                )
                logger.warning(
                    f"[ToolPolicy] Violated — minimum calls not met | tool={req.tool_name} calls_made={len(matching)} min_required={req.min_calls} decision={decision.value}"
                )
                status._tool_policy_cache = (decision, msg)
                return decision, msg

            if req.require_success:
                success_count = sum(
                    1 for r in matching if r.status == ToolStatus.SUCCESS
                )
                if success_count < req.min_calls:
                    msg = req.error_message or (
                        f"Tool '{req.tool_name}' failed. "
                        "Retry and ensure the result is SUCCESS."
                    )
                    decision = (
                        HookDecision.FAIL
                        if status.iteration >= status.agent.max_iterations
                        else HookDecision.RETRY
                    )
                    logger.warning(
                        f"[ToolPolicy] Violated — insufficient successful calls | tool={req.tool_name} success_count={success_count} min_required={req.min_calls} decision={decision.value}"
                    )
                    status._tool_policy_cache = (decision, msg)
                    return decision, msg

        status._tool_policy_cache = (None, None)
        return None, None

    def _should_retry(self, status: AgentStatus) -> bool:
        decision, _ = self._get_violation(status)
        return decision == HookDecision.RETRY

    def _should_fail(self, status: AgentStatus) -> bool:
        decision, _ = self._get_violation(status)
        return decision == HookDecision.FAIL

    def _inject_message(self, status: AgentStatus) -> None:
        _, msg = self._get_violation(status)
        if msg:
            status.agent.conversation_history.append(SystemMessage(content=msg))


def _collect_tool_results(
    history: list[StandardMessage],
) -> list[ToolResult]:
    """Collect all ToolResults from a message history."""
    results: list[ToolResult] = []
    for message in history:
        if isinstance(message, ToolMessage):
            results.extend(message.tool_results)
    return results
