"""SkillTool: built-in tool exposed to the LLM to invoke skills.

Task 7.2: execute() wires in placeholder substitution via
`substitute_placeholders` and propagates substitution errors. Unknown skills
and invalid args are raised as exceptions so the @tool decorator converts
them to ToolStatus.ERROR.

Task 7.3: adds idempotent invocation (same skill invoked twice in the same
query returns an "already active" marker instead of re-injecting its body)
and registers frontmatter `hooks:` on the parent agent at first invoke.

Task 7.4: installs a QUERY_END cleanup hook that unregisters all
skill-scoped hooks at the end of each query, resets idempotence state,
and re-arms itself for the next query cycle.

Task 7.5: adds fork execution — when a skill declares ``context: fork``
its body is not returned inline but instead drives an isolated
``BaseAgent`` wrapped in ``SubAgentWrapper`` (stateless). The inner agent
inherits the parent's provider, registered tools, and max_iterations;
memory is NOT inherited. Only the sub-agent's last AssistantMessage
content flows back as the tool result.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import Field

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.agent.subagent_wrapper import SubAgentWrapper
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.tool_message import ToolCall
from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.core.skill.substitution import (
    SkillInvocationError,
    substitute_placeholders,
)
from obelix.core.tool.tool_decorator import tool

logger = logging.getLogger(__name__)

# ~1% of a 200k-token context window, expressed in characters.
DEFAULT_LISTING_BUDGET = 8_000


def _register_skill_hooks(agent, registered: list, skill: Skill) -> None:
    """Register skill.hooks on the agent, tracking them for later cleanup.

    Each hook uses CONTINUE decision plus a side-effect that appends a
    HumanMessage to the agent's conversation history. This matches the
    Hook fluent API (`.handle(decision, value, effects)`) and is neutral
    to the per-event value shape — we never attempt to transform the
    handler's value, we only nudge the model by injecting an instruction.
    """

    def _make_effect(text: str):
        def _effect(status):
            status.agent.conversation_history.append(HumanMessage(content=text))

        return _effect

    for event_name, instruction in skill.hooks.items():
        try:
            event = AgentEvent(event_name)
        except ValueError:
            # Unknown event — validator should have caught this at boot.
            # Log and skip here rather than crash.
            logger.warning(
                "[skill] unknown hook event '%s' on skill '%s' — skipping",
                event_name,
                skill.name,
            )
            continue
        hook = agent.on(event)
        hook.handle(HookDecision.CONTINUE, effects=[_make_effect(instruction)])
        registered.append((event, hook))


def _make_fork_agent(parent_agent, rendered_body: str) -> BaseAgent:
    """Build the ephemeral inner agent for fork execution.

    The inner agent starts from a fresh system message (the rendered skill
    body) and inherits the parent's provider, registered tools, and
    ``max_iterations`` (default 15). Memory and conversation history are
    NOT inherited — this is the isolation contract of a forked skill.

    Raises whatever ``BaseAgent.__init__`` raises on a misconfigured
    parent (e.g. a provider without a string ``model_id``). Callers treat
    this as a skill-invocation failure.
    """
    inner = BaseAgent(
        system_message=rendered_body,
        provider=parent_agent.provider,
        max_iterations=getattr(parent_agent, "max_iterations", 15),
    )
    for t in getattr(parent_agent, "registered_tools", []):
        inner.register_tool(t)
    return inner


async def _execute_fork(skill: Skill, rendered_body: str, parent_agent) -> str:
    """Run the skill in an isolated sub-agent; return last-message content.

    The inner agent uses the rendered skill body as its system message and
    inherits the parent's provider, registered tools, and
    ``max_iterations`` (default 15). Memory is NOT inherited —
    ``SubAgentWrapper(stateless=True)`` ensures the parent's history is
    not mutated. Skill-level ``hooks`` apply only to the ephemeral inner
    agent; no cleanup hook is needed since the agent dies with the call.
    """
    inner = _make_fork_agent(parent_agent, rendered_body)

    if skill.hooks:
        # Skill hooks apply to the inner agent only; it's ephemeral so no
        # cleanup hook is needed — the agent dies when execute completes.
        _register_skill_hooks(inner, [], skill)

    sub = SubAgentWrapper(
        inner,
        name=skill.name,
        description=skill.description,
        stateless=True,
    )
    call = ToolCall(
        id=str(uuid.uuid4()),
        name=skill.name,
        arguments={"query": "Begin executing the skill as specified above."},
    )
    result = await sub.execute(call)
    if result.error:
        raise SkillInvocationError(
            f"Skill '{skill.name}' (fork) failed: {result.error}"
        )
    if not result.result:
        # A successful fork with empty content is almost always a bug (e.g.,
        # the sub-agent exhausted iterations or produced no synthesis). Log
        # and surface as error so the caller isn't left holding an empty
        # string silently.
        raise SkillInvocationError(f"Skill '{skill.name}' (fork) produced no content")
    return result.result


def _install_cleanup_hook(
    agent, registered_hooks: list, active_skills: set, cleanup_flag: list
) -> None:
    """Install a QUERY_END hook that unregisters all skill-scoped hooks.

    The cleanup hook ALSO removes itself so subsequent queries aren't
    polluted. Active-skills set is cleared so idempotence resets per query.
    `cleanup_flag` is a 1-element mutable list wrapping a bool, reset to
    False inside the effect so the next query can re-install cleanup.
    """
    cleanup_hook = agent.on(AgentEvent.QUERY_END)

    def _cleanup_effect(status):
        # Unregister each skill-scoped hook
        for event, hook in registered_hooks:
            status.agent._unregister_hook(event, hook)
        # Remove the cleanup hook itself
        status.agent._unregister_hook(AgentEvent.QUERY_END, cleanup_hook)
        # Reset per-query state so next invocation starts clean
        registered_hooks.clear()
        active_skills.clear()
        cleanup_flag[0] = False

    cleanup_hook.handle(HookDecision.CONTINUE, effects=[_cleanup_effect])


def make_skill_tool(
    manager: SkillManager,
    listing_budget: int = DEFAULT_LISTING_BUDGET,
    session_id: str | None = None,
    parent_agent=None,
):
    """Produce a SkillTool instance bound to the given SkillManager.

    Returns a tool-decorated instance ready to be `register_tool()`-ed on an
    agent. The instance's `system_prompt_fragment()` yields the listing to
    inject at construction time.

    Each call produces a fresh tool class closing over `manager`,
    `listing_budget`, `session_id`, and `parent_agent`; do not rely on class
    identity across invocations.

    `session_id`, when provided, is substituted into ${OBELIX_SESSION_ID}.
    When omitted, a UUID is generated at construction and reused across every
    invocation of this tool instance — simpler and race-free compared to
    lazy generation.

    `parent_agent`, when provided, receives frontmatter `hooks:` on first
    invocation of each skill (via duck-typed `.on(event)` returning a
    Hook-like with `.handle(decision, effects=...)`). When None, hook
    registration is silently skipped.
    """
    resolved_session_id: str = session_id or str(uuid.uuid4())
    # Per-tool-instance idempotence state: obtain a fresh tool via
    # make_skill_tool() per query to reset these. Task 7.4 consumes
    # _registered_hooks to unregister skill-scoped hooks at QUERY_END.
    active_skills: set[str] = set()
    registered_hooks: list = []
    cleanup_installed = [False]  # list-wrapped for mutability in nested closure

    @tool(
        name="Skill",
        description=(
            "Execute a skill within the main conversation. "
            "Use when a skill from the listing matches the user's request."
        ),
    )
    class SkillTool:
        name: str = Field(..., description="Name of the skill to invoke")
        args: str = Field(
            default="", description="Optional arguments passed as $ARGUMENTS"
        )

        async def execute(self) -> str:
            skill = manager.load(self.name)
            if skill is None:
                names = [s.name for s in manager.list_all()]
                tail = (
                    f"Available: {', '.join(names)}"
                    if names
                    else "No skills registered."
                )
                raise SkillInvocationError(f"Skill '{self.name}' not found. {tail}")

            # Idempotence: second invocation in the same query returns a
            # short marker instead of re-injecting the body. Applies to
            # both inline and fork skills.
            if self.name in active_skills:
                return (
                    f"Skill '{self.name}' is already active — "
                    "continue following its instructions."
                )

            try:
                rendered = substitute_placeholders(
                    skill.body,
                    self.args,
                    skill.arguments,
                    skill.base_dir,
                    resolved_session_id,
                )
            except SkillInvocationError as e:
                # Prefix the skill name so the caller knows which skill failed.
                # Do NOT mark the skill active on failure — retry is allowed.
                raise SkillInvocationError(f"Skill '{self.name}': {e}") from e

            # Fork path: run the body in an isolated sub-agent and return
            # the last AssistantMessage content as the tool result. Hooks
            # (if any) are registered on the ephemeral inner agent inside
            # _execute_fork — not on the parent.
            if skill.context == "fork":
                if parent_agent is None:
                    raise SkillInvocationError(
                        f"Skill '{self.name}': fork execution requires a parent agent"
                    )
                # Look up the (possibly monkey-patched) SubAgentWrapper from
                # this module so tests can swap it. Same applies to BaseAgent.
                result_text = await _execute_fork(skill, rendered, parent_agent)
                active_skills.add(self.name)
                return result_text

            # Inline path: register hooks on the parent and return the body.
            if skill.hooks and parent_agent is not None:
                _register_skill_hooks(parent_agent, registered_hooks, skill)
                if not cleanup_installed[0]:
                    _install_cleanup_hook(
                        parent_agent,
                        registered_hooks,
                        active_skills,
                        cleanup_installed,
                    )
                    cleanup_installed[0] = True
            active_skills.add(self.name)
            return rendered

        def system_prompt_fragment(self) -> str:
            listing = manager.format_listing(listing_budget)
            if not listing:
                return ""
            return (
                "\n\n## Available skills\n\n"
                "Use the Skill tool to invoke one of the following:\n\n"
                f"{listing}\n\n"
                "When a skill matches, invoke it BEFORE generating your response.\n"
            )

    return SkillTool()
