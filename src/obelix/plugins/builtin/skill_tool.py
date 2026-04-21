"""SkillTool: built-in tool exposed to the LLM to invoke skills.

Task 7.2: execute() wires in placeholder substitution via
`substitute_placeholders` and propagates substitution errors. Unknown skills
and invalid args are raised as exceptions so the @tool decorator converts
them to ToolStatus.ERROR.

Task 7.3: adds idempotent invocation (same skill invoked twice in the same
query returns an "already active" marker instead of re-injecting its body)
and registers frontmatter `hooks:` on the parent agent at first invoke.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import Field

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
    """Register skill.hooks on the agent, tracking them for later cleanup."""
    from obelix.core.agent.hooks import AgentEvent
    from obelix.core.model.human_message import HumanMessage

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
        hook.inject(lambda _status, msg=instruction: HumanMessage(content=msg))
        registered.append((event, hook))


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
    Hook-like with `.inject(...)`). When None, hook registration is silently
    skipped.
    """
    resolved_session_id: str = session_id or str(uuid.uuid4())
    _active_skills: set[str] = set()
    _registered_hooks: list = []

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

        def execute(self) -> str:
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
            # short marker instead of re-injecting the body.
            if self.name in _active_skills:
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

            # Only after successful substitution: register hooks and mark active.
            if skill.hooks and parent_agent is not None:
                _register_skill_hooks(parent_agent, _registered_hooks, skill)
            _active_skills.add(self.name)
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
