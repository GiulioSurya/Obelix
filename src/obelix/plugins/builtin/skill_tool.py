"""SkillTool: built-in tool exposed to the LLM to invoke skills.

Task 7.2: execute() wires in placeholder substitution via
`substitute_placeholders` and propagates substitution errors. Unknown skills
and invalid args are raised as exceptions so the @tool decorator converts
them to ToolStatus.ERROR.
"""

from __future__ import annotations

import uuid

from pydantic import Field

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.substitution import (
    SkillInvocationError,
    substitute_placeholders,
)
from obelix.core.tool.tool_decorator import tool

# ~1% of a 200k-token context window, expressed in characters.
DEFAULT_LISTING_BUDGET = 8_000


def make_skill_tool(
    manager: SkillManager,
    listing_budget: int = DEFAULT_LISTING_BUDGET,
    session_id: str | None = None,
):
    """Produce a SkillTool instance bound to the given SkillManager.

    Returns a tool-decorated instance ready to be `register_tool()`-ed on an
    agent. The instance's `system_prompt_fragment()` yields the listing to
    inject at construction time.

    Each call produces a fresh tool class closing over `manager`,
    `listing_budget`, and `session_id`; do not rely on class identity across
    invocations.

    `session_id`, when provided, is substituted into ${OBELIX_SESSION_ID}.
    When omitted, a UUID is generated lazily on first use and cached in the
    closure, so every invocation of the same tool instance sees the same id.
    """
    # Stable session id for this tool instance — generated on first need
    # when the caller did not provide one, then cached inside the closure.
    _session_state = {"id": session_id}

    def _session() -> str:
        if _session_state["id"] is None:
            _session_state["id"] = str(uuid.uuid4())
        return _session_state["id"]

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
                available = ", ".join(s.name for s in manager.list_all())
                raise RuntimeError(
                    f"Skill '{self.name}' not found. Available: {available}"
                )
            try:
                return substitute_placeholders(
                    skill.body,
                    self.args,
                    skill.arguments,
                    skill.base_dir,
                    _session(),
                )
            except SkillInvocationError as e:
                # Re-raise as RuntimeError so the @tool wrapper converts to
                # ToolStatus.ERROR with this message.
                raise RuntimeError(str(e)) from e

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
