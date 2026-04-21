"""SkillTool: built-in tool exposed to the LLM to invoke skills.

v1 scaffold: renders the listing via system_prompt_fragment and exposes a
minimal tool schema. Actual invocation logic (inline, fork, hooks) lands in
subsequent tasks (7.2 through 7.5).
"""

from __future__ import annotations

from pydantic import Field

from obelix.core.skill.manager import SkillManager
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

    Each call produces a fresh tool class closing over `manager` and
    `listing_budget`; do not rely on class identity across invocations.

    `session_id` is accepted for forward compatibility with Task 7.2
    (placeholder substitution). The scaffold ignores it — no eager UUID
    generation, no unused binding. It becomes load-bearing at 7.2.
    """
    # session_id intentionally unused at this stage — see 7.2.
    _ = session_id  # noqa: F841 — keeps the parameter reachable for tooling.

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
            # Actual invocation logic lands in Task 7.2. For now a clear stub
            # so register_tool() works and tests can verify wiring.
            skill = manager.load(self.name)
            if skill is None:
                available = ", ".join(s.name for s in manager.list_all())
                return f"Skill '{self.name}' not found. Available: {available}"
            return skill.body

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
