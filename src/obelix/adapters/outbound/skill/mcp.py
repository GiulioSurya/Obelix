"""MCP-backed skill provider.

Turns MCP prompts exposed by connected servers into Skill objects.
Malformed prompts are logged and skipped, not raised — Obelix does not
control remote servers.
"""

from __future__ import annotations

import logging

from obelix.adapters.outbound.mcp.manager import MCPManager, MCPPrompt
from obelix.core.skill.skill import Skill, SkillCandidate
from obelix.core.skill.validation import DEFAULT_VALIDATORS, run_validators
from obelix.ports.outbound.skill_provider import AbstractSkillProvider

logger = logging.getLogger(__name__)


class MCPSkillProvider(AbstractSkillProvider):
    """Wraps MCPManager.list_prompts() and renders prompts as skills.

    Accepts `None` to signal "no MCP configured" — returns an empty list.
    """

    def __init__(self, manager: MCPManager | None):
        self._manager = manager

    def discover(self) -> list[Skill]:
        if self._manager is None:
            return []

        skills: list[Skill] = []
        for prompt in self._manager.list_prompts():
            skill, reason = self._to_skill(prompt)
            if skill is None:
                logger.warning(
                    "[skill] Skipped MCP prompt %r from server %r: %s",
                    prompt.name,
                    prompt.server_name,
                    reason,
                )
                continue
            skills.append(skill)
        return skills

    def _to_skill(self, prompt: MCPPrompt) -> tuple[Skill, None] | tuple[None, str]:
        """Convert an MCPPrompt into a Skill, or return (None, reason) on skip."""
        if not prompt.description:
            return None, "missing description"
        body = prompt.template or ""
        if not body.strip():
            return None, "empty template"
        # MCP skills have no on-disk base_dir; ${OBELIX_SKILL_DIR} is meaningless.
        # Reject upfront rather than silently substituting an empty string at runtime.
        if "${OBELIX_SKILL_DIR}" in body:
            return (
                None,
                "body references ${OBELIX_SKILL_DIR} (not available for MCP skills)",
            )

        # Extract argument names from MCP PromptArgument (.name attribute)
        arg_names: list[str] = []
        for arg in prompt.arguments:
            name = getattr(arg, "name", None)
            if isinstance(name, str):
                arg_names.append(name)

        candidate = SkillCandidate(
            file_path=None,
            frontmatter={
                "description": prompt.description,
                "arguments": arg_names,
            },
            body=body,
        )
        issues = run_validators(candidate, DEFAULT_VALIDATORS)
        if issues:
            reasons = "; ".join(f"{i.field}: {i.message}" for i in issues)
            return None, f"validation failed: {reasons}"

        namespaced = f"mcp__{prompt.server_name}__{prompt.name}"
        skill = Skill.from_candidate(
            candidate, name=namespaced, base_dir=None, source="mcp"
        )
        return skill, None
