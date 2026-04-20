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
            skill = self._to_skill(prompt)
            if skill is None:
                logger.warning(
                    "[skill] Skipped malformed MCP prompt %r from server %r",
                    prompt.name,
                    prompt.server_name,
                )
                continue
            skills.append(skill)
        return skills

    def _to_skill(self, prompt: MCPPrompt) -> Skill | None:
        if not prompt.description:
            return None
        if not (prompt.template or "").strip():
            return None

        # Extract argument names from MCP PromptArgument (.name attribute)
        arg_names: list[str] = []
        for a in prompt.arguments:
            name = getattr(a, "name", None)
            if isinstance(name, str):
                arg_names.append(name)

        candidate = SkillCandidate(
            file_path=None,
            frontmatter={
                "description": prompt.description,
                "arguments": arg_names,
            },
            body=prompt.template,
        )
        issues = run_validators(candidate, DEFAULT_VALIDATORS)
        if issues:
            return None

        namespaced = f"mcp__{prompt.server_name}__{prompt.name}"
        return Skill.from_candidate(
            candidate, name=namespaced, base_dir=None, source="mcp"
        )
