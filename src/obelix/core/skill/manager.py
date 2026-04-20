"""Aggregate providers, dedup skills, format the listing under a budget."""

from __future__ import annotations

import logging

from obelix.core.skill.skill import Skill
from obelix.ports.outbound.skill_provider import AbstractSkillProvider

logger = logging.getLogger(__name__)

# Hard cap on description chars regardless of budget — prevents a pathological
# skill from monopolizing the listing when budget is generous.
_MAX_DESC_CHARS = 250
# Below this per-entry limit, truncated descriptions become useless
# (just "a few chars…") — fall back to names-only instead.
_MIN_DESC_CHARS = 20


def _source_priority(source: str) -> int:
    """Higher priority wins in collisions.

    Unknown sources get 0, so they lose to all known sources.
    """
    return {"filesystem": 2, "mcp": 1}.get(source, 0)


class SkillManager:
    """Aggregates skills from multiple providers with dedup + budget-aware listing."""

    def __init__(self, providers: list[AbstractSkillProvider]):
        self._providers = providers
        self._by_name: dict[str, Skill] = self._aggregate()

    def _aggregate(self) -> dict[str, Skill]:
        by_name: dict[str, Skill] = {}
        for provider in self._providers:
            for skill in provider.discover():
                existing = by_name.get(skill.name)
                if existing is None:
                    by_name[skill.name] = skill
                    continue
                # Collision — higher source priority wins; same priority = first wins
                if _source_priority(existing.source) >= _source_priority(skill.source):
                    logger.warning(
                        "[skill] collision on '%s': kept source=%s, "
                        "discarded source=%s",
                        skill.name,
                        existing.source,
                        skill.source,
                    )
                else:
                    logger.warning(
                        "[skill] collision on '%s': replacing source=%s with source=%s",
                        skill.name,
                        existing.source,
                        skill.source,
                    )
                    by_name[skill.name] = skill
        return by_name

    def list_all(self) -> list[Skill]:
        """All skills, sorted by name for determinism."""
        return sorted(self._by_name.values(), key=lambda s: s.name)

    def load(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def format_listing(self, char_budget: int) -> str:
        """Render the skill listing to inject into the system prompt.

        Strategy:
          1. If full rendering fits -> use it.
          2. Else truncate descriptions uniformly, preserve names.
          3. Else (extreme) names-only.
        """
        skills = self.list_all()
        if not skills or char_budget <= 0:
            return ""

        full_lines = [self._render_entry(s, truncate=None) for s in skills]
        # Sum of line lengths + separators between lines
        full_total = sum(len(ln) for ln in full_lines) + (len(full_lines) - 1)
        if full_total <= char_budget:
            return "\n".join(full_lines)

        # Compute per-desc max that lets all lines fit.
        # Each line is "- <name>: <desc>": the non-desc portion is `len(name) + 4`
        # chars ("- ", ": "). Plus one "\n" separator between each pair of lines.
        per_line_overhead = sum(len(s.name) + 4 for s in skills)
        separator_overhead = len(skills) - 1
        name_overhead = per_line_overhead + separator_overhead
        available_for_desc = char_budget - name_overhead
        if (
            available_for_desc <= 0
            or available_for_desc // len(skills) < _MIN_DESC_CHARS
        ):
            # Names-only fallback
            return "\n".join(f"- {s.name}" for s in skills)
        max_desc = available_for_desc // len(skills)
        return "\n".join(self._render_entry(s, truncate=max_desc) for s in skills)

    @staticmethod
    def _render_entry(s: Skill, truncate: int | None) -> str:
        desc = s.description
        # Append when_to_use after an em-dash when present; skipped entirely
        # when absent so we don't emit a dangling "— " on plain skills.
        if s.when_to_use:
            desc = f"{desc} \u2014 {s.when_to_use}"
        # Hard-cap first so a 10_000-char description can't consume the budget.
        desc = desc[:_MAX_DESC_CHARS]
        if truncate is not None and len(desc) > truncate:
            desc = desc[: max(truncate - 1, 0)] + "\u2026"
        return f"- {s.name}: {desc}"
