"""Port for skill providers.

Concrete providers (filesystem, MCP, ...) implement this ABC.
SkillManager depends only on this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from obelix.core.skill.skill import Skill


class AbstractSkillProvider(ABC):
    """A source of skills."""

    @abstractmethod
    def discover(self) -> list[Skill]:
        """Return all skills this provider offers.

        Implementations must raise SkillValidationError if any candidate
        fails validation, aggregating issues across all candidates.
        """
        ...
