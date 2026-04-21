"""End-to-end: invalid skills_config aggregates all issues into one error."""

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.skill.skill import SkillValidationError

from .conftest import FIXTURES
from .conftest import StubProvider as _StubProvider


def test_agent_construction_fails_cleanly_on_invalid_skills():
    """3 invalid skill fixtures -> SkillValidationError with issues spanning all 3."""
    with pytest.raises(SkillValidationError) as exc:
        BaseAgent(
            system_message="X",
            provider=_StubProvider(),
            skills_config=[
                str(FIXTURES / "invalid_missing_description"),
                str(FIXTURES / "invalid_duplicate_args"),
                str(FIXTURES / "invalid_unknown_hook"),
            ],
        )
    files = {str(i.file_path) for i in exc.value.issues if i.file_path is not None}
    assert len(files) == 3


def test_mixed_valid_invalid_aggregates_only_invalid():
    """Mix valid + invalid paths: validation error carries only invalid's issues."""
    with pytest.raises(SkillValidationError) as exc:
        BaseAgent(
            system_message="X",
            provider=_StubProvider(),
            skills_config=[
                str(FIXTURES / "minimal"),
                str(FIXTURES / "invalid_missing_description"),
            ],
        )
    files = {str(i.file_path) for i in exc.value.issues if i.file_path is not None}
    assert len(files) == 1
    assert "invalid_missing_description" in next(iter(files))
