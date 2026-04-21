"""Shared fixtures for skill integration tests."""

from pathlib import Path

from obelix.core.model.assistant_message import AssistantMessage
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

# Resolves to the `tests/fixtures/skills/` directory from
# `tests/integration/skill/conftest.py` (3 parents up + fixtures/skills).
FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider(AbstractLLMProvider):
    """AbstractLLMProvider stub that replays scripted responses in order.

    Raises RuntimeError on overrun so tests that under-script surface the
    mismatch loudly rather than deadlocking.
    """

    model_id = "stub-model"

    def __init__(self, responses: list[AssistantMessage] | None = None):
        self._responses = list(responses or [])
        self._idx = 0

    @property
    def provider_type(self):
        return Providers.ANTHROPIC

    async def invoke(self, messages, tools, response_schema=None):
        if self._idx >= len(self._responses):
            raise RuntimeError(
                "Unscripted provider call — test scripted too few responses"
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return resp
