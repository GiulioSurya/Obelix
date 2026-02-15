"""Shared fixtures for Obelix test suite."""

from unittest.mock import AsyncMock

import pytest

from obelix.core.model.assistant_message import AssistantMessage


class MockProvider:
    """Minimal mock that satisfies AbstractLLMProvider contract for BaseAgent."""

    def __init__(self, response_content: str = "Mock response") -> None:
        self.model_id = "mock-model"
        self.provider_type = "mock"
        self._response_content = response_content
        self.invoke = AsyncMock(
            return_value=AssistantMessage(content=response_content)
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory fixture for creating MockProvider with custom responses."""
    def _create(response_content: str = "Mock response") -> MockProvider:
        return MockProvider(response_content=response_content)
    return _create