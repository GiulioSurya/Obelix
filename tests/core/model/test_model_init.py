"""Tests for obelix.core.model __init__ — verify public API exports."""

import obelix.core.model as model_pkg


class TestModelPackageExports:
    """All symbols listed in __all__ should be importable from the package."""

    def test_all_symbols_are_accessible(self):
        for name in model_pkg.__all__:
            assert hasattr(model_pkg, name), f"{name} not found in obelix.core.model"

    def test_all_list_is_complete(self):
        expected = {
            "SystemMessage",
            "HumanMessage",
            "AssistantMessage",
            "AssistantResponse",
            "StreamEvent",
            "ToolCall",
            "ToolResult",
            "ToolStatus",
            "ToolRequirement",
            "ToolMessage",
            "MessageRole",
            "StandardMessage",
            "Usage",
            "AgentUsage",
            "TextContent",
            "FileContent",
            "DataContent",
            "ContentPart",
        }
        assert set(model_pkg.__all__) == expected
