"""Tests for obelix.core.tool.__init__ - public exports.

Covers:
- All expected symbols are exported from the package
- __all__ is consistent with actual exports
"""

from obelix.core.tool import Tool, tool


class TestPublicExports:
    """Verify the public API of the tool package."""

    def test_tool_protocol_exported(self):
        """Tool protocol should be importable from the package."""
        assert Tool is not None

    def test_tool_decorator_exported(self):
        """@tool decorator should be importable from the package."""
        assert tool is not None
        assert callable(tool)

    def test_all_contains_tool(self):
        """__all__ should list 'Tool'."""
        from obelix.core import tool as tool_pkg

        assert "Tool" in tool_pkg.__all__

    def test_all_contains_tool_decorator(self):
        """__all__ should list 'tool'."""
        from obelix.core import tool as tool_pkg

        assert "tool" in tool_pkg.__all__

    def test_all_has_exactly_two_exports(self):
        """__all__ should contain exactly Tool and tool."""
        from obelix.core import tool as tool_pkg

        assert len(tool_pkg.__all__) == 2

    def test_tool_is_same_as_direct_import(self):
        """Tool from package should be the same object as from tool_base."""
        from obelix.core.tool.tool_base import Tool as DirectTool

        assert Tool is DirectTool

    def test_tool_decorator_is_same_as_direct_import(self):
        """tool from package should be the same object as from tool_decorator."""
        from obelix.core.tool.tool_decorator import tool as direct_tool

        assert tool is direct_tool
