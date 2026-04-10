"""Tests for behavioral annotations on Tool protocol and @tool decorator.

Covers:
- Tool WITH annotations satisfies the Tool protocol
- Tool WITHOUT annotations still satisfies the Tool protocol (runtime_checkable only checks methods)
- Annotation values are accessible on decorated classes
- Default annotation values are None
"""

from pydantic import Field

from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool

# ---------------------------------------------------------------------------
# Helpers: tools with and without annotations
# ---------------------------------------------------------------------------


@tool(
    name="annotated_tool",
    description="A tool with all annotations set",
    read_only=True,
    destructive=False,
    idempotent=True,
    open_world=False,
)
class AnnotatedTool:
    query: str = Field(..., description="Input query")

    def execute(self) -> dict:
        return {"result": self.query}


@tool(name="plain_tool", description="A tool without any annotations")
class PlainTool:
    query: str = Field(..., description="Input query")

    def execute(self) -> dict:
        return {"result": self.query}


@tool(
    name="partial_tool",
    description="A tool with only some annotations",
    read_only=True,
    destructive=True,
)
class PartialTool:
    query: str = Field(..., description="Input query")

    def execute(self) -> dict:
        return {"result": self.query}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnnotatedToolSatisfiesProtocol:
    """A tool WITH annotations satisfies the Tool protocol."""

    def test_annotated_tool_is_instance_of_tool(self):
        """Tool with all annotations should pass isinstance check."""
        obj = AnnotatedTool()
        assert isinstance(obj, Tool)

    def test_plain_tool_is_instance_of_tool(self):
        """Tool WITHOUT annotations still satisfies the protocol.

        runtime_checkable Protocol only checks methods, not attributes.
        """
        obj = PlainTool()
        assert isinstance(obj, Tool)

    def test_partial_tool_is_instance_of_tool(self):
        """Tool with partial annotations satisfies the protocol."""
        obj = PartialTool()
        assert isinstance(obj, Tool)


class TestAnnotationValuesAccessible:
    """Verify annotation values are accessible on the class and instance."""

    def test_annotated_tool_read_only(self):
        assert AnnotatedTool.read_only is True

    def test_annotated_tool_destructive(self):
        assert AnnotatedTool.destructive is False

    def test_annotated_tool_idempotent(self):
        assert AnnotatedTool.idempotent is True

    def test_annotated_tool_open_world(self):
        assert AnnotatedTool.open_world is False

    def test_plain_tool_defaults_to_none(self):
        """All annotations default to None when not specified."""
        assert PlainTool.read_only is None
        assert PlainTool.destructive is None
        assert PlainTool.idempotent is None
        assert PlainTool.open_world is None

    def test_partial_tool_set_values(self):
        """Only specified annotations are set; others remain None."""
        assert PartialTool.read_only is True
        assert PartialTool.destructive is True
        assert PartialTool.idempotent is None
        assert PartialTool.open_world is None

    def test_annotations_accessible_on_instance(self):
        """Annotations should be accessible on instances too."""
        obj = AnnotatedTool()
        assert obj.read_only is True
        assert obj.destructive is False
        assert obj.idempotent is True
        assert obj.open_world is False

    def test_plain_annotations_accessible_on_instance(self):
        """None defaults accessible on instances."""
        obj = PlainTool()
        assert obj.read_only is None
        assert obj.destructive is None
        assert obj.idempotent is None
        assert obj.open_world is None


class TestAnnotationsPropagatedToSchema:
    """Verify annotations are propagated to MCPToolSchema via create_schema()."""

    def test_annotated_tool_schema_has_annotations(self):
        """A fully annotated tool should have all MCP-format hints in schema."""
        schema = AnnotatedTool.create_schema()
        assert schema.annotations is not None
        assert schema.annotations["readOnlyHint"] is True
        assert schema.annotations["destructiveHint"] is False
        assert schema.annotations["idempotentHint"] is True
        assert schema.annotations["openWorldHint"] is False

    def test_plain_tool_schema_annotations_is_none(self):
        """A tool without any annotations should have annotations=None."""
        schema = PlainTool.create_schema()
        assert schema.annotations is None

    def test_partial_tool_schema_has_only_set_annotations(self):
        """A partially annotated tool includes only declared hints."""
        schema = PartialTool.create_schema()
        assert schema.annotations is not None
        assert schema.annotations["readOnlyHint"] is True
        assert schema.annotations["destructiveHint"] is True
        assert "idempotentHint" not in schema.annotations
        assert "openWorldHint" not in schema.annotations

    def test_annotated_tool_schema_annotation_count(self):
        """Fully annotated tool should have exactly 4 hint keys."""
        schema = AnnotatedTool.create_schema()
        assert len(schema.annotations) == 4

    def test_partial_tool_schema_annotation_count(self):
        """Partially annotated tool should have exactly 2 hint keys."""
        schema = PartialTool.create_schema()
        assert len(schema.annotations) == 2
