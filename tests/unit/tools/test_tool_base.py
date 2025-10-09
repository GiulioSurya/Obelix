"""
Unit tests for ToolBase abstract class.

Tests verify:
- Abstract class cannot be instantiated directly
- create_schema() generates correct MCPToolSchema
- schema_class is required
- execute() must be implemented by subclasses
- Schema generation with various ToolSchema configurations
"""

import pytest
from typing import Type

from src.tools.tool_base import ToolBase
from src.tools.tool_schema import ToolSchema
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema


class TestToolBaseAbstractBehavior:
    """Test that ToolBase is properly abstract."""

    def test_tool_base_cannot_be_instantiated_directly(self):
        """Test that ToolBase cannot be instantiated without implementing execute()."""
        with pytest.raises(TypeError) as exc_info:
            ToolBase()

        # Error should mention abstract method 'execute'
        assert "abstract" in str(exc_info.value).lower()
        assert "execute" in str(exc_info.value).lower()

    def test_tool_base_requires_execute_implementation(self):
        """Test that subclass without execute() cannot be instantiated."""
        class IncompleteTool(ToolBase):
            # Missing execute() implementation
            schema_class = None

        with pytest.raises(TypeError):
            IncompleteTool()


class TestToolBaseSchemaCreation:
    """Test create_schema() method."""

    def test_create_schema_requires_schema_class(self):
        """Test that create_schema() raises error if schema_class not defined."""
        class NoSchemaClassTool(ToolBase):
            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass  # Dummy implementation

        with pytest.raises(ValueError) as exc_info:
            NoSchemaClassTool.create_schema()

        assert "schema_class" in str(exc_info.value)

    def test_create_schema_with_valid_schema_class(self):
        """Test that create_schema() generates correct MCPToolSchema."""
        # Define a schema
        class TestToolSchema(ToolSchema):
            tool_name = "test_tool"
            tool_description = "A test tool"
            param1: str
            param2: int

        # Define tool with schema
        class TestTool(ToolBase):
            schema_class = TestToolSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        # Create schema
        mcp_schema = TestTool.create_schema()

        # Verify MCPToolSchema structure
        assert isinstance(mcp_schema, MCPToolSchema)
        assert mcp_schema.name == "test_tool"
        assert mcp_schema.description == "A test tool"
        assert mcp_schema.inputSchema is not None
        assert "properties" in mcp_schema.inputSchema

    def test_create_schema_includes_input_schema(self):
        """Test that inputSchema contains correct field definitions."""
        class DetailedSchema(ToolSchema):
            tool_name = "detailed_tool"
            tool_description = "Tool with detailed schema"
            string_field: str
            int_field: int
            optional_field: str = "default"

        class DetailedTool(ToolBase):
            schema_class = DetailedSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = DetailedTool.create_schema()

        # Verify inputSchema has all fields
        input_props = mcp_schema.inputSchema["properties"]
        assert "string_field" in input_props
        assert "int_field" in input_props
        assert "optional_field" in input_props

    def test_create_schema_has_output_schema(self):
        """Test that outputSchema is included in MCPToolSchema."""
        class SimpleSchema(ToolSchema):
            tool_name = "simple_tool"
            param: str

        class SimpleTool(ToolBase):
            schema_class = SimpleSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = SimpleTool.create_schema()

        # Verify outputSchema exists
        assert mcp_schema.outputSchema is not None
        assert mcp_schema.outputSchema["type"] == "object"
        assert mcp_schema.outputSchema["additionalProperties"] is True


class TestToolBaseSchemaDerivedMetadata:
    """Test schema metadata derivation from ToolSchema."""

    def test_schema_name_derived_from_tool_schema(self):
        """Test that tool name comes from ToolSchema.get_tool_name()."""
        class NamedSchema(ToolSchema):
            tool_name = "custom_name"
            param: str

        class NamedTool(ToolBase):
            schema_class = NamedSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = NamedTool.create_schema()
        assert mcp_schema.name == "custom_name"

    def test_schema_description_derived_from_tool_schema(self):
        """Test that description comes from ToolSchema.get_tool_description()."""
        class DescribedSchema(ToolSchema):
            """This is the tool description from docstring"""
            param: str

        class DescribedTool(ToolBase):
            schema_class = DescribedSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = DescribedTool.create_schema()
        assert mcp_schema.description == "This is the tool description from docstring"

    def test_schema_name_fallback_from_class_name(self):
        """Test name fallback when tool_name not explicitly set."""
        class MyCustomToolSchema(ToolSchema):
            param: str

        class MyCustomTool(ToolBase):
            schema_class = MyCustomToolSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = MyCustomTool.create_schema()
        # Should derive from "MyCustomToolSchema" → "mycustom"
        assert "mycustom" in mcp_schema.name.lower()


class TestToolBaseMultipleTools:
    """Test behavior with multiple tool implementations."""

    def test_different_tools_have_independent_schemas(self):
        """Test that different tool classes have independent schemas."""
        class Schema1(ToolSchema):
            tool_name = "tool_one"
            param: str

        class Schema2(ToolSchema):
            tool_name = "tool_two"
            param: int

        class Tool1(ToolBase):
            schema_class = Schema1

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        class Tool2(ToolBase):
            schema_class = Schema2

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        schema1 = Tool1.create_schema()
        schema2 = Tool2.create_schema()

        assert schema1.name == "tool_one"
        assert schema2.name == "tool_two"
        assert schema1.name != schema2.name

    def test_tool_schema_is_class_level(self):
        """Test that schema_class is defined at class level, not instance level."""
        class TestSchema(ToolSchema):
            tool_name = "class_level_test"
            param: str

        class ClassLevelTool(ToolBase):
            schema_class = TestSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_name="class_level_test",
                    tool_call_id=tool_call.id,
                    result="ok",
                    status=ToolStatus.SUCCESS
                )

        # Create schema without instance
        schema_from_class = ClassLevelTool.create_schema()
        assert schema_from_class.name == "class_level_test"

        # Verify instance doesn't change class-level schema
        instance = ClassLevelTool()
        schema_from_instance = instance.create_schema()
        assert schema_from_instance.name == schema_from_class.name


class TestToolBaseSchemaComplexity:
    """Test schema creation with complex ToolSchema definitions."""

    def test_schema_with_nested_types(self):
        """Test schema creation with complex nested types."""
        from typing import Dict, List, Optional

        class ComplexSchema(ToolSchema):
            tool_name = "complex_tool"
            tool_description = "Tool with complex types"
            simple_field: str
            list_field: List[str]
            dict_field: Dict[str, int]
            optional_field: Optional[str] = None

        class ComplexTool(ToolBase):
            schema_class = ComplexSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = ComplexTool.create_schema()

        # Verify all fields are in inputSchema
        input_props = mcp_schema.inputSchema["properties"]
        assert "simple_field" in input_props
        assert "list_field" in input_props
        assert "dict_field" in input_props
        assert "optional_field" in input_props

    def test_schema_with_pydantic_field_descriptions(self):
        """Test that Pydantic Field descriptions are preserved in schema."""
        from pydantic import Field

        class DescriptiveSchema(ToolSchema):
            tool_name = "descriptive_tool"
            param_with_desc: str = Field(description="This parameter has a description")
            param_with_default: int = Field(default=42, description="Has default and description")

        class DescriptiveTool(ToolBase):
            schema_class = DescriptiveSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        mcp_schema = DescriptiveTool.create_schema()

        # Verify descriptions are in schema
        input_props = mcp_schema.inputSchema["properties"]
        assert "description" in input_props["param_with_desc"]
        assert input_props["param_with_desc"]["description"] == "This parameter has a description"


class TestToolBaseEdgeCases:
    """Test edge cases and error conditions."""

    def test_tool_with_minimal_schema(self):
        """Test tool with minimal schema (no fields)."""
        class MinimalSchema(ToolSchema):
            tool_name = "minimal"
            tool_description = "Minimal tool"
            # No fields defined

        class MinimalTool(ToolBase):
            schema_class = MinimalSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_name="minimal",
                    tool_call_id=tool_call.id,
                    result=None,
                    status=ToolStatus.SUCCESS
                )

        # Should work even with no fields
        mcp_schema = MinimalTool.create_schema()
        assert mcp_schema.name == "minimal"
        assert mcp_schema.inputSchema is not None

    def test_tool_inheritance_preserves_schema_class(self):
        """Test that tool inheritance preserves schema_class correctly."""
        class BaseSchema(ToolSchema):
            tool_name = "base_tool"
            base_param: str

        class BaseTool(ToolBase):
            schema_class = BaseSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        class ExtendedSchema(ToolSchema):
            tool_name = "extended_tool"
            extended_param: int

        class ExtendedTool(BaseTool):
            schema_class = ExtendedSchema  # Override schema

        # Extended tool should have its own schema
        extended_schema = ExtendedTool.create_schema()
        assert extended_schema.name == "extended_tool"

        # Base tool schema should be unchanged
        base_schema = BaseTool.create_schema()
        assert base_schema.name == "base_tool"

    def test_create_schema_is_idempotent(self):
        """Test that calling create_schema() multiple times returns consistent results."""
        class IdempotentSchema(ToolSchema):
            tool_name = "idempotent_tool"
            param: str

        class IdempotentTool(ToolBase):
            schema_class = IdempotentSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                pass

        schema1 = IdempotentTool.create_schema()
        schema2 = IdempotentTool.create_schema()

        # Should have same values
        assert schema1.name == schema2.name
        assert schema1.description == schema2.description
        assert schema1.inputSchema == schema2.inputSchema


class TestToolBaseExecuteSignature:
    """Test execute() method signature and requirements."""

    @pytest.mark.asyncio
    async def test_execute_receives_tool_call(self):
        """Test that execute() receives ToolCall parameter."""
        class ExecuteTestSchema(ToolSchema):
            tool_name = "execute_test"
            param: str

        class ExecuteTestTool(ToolBase):
            schema_class = ExecuteTestSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                # Verify we receive a ToolCall
                assert isinstance(tool_call, ToolCall)
                return ToolResult(
                    tool_name="execute_test",
                    tool_call_id=tool_call.id,
                    result=f"received: {tool_call.name}",
                    status=ToolStatus.SUCCESS
                )

        tool = ExecuteTestTool()
        call = ToolCall(id="test_123", name="execute_test", arguments={"param": "value"})
        result = await tool.execute(call)

        assert isinstance(result, ToolResult)
        assert "received" in result.result

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        """Test that execute() returns ToolResult."""
        class ResultTestSchema(ToolSchema):
            tool_name = "result_test"
            param: str

        class ResultTestTool(ToolBase):
            schema_class = ResultTestSchema

            async def execute(self, tool_call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_name="result_test",
                    tool_call_id=tool_call.id,
                    result="success",
                    status=ToolStatus.SUCCESS
                )

        tool = ResultTestTool()
        call = ToolCall(id="test_456", name="result_test", arguments={"param": "test"})
        result = await tool.execute(call)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "result_test"
        assert result.tool_call_id == "test_456"
        assert result.status == ToolStatus.SUCCESS