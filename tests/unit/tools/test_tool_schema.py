"""
Unit tests for ToolSchema class.

Tests verify:
- Schema creation with explicit tool_name and tool_description
- Schema creation with fallback mechanisms
- Pydantic field validation
- JSON schema generation
- ClassVar behavior for metadata
"""

import pytest
from pydantic import ValidationError, Field
from typing import Optional

from src.tools.tool_schema import ToolSchema


class TestToolSchemaBasicCreation:
    """Test basic ToolSchema creation and metadata."""

    def test_tool_schema_with_explicit_metadata(self):
        """Test schema with explicit tool_name and tool_description."""
        class TestSchema(ToolSchema):
            tool_name = "test_tool"
            tool_description = "A test tool for unit testing"
            param1: str
            param2: int

        assert TestSchema.get_tool_name() == "test_tool"
        assert TestSchema.get_tool_description() == "A test tool for unit testing"

    def test_tool_schema_name_fallback_from_class_name(self):
        """Test that tool_name falls back to class name when not specified."""
        class MyCustomToolSchema(ToolSchema):
            param: str

        # Should derive "mycustom" from "MyCustomToolSchema"
        name = MyCustomToolSchema.get_tool_name()
        assert name == "mycustom"
        assert "schema" not in name.lower()
        assert "tool" not in name.lower()

    def test_tool_schema_description_fallback_from_docstring(self):
        """Test that description falls back to docstring when not specified."""
        class TestSchema(ToolSchema):
            """This is a test schema with a docstring"""
            param: str

        description = TestSchema.get_tool_description()
        assert description == "This is a test schema with a docstring"

    def test_tool_schema_description_default_when_no_docstring(self):
        """Test default description when neither description nor docstring provided."""
        class TestSchema(ToolSchema):
            param: str

        description = TestSchema.get_tool_description()
        # Should be "Tool <tool_name>"
        assert "Tool" in description
        assert TestSchema.get_tool_name() in description


class TestToolSchemaFieldValidation:
    """Test Pydantic field validation in ToolSchema."""

    def test_tool_schema_with_required_fields(self):
        """Test schema with required parameters."""
        class RequiredFieldsSchema(ToolSchema):
            tool_name = "required_test"
            required_str: str
            required_int: int

        # Valid instantiation
        instance = RequiredFieldsSchema(required_str="test", required_int=42)
        assert instance.required_str == "test"
        assert instance.required_int == 42

    def test_tool_schema_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        class RequiredFieldsSchema(ToolSchema):
            tool_name = "required_test"
            required_str: str
            required_int: int

        with pytest.raises(ValidationError) as exc_info:
            RequiredFieldsSchema(required_str="test")  # Missing required_int

        assert "required_int" in str(exc_info.value)

    def test_tool_schema_with_optional_fields(self):
        """Test schema with optional parameters and defaults."""
        class OptionalFieldsSchema(ToolSchema):
            tool_name = "optional_test"
            required_param: str
            optional_param: Optional[str] = None
            default_param: int = 42

        # With all fields
        instance1 = OptionalFieldsSchema(
            required_param="test",
            optional_param="optional",
            default_param=100
        )
        assert instance1.optional_param == "optional"
        assert instance1.default_param == 100

        # Without optional fields
        instance2 = OptionalFieldsSchema(required_param="test")
        assert instance2.optional_param is None
        assert instance2.default_param == 42

    def test_tool_schema_type_validation(self):
        """Test that Pydantic validates field types correctly."""
        class TypedSchema(ToolSchema):
            tool_name = "typed_test"
            string_field: str
            int_field: int
            bool_field: bool

        # Valid types
        instance = TypedSchema(
            string_field="text",
            int_field=42,
            bool_field=True
        )
        assert isinstance(instance.string_field, str)
        assert isinstance(instance.int_field, int)
        assert isinstance(instance.bool_field, bool)

        # Invalid type (string instead of int)
        with pytest.raises(ValidationError):
            TypedSchema(
                string_field="text",
                int_field="not_an_int",  # Should fail
                bool_field=True
            )


class TestToolSchemaComplexTypes:
    """Test ToolSchema with complex field types."""

    def test_tool_schema_with_nested_dict(self):
        """Test schema with dictionary field."""
        from typing import Dict

        class DictSchema(ToolSchema):
            tool_name = "dict_test"
            config: Dict[str, str]

        instance = DictSchema(config={"key1": "value1", "key2": "value2"})
        assert instance.config["key1"] == "value1"
        assert len(instance.config) == 2

    def test_tool_schema_with_list_field(self):
        """Test schema with list field."""
        from typing import List

        class ListSchema(ToolSchema):
            tool_name = "list_test"
            items: List[int]

        instance = ListSchema(items=[1, 2, 3, 4, 5])
        assert len(instance.items) == 5
        assert instance.items[0] == 1

    def test_tool_schema_with_nested_pydantic_model(self):
        """Test schema with nested Pydantic model."""
        from pydantic import BaseModel

        class NestedModel(BaseModel):
            name: str
            value: int

        class NestedSchema(ToolSchema):
            tool_name = "nested_test"
            nested: NestedModel

        nested_data = NestedModel(name="test", value=42)
        instance = NestedSchema(nested=nested_data)
        assert instance.nested.name == "test"
        assert instance.nested.value == 42


class TestToolSchemaJsonGeneration:
    """Test JSON schema generation from ToolSchema."""

    def test_json_schema_generation(self):
        """Test that model_json_schema() generates correct JSON schema."""
        class JsonSchema(ToolSchema):
            tool_name = "json_test"
            tool_description = "Test JSON generation"
            string_param: str
            int_param: int
            optional_param: Optional[str] = None

        json_schema = JsonSchema.model_json_schema()

        assert "properties" in json_schema
        assert "string_param" in json_schema["properties"]
        assert "int_param" in json_schema["properties"]
        assert "optional_param" in json_schema["properties"]

    def test_json_schema_field_types(self):
        """Test that JSON schema contains correct field types."""
        class TypeSchema(ToolSchema):
            tool_name = "type_test"
            str_field: str
            int_field: int
            bool_field: bool

        json_schema = TypeSchema.model_json_schema()

        assert json_schema["properties"]["str_field"]["type"] == "string"
        assert json_schema["properties"]["int_field"]["type"] == "integer"
        assert json_schema["properties"]["bool_field"]["type"] == "boolean"

    def test_json_schema_required_fields(self):
        """Test that JSON schema marks required fields correctly."""
        class RequiredSchema(ToolSchema):
            tool_name = "required_test"
            required_field: str
            optional_field: Optional[str] = None

        json_schema = RequiredSchema.model_json_schema()

        # Required fields should be in 'required' list
        assert "required" in json_schema
        assert "required_field" in json_schema["required"]
        # Optional fields should NOT be in 'required' list
        assert "optional_field" not in json_schema.get("required", [])


class TestToolSchemaMetadataIsolation:
    """Test that ClassVar metadata doesn't interfere between schemas."""

    def test_different_schemas_have_independent_metadata(self):
        """Test that tool_name and tool_description are independent per class."""
        class Schema1(ToolSchema):
            tool_name = "schema_one"
            tool_description = "First schema"
            param: str

        class Schema2(ToolSchema):
            tool_name = "schema_two"
            tool_description = "Second schema"
            param: str

        assert Schema1.get_tool_name() == "schema_one"
        assert Schema2.get_tool_name() == "schema_two"
        assert Schema1.get_tool_description() == "First schema"
        assert Schema2.get_tool_description() == "Second schema"

    def test_schema_instances_share_class_metadata(self):
        """Test that instances share class-level metadata."""
        class TestSchema(ToolSchema):
            tool_name = "shared_test"
            tool_description = "Shared metadata"
            param: str

        instance1 = TestSchema(param="value1")
        instance2 = TestSchema(param="value2")

        # Both instances should access same class metadata
        assert instance1.get_tool_name() == instance2.get_tool_name()
        assert instance1.get_tool_description() == instance2.get_tool_description()


class TestToolSchemaEdgeCases:
    """Test edge cases and special scenarios."""

    def test_tool_schema_with_empty_string_name(self):
        """Test behavior with empty string tool_name."""
        class EmptyNameSchema(ToolSchema):
            tool_name = ""
            param: str

        # Empty string is falsy, should fallback to class name
        name = EmptyNameSchema.get_tool_name()
        assert name != ""
        assert "emptyname" in name.lower()

    def test_tool_schema_with_special_characters_in_class_name(self):
        """Test tool name derivation from class with special patterns."""
        class My_Tool_Schema_Test(ToolSchema):
            param: str

        name = My_Tool_Schema_Test.get_tool_name()
        # Should remove 'schema' and 'tool', lowercase the rest
        assert "schema" not in name.lower()
        assert "tool" not in name.lower()

    def test_tool_schema_inheritance(self):
        """Test that ToolSchema can be inherited multiple times."""
        class BaseCustomSchema(ToolSchema):
            tool_name = "base_custom"
            base_param: str

        class ExtendedSchema(BaseCustomSchema):
            tool_name = "extended_custom"
            extended_param: int

        # Extended schema should have its own metadata
        assert ExtendedSchema.get_tool_name() == "extended_custom"

        # Should have both fields
        instance = ExtendedSchema(base_param="test", extended_param=42)
        assert instance.base_param == "test"
        assert instance.extended_param == 42

    def test_tool_schema_with_field_description(self):
        """Test schema with Pydantic Field descriptions."""
        class DescribedFieldSchema(ToolSchema):
            tool_name = "described_test"
            param_with_desc: str = Field(description="A parameter with description")
            param_with_default: int = Field(default=42, description="Has default value")

        json_schema = DescribedFieldSchema.model_json_schema()

        # Field descriptions should be in JSON schema
        assert "description" in json_schema["properties"]["param_with_desc"]
        assert json_schema["properties"]["param_with_desc"]["description"] == "A parameter with description"
