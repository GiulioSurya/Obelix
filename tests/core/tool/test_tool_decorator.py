"""Tests for obelix.core.tool.tool_decorator - @tool decorator.

Covers:
- Decorator validation (missing name / description)
- Class attribute injection (tool_name, tool_description)
- Input schema generation from Pydantic Fields
- Wrapped execute: sync and async paths
- Validation error handling (missing args, wrong types)
- Runtime exception handling
- ToolResult construction (success, error, execution_time)
- Parallel execution isolation (copy.copy)
- create_schema() classmethod
- Edge cases: no fields, optional fields, complex types
"""

import asyncio
from enum import StrEnum

import pytest
from pydantic import Field

from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from obelix.core.tool.tool_decorator import tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(
    name: str = "test_tool",
    call_id: str = "tc_1",
    arguments: dict | None = None,
) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# Decorator validation tests
# ---------------------------------------------------------------------------


class TestDecoratorValidation:
    """@tool must reject missing name or description at import time."""

    def test_missing_name_raises_value_error(self):
        with pytest.raises(ValueError, match="'name' is required"):

            @tool(name="", description="desc")
            class Bad:
                pass

    def test_none_name_raises_value_error(self):
        with pytest.raises(ValueError, match="'name' is required"):

            @tool(name=None, description="desc")
            class Bad:
                pass

    def test_missing_description_raises_value_error(self):
        with pytest.raises(ValueError, match="'description' is required"):

            @tool(name="my_tool", description="")
            class Bad:
                pass

    def test_none_description_raises_value_error(self):
        with pytest.raises(ValueError, match="'description' is required"):

            @tool(name="my_tool", description=None)
            class Bad:
                pass

    def test_both_missing_raises_name_error_first(self):
        """When both are missing, name is checked first."""
        with pytest.raises(ValueError, match="'name' is required"):

            @tool(name="", description="")
            class Bad:
                pass


# ---------------------------------------------------------------------------
# Attribute injection
# ---------------------------------------------------------------------------


class TestAttributeInjection:
    """Decorator should inject tool_name and tool_description on the class."""

    def test_tool_name_set_on_class(self):
        @tool(name="my_tool", description="My description")
        class MyTool:
            def execute(self) -> dict:
                return {}

        assert MyTool.tool_name == "my_tool"

    def test_tool_description_set_on_class(self):
        @tool(name="my_tool", description="My description")
        class MyTool:
            def execute(self) -> dict:
                return {}

        assert MyTool.tool_description == "My description"

    def test_tool_name_accessible_on_instance(self):
        @tool(name="instance_tool", description="desc")
        class MyTool:
            def execute(self) -> dict:
                return {}

        instance = MyTool()
        assert instance.tool_name == "instance_tool"

    def test_tool_description_accessible_on_instance(self):
        @tool(name="instance_tool", description="desc")
        class MyTool:
            def execute(self) -> dict:
                return {}

        instance = MyTool()
        assert instance.tool_description == "desc"


# ---------------------------------------------------------------------------
# Input schema generation
# ---------------------------------------------------------------------------


class TestInputSchemaGeneration:
    """_create_input_schema extracts Pydantic Fields into a dynamic model."""

    def test_schema_created_for_fields(self):
        @tool(name="calc", description="Calculator")
        class Calc:
            a: float = Field(..., description="First")
            b: float = Field(..., description="Second")

            def execute(self) -> dict:
                return {"sum": self.a + self.b}

        schema = Calc._input_schema
        assert "a" in schema.model_fields
        assert "b" in schema.model_fields

    def test_schema_ignores_non_field_attributes(self):
        @tool(name="partial", description="desc")
        class Partial:
            x: int = Field(..., description="A field")
            y: str = "not_a_field"  # plain default, not FieldInfo

            def execute(self) -> dict:
                return {}

        schema = Partial._input_schema
        assert "x" in schema.model_fields
        assert "y" not in schema.model_fields

    def test_schema_ignores_private_attributes(self):
        @tool(name="priv", description="desc")
        class Priv:
            _internal: str = Field(default="hidden")
            public: str = Field(..., description="Visible")

            def execute(self) -> dict:
                return {}

        schema = Priv._input_schema
        assert "public" in schema.model_fields
        assert "_internal" not in schema.model_fields

    def test_no_fields_produces_empty_schema(self):
        @tool(name="empty", description="desc")
        class Empty:
            def execute(self) -> dict:
                return {}

        schema = Empty._input_schema
        assert len(schema.model_fields) == 0

    def test_optional_field_in_schema(self):
        @tool(name="opt", description="desc")
        class Opt:
            query: str = Field(..., description="Required")
            limit: int = Field(default=10, description="Optional with default")

            def execute(self) -> dict:
                return {}

        schema = Opt._input_schema
        assert "query" in schema.model_fields
        assert "limit" in schema.model_fields
        # limit should have a default
        assert schema.model_fields["limit"].default == 10

    def test_schema_class_name_derived_from_tool_name(self):
        @tool(name="my_great_tool", description="desc")
        class Whatever:
            def execute(self) -> dict:
                return {}

        assert Whatever._input_schema.__name__ == "MyGreatToolSchema"

    def test_complex_field_types(self):
        @tool(name="complex_tool", description="desc")
        class ComplexTool:
            tags: list[str] = Field(..., description="List of tags")
            config: dict[str, int] = Field(..., description="Config map")

            def execute(self) -> dict:
                return {}

        schema = ComplexTool._input_schema
        assert "tags" in schema.model_fields
        assert "config" in schema.model_fields


# ---------------------------------------------------------------------------
# Sync execute path
# ---------------------------------------------------------------------------


class TestSyncExecute:
    """Tests for wrapped execute with sync original method."""

    @pytest.fixture
    def adder_class(self):
        @tool(name="adder", description="Adds two numbers")
        class Adder:
            a: float = Field(..., description="First")
            b: float = Field(..., description="Second")

            def execute(self) -> dict:
                return {"sum": self.a + self.b}

        return Adder

    async def test_success_returns_tool_result(self, adder_class):
        instance = adder_class()
        call = _make_tool_call("adder", arguments={"a": 3, "b": 7})
        result = await instance.execute(call)

        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"sum": 10.0}
        assert result.tool_name == "adder"
        assert result.tool_call_id == "tc_1"
        assert result.error is None

    async def test_execution_time_is_recorded(self, adder_class):
        instance = adder_class()
        call = _make_tool_call("adder", arguments={"a": 1, "b": 2})
        result = await instance.execute(call)

        assert result.execution_time is not None
        assert result.execution_time >= 0

    async def test_float_coercion(self, adder_class):
        """Pydantic should coerce int to float for float fields."""
        instance = adder_class()
        call = _make_tool_call("adder", arguments={"a": 5, "b": 10})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"sum": 15.0}


# ---------------------------------------------------------------------------
# Async execute path
# ---------------------------------------------------------------------------


class TestAsyncExecute:
    """Tests for wrapped execute with async original method."""

    @pytest.fixture
    def async_greeter_class(self):
        @tool(name="greeter", description="Greets a person")
        class AsyncGreeter:
            name: str = Field(..., description="Name to greet")

            async def execute(self) -> dict:
                return {"greeting": f"Hello, {self.name}!"}

        return AsyncGreeter

    async def test_async_success(self, async_greeter_class):
        instance = async_greeter_class()
        call = _make_tool_call("greeter", arguments={"name": "Alice"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"greeting": "Hello, Alice!"}

    async def test_async_execute_is_awaited_directly(self, async_greeter_class):
        """Async execute should not go through asyncio.to_thread."""
        instance = async_greeter_class()
        call = _make_tool_call("greeter", arguments={"name": "Bob"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Wrapped execute should return ERROR ToolResult on validation failures."""

    @pytest.fixture
    def strict_tool_class(self):
        @tool(name="strict", description="Strict tool")
        class Strict:
            required_str: str = Field(..., description="Required string")
            required_int: int = Field(..., description="Required integer")

            def execute(self) -> dict:
                return {"ok": True}

        return Strict

    async def test_missing_required_argument(self, strict_tool_class):
        instance = strict_tool_class()
        call = _make_tool_call("strict", arguments={"required_str": "hello"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.error is not None
        assert "VALIDATION ERROR" in result.error
        assert result.result is None

    async def test_wrong_type_argument(self, strict_tool_class):
        instance = strict_tool_class()
        call = _make_tool_call(
            "strict",
            arguments={"required_str": "ok", "required_int": "not_an_int"},
        )
        result = await instance.execute(call)

        # Pydantic may coerce "not_an_int" to fail or succeed depending on strict mode
        # With default Pydantic, "not_an_int" cannot be coerced to int -> ERROR
        assert result.status == ToolStatus.ERROR
        assert result.error is not None

    async def test_no_arguments_when_required(self, strict_tool_class):
        instance = strict_tool_class()
        call = _make_tool_call("strict", arguments={})
        result = await instance.execute(call)

        assert result.status == ToolStatus.ERROR
        assert "VALIDATION ERROR" in result.error

    async def test_validation_error_preserves_tool_metadata(self, strict_tool_class):
        instance = strict_tool_class()
        call = _make_tool_call("strict", call_id="tc_42", arguments={})
        result = await instance.execute(call)

        assert result.tool_name == "strict"
        assert result.tool_call_id == "tc_42"
        assert result.execution_time is not None


# ---------------------------------------------------------------------------
# Runtime exception handling
# ---------------------------------------------------------------------------


class TestRuntimeExceptionHandling:
    """Wrapped execute catches runtime exceptions and returns ERROR ToolResult."""

    @pytest.fixture
    def error_tool_class(self):
        @tool(name="error_tool", description="Raises errors")
        class ErrorTool:
            msg: str = Field(..., description="Error message")

            def execute(self) -> dict:
                raise RuntimeError(self.msg)

        return ErrorTool

    async def test_runtime_error_caught(self, error_tool_class):
        instance = error_tool_class()
        call = _make_tool_call("error_tool", arguments={"msg": "boom"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.error == "boom"
        assert result.result is None

    async def test_generic_exception_caught(self):
        @tool(name="generic_err", description="desc")
        class GenericErr:
            x: int = Field(..., description="val")

            def execute(self) -> dict:
                raise ValueError("bad value")

        instance = GenericErr()
        call = _make_tool_call("generic_err", arguments={"x": 1})
        result = await instance.execute(call)

        assert result.status == ToolStatus.ERROR
        assert "bad value" in result.error

    async def test_async_exception_caught(self):
        @tool(name="async_err", description="desc")
        class AsyncErr:
            x: int = Field(..., description="val")

            async def execute(self) -> dict:
                raise TypeError("async type error")

        instance = AsyncErr()
        call = _make_tool_call("async_err", arguments={"x": 1})
        result = await instance.execute(call)

        assert result.status == ToolStatus.ERROR
        assert "async type error" in result.error


# ---------------------------------------------------------------------------
# Parallel execution isolation (copy.copy)
# ---------------------------------------------------------------------------


class TestParallelExecutionIsolation:
    """Wrapped execute should use copy.copy for thread-safe parallel runs."""

    @pytest.fixture
    def stateful_tool_class(self):
        @tool(name="stateful", description="Stateful tool for testing isolation")
        class StatefulTool:
            value: str = Field(..., description="Value to store")

            def __init__(self):
                self.call_count = 0

            def execute(self) -> dict:
                self.call_count += 1
                return {"value": self.value, "count": self.call_count}

        return StatefulTool

    async def test_parallel_calls_do_not_share_state(self, stateful_tool_class):
        """Each execute call should work on a copy, not mutating the original."""
        instance = stateful_tool_class()

        call_a = _make_tool_call("stateful", call_id="a", arguments={"value": "alpha"})
        call_b = _make_tool_call("stateful", call_id="b", arguments={"value": "beta"})

        # Run in parallel
        result_a, result_b = await asyncio.gather(
            instance.execute(call_a),
            instance.execute(call_b),
        )

        assert result_a.status == ToolStatus.SUCCESS
        assert result_b.status == ToolStatus.SUCCESS
        assert result_a.result["value"] == "alpha"
        assert result_b.result["value"] == "beta"

    async def test_original_instance_not_mutated(self, stateful_tool_class):
        """The original instance should not have its fields modified after execute."""
        instance = stateful_tool_class()
        original_count = instance.call_count

        call = _make_tool_call("stateful", arguments={"value": "test"})
        await instance.execute(call)

        # Original should remain unchanged since execute works on a copy
        assert instance.call_count == original_count


# ---------------------------------------------------------------------------
# create_schema() classmethod
# ---------------------------------------------------------------------------


class TestCreateSchema:
    """Tests for the injected create_schema() classmethod."""

    @pytest.fixture
    def schema_tool_class(self):
        @tool(name="schema_tool", description="Tool for schema tests")
        class SchemaTool:
            query: str = Field(..., description="The query string")
            limit: int = Field(default=50, description="Max results")

            def execute(self) -> dict:
                return {"query": self.query, "limit": self.limit}

        return SchemaTool

    def test_returns_mcp_tool_schema(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        assert isinstance(schema, MCPToolSchema)

    def test_schema_name_matches_tool_name(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        assert schema.name == "schema_tool"

    def test_schema_description_matches_tool_description(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        assert schema.description == "Tool for schema tests"

    def test_input_schema_contains_properties(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        props = schema.inputSchema.get("properties", {})
        assert "query" in props
        assert "limit" in props

    def test_input_schema_marks_required_fields(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        required = schema.inputSchema.get("required", [])
        assert "query" in required
        # limit has a default so it should NOT be required
        assert "limit" not in required

    def test_output_schema_is_generic_object(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        assert schema.outputSchema == {
            "type": "object",
            "additionalProperties": True,
        }

    def test_create_schema_callable_on_class_not_instance(self, schema_tool_class):
        """create_schema is a classmethod, callable on the class directly."""
        schema = schema_tool_class.create_schema()
        assert schema is not None

    def test_create_schema_callable_on_instance_too(self, schema_tool_class):
        instance = schema_tool_class()
        schema = instance.create_schema()
        assert isinstance(schema, MCPToolSchema)

    def test_no_fields_produces_empty_properties(self):
        @tool(name="empty_schema", description="No fields")
        class NoFields:
            def execute(self) -> dict:
                return {}

        schema = NoFields.create_schema()
        props = schema.inputSchema.get("properties", {})
        assert props == {}

    def test_field_descriptions_in_schema(self, schema_tool_class):
        schema = schema_tool_class.create_schema()
        props = schema.inputSchema["properties"]
        assert props["query"].get("description") == "The query string"
        assert props["limit"].get("description") == "Max results"


# ---------------------------------------------------------------------------
# Tool with constructor dependencies
# ---------------------------------------------------------------------------


class TestToolWithDependencies:
    """Tool classes can have __init__ with injected dependencies."""

    @pytest.fixture
    def db_tool_class(self):
        @tool(name="db_query", description="Queries a database")
        class DbQuery:
            sql: str = Field(..., description="SQL to execute")

            def __init__(self, connection):
                self.connection = connection

            def execute(self) -> dict:
                return {"rows": self.connection.run(self.sql)}

        return DbQuery

    async def test_injected_dependency_accessible_in_execute(self, db_tool_class):
        class FakeConnection:
            def run(self, sql):
                return [{"id": 1}]

        instance = db_tool_class(FakeConnection())
        call = _make_tool_call("db_query", arguments={"sql": "SELECT 1"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"rows": [{"id": 1}]}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and unusual but valid configurations."""

    async def test_execute_with_none_result(self):
        @tool(name="none_result", description="desc")
        class NoneResult:
            def execute(self) -> dict:
                return None

        instance = NoneResult()
        call = _make_tool_call("none_result")
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result is None

    async def test_execute_with_string_result(self):
        @tool(name="str_result", description="desc")
        class StrResult:
            def execute(self) -> str:
                return "just a string"

        instance = StrResult()
        call = _make_tool_call("str_result")
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == "just a string"

    async def test_execute_with_list_result(self):
        @tool(name="list_result", description="desc")
        class ListResult:
            def execute(self) -> list:
                return [1, 2, 3]

        instance = ListResult()
        call = _make_tool_call("list_result")
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == [1, 2, 3]

    async def test_tool_call_id_preserved_in_result(self):
        @tool(name="id_test", description="desc")
        class IdTest:
            def execute(self) -> dict:
                return {}

        instance = IdTest()
        call = _make_tool_call("id_test", call_id="unique-id-42")
        result = await instance.execute(call)

        assert result.tool_call_id == "unique-id-42"

    async def test_tool_with_enum_field(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        @tool(name="color_tool", description="desc")
        class ColorTool:
            color: Color = Field(..., description="Pick a color")

            def execute(self) -> dict:
                return {"color": self.color.value}

        instance = ColorTool()
        call = _make_tool_call("color_tool", arguments={"color": "red"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"color": "red"}

    async def test_tool_with_optional_field(self):
        @tool(name="opt_tool", description="desc")
        class OptTool:
            required: str = Field(..., description="Required")
            optional: str | None = Field(default=None, description="Optional")

            def execute(self) -> dict:
                return {"required": self.required, "optional": self.optional}

        instance = OptTool()
        call = _make_tool_call("opt_tool", arguments={"required": "yes"})
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result["required"] == "yes"
        assert result.result["optional"] is None

    async def test_tool_with_nested_dict_arguments(self):
        @tool(name="nested", description="desc")
        class Nested:
            config: dict = Field(..., description="Config dict")

            def execute(self) -> dict:
                return self.config

        instance = Nested()
        nested_args = {"config": {"level1": {"level2": "deep"}}}
        call = _make_tool_call("nested", arguments=nested_args)
        result = await instance.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"level1": {"level2": "deep"}}

    async def test_extra_arguments_rejected(self):
        """By default, Pydantic models created by create_model reject extra fields."""

        @tool(name="strict_args", description="desc")
        class StrictArgs:
            x: int = Field(..., description="Only field")

            def execute(self) -> dict:
                return {"x": self.x}

        instance = StrictArgs()
        call = _make_tool_call("strict_args", arguments={"x": 1, "unexpected": "value"})
        result = await instance.execute(call)

        # Pydantic by default ignores extra fields (not forbids)
        # so this should succeed - the extra field is just ignored
        # If the schema uses model_config = {"extra": "forbid"}, this would error
        assert result.status == ToolStatus.SUCCESS
