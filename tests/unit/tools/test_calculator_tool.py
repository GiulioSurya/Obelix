"""
Unit tests for CalculatorTool.

Tests verify:
- Schema generation and metadata
- All mathematical operations (add, subtract, multiply, divide, power, modulo)
- Error handling (division by zero, invalid operations, invalid parameters)
- ToolResult structure and fields
- Execution time tracking
- Parameter validation
"""

import pytest
from pydantic import ValidationError

from src.tools.tool.calculator_tool import CalculatorTool, CalculatorSchema, MathOperation
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus


class TestCalculatorToolSchema:
    """Test CalculatorTool schema generation and metadata."""

    def test_calculator_schema_generation(self):
        """Test that calculator generates correct MCP schema."""
        schema = CalculatorTool.create_schema()

        assert schema.name == "calculator"
        assert "matematiche" in schema.description.lower() or "math" in schema.description.lower()
        assert schema.inputSchema is not None

    def test_calculator_schema_has_required_fields(self):
        """Test that schema requires a, b, and operation."""
        schema = CalculatorTool.create_schema()
        input_props = schema.inputSchema["properties"]

        assert "a" in input_props
        assert "b" in input_props
        assert "operation" in input_props

    def test_calculator_schema_field_descriptions(self):
        """Test that fields have descriptions."""
        schema = CalculatorTool.create_schema()
        input_props = schema.inputSchema["properties"]

        assert "description" in input_props["a"]
        assert "description" in input_props["b"]
        assert "description" in input_props["operation"]

    def test_calculator_schema_validation_with_valid_data(self):
        """Test that CalculatorSchema validates correct data."""
        valid_data = {
            "a": 10.0,
            "b": 5.0,
            "operation": "add"
        }
        schema_instance = CalculatorSchema(**valid_data)

        assert schema_instance.a == 10.0
        assert schema_instance.b == 5.0
        assert schema_instance.operation == MathOperation.ADD

    def test_calculator_schema_validation_rejects_missing_fields(self):
        """Test that schema validation fails with missing required fields."""
        with pytest.raises(ValidationError):
            CalculatorSchema(a=10.0)  # Missing b and operation

    def test_calculator_schema_accepts_integers(self):
        """Test that schema accepts integers (converted to float)."""
        schema_instance = CalculatorSchema(a=10, b=5, operation="add")

        assert schema_instance.a == 10.0
        assert schema_instance.b == 5.0


class TestCalculatorToolAddition:
    """Test addition operations."""

    @pytest.mark.asyncio
    async def test_add_positive_numbers(self):
        """Test adding two positive numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_add_1",
            name="calculator",
            arguments={"a": 5, "b": 3, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 8
        assert result.error is None

    @pytest.mark.asyncio
    async def test_add_negative_numbers(self):
        """Test adding negative numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_add_2",
            name="calculator",
            arguments={"a": -5, "b": -3, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == -8

    @pytest.mark.asyncio
    async def test_add_with_zero(self):
        """Test adding with zero."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_add_3",
            name="calculator",
            arguments={"a": 10, "b": 0, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_add_decimals(self):
        """Test adding decimal numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_add_4",
            name="calculator",
            arguments={"a": 3.5, "b": 2.5, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 6  # Should be integer 6, not 6.0


class TestCalculatorToolSubtraction:
    """Test subtraction operations."""

    @pytest.mark.asyncio
    async def test_subtract_positive_numbers(self):
        """Test subtracting two positive numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_sub_1",
            name="calculator",
            arguments={"a": 10, "b": 4, "operation": "subtract"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 6

    @pytest.mark.asyncio
    async def test_subtract_resulting_in_negative(self):
        """Test subtraction that results in negative number."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_sub_2",
            name="calculator",
            arguments={"a": 5, "b": 10, "operation": "subtract"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == -5

    @pytest.mark.asyncio
    async def test_subtract_decimals(self):
        """Test subtracting decimal numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_sub_3",
            name="calculator",
            arguments={"a": 10.5, "b": 3.2, "operation": "subtract"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert abs(result.result - 7.3) < 0.001  # Float comparison


class TestCalculatorToolMultiplication:
    """Test multiplication operations."""

    @pytest.mark.asyncio
    async def test_multiply_positive_numbers(self):
        """Test multiplying two positive numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mul_1",
            name="calculator",
            arguments={"a": 7, "b": 6, "operation": "multiply"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 42

    @pytest.mark.asyncio
    async def test_multiply_by_zero(self):
        """Test multiplying by zero."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mul_2",
            name="calculator",
            arguments={"a": 100, "b": 0, "operation": "multiply"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 0

    @pytest.mark.asyncio
    async def test_multiply_negative_numbers(self):
        """Test multiplying negative numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mul_3",
            name="calculator",
            arguments={"a": -5, "b": -4, "operation": "multiply"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 20

    @pytest.mark.asyncio
    async def test_multiply_decimals(self):
        """Test multiplying decimal numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mul_4",
            name="calculator",
            arguments={"a": 2.5, "b": 4, "operation": "multiply"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 10  # Should be integer


class TestCalculatorToolDivision:
    """Test division operations."""

    @pytest.mark.asyncio
    async def test_divide_positive_numbers(self):
        """Test dividing two positive numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_div_1",
            name="calculator",
            arguments={"a": 20, "b": 5, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 4

    @pytest.mark.asyncio
    async def test_divide_resulting_in_decimal(self):
        """Test division that results in decimal."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_div_2",
            name="calculator",
            arguments={"a": 10, "b": 4, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 2.5

    @pytest.mark.asyncio
    async def test_divide_by_zero_returns_error(self):
        """Test that division by zero returns error status."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_div_3",
            name="calculator",
            arguments={"a": 10, "b": 0, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.result is None
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_divide_negative_numbers(self):
        """Test dividing negative numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_div_4",
            name="calculator",
            arguments={"a": -20, "b": 4, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == -5


class TestCalculatorToolPower:
    """Test power (exponentiation) operations."""

    @pytest.mark.asyncio
    async def test_power_positive_exponent(self):
        """Test raising to positive power."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_pow_1",
            name="calculator",
            arguments={"a": 2, "b": 3, "operation": "power"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 8

    @pytest.mark.asyncio
    async def test_power_zero_exponent(self):
        """Test raising to power of zero."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_pow_2",
            name="calculator",
            arguments={"a": 5, "b": 0, "operation": "power"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 1

    @pytest.mark.asyncio
    async def test_power_negative_exponent(self):
        """Test raising to negative power."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_pow_3",
            name="calculator",
            arguments={"a": 2, "b": -2, "operation": "power"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 0.25


class TestCalculatorToolModulo:
    """Test modulo operations."""

    @pytest.mark.asyncio
    async def test_modulo_positive_numbers(self):
        """Test modulo with positive numbers."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mod_1",
            name="calculator",
            arguments={"a": 10, "b": 3, "operation": "modulo"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 1

    @pytest.mark.asyncio
    async def test_modulo_exact_division(self):
        """Test modulo when numbers divide evenly."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mod_2",
            name="calculator",
            arguments={"a": 10, "b": 5, "operation": "modulo"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result == 0

    @pytest.mark.asyncio
    async def test_modulo_by_zero_returns_error(self):
        """Test that modulo by zero returns error."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_mod_3",
            name="calculator",
            arguments={"a": 10, "b": 0, "operation": "modulo"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert "zero" in result.error.lower()


class TestCalculatorToolErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_missing_required_argument(self):
        """Test error when required argument is missing."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_err_1",
            name="calculator",
            arguments={"a": 10, "operation": "add"}  # Missing 'b'
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.result is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_invalid_operation_type(self):
        """Test error with invalid operation."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_err_2",
            name="calculator",
            arguments={"a": 10, "b": 5, "operation": "invalid_op"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_non_numeric_arguments(self):
        """Test error when arguments are non-numeric."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_err_3",
            name="calculator",
            arguments={"a": "not_a_number", "b": 5, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR


class TestCalculatorToolResultStructure:
    """Test ToolResult structure and metadata."""

    @pytest.mark.asyncio
    async def test_result_has_correct_tool_name(self):
        """Test that result has correct tool_name."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_meta_1",
            name="calculator",
            arguments={"a": 1, "b": 1, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.tool_name == "calculator"

    @pytest.mark.asyncio
    async def test_result_has_correct_tool_call_id(self):
        """Test that result preserves tool_call_id."""
        tool = CalculatorTool()
        call = ToolCall(
            id="unique_call_id_123",
            name="calculator",
            arguments={"a": 1, "b": 1, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.tool_call_id == "unique_call_id_123"

    @pytest.mark.asyncio
    async def test_result_tracks_execution_time(self):
        """Test that execution time is tracked."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_time_1",
            name="calculator",
            arguments={"a": 100, "b": 50, "operation": "multiply"}
        )

        result = await tool.execute(call)

        assert result.execution_time is not None
        assert result.execution_time > 0
        assert result.execution_time < 1  # Should be very fast

    @pytest.mark.asyncio
    async def test_success_result_structure(self):
        """Test complete structure of successful result."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_struct_1",
            name="calculator",
            arguments={"a": 7, "b": 3, "operation": "add"}
        )

        result = await tool.execute(call)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "calculator"
        assert result.tool_call_id == "call_struct_1"
        assert result.result == 10
        assert result.status == ToolStatus.SUCCESS
        assert result.error is None
        assert result.execution_time is not None

    @pytest.mark.asyncio
    async def test_error_result_structure(self):
        """Test complete structure of error result."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_struct_2",
            name="calculator",
            arguments={"a": 10, "b": 0, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "calculator"
        assert result.tool_call_id == "call_struct_2"
        assert result.result is None
        assert result.status == ToolStatus.ERROR
        assert result.error is not None
        assert result.execution_time is not None


class TestCalculatorToolIntegerConversion:
    """Test that results are converted to integers when appropriate."""

    @pytest.mark.asyncio
    async def test_float_result_converted_to_int_when_whole(self):
        """Test that 6.0 becomes 6."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_int_1",
            name="calculator",
            arguments={"a": 3.5, "b": 2.5, "operation": "add"}
        )

        result = await tool.execute(call)

        assert result.result == 6
        assert isinstance(result.result, int)

    @pytest.mark.asyncio
    async def test_float_result_preserved_when_not_whole(self):
        """Test that 2.5 stays as 2.5."""
        tool = CalculatorTool()
        call = ToolCall(
            id="call_int_2",
            name="calculator",
            arguments={"a": 5, "b": 2, "operation": "divide"}
        )

        result = await tool.execute(call)

        assert result.result == 2.5
        assert isinstance(result.result, float)