# src/mcp/mcp_validator.py
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError, create_model
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo


class MCPValidationError(Exception):
    """Specific exception for MCP validation errors"""

    def __init__(self, tool_name: str, validation_errors: list):
        self.tool_name = tool_name
        self.validation_errors = validation_errors
        super().__init__(f"Validation failed for tool '{tool_name}': {validation_errors}")


class MCPValidator:
    """
    Centralized validator for MCP tools using dynamic Pydantic.

    Converts MCP JSON Schema into Pydantic models for automatic
    validation and type conversion (e.g. "10" → 10, "true" → True).
    """

    def __init__(self):
        self._cached_validators: Dict[str, Type[BaseModel]] = {}

    def create_pydantic_model_from_schema(self, tool_name: str, json_schema: dict) -> Type[BaseModel]:
        """
        Dynamically generate Pydantic model from MCP JSON Schema.

        Args:
            tool_name: Name of the tool (for cache and debug)
            json_schema: JSON schema of the MCP tool

        Returns:
            Type[BaseModel]: Pydantic class for validation
        """
        if tool_name in self._cached_validators:
            return self._cached_validators[tool_name]

        # Extract properties from JSON Schema
        properties = json_schema.get('properties', {})
        required_fields = set(json_schema.get('required', []))

        # Convert each property to Pydantic field
        field_definitions = {}

        for field_name, field_schema in properties.items():
            python_type, default_value = self._json_schema_to_python_type(field_schema)

            # Determine if field is required
            if field_name in required_fields:
                field_definitions[field_name] = (python_type, ...)  # ... = required
            else:
                field_definitions[field_name] = (python_type, default_value)

        # Create Pydantic model dynamically
        model_name = f"{tool_name.title()}Args"
        pydantic_model = create_model(model_name, **field_definitions)

        # Cache for reuse
        self._cached_validators[tool_name] = pydantic_model

        return pydantic_model

    def _json_schema_to_python_type(self, field_schema: dict) -> tuple:
        """
        Convert JSON Schema definition to Python type + default.

        Args:
            field_schema: Field definition from JSON Schema

        Returns:
            tuple: (python_type, default_value)
        """
        schema_type = field_schema.get('type', 'string')
        default_value = field_schema.get('default')

        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }

        python_type = type_mapping.get(schema_type, str)

        # If no explicit default, use Python type default
        if default_value is None:
            if schema_type == 'string':
                default_value = ""
            elif schema_type in ['integer', 'number']:
                default_value = 0
            elif schema_type == 'boolean':
                default_value = False
            elif schema_type == 'array':
                default_value = []
            elif schema_type == 'object':
                default_value = {}
            else:
                default_value = None

        return python_type, default_value

    def validate_and_convert(self, tool_name: str, json_schema: dict, raw_args: dict) -> dict:
        """
        Validate and convert arguments using MCP schema.

        NEW: Pre-processes stringified arrays and objects before Pydantic validation.
        """
        try:
            # NEW: Pre-processing arguments to convert stringified JSON
            preprocessed_args = self._preprocess_json_strings(json_schema, raw_args)

            # Get or create Pydantic model
            validator_model = self.create_pydantic_model_from_schema(tool_name, json_schema)

            # Pydantic validates the pre-processed data
            validated_instance = validator_model(**preprocessed_args)

            # Return dict with correct types
            return validated_instance.model_dump()

        except ValidationError as e:
            # Convert Pydantic errors to more user-friendly format
            raise MCPValidationError(tool_name, e.errors())
        except Exception as e:
            # Handle other errors
            raise MCPValidationError(tool_name, [{"error": str(e)}])

    def _preprocess_json_strings(self, json_schema: dict, raw_args: dict) -> dict:
        """
        Pre-process arguments converting JSON strings to Python types.

        Handles arrays and objects that arrive as JSON strings from the LLM.
        """
        import json

        properties = json_schema.get('properties', {})
        processed_args = {}

        for key, value in raw_args.items():
            if key not in properties:
                processed_args[key] = value
                continue

            prop_def = properties[key]
            prop_type = prop_def.get('type', 'string')

            # If it's an array and value is a string, try to convert
            if prop_type == 'array' and isinstance(value, str):
                if value.strip() in ('', '[]', 'null'):
                    processed_args[key] = []
                else:
                    try:
                        parsed = json.loads(value)
                        processed_args[key] = parsed if isinstance(parsed, list) else [parsed]
                    except (json.JSONDecodeError, TypeError):
                        processed_args[key] = []

            # If it's an object and value is a string, try to convert
            elif prop_type == 'object' and isinstance(value, str):
                if value.strip() in ('', '{}', 'null'):
                    processed_args[key] = {}
                else:
                    try:
                        parsed = json.loads(value)
                        processed_args[key] = parsed if isinstance(parsed, dict) else {}
                    except (json.JSONDecodeError, TypeError):
                        processed_args[key] = {}

            # Other types - pass through
            else:
                processed_args[key] = value

        return processed_args

    def get_validator_for_tool(self, tool_name: str, json_schema: dict) -> Type[BaseModel]:
        """
        Get cached validator for a specific tool.

        Args:
            tool_name: Name of the tool
            json_schema: JSON schema of the tool

        Returns:
            Type[BaseModel]: Pydantic class for manual validation
        """
        return self.create_pydantic_model_from_schema(tool_name, json_schema)


# Standalone validator test
if __name__ == "__main__":
    # Schema that includes arrays (similar to Tavily)
    test_schema_with_arrays = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results",
                "default": 5
            },
            "include_domains": {
                "type": "array",
                "description": "Domains to include",
                "items": {"type": "string"},
                "default": []
            },
            "exclude_domains": {
                "type": "array",
                "description": "Domains to exclude",
                "items": {"type": "string"},
                "default": []
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata",
                "default": {}
            },
            "active": {
                "type": "boolean",
                "default": True
            }
        },
        "required": ["query"]
    }

    validator = MCPValidator()

    # Test the case that was failing (Tavily)
    print("=== Test Array Conversion Fix ===")
    tavily_like_args = {
        "query": "Finmatica",
        "max_results": "10",  # String → int
        "include_domains": '["www.finmatica.it"]',  # JSON string → list
        "exclude_domains": '[]',  # Empty JSON array → empty list
        "metadata": '{"source": "test"}',  # JSON object → dict
        "active": "true"  # String → bool
    }

    try:
        converted = validator.validate_and_convert("tavily_search", test_schema_with_arrays, tavily_like_args)
        print(f"✓ Conversion successful!")
        print(f"Input types: {[(k, type(v).__name__) for k, v in tavily_like_args.items()]}")
        print(f"Output types: {[(k, type(v).__name__) for k, v in converted.items()]}")
        print(f"include_domains value: {converted['include_domains']}")
        print(f"exclude_domains value: {converted['exclude_domains']}")
        print(f"metadata value: {converted['metadata']}")
        print()

    except MCPValidationError as e:
        print(f"✗ Validation failed: {e}")
        print()

    # Test edge cases for arrays
    print("=== Test Array Edge Cases ===")
    edge_cases = [
        {"query": "test", "include_domains": "[]"},  # Empty array
        {"query": "test", "include_domains": ""},  # Empty string
        {"query": "test", "include_domains": '["a", "b"]'},  # Multi-element
        {"query": "test", "include_domains": ["already", "list"]},  # Already list
    ]

    for i, case in enumerate(edge_cases):
        try:
            result = validator.validate_and_convert("test", test_schema_with_arrays, case)
            print(f"✓ Edge case {i + 1}: {case['include_domains']} → {result['include_domains']}")
        except Exception as e:
            print(f"✗ Edge case {i + 1} failed: {e}")