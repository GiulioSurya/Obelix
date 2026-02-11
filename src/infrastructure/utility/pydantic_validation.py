# src/utility/pydantic_validation.py
"""
Shared Pydantic validation utilities.

Provides consistent error formatting for ValidationError across:
- Tool argument validation (tool_decorator.py)
- ToolCall creation (tool_extr_fall_back.py)
- Structured output validation (future)
"""
from pydantic import ValidationError


def get_validation_action(err: dict, field_name: str) -> str:
    """
    Get actionable message for a Pydantic validation error.

    Args:
        err: Single error dict from ValidationError.errors()
        field_name: Name of the field that failed validation

    Returns:
        Human-readable action to fix the error
    """
    err_type = err["type"]

    if err_type == "missing":
        return f"Add '{field_name}' - it is required."
    elif err_type == "dict_type":
        return f"Provide a valid dictionary/object for '{field_name}'."
    elif err_type.endswith("_type"):
        expected = err_type.replace("_type", "")
        return f"Provide a {expected} value for '{field_name}'."
    elif err_type == "extra_forbidden":
        return f"Remove '{field_name}' - it is not a valid parameter."
    elif err_type in ("too_short", "too_long", "string_too_short", "string_too_long"):
        return f"Adjust length of '{field_name}'."
    elif err_type in ("greater_than", "less_than", "greater_than_equal", "less_than_equal"):
        return f"Adjust value of '{field_name}'."
    elif err_type in ("enum", "literal_error"):
        return f"Use an allowed value for '{field_name}'."
    elif err_type == "json_invalid":
        return f"Provide valid JSON for '{field_name}'."
    else:
        return f"Fix '{field_name}'."


def format_validation_error(e: ValidationError, context: str) -> str:
    """
    Format a Pydantic ValidationError into a readable, actionable message.

    Args:
        e: The ValidationError exception
        context: Description of what was being validated (e.g., "tool 'calculator'")

    Returns:
        Formatted error message with details and suggested actions
    """
    error_details = []

    for err in e.errors():
        # Build readable path: e.g., questions[0].question
        path_parts = []
        for item in err["loc"]:
            if isinstance(item, int):
                path_parts.append(f"[{item}]")
            else:
                if path_parts:
                    path_parts.append(f".{item}")
                else:
                    path_parts.append(str(item))
        loc = "".join(path_parts) or "root"

        msg = err["msg"]
        field_name = err["loc"][-1] if err["loc"] else "unknown"
        action = get_validation_action(err, field_name)

        detail = f"  - '{loc}': {msg}\n    Action: {action}"
        error_details.append(detail)

    return f"VALIDATION ERROR for {context}:\n\n" + "\n\n".join(error_details)