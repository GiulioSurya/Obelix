# src/infrastructure/utility/pydantic_validation.py
"""
Shared Pydantic validation utilities.

Provides consistent error formatting for ValidationError across:
- Tool argument validation (tool_decorator.py)
- ToolCall creation (provider-level extraction)
- Structured output validation (future)
"""

import difflib
import re

from pydantic import ValidationError


def _extract_invalid_value(err: dict) -> str | None:
    """Extract the invalid value from a validation error's input context."""
    ctx = err.get("ctx", {})
    input_val = err.get("input") or ctx.get("input")
    if input_val is not None:
        return str(input_val)
    return None


def _extract_allowed_values(err: dict) -> list[str]:
    """Extract allowed values from a literal_error or enum error."""
    ctx = err.get("ctx", {})
    expected = ctx.get("expected")
    if expected and isinstance(expected, str):
        return re.findall(r"'([^']+)'", expected)
    return []


def _suggest_similar(invalid: str, allowed: list[str], max_suggestions: int = 5) -> list[str]:
    """Find allowed values most similar to the invalid one."""
    return difflib.get_close_matches(invalid, allowed, n=max_suggestions, cutoff=0.4)


def _resolve_field_name(loc: tuple) -> str:
    """Resolve a human-readable field name from a Pydantic error location.

    For array paths like ('VISTA_BILANCIO_SPESA_AI', 33), returns the
    parent field name instead of the numeric index.
    """
    if not loc:
        return "unknown"
    last = loc[-1]
    if isinstance(last, int) and len(loc) >= 2:
        return str(loc[-2])
    return str(last)


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
    elif err_type in (
        "greater_than",
        "less_than",
        "greater_than_equal",
        "less_than_equal",
    ):
        return f"Adjust value of '{field_name}'."
    elif err_type in ("enum", "literal_error"):
        invalid = _extract_invalid_value(err)
        allowed = _extract_allowed_values(err)
        if invalid and allowed:
            suggestions = _suggest_similar(invalid, allowed)
            if suggestions:
                return (
                    f"'{invalid}' is not valid for '{field_name}'. "
                    f"Did you mean: {', '.join(suggestions)}?"
                )
            return f"'{invalid}' is not valid for '{field_name}'. Check exact spelling."
        return f"Use an allowed value for '{field_name}'."
    elif err_type == "json_invalid":
        return f"Provide valid JSON for '{field_name}'."
    else:
        return f"Fix '{field_name}'."


def _format_literal_message(err: dict) -> str:
    """Format a concise message for literal/enum errors without dumping all values."""
    invalid = _extract_invalid_value(err)
    if invalid:
        return f"Invalid value: '{invalid}'"
    return err["msg"]


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

        field_name = _resolve_field_name(err["loc"])
        action = get_validation_action(err, field_name)

        if err["type"] in ("enum", "literal_error"):
            msg = _format_literal_message(err)
        else:
            msg = err["msg"]

        detail = f"  - '{loc}': {msg}\n    Action: {action}"
        error_details.append(detail)

    return f"VALIDATION ERROR for {context}:\n\n" + "\n\n".join(error_details)
