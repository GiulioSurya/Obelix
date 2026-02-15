# src/logging_config.py
"""
Logging configuration module for SophIA.

Uses Loguru as backend. This module provides two main functions:
- setup_logging(): configures the logger at application startup
- get_logger(name): gets a logger "bound" with the module name

=============================================================================
LOGURU - BASIC CONCEPTS
=============================================================================

Loguru has ONE single global logger called `logger`. You don't create multiple
instances like with standard logging. Instead, you use `bind()` to add context.

Conceptual example:
    from loguru import logger

    # This is ALWAYS the same global logger
    logger.info("message")

    # bind() adds "extra" to context, but it's still the same logger
    logger_module = logger.bind(name="my_module")
    logger_module.info("message")  # Now includes name="my_module"

=============================================================================
LOG LEVELS (from least to most severe)
=============================================================================

1. TRACE (5)    - Extreme details, for deep debugging
                  Example: "Entering function X with param Y"

2. DEBUG (10)   - Useful info during development
                  Example: "Generated SQL query: SELECT..."

3. INFO (20)    - Normal application events
                  Example: "SQLGenerator Agent initialized"

4. SUCCESS (25) - Operations completed successfully (Loguru specific)
                  Example: "Query executed, 150 rows returned"

5. WARNING (30) - Anomalous but handled situations
                  Example: "LLM timeout, retrying..."

6. ERROR (40)   - Errors that prevent an operation
                  Example: "Unable to connect to DB"

7. CRITICAL (50)- Fatal errors, application cannot continue
                  Example: "Configuration file missing"

=============================================================================
WHEN TO USE EACH LEVEL
=============================================================================

DEBUG:
    - Variable values during execution
    - SQL queries before execution
    - API request/response payloads
    - Internal object state

    logger.debug(f"Tool call received: {tool_call.name}")
    logger.debug(f"Parameters: {tool_call.arguments}")

INFO:
    - Component startup/shutdown
    - Completed business operations
    - Significant state changes

    logger.info(f"Agent {self.agent_name} initialized")
    logger.info("Execution pipeline started")

WARNING:
    - Automatic retries
    - Missing configuration with default used
    - Rate limiting applied
    - Deprecation notices

    logger.warning(f"Attempt {attempt}/3 failed, retrying...")
    logger.warning("Parameter X not specified, using default Y")

ERROR:
    - Caught and handled exceptions
    - Failed operations (but app continues)
    - Unavailable resources

    logger.error(f"Tool {tool_name} not found: {e}")
    logger.error("DB connection failed", exc_info=True)

CRITICAL:
    - Errors requiring shutdown
    - Unrecoverable state corruption
    - Security violations

    logger.critical("Unable to initialize LLM provider")

=============================================================================
HANDLER CONFIGURATION (add)
=============================================================================

Loguru starts with a default handler (stderr). To customize:

    logger.remove()  # Remove default handler

    # Add custom handler
    logger.add(
        sink,           # Where to write: file path, sys.stderr, custom function
        level,          # Minimum level: "DEBUG", "INFO", etc.
        format,         # Message format
        rotation,       # When to rotate file: "50 MB", "1 day", "12:00"
        retention,      # How long to keep old files: "7 days", "1 week"
        colorize,       # True/False for colors (console only)
        serialize,      # True for JSON output
        filter,         # Function to filter messages
    )

The add() method returns an ID you can use to remove the handler:

    handler_id = logger.add("file.log")
    logger.remove(handler_id)  # Remove only this handler

=============================================================================
"""

from loguru import logger
from pathlib import Path
import sys


# Flag to prevent multiple setups
_is_configured = False


def setup_logging(
    level: str = "INFO",
    console_level: str = "DEBUG",
    log_dir: str = "logs",
    log_filename: str = "sophia.log"
) -> None:
    """
    Configure logging for the application.

    Call this function ONCE at app startup (e.g. in main.py).
    Subsequent calls are ignored.

    Args:
        level: Minimum level for FILE. Default: "DEBUG" (captures everything)
        console_level: Minimum level for CONSOLE. Default: "INFO"
                       Values: "TRACE", "DEBUG", "INFO", "SUCCESS",
                       "WARNING", "ERROR", "CRITICAL"
        log_dir: Directory for log files. Created if it doesn't exist.
                 Default: "logs"
        log_filename: Name of log file. Default: "sophia.log"

    Example:
        # In main.py
        from obelix.infrastructure.logging import setup_logging

        setup_logging()  # Default: file=DEBUG, console=INFO

        # Only errors on console, everything to file
        setup_logging(console_level="WARNING")

        # Debug also on console
        setup_logging(console_level="DEBUG")

    Behavior:
        - Removes Loguru's default handler
        - Adds FILE handler (level=DEBUG, with 50 MB rotation)
        - Adds CONSOLE handler (level=INFO, with colors)
    """
    global _is_configured

    if _is_configured:
        return

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler (stderr)
    logger.remove()

    # Log format: timestamp | level | module:function:line | message
    # {name} comes from bind() that we do in get_logger()
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{extra[name]}:{function}:{line} | "
        "{message}"
    )

    # File handler with rotation
    logger.add(
        sink=log_path / log_filename,  # Path to file
        level=level,                    # Minimum level (DEBUG)
        format=log_format,              # Format defined above
        rotation="50 MB",               # Rotate when file exceeds 50 MB
        retention="7 days",             # Keep files for 7 days
        encoding="utf-8",               # File encoding
    )

    # Console format: more compact, no full timestamp
    console_format = (
        "<level>{level:<8}</level> | "
        "<cyan>{extra[name]}</cyan>:<cyan>{function}</cyan> | "
        "{message}"
    )

    # Console handler with colors
    logger.add(
        sink=sys.stderr,                # Output: console (stderr)
        level=console_level,            # Minimum level (INFO by default)
        format=console_format,          # Compact format
        colorize=True,                  # Colors active
    )

    _is_configured = True

    # Initial log to confirm setup
    logger.bind(name="logging_config").info(
        f"Logging configured - file={level}, console={console_level}, path={log_path / log_filename}"
    )


def get_logger(name: str):
    """
    Get a logger with the module name bound.

    The name is included in every log message, allowing identification of
    which module the log came from.

    Args:
        name: Module name. Always use __name__ for consistency.

    Returns:
        Loguru logger with bound name.

    Example:
        # At the beginning of the module
        from obelix.infrastructure.logging import get_logger

        logger = get_logger(__name__)

        # Then in code
        logger.debug("This is a debug message")
        logger.info("Operation completed")
        logger.warning("Warning: unexpected value")
        logger.error("Error during execution")

    Note:
        - __name__ returns module path (e.g. "src.domain.agent.base_agent")
        - Returned logger is always the same global Loguru logger,
          but with additional context (the name)
        - If setup_logging() wasn't called, behavior is Loguru's default (stderr)
    """
    return logger.bind(name=name)


def format_message_for_trace(message, max_chars: int = 2500) -> str:
    """
    Format a StandardMessage for TRACE logging with full details.

    Shows complete message content including tool_calls for AssistantMessage
    and tool_results for ToolMessage, with configurable truncation.

    Args:
        message: A StandardMessage instance (HumanMessage, SystemMessage,
                 AssistantMessage, or ToolMessage)
        max_chars: Maximum characters before truncation. Default: 2500

    Returns:
        Formatted string representation of the message

    Example output:
        SystemMessage: "You are an expert..."
        HumanMessage: "Calculate 2+2"
        AssistantMessage: content="" | tool_calls=[calculator(a=2, b=2, op="add")]
        ToolMessage: [calculator: SUCCESS result={'result': 4}]
    """
    msg_type = type(message).__name__

    # Import here to avoid circular imports
    from obelix.domain.model.assistant_message import AssistantMessage
    from obelix.domain.model.tool_message import ToolMessage

    if isinstance(message, AssistantMessage):
        parts = []
        content = message.content or ""
        if content:
            parts.append(f'content="{content}"')
        else:
            parts.append('content=""')

        if message.tool_calls:
            tool_strs = []
            for tc in message.tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
                tool_strs.append(f"{tc.name}({args_str})")
            parts.append(f"tool_calls=[{', '.join(tool_strs)}]")

        formatted = f"{msg_type}: {' | '.join(parts)}"

    elif isinstance(message, ToolMessage):
        results_strs = []
        for tr in message.tool_results:
            status = tr.status.value if hasattr(tr.status, 'value') else str(tr.status)
            result_repr = repr(tr.result)
            if tr.error:
                results_strs.append(f"{tr.tool_name}: {status} error={tr.error!r}")
            else:
                results_strs.append(f"{tr.tool_name}: {status} result={result_repr}")
        formatted = f"{msg_type}: [{'; '.join(results_strs)}]"

    else:
        # SystemMessage, HumanMessage, or other
        content = message.content if hasattr(message, 'content') else ""
        formatted = f"{msg_type}: {content}"

    # Truncate if too long
    if len(formatted) > max_chars:
        truncate_marker = " ... [TRUNCATED]"
        formatted = formatted[:max_chars - len(truncate_marker)] + truncate_marker

    return formatted


# =============================================================================
# QUICK USAGE EXAMPLES
# =============================================================================
#
# --- In main.py (only once) ---
#
# from obelix.infrastructure.logging import setup_logging
# setup_logging()
#
# --- In any other module ---
#
# from obelix.infrastructure.logging import get_logger
# logger = get_logger(__name__)
#
# def my_function():
#     logger.debug("Function start")
#     try:
#         # ... code ...
#         logger.info("Operation completed")
#     except Exception as e:
#         logger.error(f"Error: {e}")
#         raise
#
# --- Logging with variables (f-string or format) ---
#
# logger.debug(f"Value x={x}, y={y}")
# logger.info("User {} logged in", username)  # Lazy formatting
#
# --- Logging with exception (full traceback) ---
#
# try:
#     risky_operation()
# except Exception:
#     logger.exception("Operation failed")  # Include traceback
#
# --- Logging with structured data ---
#
# logger.info("Request received", extra={"method": "POST", "path": "/api"})
#
