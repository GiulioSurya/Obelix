"""Tests for obelix.infrastructure.logging -- logging configuration and utilities."""

import pytest

import obelix.infrastructure.logging as logging_mod
from obelix.infrastructure.logging import (
    format_message_for_trace,
    get_logger,
    restore_console,
    setup_logging,
    suppress_console,
)

# ---------------------------------------------------------------------------
# Helpers to reset module-level state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level _is_configured flag before each test.

    This prevents setup_logging() from short-circuiting in subsequent tests
    and avoids side-effects leaking between tests.
    """
    original_configured = logging_mod._is_configured
    original_handler_id = logging_mod._console_handler_id
    original_format = logging_mod._console_format
    original_level = logging_mod._console_level

    logging_mod._is_configured = False
    logging_mod._console_handler_id = None
    logging_mod._console_format = None
    logging_mod._console_level = "INFO"

    yield

    # Restore originals so we don't break anything
    logging_mod._is_configured = original_configured
    logging_mod._console_handler_id = original_handler_id
    logging_mod._console_format = original_format
    logging_mod._console_level = original_level


# ---------------------------------------------------------------------------
# TestGetLogger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger()."""

    def test_returns_bound_logger(self):
        log = get_logger("my_module")
        # Loguru's bind() returns a logger that carries extra context
        assert log is not None

    def test_bound_name_is_set(self):
        """The bound logger should carry the name in its extra dict."""
        log = get_logger("test.module.name")
        # Loguru's bind returns a Logger with _core extra; verify by logging
        # through a sink that captures the extra.
        captured = []

        from loguru import logger

        handler_id = logger.add(
            lambda msg: captured.append(msg),
            format="{extra[name]} | {message}",
            level="DEBUG",
        )
        try:
            log.info("hello")
            assert len(captured) == 1
            assert "test.module.name" in str(captured[0])
        finally:
            logger.remove(handler_id)

    def test_different_names_produce_different_loggers(self):
        log1 = get_logger("module_a")
        log2 = get_logger("module_b")
        # They are both loguru Logger instances but with different bindings.
        # We can't compare them directly; just ensure both are usable.
        assert log1 is not None
        assert log2 is not None


# ---------------------------------------------------------------------------
# TestSetupLogging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_creates_log_directory(self, tmp_path):
        log_dir = str(tmp_path / "new_logs")
        setup_logging(log_dir=log_dir)
        assert (tmp_path / "new_logs").is_dir()

    def test_creates_nested_log_directory(self, tmp_path):
        log_dir = str(tmp_path / "a" / "b" / "c")
        setup_logging(log_dir=log_dir)
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_idempotent_second_call_ignored(self, tmp_path):
        """Calling setup_logging() twice should not reconfigure."""
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir)
        assert logging_mod._is_configured is True

        # Second call should be a no-op
        setup_logging(log_dir=str(tmp_path / "other_logs"))
        # The second directory should NOT have been created
        assert not (tmp_path / "other_logs").exists()

    def test_sets_configured_flag(self, tmp_path):
        assert logging_mod._is_configured is False
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert logging_mod._is_configured is True

    def test_stores_console_handler_id(self, tmp_path):
        assert logging_mod._console_handler_id is None
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert logging_mod._console_handler_id is not None

    def test_stores_console_format(self, tmp_path):
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert logging_mod._console_format is not None
        assert "extra[name]" in logging_mod._console_format

    def test_stores_console_level(self, tmp_path):
        setup_logging(log_dir=str(tmp_path / "logs"), console_level="WARNING")
        assert logging_mod._console_level == "WARNING"

    def test_default_console_level(self, tmp_path):
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert logging_mod._console_level == "DEBUG"

    def test_custom_log_filename(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir, log_filename="custom.log")
        # After first log message, file should exist
        log = get_logger("test")
        log.info("test message")
        # The file should be created (loguru creates on first write)
        assert (tmp_path / "logs" / "custom.log").exists()


# ---------------------------------------------------------------------------
# TestSuppressAndRestoreConsole
# ---------------------------------------------------------------------------


class TestSuppressAndRestoreConsole:
    """Tests for suppress_console() and restore_console()."""

    def test_suppress_when_no_handler_is_noop(self):
        """suppress_console() should not raise when _console_handler_id is None."""
        logging_mod._console_handler_id = None
        suppress_console()  # Should not raise

    def test_restore_when_no_handler_is_noop(self):
        """restore_console() should not raise when _console_handler_id is None."""
        logging_mod._console_handler_id = None
        restore_console()  # Should not raise

    def test_suppress_changes_handler(self, tmp_path):
        setup_logging(log_dir=str(tmp_path / "logs"), console_level="DEBUG")
        original_id = logging_mod._console_handler_id
        suppress_console()
        # Handler ID should change because old was removed and new was added
        assert logging_mod._console_handler_id != original_id

    def test_restore_changes_handler(self, tmp_path):
        setup_logging(log_dir=str(tmp_path / "logs"), console_level="DEBUG")
        suppress_console()
        suppressed_id = logging_mod._console_handler_id
        restore_console()
        assert logging_mod._console_handler_id != suppressed_id


# ---------------------------------------------------------------------------
# TestFormatMessageForTrace -- AssistantMessage
# ---------------------------------------------------------------------------


class TestFormatAssistantMessage:
    """Tests for format_message_for_trace() with AssistantMessage."""

    def test_assistant_with_content_only(self):
        from obelix.core.model.assistant_message import AssistantMessage

        msg = AssistantMessage(content="Hello world")
        result = format_message_for_trace(msg)
        assert result.startswith("AssistantMessage:")
        assert 'content="Hello world"' in result

    def test_assistant_empty_content(self):
        from obelix.core.model.assistant_message import AssistantMessage

        msg = AssistantMessage(content="")
        result = format_message_for_trace(msg)
        assert 'content=""' in result

    def test_assistant_with_tool_calls(self):
        from obelix.core.model.assistant_message import AssistantMessage
        from obelix.core.model.tool_message import ToolCall

        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="tc1", name="calculator", arguments={"a": 2, "b": 3}),
            ],
        )
        result = format_message_for_trace(msg)
        assert "tool_calls=" in result
        assert "calculator(" in result
        assert "a=2" in result
        assert "b=3" in result

    def test_assistant_with_multiple_tool_calls(self):
        from obelix.core.model.assistant_message import AssistantMessage
        from obelix.core.model.tool_message import ToolCall

        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="tc1", name="tool_a", arguments={"x": 1}),
                ToolCall(id="tc2", name="tool_b", arguments={"y": "hello"}),
            ],
        )
        result = format_message_for_trace(msg)
        assert "tool_a(" in result
        assert "tool_b(" in result

    def test_assistant_content_and_tool_calls(self):
        from obelix.core.model.assistant_message import AssistantMessage
        from obelix.core.model.tool_message import ToolCall

        msg = AssistantMessage(
            content="Let me calculate",
            tool_calls=[
                ToolCall(id="tc1", name="calc", arguments={"n": 5}),
            ],
        )
        result = format_message_for_trace(msg)
        assert 'content="Let me calculate"' in result
        assert "tool_calls=" in result
        assert " | " in result  # Parts joined by pipe

    def test_assistant_no_tool_calls_no_tool_section(self):
        from obelix.core.model.assistant_message import AssistantMessage

        msg = AssistantMessage(content="Just text")
        result = format_message_for_trace(msg)
        assert "tool_calls" not in result


# ---------------------------------------------------------------------------
# TestFormatMessageForTrace -- ToolMessage
# ---------------------------------------------------------------------------


class TestFormatToolMessage:
    """Tests for format_message_for_trace() with ToolMessage."""

    def test_tool_message_success(self):
        from obelix.core.model.tool_message import (
            ToolMessage,
            ToolResult,
            ToolStatus,
        )

        msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="calculator",
                    tool_call_id="tc1",
                    result={"sum": 5},
                    status=ToolStatus.SUCCESS,
                ),
            ]
        )
        result = format_message_for_trace(msg)
        assert result.startswith("ToolMessage:")
        assert "calculator" in result
        assert "success" in result
        assert "result=" in result

    def test_tool_message_error(self):
        from obelix.core.model.tool_message import (
            ToolMessage,
            ToolResult,
            ToolStatus,
        )

        msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="fetcher",
                    tool_call_id="tc2",
                    result=None,
                    status=ToolStatus.ERROR,
                    error="Connection timeout",
                ),
            ]
        )
        result = format_message_for_trace(msg)
        assert "fetcher" in result
        assert "error" in result.lower()
        assert "Connection timeout" in result

    def test_tool_message_multiple_results(self):
        from obelix.core.model.tool_message import (
            ToolMessage,
            ToolResult,
            ToolStatus,
        )

        msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="tool_a",
                    tool_call_id="tc1",
                    result="ok",
                    status=ToolStatus.SUCCESS,
                ),
                ToolResult(
                    tool_name="tool_b",
                    tool_call_id="tc2",
                    result=None,
                    status=ToolStatus.ERROR,
                    error="fail",
                ),
            ]
        )
        result = format_message_for_trace(msg)
        assert "tool_a" in result
        assert "tool_b" in result
        # Results separated by semicolons
        assert ";" in result

    def test_tool_message_status_enum_value_used(self):
        """ToolStatus is a StrEnum; format should use its .value."""
        from obelix.core.model.tool_message import (
            ToolMessage,
            ToolResult,
            ToolStatus,
        )

        msg = ToolMessage(
            tool_results=[
                ToolResult(
                    tool_name="t",
                    tool_call_id="tc",
                    result=42,
                    status=ToolStatus.TIMEOUT,
                ),
            ]
        )
        result = format_message_for_trace(msg)
        assert "timeout" in result


# ---------------------------------------------------------------------------
# TestFormatMessageForTrace -- SystemMessage, HumanMessage, others
# ---------------------------------------------------------------------------


class TestFormatOtherMessages:
    """Tests for format_message_for_trace() with SystemMessage, HumanMessage, etc."""

    def test_system_message(self):
        from obelix.core.model.system_message import SystemMessage

        msg = SystemMessage(content="You are a helpful assistant.")
        result = format_message_for_trace(msg)
        assert result == "SystemMessage: You are a helpful assistant."

    def test_human_message(self):
        from obelix.core.model.human_message import HumanMessage

        msg = HumanMessage(content="What is 2+2?")
        result = format_message_for_trace(msg)
        assert result == "HumanMessage: What is 2+2?"

    def test_unknown_type_with_content(self):
        """An arbitrary object with a content attribute should be handled."""

        class CustomMessage:
            content = "custom content"

        msg = CustomMessage()
        result = format_message_for_trace(msg)
        assert "CustomMessage" in result
        assert "custom content" in result

    def test_unknown_type_without_content(self):
        """An object without a content attribute should produce empty string."""

        class NoContentMessage:
            pass

        msg = NoContentMessage()
        result = format_message_for_trace(msg)
        assert "NoContentMessage" in result


# ---------------------------------------------------------------------------
# TestFormatMessageForTrace -- Truncation
# ---------------------------------------------------------------------------


class TestFormatTruncation:
    """Tests for format_message_for_trace() truncation behavior."""

    def test_short_message_not_truncated(self):
        from obelix.core.model.human_message import HumanMessage

        msg = HumanMessage(content="short")
        result = format_message_for_trace(msg)
        assert "[TRUNCATED]" not in result

    def test_long_message_truncated_at_default(self):
        from obelix.core.model.human_message import HumanMessage

        long_content = "x" * 3000
        msg = HumanMessage(content=long_content)
        result = format_message_for_trace(msg)
        assert "[TRUNCATED]" in result
        assert len(result) <= 2500

    def test_custom_max_chars(self):
        from obelix.core.model.human_message import HumanMessage

        msg = HumanMessage(content="a" * 200)
        result = format_message_for_trace(msg, max_chars=50)
        assert "[TRUNCATED]" in result
        assert len(result) <= 50

    def test_exact_boundary_not_truncated(self):
        """A message exactly at max_chars should not be truncated."""
        from obelix.core.model.human_message import HumanMessage

        # "HumanMessage: " prefix is 15 chars
        prefix_len = len("HumanMessage: ")
        content_len = 100 - prefix_len
        msg = HumanMessage(content="a" * content_len)
        result = format_message_for_trace(msg, max_chars=100)
        assert "[TRUNCATED]" not in result
        assert len(result) == 100

    def test_one_over_boundary_is_truncated(self):
        from obelix.core.model.human_message import HumanMessage

        prefix_len = len("HumanMessage: ")
        content_len = 101 - prefix_len
        msg = HumanMessage(content="a" * content_len)
        result = format_message_for_trace(msg, max_chars=100)
        assert "[TRUNCATED]" in result
        assert len(result) <= 100

    def test_truncated_assistant_with_tool_calls(self):
        """Even a complex AssistantMessage should respect max_chars."""
        from obelix.core.model.assistant_message import AssistantMessage
        from obelix.core.model.tool_message import ToolCall

        msg = AssistantMessage(
            content="x" * 3000,
            tool_calls=[
                ToolCall(id="tc1", name="tool", arguments={"key": "v" * 1000}),
            ],
        )
        result = format_message_for_trace(msg, max_chars=500)
        assert "[TRUNCATED]" in result
        assert len(result) <= 500
