"""Tracer exporters for outputting trace data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession
from obelix.infrastructure.logging import get_logger, restore_console, suppress_console

logger = get_logger(__name__)


@dataclass
class _TraceStats:
    """Accumulated stats for a single trace (used by ConsoleExporter footer)."""

    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    start_time: datetime | None = None


class TracerExporter(ABC):
    @abstractmethod
    async def export(self, trace: TraceSession) -> None: ...

    @abstractmethod
    async def export_span(self, span: Span, service_name: str) -> None: ...

    @abstractmethod
    async def start_trace(self, trace: TraceSession, service_name: str) -> None: ...

    @abstractmethod
    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...


class NoOpExporter(TracerExporter):
    async def export(self, trace: TraceSession) -> None:
        pass

    async def export_span(self, span: Span, service_name: str) -> None:
        pass

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:
        pass

    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class ConsoleExporter(TracerExporter):
    """Console exporter with 3 verbosity levels and colored output.

    Verbosity levels:
        1 (minimal):  icon + name + duration + error
        2 (standard): + model_id, token usage, tool args/result (truncated)
        3 (debug):    + reasoning, full output, full tool args/result
    """

    # ANSI color codes per span type
    _COLORS: dict[str, str] = {
        "agent": "\033[1;36m",  # cyan bold
        "llm": "\033[33m",  # yellow
        "tool": "\033[32m",  # green
        "sub_agent": "\033[35m",  # magenta
        "human": "\033[2m",  # dim/gray
        "assistant": "\033[37m",  # white
        "error": "\033[31m",  # red
        "reset": "\033[0m",
        "dim": "\033[2m",
        "bold": "\033[1m",
    }

    # Icons per span type
    _ICONS: dict[str, str] = {
        "agent": "▶",
        "llm": "◇",
        "tool": "●",
        "sub_agent": "▷",
        "human": "◆",
        "assistant": "◁",
        "memory": "○",
        "hook": "⚡",
    }

    def __init__(self, verbosity: int = 1, use_color: bool = True):
        self._verbosity = max(1, min(3, verbosity))
        self._use_color = use_color
        self._depth: dict[str, int] = {}
        self._stats: dict[str, _TraceStats] = {}
        self._parent_of: dict[str, str | None] = {}  # span_id → parent_span_id
        self._agent_name: dict[str, str] = {}  # span_id → agent name (for agent spans)

    _PREFIX = "┃ "

    # -- helpers --

    def _print(self, text: str = "") -> None:
        """Print a line with the trace prefix for visual separation from logs."""
        prefix = self._colorize(self._PREFIX, "dim") if text else self._PREFIX.rstrip()
        print(f"{prefix}{text}")

    def _colorize(self, text: str, color_key: str) -> str:
        if not self._use_color:
            return text
        code = self._COLORS.get(color_key, "")
        return f"{code}{text}{self._COLORS['reset']}" if code else text

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        """Format token count: 1234 → '1.2k', 500 → '500'."""
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def _get_icon(self, span_type: str) -> str:
        return self._ICONS.get(span_type, "○")

    def _get_indent(self, span_id: str) -> str:
        depth = self._depth.get(span_id, 0)
        return "  " * depth

    def _resolve_agent_tag(self, span_id: str) -> str:
        """Walk up the parent chain to find the owning agent name."""
        current = self._parent_of.get(span_id)
        while current:
            name = self._agent_name.get(current)
            if name:
                return f"  {self._colorize(f'[{name}]', 'agent')}"
            current = self._parent_of.get(current)
        return ""

    # -- span detail formatters --

    def _fmt_llm(self, span: Span) -> str:
        """Format LLM span detail (level 2+)."""
        parts: list[str] = []
        inp = span.input if isinstance(span.input, dict) else {}
        out = span.output if isinstance(span.output, dict) else {}

        model_id = inp.get("model_id")
        if model_id:
            parts.append(self._colorize(str(model_id), "dim"))

        usage = out.get("usage")
        if isinstance(usage, dict):
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
            parts.append(f"tokens: {self._fmt_tokens(tin)}→{self._fmt_tokens(tout)}")

        tc = out.get("tool_calls", 0)
        if tc:
            parts.append(f"tool_calls: {tc}")

        return "  ".join(parts)

    def _fmt_llm_debug(self, span: Span, indent: str) -> list[str]:
        """Format LLM debug lines (level 3). Output goes on assistant span instead."""
        lines: list[str] = []
        out = span.output if isinstance(span.output, dict) else {}

        reasoning = out.get("reasoning")
        if reasoning:
            lines.append(
                f"{indent}  {self._colorize('↳ reasoning:', 'dim')} {reasoning}"
            )

        return lines

    def _fmt_tool(self, span: Span) -> str:
        """Format tool span detail (level 2)."""
        parts: list[str] = []
        inp = span.input if isinstance(span.input, dict) else {}
        out = span.output if isinstance(span.output, dict) else {}

        args = inp.get("arguments")
        if args:
            args_str = str(args) if not isinstance(args, str) else args
            parts.append(f"args={self._truncate(args_str, 200)}")

        status = out.get("status", "")
        result = out.get("result")
        if result:
            parts.append(f"→ {status} ({self._truncate(str(result), 200)})")
        elif status:
            parts.append(f"→ {status}")

        return "  ".join(parts)

    def _fmt_tool_debug(self, span: Span, indent: str) -> list[str]:
        """Format tool debug lines (level 3)."""
        lines: list[str] = []
        inp = span.input if isinstance(span.input, dict) else {}
        out = span.output if isinstance(span.output, dict) else {}

        args = inp.get("arguments")
        if args:
            lines.append(f"{indent}  {self._colorize('↳ args:', 'dim')} {args}")

        result = out.get("result")
        if result:
            lines.append(f"{indent}  {self._colorize('↳ result:', 'dim')} {result}")

        return lines

    def _fmt_assistant(self, span: Span) -> str:
        """Format assistant span: show the LLM response content."""
        out = span.output if isinstance(span.output, dict) else {}
        content = out.get("content")
        if not content:
            return ""
        if self._verbosity >= 3:
            return str(content)
        return self._truncate(str(content), 200)

    # -- accumulate stats --

    def _update_stats(self, span: Span) -> None:
        stats = self._stats.get(span.trace_id)
        if not stats:
            return

        if span.span_type == SpanType.llm:
            stats.llm_calls += 1
            out = span.output if isinstance(span.output, dict) else {}
            usage = out.get("usage")
            if isinstance(usage, dict):
                stats.tokens_in += usage.get("input_tokens", 0)
                stats.tokens_out += usage.get("output_tokens", 0)

        elif span.span_type in (SpanType.tool, SpanType.sub_agent):
            stats.tool_calls += 1

    # -- TracerExporter interface --

    async def export(self, trace: TraceSession) -> None:
        pass

    async def export_span(self, span: Span, service_name: str) -> None:
        if span.end_time is None:
            # Start call: only track depth and hierarchy, don't print
            parent_depth = self._depth.get(span.parent_span_id or "", 0)
            self._depth[span.span_id] = parent_depth + 1
            self._parent_of[span.span_id] = span.parent_span_id
            if span.span_type == SpanType.agent:
                self._agent_name[span.span_id] = span.name
            return

        # End call: print formatted line + update stats
        self._update_stats(span)

        indent = self._get_indent(span.span_id)
        icon = self._get_icon(span.span_type)
        color_key = "error" if span.status == SpanStatus.error else span.span_type
        duration = f" {span.duration_ms:.1f}ms" if span.duration_ms else ""
        error_info = (
            f" {self._colorize('ERROR: ' + span.error, 'error')}" if span.error else ""
        )

        # Resolve owning agent name for LLM spans
        agent_tag = ""
        if span.span_type == SpanType.llm and self._verbosity >= 2:
            agent_tag = self._resolve_agent_tag(span.span_id)

        # Base line (all levels)
        name_str = self._colorize(f"{icon} {span.name}", color_key)
        line = f"{indent}{name_str}{duration}{agent_tag}{error_info}"

        # Level 2+: append detail
        detail = ""
        if self._verbosity >= 2:
            if span.span_type == SpanType.llm:
                detail = self._fmt_llm(span)
            elif span.span_type in (SpanType.tool, SpanType.sub_agent):
                detail = self._fmt_tool(span)
            elif span.span_type == SpanType.assistant:
                detail = self._fmt_assistant(span)

        if detail:
            line += f"  {detail}"

        self._print(line)

        # Level 3: debug lines on separate rows
        if self._verbosity >= 3:
            debug_lines: list[str] = []
            if span.span_type == SpanType.llm:
                debug_lines = self._fmt_llm_debug(span, indent)
            elif span.span_type in (SpanType.tool, SpanType.sub_agent):
                debug_lines = self._fmt_tool_debug(span, indent)

            for dl in debug_lines:
                self._print(dl)

        # Cleanup depth tracking
        self._depth.pop(span.span_id, None)

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:
        suppress_console()
        self._stats[trace.trace_id] = _TraceStats(start_time=trace.start_time)
        header = self._colorize(
            f"--- Trace: {trace.name} [{trace.trace_id[:8]}] "
            f"service={service_name} ---",
            "bold",
        )
        print()
        self._print(header)

    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None:
        stats = self._stats.pop(trace_id, None)

        parts: list[str] = [f"[{trace_id[:8]}]"]

        if stats and stats.start_time and end_time:
            dur_s = (end_time - stats.start_time).total_seconds()
            parts.append(f"{dur_s:.1f}s")

        status_color = "error" if status == SpanStatus.error else "bold"
        parts.append(self._colorize(f"status={status}", status_color))

        if stats and stats.llm_calls:
            total_tok = stats.tokens_in + stats.tokens_out
            parts.append(
                f"llm: {stats.llm_calls} calls, ~{self._fmt_tokens(total_tok)} tokens"
            )

        if stats and stats.tool_calls:
            parts.append(f"tools: {stats.tool_calls} calls")

        self._print(f"--- Trace {' | '.join(parts)} ---")
        print()
        restore_console()

    async def shutdown(self) -> None:
        restore_console()
        self._depth.clear()
        self._stats.clear()
        self._parent_of.clear()
        self._agent_name.clear()


class HTTPExporter(TracerExporter):
    def __init__(self, endpoint: str, timeout: float = 10.0):
        self._endpoint = endpoint
        self._base_url = (
            endpoint.rsplit("/ingest", 1)[0]
            if "/ingest" in endpoint
            else endpoint.rstrip("/")
        )
        self._timeout = timeout
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @staticmethod
    def _dt_to_epoch(dt: datetime | None) -> float | None:
        """Convert datetime to epoch float for the ingest API."""
        return dt.timestamp() if dt else None

    def _span_to_payload(self, span: Span) -> dict:
        """Convert a Span to the ingest API format."""
        return {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_span_id": span.parent_span_id,
            "span_type": span.span_type,
            "name": span.name,
            "start_time": self._dt_to_epoch(span.start_time),
            "end_time": self._dt_to_epoch(span.end_time),
            "duration_ms": span.duration_ms,
            "input": span.input,
            "output": span.output,
            "status": span.status,
            "error": span.error,
            "metadata": span.metadata,
        }

    def _to_ingest_payload(self, trace: TraceSession) -> dict:
        """Convert TraceSession to the ingest API format (epoch timestamps)."""
        return {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "service_name": trace.service_name,
            "start_time": self._dt_to_epoch(trace.start_time),
            "end_time": self._dt_to_epoch(trace.end_time),
            "status": trace.status,
            "metadata": trace.metadata,
            "spans": [self._span_to_payload(s) for s in trace.spans],
        }

    async def export(self, trace: TraceSession) -> None:
        """Batch export: send complete trace with all spans (backward compatible)."""
        try:
            client = await self._get_client()
            payload = self._to_ingest_payload(trace)
            response = await client.post(f"{self._base_url}/ingest", json=payload)
            response.raise_for_status()
            logger.debug(f"Trace {trace.trace_id[:8]} exported (batch)")
        except Exception as e:
            logger.error(f"Failed to export trace {trace.trace_id[:8]}: {e}")

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:
        """Streaming: send trace header (no spans) to create the trace record."""
        try:
            client = await self._get_client()
            payload = {
                "trace_id": trace.trace_id,
                "name": trace.name,
                "service_name": service_name,
                "start_time": self._dt_to_epoch(trace.start_time),
                "end_time": None,
                "status": trace.status,
                "metadata": trace.metadata,
                "spans": [],
            }
            response = await client.post(f"{self._base_url}/ingest/trace", json=payload)
            response.raise_for_status()
            logger.debug(f"Trace {trace.trace_id[:8]} header sent")
        except Exception as e:
            logger.error(f"Failed to send trace header {trace.trace_id[:8]}: {e}")

    async def export_span(self, span: Span, service_name: str) -> None:
        """Streaming: send a single span as it completes."""
        try:
            client = await self._get_client()
            payload = self._span_to_payload(span)
            response = await client.post(f"{self._base_url}/ingest/span", json=payload)
            response.raise_for_status()
            logger.debug(
                f"Span {span.span_id[:8]} exported ({span.span_type}:{span.name})"
            )
        except Exception as e:
            logger.error(f"Failed to export span {span.span_id[:8]}: {e}")

    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None:
        """Streaming: update trace status when execution completes."""
        try:
            client = await self._get_client()
            payload = {
                "status": status.value,
                "end_time": self._dt_to_epoch(end_time),
            }
            response = await client.patch(
                f"{self._base_url}/ingest/trace/{trace_id}", json=payload
            )
            response.raise_for_status()
            logger.debug(f"Trace {trace_id[:8]} end sent (status={status})")
        except Exception as e:
            logger.error(f"Failed to end trace {trace_id[:8]}: {e}")

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
