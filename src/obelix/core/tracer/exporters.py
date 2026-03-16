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
        1 (minimal):  agent headers + LLM/tool/sub_agent with duration
        2 (standard): + token usage, tool args/result (truncated), user/assistant content
        3 (debug):    + reasoning, full output, full tool args/result
    """

    # ANSI color codes
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

    _ICONS: dict[str, str] = {
        "llm": "[LLM]",
        "tool": "[TOL]",
        "sub_agent": "[SUB]",
        "human": "[USR]",
        "assistant": "[AST]",
        "memory": "[MEM]",
        "hook": "[HKS]",
    }

    _PREFIX = "| "

    def __init__(self, verbosity: int = 2, use_color: bool = True):
        self._verbosity = max(1, min(3, verbosity))
        self._use_color = use_color
        # Tracking state
        self._depth: dict[str, int] = {}
        self._stats: dict[str, _TraceStats] = {}
        self._parent_of: dict[str, str | None] = {}
        self._agent_name: dict[str, str] = {}
        self._in_subagent: dict[str, bool] = {}  # span_id -> is inside a sub-agent

    # -- helpers --

    def _print(self, text: str = "") -> None:
        prefix = self._colorize(self._PREFIX, "dim") if text else self._PREFIX.rstrip()
        print(f"{prefix}{text}")

    def _colorize(self, text: str, color_key: str) -> str:
        if not self._use_color:
            return text
        code = self._COLORS.get(color_key, "")
        return f"{code}{text}{self._COLORS['reset']}" if code else text

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = text.replace("\n", "\n| ")
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    @staticmethod
    def _fmt_duration(ms: float | None) -> str:
        if ms is None:
            return ""
        if ms >= 1000:
            return f"{ms / 1000:.1f}s"
        return f"{ms:.0f}ms"

    def _get_indent(self, span_id: str) -> str:
        """Build indent string. Uses ':' marker when inside a sub-agent."""
        depth = self._depth.get(span_id, 0)
        if depth <= 1:
            return ""
        # Check if this span is inside a sub-agent
        if self._is_in_subagent(span_id):
            # One level for the base agent, then ': ' for sub-agent nesting
            base = "  " * (depth - 2)
            return base + ":  "
        return "  " * (depth - 1)

    def _resolve_parent_agent_name(self, span_id: str) -> str | None:
        """Find the name of the parent agent for a given span."""
        current = self._parent_of.get(span_id)
        while current:
            name = self._agent_name.get(current)
            if name:
                return name
            current = self._parent_of.get(current)
        return None

    def _is_in_subagent(self, span_id: str) -> bool:
        """Check if a span is nested inside a sub-agent (not the root agent)."""
        cached = self._in_subagent.get(span_id)
        if cached is not None:
            return cached
        # Walk up the parent chain looking for a sub_agent span
        current = self._parent_of.get(span_id)
        while current:
            if current in self._agent_name and current != span_id:
                # Found an agent span that's not the root — check if it has
                # a sub_agent parent
                parent = self._parent_of.get(current)
                # If this agent's parent exists in our tracking, it's nested
                if parent and parent in self._depth:
                    self._in_subagent[span_id] = True
                    return True
            current = self._parent_of.get(current)
        self._in_subagent[span_id] = False
        return False

    # -- span formatters --

    def _fmt_llm_line(self, span: Span) -> str:
        """Format a complete LLM span line."""
        out = span.output if isinstance(span.output, dict) else {}

        parts: list[str] = []

        # Provider (short name: "llm.Providers.LITELLM" -> "litellm")
        provider = span.name.split(".")[-1].lower() if "." in span.name else span.name
        parts.append(self._colorize(f"[LLM] {provider}", "llm"))

        # Duration
        dur = self._fmt_duration(span.duration_ms)
        if dur:
            parts.append(dur)

        # Token usage (level 2+)
        if self._verbosity >= 2:
            usage = out.get("usage")
            if isinstance(usage, dict):
                tin = usage.get("input_tokens", 0)
                tout = usage.get("output_tokens", 0)
                parts.append(f"{self._fmt_tokens(tin)}->{self._fmt_tokens(tout)} tok")

            tc = out.get("tool_calls", 0)
            if tc:
                parts.append(f"{tc} calls")

        return "  ".join(parts)

    def _fmt_tool_line(self, span: Span) -> str:
        """Format a complete tool span line."""
        inp = span.input if isinstance(span.input, dict) else {}
        out = span.output if isinstance(span.output, dict) else {}

        parts: list[str] = []

        color = "error" if span.status == SpanStatus.error else "tool"
        parts.append(self._colorize(f"[TOL] {span.name}", color))

        dur = self._fmt_duration(span.duration_ms)
        if dur:
            parts.append(dur)

        # Level 2+: args and result
        if self._verbosity >= 2:
            args = inp.get("arguments")
            if args:
                args_str = str(args) if not isinstance(args, str) else args
                max_len = 500 if self._verbosity >= 3 else 120
                parts.append(self._truncate(args_str, max_len))

            status = out.get("status", "")
            result = out.get("result")
            if result:
                max_len = 500 if self._verbosity >= 3 else 120
                parts.append(f"-> {status} ({self._truncate(str(result), max_len)})")
            elif status:
                parts.append(f"-> {status}")

        if span.error:
            parts.append(self._colorize(f"ERROR: {span.error}", "error"))

        return "  ".join(parts)

    def _fmt_subagent_line(self, span: Span) -> str:
        """Format a sub-agent invocation line."""
        parts: list[str] = []

        color = "error" if span.status == SpanStatus.error else "sub_agent"
        parts.append(self._colorize(f"[SUB] {span.name}", color))

        dur = self._fmt_duration(span.duration_ms)
        if dur:
            parts.append(dur)

        if span.error:
            parts.append(self._colorize(f"ERROR: {span.error}", "error"))

        return "  ".join(parts)

    def _fmt_human_line(self, span: Span) -> str:
        """Format a human input line showing the actual content."""
        content = ""
        if isinstance(span.input, str):
            content = span.input
        elif isinstance(span.output, str):
            content = span.output

        if not content:
            return "HumanMessage: (empty)"

        max_len = 500 if self._verbosity >= 3 else 150
        preview = self._truncate(content, max_len)
        return f'HumanMessage: "{preview}"'

    def _fmt_assistant_line(self, span: Span) -> str:
        """Format an assistant response line showing the actual content."""
        out = span.output if isinstance(span.output, dict) else {}
        content = out.get("content")

        if not content:
            return ""

        max_len = 500 if self._verbosity >= 3 else 150
        preview = self._truncate(str(content), max_len)
        return self._colorize(f'[AST] "{preview}"', "assistant")

    # -- debug lines (level 3) --

    def _fmt_debug_lines(self, span: Span, indent: str) -> list[str]:
        """Format extra debug lines for level 3."""
        lines: list[str] = []
        out = span.output if isinstance(span.output, dict) else {}
        inp = span.input if isinstance(span.input, dict) else {}

        if span.span_type == SpanType.llm:
            reasoning = out.get("reasoning")
            if reasoning:
                # Replace newlines so reasoning renders multi-line with prefix
                reasoning_text = str(reasoning).replace("\n", f"\n{self._PREFIX}  ")
                lines.append(
                    f"{indent}  {self._colorize('  reasoning:', 'dim')} {reasoning_text}"
                )

        elif span.span_type == SpanType.tool:
            args = inp.get("arguments")
            if args:
                lines.append(f"{indent}  {self._colorize('  args:', 'dim')} {args}")
            result = out.get("result")
            if result:
                lines.append(f"{indent}  {self._colorize('  result:', 'dim')} {result}")

        elif span.span_type == SpanType.sub_agent:
            args = inp.get("arguments")
            if args:
                lines.append(f"{indent}  {self._colorize('  args:', 'dim')} {args}")
            result = out.get("result")
            if result:
                lines.append(f"{indent}  {self._colorize('  result:', 'dim')} {result}")

        return lines

    # -- stats --

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
            # Start: track hierarchy, print agent headers
            parent_depth = self._depth.get(span.parent_span_id or "", 0)
            self._depth[span.span_id] = parent_depth + 1
            self._parent_of[span.span_id] = span.parent_span_id

            if span.span_type == SpanType.agent:
                self._agent_name[span.span_id] = span.name
                # Defer root agent header — printed after HumanMessage
                if parent_depth == 0:
                    self._pending_agent_header = span.name
            return

        # End: format and print
        self._update_stats(span)

        indent = self._get_indent(span.span_id)

        # Skip agent end spans (they're just structural containers)
        if span.span_type == SpanType.agent:
            self._depth.pop(span.span_id, None)
            return

        # Human spans: only show for the root agent (depth <= 2).
        # Sub-agent human spans are redundant — the query is in the [SUB] args.
        if span.span_type == SpanType.human:
            depth = self._depth.get(span.span_id, 0)
            if depth <= 2:
                line = self._fmt_human_line(span)
                self._print(line)
                # Print deferred agent header after the human message
                header = getattr(self, "_pending_agent_header", None)
                if header:
                    self._print()
                    self._print(self._colorize(f"{header}:", "agent"))
                    self._pending_agent_header = None
            self._depth.pop(span.span_id, None)
            self._in_subagent.pop(span.span_id, None)
            return

        # If we're resuming the parent agent after a SUB, print a header
        resume = getattr(self, "_pending_resume_agent", None)
        if resume and span.span_type in (
            SpanType.llm,
            SpanType.tool,
            SpanType.assistant,
        ):
            self._print(self._colorize(f"{resume} (contd.):", "agent"))
            self._pending_resume_agent = None

        # Format line based on type
        line = ""
        if span.span_type == SpanType.llm:
            line = self._fmt_llm_line(span)
        elif span.span_type == SpanType.tool:
            line = self._fmt_tool_line(span)
        elif span.span_type == SpanType.sub_agent:
            line = self._fmt_subagent_line(span)
            # After printing SUB, re-print parent agent header to show we're back
            self._pending_resume_agent = self._resolve_parent_agent_name(span.span_id)
        elif span.span_type == SpanType.assistant:
            # Skip assistant spans inside sub-agents — result is in [SUB] debug lines
            if self._is_in_subagent(span.span_id):
                self._depth.pop(span.span_id, None)
                self._in_subagent.pop(span.span_id, None)
                return
            line = self._fmt_assistant_line(span)
            if not line:
                self._depth.pop(span.span_id, None)
                return
        else:
            icon = self._ICONS.get(span.span_type, "[???]")
            line = f"{icon} {span.name}"

        self._print(f"{indent}{line}")

        # Level 3: debug lines
        if self._verbosity >= 3:
            for dl in self._fmt_debug_lines(span, indent):
                self._print(dl)

        # Cleanup
        self._depth.pop(span.span_id, None)
        self._in_subagent.pop(span.span_id, None)

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
                f"llm: {stats.llm_calls} calls ~{self._fmt_tokens(total_tok)} tok"
            )

        if stats and stats.tool_calls:
            parts.append(f"tools: {stats.tool_calls} calls")

        self._print()
        self._print(f"--- Trace {' | '.join(parts)} ---")
        print()
        restore_console()

    async def shutdown(self) -> None:
        restore_console()
        self._depth.clear()
        self._stats.clear()
        self._parent_of.clear()
        self._agent_name.clear()
        self._in_subagent.clear()


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
