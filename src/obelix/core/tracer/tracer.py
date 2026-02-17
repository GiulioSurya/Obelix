"""Main Tracer class for instrumentation."""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from obelix.core.tracer.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from obelix.core.tracer.exporters import TracerExporter
from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession
from obelix.infrastructure.logging import get_logger


logger = get_logger(__name__)


def _serialize(value: Any) -> Any:
    """Serialize a value for storage in a span."""
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict | list):
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return str(value)
    return str(value)


class Tracer:
    def __init__(
        self,
        exporter: TracerExporter,
        service_name: str = "obelix",
    ):
        self._exporter = exporter
        self.service_name = service_name

    async def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSession:
        trace = TraceSession(
            name=name,
            service_name=self.service_name,
            metadata={"service": self.service_name, **(metadata or {})},
        )
        set_current_trace(trace)
        set_current_span(None)
        logger.debug(f"Trace started: {name} [{trace.trace_id[:8]}]")

        # Streaming: send trace header immediately
        await self._exporter.start_trace(trace, self.service_name)

        return trace

    async def end_trace(
        self,
        status: SpanStatus = SpanStatus.ok,
        error: str | None = None,
    ) -> None:
        trace = get_current_trace()
        if trace is None:
            return
        trace.end_time = datetime.now(UTC)
        trace.status = status
        if error:
            trace.metadata["error"] = error

        # Streaming: send trace end update
        await self._exporter.end_trace(trace.trace_id, status, trace.end_time)

        set_current_trace(None)
        set_current_span(None)
        logger.debug(
            f"Trace ended: {trace.name} [{trace.trace_id[:8]}] "
            f"status={status} spans={len(trace.spans)}"
        )

    def start_span(
        self,
        span_type: SpanType,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        trace = get_current_trace()
        if trace is None:
            raise RuntimeError("Cannot start span without an active trace")

        parent = get_current_span()
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            span_type=span_type,
            name=name,
            input=_serialize(input),
            metadata=metadata or {},
        )
        trace.spans.append(span)
        set_current_span(span)
        return span

    async def end_span(
        self,
        output: Any = None,
        status: SpanStatus = SpanStatus.ok,
        error: str | None = None,
    ) -> None:
        span = get_current_span()
        if span is None:
            return
        span.end_time = datetime.now(UTC)
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.output = _serialize(output)
        span.status = status
        span.error = error

        # Streaming: export span immediately
        await self._exporter.export_span(span, self.service_name)

        # Restore parent span as current
        trace = get_current_trace()
        if trace and span.parent_span_id:
            parent = next(
                (s for s in trace.spans if s.span_id == span.parent_span_id),
                None,
            )
            set_current_span(parent)
        else:
            set_current_span(None)

    @asynccontextmanager
    async def trace_context(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TraceSession]:
        trace = await self.start_trace(name, metadata)
        try:
            yield trace
        except Exception as e:
            await self.end_trace(status=SpanStatus.error, error=str(e))
            raise
        else:
            await self.end_trace()

    @asynccontextmanager
    async def span_context(
        self,
        span_type: SpanType,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Span]:
        span = self.start_span(span_type, name, input, metadata)
        try:
            yield span
        except Exception as e:
            await self.end_span(status=SpanStatus.error, error=str(e))
            raise
        else:
            await self.end_span()
