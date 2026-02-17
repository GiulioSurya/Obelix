"""Tracer exporters for outputting trace data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from obelix.core.tracer.models import Span, SpanStatus, TraceSession
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


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
    def __init__(self, use_color: bool = True):
        self._use_color = use_color

    async def export(self, trace: TraceSession) -> None:
        header = (
            f"Trace: {trace.name} [{trace.trace_id[:8]}] "
            f"status={trace.status} "
            f"spans={len(trace.spans)}"
        )
        if trace.end_time and trace.start_time:
            duration = (trace.end_time - trace.start_time).total_seconds() * 1000
            header += f" duration={duration:.1f}ms"

        print(f"\n{'=' * 60}")
        print(header)
        print(f"{'=' * 60}")

        # Build parent->children map for tree rendering
        children: dict[str | None, list] = {}
        for span in trace.spans:
            children.setdefault(span.parent_span_id, []).append(span)

        # Print tree starting from root spans
        self._print_tree(children, parent_id=None, indent=0)
        print(f"{'=' * 60}\n")

    async def export_span(self, span: Span, service_name: str) -> None:
        status_marker = "x" if span.status == SpanStatus.error else "v"
        duration = f" {span.duration_ms:.1f}ms" if span.duration_ms else ""
        print(f"  [{status_marker}] {span.span_type}:{span.name}{duration}")

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:
        print(f"\n--- Trace started: {trace.name} [{trace.trace_id[:8]}] service={service_name} ---")

    async def end_trace(
        self, trace_id: str, status: SpanStatus, end_time: datetime | None
    ) -> None:
        print(f"--- Trace ended: [{trace_id[:8]}] status={status} ---\n")

    def _print_tree(
        self,
        children: dict[str | None, list],
        parent_id: str | None,
        indent: int,
    ) -> None:
        for span in children.get(parent_id, []):
            prefix = "  " * indent
            status_marker = "x" if span.status == SpanStatus.error else "v"
            duration = f" {span.duration_ms:.1f}ms" if span.duration_ms else ""
            error_info = f" ERROR: {span.error}" if span.error else ""
            print(
                f"{prefix}[{status_marker}] {span.span_type}:{span.name}"
                f"{duration}{error_info}"
            )
            self._print_tree(children, span.span_id, indent + 1)

    async def shutdown(self) -> None:
        pass


class HTTPExporter(TracerExporter):
    def __init__(self, endpoint: str, timeout: float = 10.0):
        self._endpoint = endpoint
        self._base_url = endpoint.rsplit("/ingest", 1)[0] if "/ingest" in endpoint else endpoint.rstrip("/")
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
            logger.debug(f"Span {span.span_id[:8]} exported ({span.span_type}:{span.name})")
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
