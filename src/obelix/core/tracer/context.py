"""Async-safe context management for tracing using contextvars."""

from contextvars import ContextVar

from obelix.core.tracer.models import Span, TraceSession

_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)
_current_trace: ContextVar[TraceSession | None] = ContextVar(
    "_current_trace", default=None
)


def get_current_span() -> Span | None:
    return _current_span.get()


def get_current_trace() -> TraceSession | None:
    return _current_trace.get()


def set_current_span(span: Span | None) -> None:
    _current_span.set(span)


def set_current_trace(trace: TraceSession | None) -> None:
    _current_trace.set(trace)
