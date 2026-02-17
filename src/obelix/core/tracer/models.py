"""Tracer data models."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SpanType(StrEnum):
    agent = "agent"
    llm = "llm"
    tool = "tool"
    sub_agent = "sub_agent"
    memory = "memory"
    hook = "hook"


class SpanStatus(StrEnum):
    ok = "ok"
    error = "error"
    timeout = "timeout"


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    parent_span_id: str | None = None
    span_type: SpanType
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = None
    input: Any | None = None
    output: Any | None = None
    status: SpanStatus = SpanStatus.ok
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceSession(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    service_name: str = "obelix"
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.ok
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
