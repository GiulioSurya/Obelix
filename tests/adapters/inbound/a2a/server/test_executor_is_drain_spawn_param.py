"""Verify executor accepts is_drain_spawn keyword argument.

Parameter-acceptance test only; behavior switch (tracer reuse, webhook patch)
covered by tasks 4, 7. Here we verify the kwarg flows through
_run_agent → _run_agent_impl without raising TypeError.
"""

from __future__ import annotations

import inspect

from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


def test_run_agent_signature_has_is_drain_spawn():
    sig = inspect.signature(ObelixAgentExecutor._run_agent)
    assert "is_drain_spawn" in sig.parameters
    p = sig.parameters["is_drain_spawn"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is False


def test_run_agent_impl_signature_has_is_drain_spawn():
    sig = inspect.signature(ObelixAgentExecutor._run_agent_impl)
    assert "is_drain_spawn" in sig.parameters
    p = sig.parameters["is_drain_spawn"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is False
