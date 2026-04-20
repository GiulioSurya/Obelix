"""Tests for the OpenShellDeployer port-forward watchdog.

See design spec:
docs/superpowers/specs/2026-04-13-port-forward-watchdog-design.md

All tests mock `asyncio.open_connection` and the forward start/stop methods.
No real gateway, no real socket, no real sandbox.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obelix.adapters.outbound.openshell import deployer as deployer_module
from obelix.adapters.outbound.openshell.deployer import OpenShellDeployer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_watchdog(monkeypatch):
    """Speed up the watchdog intervals so tests complete quickly."""
    monkeypatch.setattr(deployer_module, "_WATCHDOG_INTERVAL", 0.01)
    monkeypatch.setattr(deployer_module, "_RESTART_BACKOFF", 0.01)
    monkeypatch.setattr(deployer_module, "_PROBE_TIMEOUT", 0.05)


def _make_deployer() -> OpenShellDeployer:
    """Build a deployer instance sufficient for watchdog testing.

    We bypass the full constructor validation: the watchdog only needs
    ``_port`` and the private stop/start methods that we will patch.
    """
    inst = OpenShellDeployer.__new__(OpenShellDeployer)
    inst._port = 8002
    inst._sandbox_name = "test-sandbox"
    inst._forward_proc = None
    inst._watchdog_task = None
    inst._destroyed = False
    inst._client = None
    return inst


def _fake_alive_connection():
    """Build a mock (reader, writer) pair simulating a healthy connection."""
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return reader, writer


# ---------------------------------------------------------------------------
# _is_forward_alive()
# ---------------------------------------------------------------------------


class TestIsForwardAlive:
    @pytest.mark.asyncio
    async def test_returns_true_on_successful_connection(self):
        d = _make_deployer()
        reader, writer = _fake_alive_connection()
        with patch(
            "asyncio.open_connection",
            AsyncMock(return_value=(reader, writer)),
        ):
            assert await d._is_forward_alive() is True
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_refused(self):
        d = _make_deployer()
        with patch(
            "asyncio.open_connection",
            AsyncMock(side_effect=ConnectionRefusedError("nope")),
        ):
            assert await d._is_forward_alive() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        d = _make_deployer()

        async def hang(*_a, **_kw):
            await asyncio.sleep(10)

        with patch("asyncio.open_connection", side_effect=hang):
            assert await d._is_forward_alive() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_oserror(self):
        d = _make_deployer()
        with patch(
            "asyncio.open_connection",
            AsyncMock(side_effect=OSError("network unreachable")),
        ):
            assert await d._is_forward_alive() is False

    @pytest.mark.asyncio
    async def test_closes_writer_even_on_wait_closed_error(self):
        """If wait_closed() raises, _is_forward_alive still returns True."""
        d = _make_deployer()
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock(side_effect=RuntimeError("late close"))
        with patch(
            "asyncio.open_connection",
            AsyncMock(return_value=(reader, writer)),
        ):
            assert await d._is_forward_alive() is True
        writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# _watchdog_loop() via _start_watchdog() / _stop_watchdog()
# ---------------------------------------------------------------------------


async def _let_loop_run(iterations: int = 5) -> None:
    """Yield control to the event loop long enough for a few watchdog cycles."""
    # Each cycle is ~0.01s; sleep proportionally but with comfortable margin.
    await asyncio.sleep(deployer_module._WATCHDOG_INTERVAL * iterations + 0.05)


class TestWatchdogLoop:
    @pytest.mark.asyncio
    async def test_no_restart_when_forward_alive(self):
        d = _make_deployer()
        d._is_forward_alive = AsyncMock(return_value=True)
        d._stop_forward = AsyncMock()
        d._start_forward = AsyncMock()

        d._start_watchdog()
        await _let_loop_run(iterations=5)
        await d._stop_watchdog()

        d._stop_forward.assert_not_awaited()
        d._start_forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restarts_on_dead_forward(self):
        d = _make_deployer()
        # First probe: dead. Subsequent probes: alive (so we don't loop-restart).
        d._is_forward_alive = AsyncMock(side_effect=[False, True, True, True, True])
        parent = MagicMock()
        d._stop_forward = AsyncMock(side_effect=lambda: parent.stop())
        d._start_forward = AsyncMock(side_effect=lambda: parent.start())

        d._start_watchdog()
        await _let_loop_run(iterations=6)
        await d._stop_watchdog()

        d._stop_forward.assert_awaited_once()
        d._start_forward.assert_awaited_once()
        # stop must be called BEFORE start
        assert parent.method_calls == [
            ("stop", (), {}),
            ("start", (), {}),
        ]

    @pytest.mark.asyncio
    async def test_restart_failure_then_recovery(self):
        """A failed restart does not kill the loop; next cycle retries."""
        d = _make_deployer()

        # Stateful probe: dead for the first two checks, alive thereafter.
        probe_calls = {"n": 0}

        async def probe():
            probe_calls["n"] += 1
            return probe_calls["n"] > 2

        d._is_forward_alive = probe
        d._stop_forward = AsyncMock()
        # First _start_forward raises; subsequent calls succeed.
        start_calls = {"n": 0}

        async def start():
            start_calls["n"] += 1
            if start_calls["n"] == 1:
                raise RuntimeError("sandbox gone")

        d._start_forward = start

        d._start_watchdog()
        await _let_loop_run(iterations=10)

        # Task must still be running (not crashed by the first failure).
        assert d._watchdog_task is not None
        assert not d._watchdog_task.done()

        await d._stop_watchdog()

        # Stop must have been called at least once for each restart attempt.
        assert d._stop_forward.await_count >= 2
        # Start must have been called at least twice (fail, then succeed).
        assert start_calls["n"] >= 2

    @pytest.mark.asyncio
    async def test_shutdown_graceful_on_cancel(self):
        d = _make_deployer()
        d._is_forward_alive = AsyncMock(return_value=True)
        d._stop_forward = AsyncMock()
        d._start_forward = AsyncMock()

        d._start_watchdog()
        task = d._watchdog_task
        await asyncio.sleep(0.02)
        await d._stop_watchdog()

        assert task is not None
        assert task.done()
        assert task.cancelled()
        # _stop_watchdog() cleared the attribute
        assert d._watchdog_task is None

    @pytest.mark.asyncio
    async def test_stop_watchdog_is_idempotent(self):
        d = _make_deployer()
        d._is_forward_alive = AsyncMock(return_value=True)
        d._stop_forward = AsyncMock()
        d._start_forward = AsyncMock()

        d._start_watchdog()
        await d._stop_watchdog()
        # Second stop must not raise.
        await d._stop_watchdog()

    @pytest.mark.asyncio
    async def test_start_watchdog_is_idempotent(self):
        """Calling _start_watchdog twice should not spawn two tasks."""
        d = _make_deployer()
        d._is_forward_alive = AsyncMock(return_value=True)
        d._stop_forward = AsyncMock()
        d._start_forward = AsyncMock()

        d._start_watchdog()
        first_task = d._watchdog_task
        d._start_watchdog()
        second_task = d._watchdog_task

        assert first_task is second_task

        await d._stop_watchdog()


# ---------------------------------------------------------------------------
# destroy() ordering
# ---------------------------------------------------------------------------


class TestDestroyOrdering:
    @pytest.mark.asyncio
    async def test_destroy_cancels_watchdog_before_stopping_forward(self):
        """destroy() must stop the watchdog BEFORE stopping the forward,
        so the watchdog does not observe the forward going down and try
        to restart it during shutdown."""
        d = _make_deployer()

        parent = MagicMock()
        d._stop_watchdog = AsyncMock(side_effect=lambda: parent.stop_watchdog())
        d._stop_forward = AsyncMock(side_effect=lambda: parent.stop_forward())

        await d.destroy()

        calls = [c[0] for c in parent.method_calls]
        assert calls.index("stop_watchdog") < calls.index("stop_forward")

    @pytest.mark.asyncio
    async def test_destroy_is_idempotent(self):
        d = _make_deployer()
        d._stop_watchdog = AsyncMock()
        d._stop_forward = AsyncMock()

        await d.destroy()
        await d.destroy()  # second call must not raise

        d._stop_watchdog.assert_awaited_once()
        d._stop_forward.assert_awaited_once()
