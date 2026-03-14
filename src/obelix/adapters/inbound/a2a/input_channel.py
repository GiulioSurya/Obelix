"""Async channel for tool <-> executor communication during A2A input-required flow.

The InputChannel bridges two async contexts that don't know each other:

- **Tool side** (inside BaseAgent._execute_loop):
  calls `await channel.request_input(question)` which suspends
  until the executor delivers the client's response.

- **Executor side** (ObelixAgentExecutor.execute):
  calls `await channel.wait_for_request()` to detect when a tool
  needs input, then later `channel.provide_input(text)` to deliver
  the client's response and resume the tool.

Thread safety: all operations are asyncio-safe (single event loop).
The channel is accessed via a ContextVar so the tool doesn't need
a direct reference to the executor.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

INPUT_TIMEOUT_SECONDS = 60.0

# ContextVar allows the tool to find the channel without coupling to the executor
input_channel_var: ContextVar[InputChannel | None] = ContextVar(
    "a2a_input_channel", default=None
)


class InputChannel:
    """Async channel for a single input-required exchange.

    Lifecycle:
    1. Executor creates InputChannel, sets it in ContextVar
    2. Tool calls request_input(question) -> suspends on Future
    3. Executor detects via wait_for_request() -> emits input-required
    4. Next A2A request calls provide_input(text) -> Future resolves
    5. Tool resumes with the user's answer
    """

    __slots__ = ("_request_event", "_question", "_response_future")

    def __init__(self) -> None:
        self._request_event = asyncio.Event()
        self._question: str | None = None
        self._response_future: asyncio.Future[str] | None = None

    # ── Tool side ─────────────────────────────────────────────────────────

    async def request_input(self, question: str) -> str:
        """Suspend the tool until the client provides input.

        Args:
            question: The question to present to the client.

        Returns:
            The client's response text.

        Raises:
            TimeoutError: If no response within INPUT_TIMEOUT_SECONDS.
        """
        loop = asyncio.get_running_loop()
        self._question = question
        self._response_future = loop.create_future()
        self._request_event.set()  # signal the executor
        return await asyncio.wait_for(
            self._response_future, timeout=INPUT_TIMEOUT_SECONDS
        )

    # ── Executor side ─────────────────────────────────────────────────────

    async def wait_for_request(self) -> str:
        """Block until the tool requests input.

        Returns:
            The question the tool wants to ask.
        """
        await self._request_event.wait()
        return self._question

    def provide_input(self, text: str) -> None:
        """Deliver the client's response, resuming the suspended tool.

        Safe to call even if the Future is already done (e.g. timeout).
        """
        if self._response_future is not None and not self._response_future.done():
            self._response_future.set_result(text)

    def is_waiting(self) -> bool:
        """Check if a tool is currently suspended waiting for input."""
        return self._response_future is not None and not self._response_future.done()

    @property
    def question(self) -> str | None:
        return self._question
