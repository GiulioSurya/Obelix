# src/obelix/adapters/outbound/shell/client_executor.py
"""Remote shell executor — delegates command execution to the A2A client.

Used with BashTool in deferred mode: the server never executes commands,
but needs the client's shell environment info to generate the correct
system_prompt_fragment so the LLM produces compatible commands.

The client sends its shell_info via Message.metadata["client_info"]
on the first message. The A2A executor extracts it and calls
set_shell_info() before the agent runs.
"""

from __future__ import annotations

from obelix.ports.outbound.shell_executor import AbstractShellExecutor


class ClientShellExecutor(AbstractShellExecutor):
    """Executor for client-side (deferred) shell command execution.

    Does NOT execute commands — that happens on the client. This adapter
    exists to carry the client's shell environment info so that
    BashTool.system_prompt_fragment() can inject the correct context
    into the agent's system message.
    """

    def __init__(self) -> None:
        self._info: dict = {}

    def set_shell_info(self, info: dict) -> None:
        """Populate shell environment info received from the client."""
        self._info = info

    @property
    def shell_info(self) -> dict:
        """Client's shell environment details for system message injection."""
        return self._info

    @property
    def is_remote(self) -> bool:
        """Always True — commands are executed by the remote client."""
        return True

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        working_directory: str | None = None,
    ) -> dict:
        """Not implemented — the client executes commands via deferred protocol."""
        raise NotImplementedError(
            "ClientShellExecutor does not execute commands. "
            "Commands are delegated to the A2A client via the deferred tool protocol."
        )
