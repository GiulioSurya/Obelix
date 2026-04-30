"""Internal base classes / mixins for A2A outbound tools.

Not exported from the package — these are implementation details for
reducing duplication across the LLM-facing tool implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry


class _ContextAware:
    """Mixin for A2A tools that need the per-request ContextEntry injected
    before execute() is called.

    Subclasses must call ``self._require_context(tool_name)`` at the top
    of execute() to fail-fast if the executor's _inject_context_entry
    helper wasn't run before invocation.
    """

    _ctx_entry: ContextEntry | None = None

    def set_context_entry(self, entry: ContextEntry) -> None:
        self._ctx_entry = entry

    def _require_context(self, tool_name: str) -> ContextEntry:
        if self._ctx_entry is None:
            raise RuntimeError(f"{tool_name}: context entry not injected")
        return self._ctx_entry
