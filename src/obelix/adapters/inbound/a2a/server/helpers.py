"""Shared helpers for the A2A server package."""

from __future__ import annotations

import uuid

from a2a.types import Message, Part, Role, TextPart

DEFAULT_MAX_CONTEXTS = 1024


def agent_message(text: str) -> Message:
    """Build an A2A Message with agent role."""
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid.uuid4()),
    )
