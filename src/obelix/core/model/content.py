"""Content types for multi-part messages.

Obelix internal representation of structured content that can flow
through the system: text, files (images/PDFs/audio/video), and
structured data (JSON). These map bidirectionally to A2A Part types
(TextPart, FilePart, DataPart) via the part converter in the A2A adapter.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class FileContent(BaseModel):
    """File content — inline base64 bytes or URL reference."""

    type: Literal["file"] = "file"
    data: str = Field(description="Base64-encoded bytes or URL")
    mime_type: str = Field(description="MIME type, e.g. image/png, application/pdf")
    filename: str | None = None
    is_url: bool = Field(
        default=False, description="True if data is a URL, False if base64"
    )


class DataContent(BaseModel):
    """Structured JSON data (machine-readable)."""

    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


ContentPart = TextContent | FileContent | DataContent
