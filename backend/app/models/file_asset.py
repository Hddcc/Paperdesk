"""User-uploaded workspace file asset models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


FileKind = Literal["txt", "md", "docx", "pdf", "unsupported"]
FileSource = Literal["upload"]
FileScope = Literal["session"]
FileStatus = Literal["uploaded", "processing", "ready", "failed", "unsupported"]
FileTextExtractionStatus = Literal["pending", "ready", "failed", "skipped"]


class FileAsset(BaseModel):
    """Session-scoped user file that is separate from the paper library."""

    id: str
    filename: str
    display_name: str
    mime_type: str | None = None
    extension: str
    size_bytes: int
    sha256: str
    storage_path: str
    source: FileSource = "upload"
    scope: FileScope = "session"
    session_id: str
    kind: FileKind
    status: FileStatus = "uploaded"
    text_extract_status: FileTextExtractionStatus = "pending"
    preview_text: str | None = None
    text_char_count: int = 0
    failure_reason: str | None = None
    created_at: datetime
