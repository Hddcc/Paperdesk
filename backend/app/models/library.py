"""Library and chunk models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LibraryDocument(BaseModel):
    """Uploaded local PDF document metadata."""

    id: str
    filename: str
    display_name: str | None = None
    title: str | None = None
    file_path: str
    sha256: str = ""
    page_count: int = 0
    status: str = "pending"
    parser_status: str = "pending"
    failure_reason: str | None = None
    indexed_at: datetime | None = None
    version: int = 1
    created_at: datetime
    uploaded_at: datetime | None = None

    @model_validator(mode="after")
    def sync_compatibility_fields(self) -> "LibraryDocument":
        if self.display_name is None:
            self.display_name = self.title or self.filename
        if self.title is None:
            self.title = self.display_name
        if self.uploaded_at is None:
            self.uploaded_at = self.created_at
        return self


class ChunkRecord(BaseModel):
    """Parsed library chunk metadata and content."""

    id: str
    chunk_id: str | None = None
    document_id: str
    source: str | None = None
    page_number: int = 0
    chunk_index: int
    section: str | None = None
    title: str | None = None
    sha256: str | None = None
    version: int = 1
    text: str
    content: str | None = None
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_compatibility_fields(self) -> "ChunkRecord":
        if self.chunk_id is None:
            self.chunk_id = self.id
        if self.content is None:
            self.content = self.text
        if self.source is None:
            candidate = self.metadata.get("source") or self.metadata.get("file_path")
            if isinstance(candidate, str) and candidate.strip():
                self.source = candidate.strip()
        if self.title is None:
            candidate = self.metadata.get("title")
            if isinstance(candidate, str) and candidate.strip():
                self.title = candidate.strip()
        if self.section is None:
            candidate = self.metadata.get("section")
            if isinstance(candidate, str) and candidate.strip():
                self.section = candidate.strip()
        if self.sha256 is None:
            candidate = self.metadata.get("sha256")
            if isinstance(candidate, str) and candidate.strip():
                self.sha256 = candidate.strip()
        version = self.metadata.get("version")
        if isinstance(version, int) and version > 0:
            self.version = version
        return self
