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
    """Chunk placeholder for future vector indexing."""

    id: str
    chunk_id: str | None = None
    document_id: str
    page_number: int = 0
    chunk_index: int
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
        return self

