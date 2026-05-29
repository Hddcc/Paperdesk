"""Agent/system generated workspace file models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


WorkspaceFileCreator = Literal["agent", "user", "system"]
WorkspaceFileStatus = Literal["ready", "failed"]
WorkspaceFileListSource = Literal["generated", "filesystem", "unknown"]


class WorkspaceFile(BaseModel):
    """Session-scoped generated file stored under the local workspace root."""

    id: str
    session_id: str
    source_message_id: str | None = None
    created_by: WorkspaceFileCreator = "agent"
    file_kind: str
    display_name: str
    relative_path: str
    storage_path: str = Field(exclude=True)
    mime_type: str | None = None
    size_bytes: int
    checksum: str
    status: WorkspaceFileStatus = "ready"
    source_file_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None = None


class WorkspaceFileListItem(BaseModel):
    """Safe read-only workspace file listing item."""

    id: str | None = None
    display_name: str
    relative_path: str
    file_kind: str
    mime_type: str | None = None
    size_bytes: int
    modified_at: datetime | None = None
    created_at: datetime | None = None
    source: WorkspaceFileListSource = "filesystem"
    is_directory: bool = False
    status: str = "ready"
    readable: bool = False
    reason: str | None = None


class WorkspaceFileListResponse(BaseModel):
    """Safe read-only workspace file listing response."""

    session_id: str
    path: str = ""
    recursive: bool = False
    files: list[WorkspaceFileListItem] = Field(default_factory=list)


class WorkspaceFileReadResult(BaseModel):
    """Safe read-only text content preview for one workspace file."""

    relative_path: str
    display_name: str
    mime_type: str | None = None
    size_bytes: int
    content: str = ""
    char_count: int = 0
    included_chars: int = 0
    truncated: bool = False
    checksum: str | None = None
    status: str = "ready"
    reason: str | None = None
