"""Pydantic data models shared across the PaperDesk skeleton."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .enums import EvidenceSourceType, ResearchRunStatus


class ResearchRequest(BaseModel):
    """Incoming request for starting a research run."""

    topic: str = Field(..., min_length=3, description="Research topic entered by the user")
    top_k_online: int = Field(default=3, ge=1, le=10)
    top_k_local: int = Field(default=3, ge=1, le=10)
    notes: str | None = None


class TodoTask(BaseModel):
    """Single TODO item produced by the planner."""

    id: str
    title: str
    intent: str
    query: str
    status: ResearchRunStatus = ResearchRunStatus.CREATED
    summary: str | None = None


class PaperRecord(BaseModel):
    """Normalized online paper metadata."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str
    url: str | None = None
    doi: str | None = None
    source_type: EvidenceSourceType = EvidenceSourceType.ONLINE_PAPER


class LibraryDocument(BaseModel):
    """Uploaded local PDF document metadata."""

    id: str
    filename: str
    display_name: str
    file_path: str
    status: str = "uploaded"
    uploaded_at: datetime


class ChunkRecord(BaseModel):
    """Chunk placeholder for future vector indexing."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Unified evidence item used by the summarizer."""

    id: str
    source_type: EvidenceSourceType
    source_id: str
    quote: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSummary(BaseModel):
    """Task-level merged summary."""

    task_id: str
    title: str
    intent: str
    summary: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    paper_records: list[PaperRecord] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final report payload."""

    id: str
    topic: str
    markdown: str
    task_summaries: list[TaskSummary] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    created_at: datetime


class ResearchRun(BaseModel):
    """Persisted research run metadata."""

    id: str
    topic: str
    status: ResearchRunStatus
    created_at: datetime
    updated_at: datetime


class ReportListItem(BaseModel):
    """Compact report metadata for the report list view."""

    id: str
    topic: str
    created_at: datetime
