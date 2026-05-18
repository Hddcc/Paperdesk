"""Report-domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .paper import EvidenceItem, PaperRecord


ReportLifecycleStatus = Literal["chat_answer", "report_draft", "saved_report", "exported_markdown"]
ReportSource = Literal["knowledge_answer", "research_task", "manual_save"]


class CitationRecord(BaseModel):
    """Structured citation entry persisted with a report."""

    citation_label: str
    source_type: str
    title: str
    url: str | None = None
    doi: str | None = None
    document_id: str | None = None
    page_number: int | None = None


class TaskSummary(BaseModel):
    """Task-level merged summary."""

    task_id: str
    title: str
    intent: str
    summary: str
    summary_markdown: str | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    paper_records: list[PaperRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_summary_fields(self) -> "TaskSummary":
        if self.summary_markdown is None:
            self.summary_markdown = self.summary
        return self


class ResearchReport(BaseModel):
    """Final report payload."""

    id: str
    report_id: str | None = None
    topic: str
    markdown: str
    lifecycle_status: ReportLifecycleStatus = "saved_report"
    source: ReportSource = "research_task"
    source_message_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    category_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    task_summaries: list[TaskSummary] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    citation_items: list[CitationRecord] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def sync_report_id(self) -> "ResearchReport":
        if self.report_id is None:
            self.report_id = self.id
        if self.updated_at is None:
            self.updated_at = self.created_at
        return self


class ReportListItem(BaseModel):
    """Compact report metadata for the report list view."""

    id: str
    topic: str
    created_at: datetime
