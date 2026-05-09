"""Paper and evidence models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import EvidenceSourceType


class PaperRecord(BaseModel):
    """Normalized online paper metadata."""

    paper_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str = "mock"
    source_type: EvidenceSourceType = EvidenceSourceType.ONLINE_PAPER


class EvidenceItem(BaseModel):
    """Unified evidence item used by the summarizer."""

    id: str
    evidence_id: str | None = None
    source_type: EvidenceSourceType
    source_id: str
    title: str = ""
    snippet: str = ""
    quote: str = ""
    citation_label: str
    url: str | None = None
    document_id: str | None = None
    page_number: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_compatibility_fields(self) -> "EvidenceItem":
        if self.evidence_id is None:
            self.evidence_id = self.id
        if not self.snippet and self.quote:
            self.snippet = self.quote
        if not self.quote and self.snippet:
            self.quote = self.snippet
        if self.document_id is None:
            metadata_document_id = self.metadata.get("document_id")
            if isinstance(metadata_document_id, str) and metadata_document_id:
                self.document_id = metadata_document_id
        if not self.title:
            candidate = self.metadata.get("filename") or self.citation_label
            self.title = str(candidate)
        return self

