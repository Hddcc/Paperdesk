"""Paper-domain models and normalization helpers."""

from __future__ import annotations

import re
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import EvidenceSourceType

SUPPORTED_SEARCH_PROVIDERS = {"all", "auto", "openalex", "arxiv"}


def normalize_search_provider(value: str | None) -> str | None:
    """Normalize the configured search provider name."""

    if value is None:
        return None

    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned not in SUPPORTED_SEARCH_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_SEARCH_PROVIDERS))
        raise ValueError(f"Unsupported search_provider: {cleaned}. Expected one of: {supported}")
    return cleaned


def normalize_doi_value(value: str | None) -> str | None:
    """Normalize DOI text for display and deduplication."""

    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^doi:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().lower()
    return cleaned or None


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
    source: str = "unknown"
    source_type: EvidenceSourceType = EvidenceSourceType.ONLINE_PAPER

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi(cls, value: str | None) -> str | None:
        return normalize_doi_value(value)


class PaperSearchRequest(BaseModel):
    """Minimal request model for the standalone online paper search route."""

    topic: str = Field(..., min_length=3)
    search_provider: str | None = None
    top_k_online: int = Field(default=5, ge=1, le=10)

    @field_validator("search_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        return normalize_search_provider(value)


class PaperSearchResponse(BaseModel):
    """Response model for normalized online paper search results."""

    items: list[PaperRecord]


class RagAskRequest(BaseModel):
    """Request payload for standalone knowledge-base Q&A."""

    question: str = Field(..., min_length=3)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=10)
    notes: str | None = None


class RagAskResponse(BaseModel):
    """Answer payload for standalone knowledge-base Q&A."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    retrieval_count: int = 0
    confidence: float | None = None
    evidence_items: list["EvidenceItem"] = Field(default_factory=list)


class PaperAnalysisRequest(BaseModel):
    """Request payload for local paper analysis/comparison."""

    document_ids: list[str] = Field(..., min_length=1)
    mode: Literal["single", "compare"] = "single"
    question: str | None = None


class PaperAnalysisSection(BaseModel):
    """Structured section in a paper analysis response."""

    title: str
    content: str


class PaperAnalysisResponse(BaseModel):
    """Structured output for local paper analysis/comparison."""

    mode: Literal["single", "compare"]
    answer: str
    sections: list[PaperAnalysisSection] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    evidence_items: list["EvidenceItem"] = Field(default_factory=list)
    retrieval_count: int = 0


class PaperCurationRequest(BaseModel):
    """Request payload for paper curation suggestions."""

    topic: str = Field(..., min_length=3)
    search_provider: str | None = None
    top_k_online: int = Field(default=5, ge=1, le=10)

    @field_validator("search_provider", mode="before")
    @classmethod
    def normalize_provider_for_curation(cls, value: str | None) -> str | None:
        return normalize_search_provider(value)


class PaperCurationItem(BaseModel):
    """Single curated paper recommendation."""

    paper: PaperRecord
    decision: Literal["recommended", "consider", "skip"]
    reason: str


class PaperCurationResponse(BaseModel):
    """Response payload for paper curation suggestions."""

    items: list[PaperCurationItem] = Field(default_factory=list)


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


RagAskResponse.model_rebuild()
PaperAnalysisResponse.model_rebuild()
