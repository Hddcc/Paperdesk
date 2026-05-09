"""Research-domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from .enums import ResearchRunStatus


class ResearchRequest(BaseModel):
    """Incoming request for starting a research run."""

    topic: str = Field(..., min_length=3, description="Research topic entered by the user")
    top_k_online: int = Field(default=3, ge=1, le=10)
    top_k_local: int = Field(default=3, ge=1, le=10)
    search_provider: str | None = None
    notes: str | None = None


class TodoTask(BaseModel):
    """Single TODO item produced by the planner."""

    id: str
    title: str
    intent: str
    query: str
    status: ResearchRunStatus = ResearchRunStatus.CREATED
    summary: str | None = None
    summary_markdown: str | None = None

    @model_validator(mode="after")
    def sync_summary_fields(self) -> "TodoTask":
        if self.summary_markdown is None and self.summary is not None:
            self.summary_markdown = self.summary
        if self.summary is None and self.summary_markdown is not None:
            self.summary = self.summary_markdown
        return self


class ResearchRun(BaseModel):
    """Persisted research run metadata."""

    id: str
    topic: str
    status: ResearchRunStatus
    created_at: datetime
    updated_at: datetime

