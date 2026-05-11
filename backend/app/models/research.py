"""Research-domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import (
    ResearchRunStatus,
    TodoTaskStatus,
    coerce_research_run_status,
    coerce_todo_task_status,
)
from .paper import normalize_search_provider
from .report import ResearchReport, TaskSummary


class ResearchRequest(BaseModel):
    """Incoming request for starting a research run."""

    topic: str = Field(..., min_length=3, description="Research topic entered by the user")
    top_k_online: int = Field(default=3, ge=1, le=10)
    top_k_local: int = Field(default=3, ge=1, le=10)
    search_provider: str | None = None
    notes: str | None = None

    @field_validator("search_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        return normalize_search_provider(value)


class TodoTask(BaseModel):
    """Single TODO item produced by the planner."""

    id: str
    title: str
    intent: str
    query: str
    status: TodoTaskStatus = TodoTaskStatus.PENDING
    summary: str | None = None
    summary_markdown: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: TodoTaskStatus | str) -> TodoTaskStatus:
        return coerce_todo_task_status(value)

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

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: ResearchRunStatus | str) -> ResearchRunStatus:
        return coerce_research_run_status(value)


class ResearchState(BaseModel):
    """In-memory execution state for a single fixed research workflow run."""

    run_id: str
    topic: str
    status: ResearchRunStatus
    todo_tasks: list[TodoTask] = Field(default_factory=list)
    task_summaries: list[TaskSummary] = Field(default_factory=list)
    report: ResearchReport | None = None


class ResearchRunDetail(BaseModel):
    """Fallback payload for reloading a previously created research run."""

    run: ResearchRun
    tasks: list[TodoTask] = Field(default_factory=list)
    task_summaries: list[TaskSummary] = Field(default_factory=list)
    report: ResearchReport | None = None
