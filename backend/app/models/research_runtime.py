"""Research runtime state models for the phase-13 main-agent loop."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from .enums import TodoTaskStatus
from .paper import EvidenceItem, PaperRecord
from .report import TaskSummary
from .runtime import TaskArtifactRef


class ResearchActionType(str, Enum):
    """Main-agent actions supported by the phase-13 loop."""

    PLAN = "plan"
    SEARCH_ONLINE = "search_online"
    SEARCH_LOCAL = "search_local"
    SUMMARIZE_EVIDENCE = "summarize_evidence"
    REVISE_PLAN = "revise_plan"
    FINALIZE_REPORT = "finalize_report"
    FINISH = "finish"
    FAIL = "fail"


class ResearchStepStatus(str, Enum):
    """Lifecycle status for a single main-agent step."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchToolResultStatus(str, Enum):
    """Normalized tool result status."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchRuntimePhase(str, Enum):
    """High-level runtime phase exposed in checkpoints and detail payloads."""

    PLANNING = "planning"
    EXECUTING = "executing"
    SUMMARIZING = "summarizing"
    WRITING_REPORT = "writing_report"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchContextStage(str, Enum):
    """Context assembly stages for the research main-agent prompt."""

    NORMAL = "normal"
    EVIDENCE_COMPACTED = "evidence_compacted"
    HISTORY_COMPACTED = "history_compacted"
    TRUNCATED = "truncated"


class ResearchCompactedEvidenceItem(BaseModel):
    """Decision-oriented evidence view produced from the raw evidence buffer."""

    task_id: str
    source_key: str
    source_type: str
    citation: str
    title: str
    page_number: int | None = None
    excerpt: str = ""
    relevance: str = "unknown"
    coverage: list[str] = Field(default_factory=list)
    potential_conflict: bool = False
    visible: bool = True


class ResearchEvidenceAssessment(BaseModel):
    """Lightweight quality signals used by the main-agent decision loop."""

    total_item_count: int = 0
    paper_count: int = 0
    local_evidence_count: int = 0
    relevant_item_count: int = 0
    visible_item_count: int = 0
    compacted_item_count: int = 0
    sufficiency_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    coverage: list[str] = Field(default_factory=list)
    conflict_detected: bool = False
    has_relevant_evidence: bool = False
    rationale: str = "尚未形成证据质量判断。"


class ResearchContextState(BaseModel):
    """Observable state for research prompt assembly and context control."""

    stage: ResearchContextStage = ResearchContextStage.NORMAL
    estimated_tokens: int = 0
    budget_tokens: int = 0
    sources: list[str] = Field(default_factory=list)
    last_compacted_at: datetime | None = None
    active_task_id: str | None = None
    visible_step_count: int = 0
    evidence_items_compacted: int = 0
    history_compacted: bool = False


class ResearchRuntimeStep(BaseModel):
    """Currently active main-agent step."""

    step_id: str
    action: ResearchActionType
    task_id: str | None = None
    attempt: int = 1
    status: ResearchStepStatus = ResearchStepStatus.RUNNING
    started_at: datetime


class ResearchToolCallRecord(BaseModel):
    """History entry for a completed or failed tool call."""

    step_id: str
    action: ResearchActionType
    task_id: str | None = None
    status: ResearchToolResultStatus
    summary: str
    retryable: bool = False
    error: str | None = None
    paper_count: int = 0
    evidence_count: int = 0
    artifact_refs: list[TaskArtifactRef] = Field(default_factory=list)
    created_at: datetime


class ResearchEvidenceBufferItem(BaseModel):
    """Accumulated evidence for a single plan item."""

    task_id: str
    paper_records: list[PaperRecord] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    compacted_evidence: list[ResearchCompactedEvidenceItem] = Field(default_factory=list)
    evidence_assessment: ResearchEvidenceAssessment = Field(default_factory=ResearchEvidenceAssessment)
    online_completed: bool = False
    local_completed: bool = False
    degraded: bool = False


class ResearchPlanItem(BaseModel):
    """Plan item tracked by the runtime loop."""

    task_id: str
    title: str
    intent: str
    query: str
    status: TodoTaskStatus = TodoTaskStatus.PENDING
    revise_count: int = 0
    query_history: list[str] = Field(default_factory=list)
    summary: str | None = None
    summary_markdown: str | None = None
    degraded: bool = False

    def to_task_summary(self, evidence: ResearchEvidenceBufferItem) -> TaskSummary:
        return TaskSummary(
            task_id=self.task_id,
            title=self.title,
            intent=self.intent,
            summary=self.summary or "",
            summary_markdown=self.summary_markdown or self.summary or "",
            paper_records=evidence.paper_records,
            evidence_items=evidence.evidence_items,
        )


class ResearchToolResult(BaseModel):
    """Unified tool output consumed by the main-agent runtime."""

    status: ResearchToolResultStatus
    summary: str
    payload: dict = Field(default_factory=dict)
    artifacts: list[TaskArtifactRef] = Field(default_factory=list)
    retryable: bool = False
    error: str | None = None


class ResearchRuntimeState(BaseModel):
    """Checkpointable state for a single research run."""

    run_id: str
    goal: str
    current_phase: ResearchRuntimePhase
    plan_items: list[ResearchPlanItem] = Field(default_factory=list)
    completed_items: list[str] = Field(default_factory=list)
    active_step: ResearchRuntimeStep | None = None
    tool_history: list[ResearchToolCallRecord] = Field(default_factory=list)
    evidence_buffer: list[ResearchEvidenceBufferItem] = Field(default_factory=list)
    context_state: ResearchContextState = Field(default_factory=ResearchContextState)
    working_summary: str = ""
    failure_count: int = 0
    stop_reason: str | None = None
    last_checkpoint_at: datetime | None = None
    step_count: int = 0
    report_id: str | None = None
