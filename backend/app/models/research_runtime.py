"""Research runtime state models for the phase-13 main-agent loop."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .enums import TodoTaskStatus
from .paper import EvidenceItem, PaperRecord
from .report import TaskSummary
from .runtime import TaskArtifactRef
from .task_routing import ResearchTaskRoute


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


class ResearchToolResultClassification(str, Enum):
    """Decision-facing classification for a completed tool observation."""

    SUCCESS_SUFFICIENT = "success_sufficient"
    SUCCESS_INSUFFICIENT = "success_insufficient"
    RETRYABLE_ERROR = "retryable_error"
    NON_RETRYABLE_ERROR = "non_retryable_error"
    NO_INCREMENT = "no_increment"


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


class PlannerProviderType(str, Enum):
    """Planner candidate providers available to the main-agent runtime."""

    RULE_BASED = "rule_based"
    LLM_CANDIDATE = "llm_candidate"
    HYBRID_CANDIDATE = "hybrid_candidate"


class ResearchPlanOperationType(str, Enum):
    """Plan structure operations proposed by planner candidates."""

    REWRITE_QUERY = "rewrite_query"
    INSERT_ITEM = "insert_item"
    SPLIT_ITEM = "split_item"
    MERGE_ITEMS = "merge_items"
    REORDER_ITEMS = "reorder_items"
    CLOSE_ITEM = "close_item"


class ResearchToolStrategy(BaseModel):
    """Concrete strategy selected under a high-level research action."""

    strategy_id: str
    action_type: ResearchActionType
    label: str
    parameters: dict = Field(default_factory=dict)
    rationale: str = ""


class ResearchPlanOperation(BaseModel):
    """Validated plan operation that can be persisted in runtime history."""

    operation_type: ResearchPlanOperationType
    target_task_id: str | None = None
    source_task_ids: list[str] = Field(default_factory=list)
    new_task_id: str | None = None
    title: str | None = None
    intent: str | None = None
    query: str | None = None
    priority: int | None = None
    ordered_task_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    applied_at: datetime | None = None


class ResearchPlannerCandidate(BaseModel):
    """Candidate suggestion from a planner provider before runtime adjudication."""

    candidate_action: ResearchActionType
    candidate_tool: str
    candidate_plan_ops: list[ResearchPlanOperation] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    provider: PlannerProviderType = PlannerProviderType.RULE_BASED
    fallback_used: bool = False


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
    selected_tool: str | None = None
    tool_strategy: ResearchToolStrategy | None = None
    reason: str = ""
    status: ResearchStepStatus = ResearchStepStatus.RUNNING
    started_at: datetime


class ResearchToolCallRecord(BaseModel):
    """History entry for a completed or failed tool call."""

    step_id: str
    action: ResearchActionType
    task_id: str | None = None
    status: ResearchToolResultStatus
    summary: str
    selected_tool: str | None = None
    tool_strategy: ResearchToolStrategy | None = None
    decision_reason: str = ""
    result_classification: ResearchToolResultClassification | None = None
    planner_provider: PlannerProviderType | None = None
    planner_fallback_used: bool = False
    plan_operations: list[ResearchPlanOperation] = Field(default_factory=list)
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
    objective: str = ""
    done_criteria: str = ""
    status: TodoTaskStatus = TodoTaskStatus.PENDING
    priority: int = 100
    suggested_tools: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    notes: list[str] = Field(default_factory=list)
    revise_count: int = 0
    query_history: list[str] = Field(default_factory=list)
    summary: str | None = None
    summary_markdown: str | None = None
    degraded: bool = False

    @model_validator(mode="after")
    def fill_dynamic_planning_defaults(self) -> "ResearchPlanItem":
        if not self.objective:
            self.objective = self.intent or self.title
        if not self.done_criteria:
            self.done_criteria = "形成有引用依据的任务级总结，或在证据不足时明确降级边界。"
        if not self.suggested_tools:
            self.suggested_tools = ["search_online", "search_local", "summarize_evidence"]
        if not self.required_evidence:
            self.required_evidence = ["online_paper", "local_document"]
        if not self.query_history and self.query:
            self.query_history = [self.query]
        return self

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
    classification: ResearchToolResultClassification | None = None
    payload: dict = Field(default_factory=dict)
    artifacts: list[TaskArtifactRef] = Field(default_factory=list)
    retryable: bool = False
    error: str | None = None


class ResearchActionDecision(BaseModel):
    """Structured decision produced by the main-agent controller."""

    action_type: ResearchActionType
    selected_tool: str | None = None
    tool_strategy: ResearchToolStrategy | None = None
    reason: str
    target_task_id: str | None = None


class ResearchRuntimeState(BaseModel):
    """Checkpointable state for a single research run."""

    run_id: str
    goal: str
    current_phase: ResearchRuntimePhase
    task_route: ResearchTaskRoute | None = None
    plan_items: list[ResearchPlanItem] = Field(default_factory=list)
    completed_items: list[str] = Field(default_factory=list)
    active_step: ResearchRuntimeStep | None = None
    tool_history: list[ResearchToolCallRecord] = Field(default_factory=list)
    evidence_buffer: list[ResearchEvidenceBufferItem] = Field(default_factory=list)
    context_state: ResearchContextState = Field(default_factory=ResearchContextState)
    working_summary: str = ""
    failure_count: int = 0
    replan_count: int = 0
    no_progress_count: int = 0
    same_tool_streak: int = 0
    last_tool_signature: str | None = None
    last_decision: ResearchActionDecision | None = None
    plan_revision_history: list[ResearchPlanOperation] = Field(default_factory=list)
    last_plan_operation: ResearchPlanOperation | None = None
    planner_provider: PlannerProviderType = PlannerProviderType.RULE_BASED
    planner_fallback_used: bool = False
    stop_reason: str | None = None
    last_checkpoint_at: datetime | None = None
    step_count: int = 0
    report_id: str | None = None
