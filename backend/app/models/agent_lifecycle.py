"""PaperDesk Agent lifecycle contracts.

These models describe the system-level Agent path used by the lightweight
refactor. They are transport-neutral so API routes, services, runtime adapters,
tools, RAG, and docs can share one vocabulary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .chat import ChatAttachment
from .skills import ToolDeclaration


class PaperDeskRoute(str, Enum):
    """Product routes in the refactored PaperDesk Agent lifecycle."""

    DIRECT_CHAT = "direct_chat"
    PAPER_RAG = "paper_rag"
    LIBRARY_READ = "library_read"
    TOOL_ACTION = "tool_action"
    WRITE_PENDING = "write_pending"
    WRITE_CONFIRMED = "write_confirmed"
    REPORT_ACTION = "report_action"
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    EXPERIMENTAL_RESEARCH = "experimental_research"


class PaperDeskRuntimeKind(str, Enum):
    """Runtime targets used by the lifecycle dispatcher."""

    DIRECT_CHAT = "DirectChatRuntime"
    PAPER_RAG = "PaperRagRuntime"
    TOOL_ACTION = "ToolActionRuntime"
    CONFIRMED_WRITE = "ConfirmedWriteRuntime"
    REPORT_ACTION = "ReportActionRuntime"
    WORKSPACE_ACTION = "WorkspaceActionRuntime"
    EXPERIMENTAL = "ExperimentalRuntime"


class AgentOrchestrationPattern(str, Enum):
    """Primary orchestration pattern selected for one Agent request."""

    SINGLE_TURN = "single-turn"
    RETRIEVE_THEN_SYNTHESIZE = "retrieve-then-synthesize"
    BOUNDED_REACT = "bounded-react"
    PREVIEW_CONFIRM_EXECUTE_VERIFY = "preview-confirm-execute-verify"
    SERVICE_WORKFLOW = "service-workflow"
    PLAN_EXECUTE_REPLAN = "plan-execute-replan"


class AgentLifecycleStage(str, Enum):
    """Traceable stages in the end-to-end Agent lifecycle."""

    INGRESS = "ingress"
    ROUTE = "route"
    SKILL = "skill"
    CONTEXT = "context"
    RUNTIME = "runtime"
    RAG = "rag"
    TOOL_POLICY = "tool_policy"
    TOOL_CALL = "tool_call"
    WRITE_SAFETY = "write_safety"
    TRACE = "trace"
    RESPONSE = "response"


class WriteOperationLevel(str, Enum):
    """Normalized write scope classes for PaperDesk mutations."""

    NONE = "none"
    ENTITY = "entity-level"
    RELATION = "relation-level"
    CONTENT = "content-level"
    QUERY = "query-level"


class AgentLifecycleTraceEvent(BaseModel):
    """One lifecycle trace event suitable for API and test assertions."""

    stage: AgentLifecycleStage
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActiveSkillState(BaseModel):
    """Prompt-safe active skill summary carried through a request."""

    skill_id: str | None = None
    name: str | None = None
    source: str | None = None
    confidence: float = 0.0
    trigger_reason: str = ""
    allowed_tool_ids: list[str] = Field(default_factory=list)


class RouteDecisionPacket(BaseModel):
    """Route decision emitted by the lifecycle route stage."""

    route: PaperDeskRoute = PaperDeskRoute.DIRECT_CHAT
    reason: str
    confidence: float = 0.0
    requires_tools: bool = False
    requires_rag: bool = False
    requires_confirmation: bool = False
    write_operation_level: WriteOperationLevel = WriteOperationLevel.NONE
    target_runtime: PaperDeskRuntimeKind = PaperDeskRuntimeKind.DIRECT_CHAT
    orchestration_pattern: AgentOrchestrationPattern = AgentOrchestrationPattern.SINGLE_TURN
    selected_document_ids: list[str] = Field(default_factory=list)
    target_scope: dict[str, Any] = Field(default_factory=dict)


class ContextPacket(BaseModel):
    """Context assembled for a runtime turn."""

    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    selected_document_ids: list[str] = Field(default_factory=list)
    selected_file_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    workspace_scope: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    token_budget: int | None = None


class ToolPolicyDecision(BaseModel):
    """Tool exposure result after route, skill, scope, and risk filtering."""

    allowed_tools: list[ToolDeclaration] = Field(default_factory=list)
    filtered_tools: dict[str, str] = Field(default_factory=dict)
    confirmation_required: bool = False
    reason: str = ""


class RuntimeRequest(BaseModel):
    """Packet sent from lifecycle dispatcher to a runtime adapter."""

    session_id: str
    message_id: str
    user_prompt: str
    route: RouteDecisionPacket
    active_skill: ActiveSkillState | None = None
    context: ContextPacket = Field(default_factory=ContextPacket)
    tool_policy: ToolPolicyDecision = Field(default_factory=ToolPolicyDecision)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    trace: list[AgentLifecycleTraceEvent] = Field(default_factory=list)

    def add_trace(
        self,
        stage: AgentLifecycleStage,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            AgentLifecycleTraceEvent(
                stage=stage,
                message=message,
                payload=payload or {},
            )
        )


class RuntimeResult(BaseModel):
    """Normalized lifecycle runtime result."""

    route: PaperDeskRoute
    runtime: PaperDeskRuntimeKind
    status: str = "completed"
    response_text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace: list[AgentLifecycleTraceEvent] = Field(default_factory=list)
    error: str | None = None


class WriteActionPlan(BaseModel):
    """Preview plan for a scoped PaperDesk write."""

    action_id: str
    route: PaperDeskRoute
    operation_level: WriteOperationLevel
    description: str
    target_scope: dict[str, Any] = Field(default_factory=dict)
    affected_objects: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str
    executable: bool = False
    requires_confirmation: bool = True
    reason: str = ""


class PendingWriteAction(BaseModel):
    """Pending write action stored before confirmed execution."""

    action_id: str
    route: PaperDeskRoute
    operation_level: WriteOperationLevel
    target_scope: dict[str, Any]
    affected_objects: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
