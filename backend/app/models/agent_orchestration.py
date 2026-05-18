"""Models for chat-side agent orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .chat import ChatAttachment, MemorySnapshot
from .skills import SkillManifest, ToolDeclaration


class AgentRunMode(str, Enum):
    """Execution compatibility modes used to dispatch to existing runtimes."""

    DIRECT = "DIRECT"
    REACT = "REACT"
    PLANNER = "PLANNER"
    REFLECTION = "REFLECTION"


class KnowledgeRoute(str, Enum):
    """Product-semantic PaperDesk Knowledge Chat route names.

    Keep product behavior described in this vocabulary first. AgentRunMode is
    the lower-level compatibility layer that maps these routes to runtimes.
    """

    DIRECT_ANSWER = "DirectAnswer"
    TOOL_ACTION = "ToolAction"
    CONFIRMED_WRITE = "ConfirmedWrite"
    OPTIONAL_PLANNER = "OptionalPlanner"
    OPTIONAL_REFLECTION = "OptionalReflection"


class KnowledgeIntent(str, Enum):
    """Coarse user intent for Knowledge Chat routing and observability."""

    CHAT = "chat"
    PAPER_QA = "paper_qa"
    PAPER_COMPARE = "paper_compare"
    TAG_QUERY = "tag_query"
    TAG_WRITE = "tag_write"
    REPORT_QUERY = "report_query"
    REPORT_SAVE = "report_save"
    CORRECTION = "correction"
    LONG_RESEARCH_TASK = "long_research_task"


class KnowledgeRiskLevel(str, Enum):
    """Normalized route risk level."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class KnowledgeTargetObject(BaseModel):
    """Object scope resolved before Knowledge tools are called."""

    type: str
    ids: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)


class AgentOrchestratorInput(BaseModel):
    """Decision packet consumed by AgentOrchestrator."""

    session_id: str
    message_id: str
    user_prompt: str
    selected_document_ids: list[str] = Field(default_factory=list)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    conversation_referents: dict[str, Any] = Field(default_factory=dict)
    memory_snapshot: MemorySnapshot = Field(default_factory=MemorySnapshot)
    available_tools: list[ToolDeclaration] = Field(default_factory=list)
    available_skills: list[SkillManifest] = Field(default_factory=list)
    runtime_context: dict[str, Any] = Field(default_factory=dict)


class AgentModeDecision(BaseModel):
    """Auditable mode decision returned by AgentOrchestrator."""

    mode: AgentRunMode
    route: KnowledgeRoute = KnowledgeRoute.DIRECT_ANSWER
    intent: KnowledgeIntent = KnowledgeIntent.CHAT
    reason: str
    confidence: float = 0.0
    target_runtime: str
    requires_tools: bool = False
    requires_rag: bool = False
    requires_confirmation: bool = False
    risk_level: KnowledgeRiskLevel = KnowledgeRiskLevel.NONE
    target_objects: list[KnowledgeTargetObject] = Field(default_factory=list)
    initial_context: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    trace_id: str
    fallback_used: bool = False
