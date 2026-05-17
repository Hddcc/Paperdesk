"""Models for chat-side agent orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .chat import ChatAttachment, MemorySnapshot
from .skills import SkillManifest, ToolDeclaration


class AgentRunMode(str, Enum):
    """Supported execution modes selected by the chat orchestrator."""

    DIRECT = "DIRECT"
    REACT = "REACT"
    PLANNER = "PLANNER"
    REFLECTION = "REFLECTION"


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
    reason: str
    confidence: float = 0.0
    target_runtime: str
    initial_context: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    trace_id: str
    fallback_used: bool = False

