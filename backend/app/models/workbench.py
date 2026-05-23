"""Workbench-facing configuration and file context models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .library import LibraryDocument


class WorkbenchModelOption(BaseModel):
    """Read-only model option exposed to the Workbench UI."""

    id: str
    label: str
    is_current: bool = False


class WorkbenchAgentProfile(BaseModel):
    """Static agent profile descriptor used as a chat hint."""

    id: str
    label: str
    description: str


class WorkbenchConfigResponse(BaseModel):
    """Read-only Workbench runtime configuration."""

    current_model: str
    masked_base_url: str | None = None
    available_models: list[WorkbenchModelOption] = Field(default_factory=list)
    agent_profiles: list[WorkbenchAgentProfile] = Field(default_factory=list)


class WorkbenchFileContextResponse(BaseModel):
    """Aggregated file context for one chat session."""

    session_id: str
    library_documents: list[LibraryDocument] = Field(default_factory=list)
    selected_document_ids: list[str] = Field(default_factory=list)
    attachment_document_ids: list[str] = Field(default_factory=list)
    recent_document_ids: list[str] = Field(default_factory=list)
    used_document_ids: list[str] = Field(default_factory=list)
    report_referenced_document_ids: list[str] = Field(default_factory=list)
    referents: dict[str, Any] = Field(default_factory=dict)


class WorkbenchTraceArtifactStatus(BaseModel):
    """User-facing artifact state derived from an assistant message."""

    report_saved: bool = False
    can_save_report: bool = False
    report_id: str | None = None


class WorkbenchTraceToolStep(BaseModel):
    """Compact tool execution row safe for Workbench display."""

    tool_name: str
    display_name: str
    status: str
    summary: str = ""
    evidence_count: int = 0
    risk_level: str = "unknown"


class WorkbenchCompactTraceStep(BaseModel):
    """Compact user-facing trace event without raw runtime payloads."""

    kind: str
    label: str
    status: str
    detail: str = ""
    created_at: str


class WorkbenchMessageTraceSummary(BaseModel):
    """Read-only compact execution summary for one chat assistant message."""

    message_id: str
    trace_id: str | None = None
    route: str | None = None
    action_status: str | None = None
    retrieval_status: str | None = None
    used_document_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    tool_steps: list[WorkbenchTraceToolStep] = Field(default_factory=list)
    risk_level: str = "unknown"
    confirmation_status: str = "none"
    saved_report_id: str | None = None
    artifact_status: WorkbenchTraceArtifactStatus = Field(default_factory=WorkbenchTraceArtifactStatus)
    compact_steps: list[WorkbenchCompactTraceStep] = Field(default_factory=list)


class WorkbenchCapability(BaseModel):
    """Safe display model for a Workbench-visible tool capability."""

    id: str
    name: str
    description: str
    group: str
    scope: str
    maturity: str
    source: str
    operation_level: str
    io_type: str
    write_type: str
    destructive: bool = False
    requires_confirmation: bool = False
    available_by_default: bool = False
    current_available: bool = False
    slash_command: str | None = None
    user_hint: str = ""


class WorkbenchExperimentalCapability(BaseModel):
    """Safe summary for future or experimental Workbench capabilities."""

    id: str
    name: str
    description: str
    scope: str = "experimental"
    maturity: str = "experimental"
    available_by_default: bool = False
    current_available: bool = False
    feature_flag: str | None = None
    user_hint: str = ""


class WorkbenchSlashCommand(BaseModel):
    """Workbench slash command suggestion shown alongside capabilities."""

    id: str
    label: str
    description: str


class WorkbenchCapabilitiesResponse(BaseModel):
    """Read-only Workbench capability summary."""

    stable_capabilities: list[WorkbenchCapability] = Field(default_factory=list)
    confirmation_required: list[WorkbenchCapability] = Field(default_factory=list)
    experimental_capabilities: list[WorkbenchExperimentalCapability] = Field(default_factory=list)
    slash_commands: list[WorkbenchSlashCommand] = Field(default_factory=list)
