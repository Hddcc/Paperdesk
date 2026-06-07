"""Backend skill and tool capability models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .skill_selection import SkillSelection, SkillSelectionResult
from .task_routing import ResearchArtifactProtocol, ResearchTaskType


class SkillExecutionMode(str, Enum):
    """Preferred execution path for a backend skill."""

    LIGHTWEIGHT = "lightweight"
    MAIN_AGENT = "main_agent"


class SkillScope(str, Enum):
    """Runtime surfaces where a skill can be considered."""

    KNOWLEDGE = "knowledge"
    RESEARCH = "research"
    SHARED = "shared"


class SkillSource(str, Enum):
    """Origin of a file-backed skill declaration."""

    BUILTIN = "builtin"
    CUSTOM = "custom"
    MCP = "mcp"


class SkillMaturity(str, Enum):
    """Default exposure maturity for skill selection metadata."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


class SkillDocumentCountConstraint(BaseModel):
    """Optional document-count bounds used only for skill trigger matching."""

    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)


class SkillTriggerMetadata(BaseModel):
    """Manifest-owned trigger metadata for automatic skill selection.

    These fields influence skill selection traces and output protocol choice
    only. They do not grant tool permissions or bypass runtime guardrails.
    """

    keywords: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    intent_hints: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    attachment_kinds: list[str] = Field(default_factory=list)
    document_count: SkillDocumentCountConstraint | None = None
    fallback: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolSource(str, Enum):
    """Origin of a normalized tool declaration."""

    BUILTIN = "builtin"
    MCP = "mcp"


class ToolSpec(BaseModel):
    """Machine-readable safety and routing metadata for an Agent tool.

    ToolSpec is descriptive metadata, not an execution contract. Runtime code
    still owns tool invocation, payload shape, preview/confirmation, and
    verification behavior.

    Field boundaries:
    - scope separates knowledge, research, mcp, and experimental declarations.
    - maturity and available_by_default only describe default exposure
      eligibility; registry filters still decide whether a tool is listed.
    - io_type, write_type, destructive, and requires_confirmation describe
      safety posture for routing and auditing, not permission to bypass runtime
      guardrails.
    - feature_flag documents the configuration gate expected before exposing an
      optional or external capability.
    """

    name: str
    display_name: str = ""
    description: str
    capability_id: str = "paper"
    integration_source: str | None = None
    scope: str = "experimental"
    maturity: str = "stable"
    operation_level: str
    io_type: str
    write_type: str = "none"
    destructive: bool = False
    requires_confirmation: bool = False
    input_object_types: list[str] = Field(default_factory=list)
    output_observation_type: str = "tool_observation"
    requires_post_read_verification: bool = False
    verification_tool: str | None = None
    available_by_default: bool = False
    feature_flag: str | None = None
    source: str = "builtin"
    metrics_fields: list[str] = Field(default_factory=list)


class ToolObservationError(BaseModel):
    """Structured error payload for failed tool observations."""

    code: str
    message: str
    recoverable: bool = True
    suggested_next_action: str = "ask_user_to_clarify"


class ToolVerification(BaseModel):
    """Structured post-read verification status."""

    performed: bool = False
    success: bool = False
    method: str = "none"
    details: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """Normalized tool observation envelope used by Agent runtimes."""

    tool_name: str
    success: bool
    capability_id: str = ""
    operation_level: str
    io_type: str
    write_type: str = "none"
    target_objects: list[dict[str, Any]] = Field(default_factory=list)
    affected_objects: list[dict[str, Any]] = Field(default_factory=list)
    counts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    requires_followup: bool = False
    requires_confirmation: bool = False
    verification: ToolVerification | None = None
    error: ToolObservationError | None = None
    message: str = ""


class ToolDeclaration(BaseModel):
    """Stable declaration for a tool that can be selected by the main agent."""

    tool_id: str
    source: ToolSource = ToolSource.BUILTIN
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True
    enabled: bool = True
    spec: ToolSpec | None = None


class SkillDefinition(BaseModel):
    """File-backed backend capability definition."""

    skill_id: str
    name: str
    enabled: bool = True
    supported_task_types: list[ResearchTaskType] = Field(default_factory=list)
    default_execution_mode: SkillExecutionMode = SkillExecutionMode.MAIN_AGENT
    description: str
    body: str
    available_tools: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifact_protocol: ResearchArtifactProtocol
    version: str = "1.0.0"
    priority: int = 100


class SkillManifest(BaseModel):
    """Lightweight skill index loaded during discovery."""

    skill_id: str
    name: str
    enabled: bool = True
    supported_task_types: list[ResearchTaskType] = Field(default_factory=list)
    default_execution_mode: SkillExecutionMode = SkillExecutionMode.MAIN_AGENT
    description: str
    artifact_protocol: ResearchArtifactProtocol
    version: str = "1.0.0"
    priority: int = 100
    skill_file: str = "SKILL.md"
    scope: SkillScope = SkillScope.SHARED
    source: SkillSource = SkillSource.BUILTIN
    maturity: SkillMaturity = SkillMaturity.STABLE
    available_by_default: bool = True
    trigger: SkillTriggerMetadata | None = None
    allowed_tool_ids: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)


class SkillContextSummary(BaseModel):
    """Prompt-safe skill context summary for trace observability only."""

    skill_id: str
    name: str
    short_description: str = ""
    artifact_protocol: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)
    output_expectations: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    trigger_reason: str = ""
    char_count: int = 0
