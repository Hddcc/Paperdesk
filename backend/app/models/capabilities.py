"""Capability contracts for the extension-ready Agent architecture."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CapabilityMaturity(str, Enum):
    """Exposure state for a capability pack."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


class CapabilityRouteBinding(BaseModel):
    """Runtime binding for one capability route surface."""

    route: str
    runtime: str
    orchestration_pattern: str
    enabled: bool = True


class CapabilityToolBinding(BaseModel):
    """Tool ownership metadata declared by a capability."""

    tool_id: str
    read_only: bool = True
    risk_level: str = "low"
    requires_confirmation: bool = False


class CapabilityDeclaration(BaseModel):
    """Declarative metadata for one extension capability."""

    capability_id: str
    display_name: str
    description: str = ""
    domain_package: str = ""
    routes: list[CapabilityRouteBinding] = Field(default_factory=list)
    tools: list[CapabilityToolBinding] = Field(default_factory=list)
    infrastructure_dependencies: list[str] = Field(default_factory=list)
    skill_scopes: list[str] = Field(default_factory=list)
    maturity: CapabilityMaturity = CapabilityMaturity.STABLE
    enabled: bool = True
    documentation_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityResolution(BaseModel):
    """Resolved capability for a lifecycle request."""

    capability_id: str
    declaration: CapabilityDeclaration | None = None
    enabled: bool = False
    reason: str = ""


class RuntimeMetricsEnvelope(BaseModel):
    """Compact runtime metrics that can be safely exposed in traces."""

    route: str = ""
    runtime: str = ""
    capability_id: str = ""
    status: str = ""
    latency_ms: int | None = None
    evidence_count: int = 0
    tool_call_count: int = 0
    allowed_tool_count: int = 0
    filtered_tool_count: int = 0
    selected_document_count: int = 0
    selected_file_count: int = 0
    token_usage: dict[str, int] | None = None
    token_usage_available: bool = False
    cost: dict[str, Any] | None = None
    error_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def unavailable_tokens(cls, **kwargs: Any) -> "RuntimeMetricsEnvelope":
        """Create metrics with an explicit unavailable token marker."""

        return cls(token_usage=None, token_usage_available=False, **kwargs)
