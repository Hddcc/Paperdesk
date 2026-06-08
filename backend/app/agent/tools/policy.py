"""Tool policy resolution for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.models import (
    ActiveSkillState,
    PaperDeskRoute,
    RouteDecisionPacket,
    ToolDeclaration,
    ToolObservation,
    ToolObservationError,
    ToolPolicyDecision,
    ToolVerification,
)


class AgentToolPolicyResolver:
    """Resolve tool exposure from route, skill, scope, and registry metadata."""

    def __init__(self, tool_registry: Any) -> None:
        self.tool_registry = tool_registry

    def resolve(
        self,
        *,
        route: RouteDecisionPacket,
        active_skill: ActiveSkillState | None = None,
        capability_id: str | None = None,
        confirmation_granted: bool = False,
        feature_flags: Iterable[str] | None = None,
        bound_external_tool_ids: Iterable[str] | None = None,
    ) -> ToolPolicyDecision:
        enabled_flags = set(feature_flags or [])
        bound_external_ids = set(bound_external_tool_ids or [])
        active_capability = capability_id or route.capability_id
        requested_ids = list(active_skill.allowed_tool_ids) if active_skill and active_skill.allowed_tool_ids else []
        candidates = (
            self.tool_registry.filter_enabled(requested_ids)
            if requested_ids
            else self.tool_registry.list_default_candidates(
                scope=self._scope_for_route(route.route),
                capability_id=active_capability,
            )
        )
        by_id = {tool.tool_id: tool for tool in candidates}
        filtered: dict[str, str] = {}

        for tool_id in requested_ids:
            if self.tool_registry.get(tool_id) is None:
                filtered[tool_id] = "tool is not registered or not enabled"
            elif tool_id not in by_id:
                filtered[tool_id] = "tool is outside registry default candidates"

        allowed: list[ToolDeclaration] = []
        confirmation_required = False
        for tool in candidates:
            reason = self._filter_reason(
                tool,
                route=route,
                capability_id=active_capability,
                confirmation_granted=confirmation_granted,
                feature_flags=enabled_flags,
                bound_external_tool_ids=bound_external_ids,
            )
            if reason:
                filtered[tool.tool_id] = reason
                if "confirmation" in reason:
                    confirmation_required = True
                continue
            allowed.append(tool)

        return ToolPolicyDecision(
            allowed_tools=allowed,
            filtered_tools=filtered,
            filter_reasons=dict(filtered),
            confirmation_required=confirmation_required or route.requires_confirmation,
            capability_id=active_capability,
            reason=self._reason(route, active_skill, allowed, filtered),
        )

    def _filter_reason(
        self,
        tool: ToolDeclaration,
        *,
        route: RouteDecisionPacket,
        capability_id: str,
        confirmation_granted: bool,
        feature_flags: set[str],
        bound_external_tool_ids: set[str],
    ) -> str:
        spec = tool.spec
        if spec is None:
            return "tool has no safety metadata"
        if spec.source == "mcp" and tool.tool_id not in bound_external_tool_ids:
            return "external MCP tool is not bound or configured for this user"
        if spec.feature_flag and spec.feature_flag not in feature_flags:
            return f"feature flag required: {spec.feature_flag}"
        if spec.capability_id not in {capability_id, "shared", "common"}:
            return f"tool capability {spec.capability_id} is not active for capability {capability_id}"
        if spec.maturity != "stable" and spec.maturity != "experimental":
            return "tool maturity is not available"
        if spec.scope not in self._visible_scopes(route.route):
            return f"tool scope {spec.scope} is not visible for route {route.route.value}"
        if spec.requires_confirmation and not confirmation_granted:
            return "tool requires pending-action confirmation"
        if route.requires_confirmation and spec.io_type == "write" and not confirmation_granted:
            return "write route requires confirmation before write tools are exposed"
        return ""

    @staticmethod
    def _scope_for_route(route: PaperDeskRoute) -> str:
        if route in {PaperDeskRoute.WORKSPACE_READ, PaperDeskRoute.WORKSPACE_WRITE}:
            return "workspace"
        if route == PaperDeskRoute.EXPERIMENTAL_RESEARCH:
            return "research"
        return "knowledge"

    @classmethod
    def _visible_scopes(cls, route: PaperDeskRoute) -> set[str]:
        scope = cls._scope_for_route(route)
        visible = {scope, "shared", "common"}
        if scope == "knowledge":
            visible.add("workspace")
        return visible

    @staticmethod
    def _reason(
        route: RouteDecisionPacket,
        active_skill: ActiveSkillState | None,
        allowed: list[ToolDeclaration],
        filtered: dict[str, str],
    ) -> str:
        skill_label = active_skill.skill_id if active_skill and active_skill.skill_id else "no active skill"
        return (
            f"Resolved {len(allowed)} allowed tools and {len(filtered)} filtered tools "
            f"for route {route.route.value} with {skill_label}."
        )


class AgentToolObservationFactory:
    """Create normalized tool observations for lifecycle traces."""

    @staticmethod
    def success(
        *,
        tool: ToolDeclaration,
        message: str = "",
        data: dict[str, Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        affected_objects: list[dict[str, Any]] | None = None,
        counts: dict[str, Any] | None = None,
        verification: ToolVerification | None = None,
    ) -> ToolObservation:
        spec = tool.spec
        return ToolObservation(
            tool_name=tool.tool_id,
            success=True,
            capability_id=spec.capability_id if spec else "",
            operation_level=spec.operation_level if spec else "unknown",
            io_type=spec.io_type if spec else "unknown",
            write_type=spec.write_type if spec else "none",
            affected_objects=affected_objects or [],
            counts=counts or {},
            data=data or {},
            evidence=evidence or [],
            verification=verification,
            message=message,
        )

    @staticmethod
    def error(
        *,
        tool_name: str,
        code: str,
        message: str,
        recoverable: bool = True,
        operation_level: str = "unknown",
        io_type: str = "unknown",
        capability_id: str = "",
    ) -> ToolObservation:
        return ToolObservation(
            tool_name=tool_name,
            success=False,
            capability_id=capability_id,
            operation_level=operation_level,
            io_type=io_type,
            error=ToolObservationError(
                code=code,
                message=message,
                recoverable=recoverable,
            ),
            message=message,
        )
