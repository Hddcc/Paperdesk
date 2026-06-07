"""Chat lifecycle adapter for the PaperDesk Agent entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.capabilities import CapabilityRegistry, default_capability_registry
from app.models import AgentLifecycleStage, ChatMessageRequest, RuntimeRequest, RuntimeResult

from app.agent.memory.context import AgentContextLifecycleService
from app.agent.runtimes.service_dispatch import AgentRuntimeDispatchService
from app.agent.skills.lifecycle import AgentSkillLifecycleService
from app.agent.skills.selector import SkillSelector
from app.agent.tools.policy import AgentToolPolicyResolver

from .ingress import AgentIngressService
from .router import AgentRouteDecisionService


@dataclass(slots=True)
class AgentLifecycleResult:
    """Lifecycle artifacts prepared for one chat turn."""

    request: RuntimeRequest
    dispatch_result: RuntimeResult


AgentLifecycleAdapterResult = AgentLifecycleResult


class _EmptyToolRegistry:
    """Fallback registry for lifecycle traces when no orchestrator is wired."""

    def filter_enabled(self, tool_ids):
        return []

    def list_default_candidates(self, *, scope: str = "knowledge", capability_id=None, operation_levels=None):
        return []

    def get(self, tool_id: str):
        return None


class _EmptySkillRegistry:
    """Fallback skill registry for lifecycle traces."""

    def list_enabled(self):
        return []


class AgentLifecycleService:
    """Single Agent lifecycle entry after chat session/message persistence."""

    def __init__(
        self,
        *,
        tool_registry: Any | None = None,
        skill_registry: Any | None = None,
        capability_registry: CapabilityRegistry | None = None,
        ingress_service: AgentIngressService | None = None,
        route_decision_service: AgentRouteDecisionService | None = None,
        context_service: AgentContextLifecycleService | None = None,
        skill_selector: SkillSelector | None = None,
        skill_lifecycle_service: AgentSkillLifecycleService | None = None,
        tool_policy_resolver: AgentToolPolicyResolver | None = None,
        runtime_dispatcher: AgentRuntimeDispatchService | None = None,
    ) -> None:
        registry = tool_registry or _EmptyToolRegistry()
        skills = skill_registry or _EmptySkillRegistry()
        self.ingress_service = ingress_service or AgentIngressService()
        self.route_decision_service = route_decision_service or AgentRouteDecisionService()
        self.context_service = context_service or AgentContextLifecycleService()
        self.skill_selector = skill_selector or SkillSelector()
        self.skill_lifecycle_service = skill_lifecycle_service or AgentSkillLifecycleService(skills)
        self.skill_registry = skills
        self.capability_registry = capability_registry or default_capability_registry()
        self.tool_policy_resolver = tool_policy_resolver or AgentToolPolicyResolver(registry)
        self.runtime_dispatcher = runtime_dispatcher or AgentRuntimeDispatchService()

    def prepare_chat_request(
        self,
        *,
        session_id: str,
        message_id: str,
        request: ChatMessageRequest,
        selected_document_ids: list[str] | None = None,
        selected_file_ids: list[str] | None = None,
        pending_action: dict[str, Any] | None = None,
        confirmation_received: bool = False,
    ) -> AgentLifecycleResult:
        lifecycle_request = self.ingress_service.build_request(
            session_id=session_id,
            message_id=message_id,
            request=request,
            pending_action=pending_action,
        )
        lifecycle_route = self.route_decision_service.decide(
            request,
            has_pending_action=lifecycle_request.context.pending_action is not None,
            confirmation_received=confirmation_received,
        )
        lifecycle_request.route = lifecycle_route
        capability_resolution = self.capability_registry.resolve(lifecycle_route.capability_id)
        lifecycle_request.capability = capability_resolution.declaration
        lifecycle_request.add_trace(
            AgentLifecycleStage.ROUTE,
            "route selected for chat request",
            {
                "route": lifecycle_route.route.value,
                "capability_id": lifecycle_route.capability_id,
                "runtime": lifecycle_route.target_runtime.value,
                "orchestration_pattern": lifecycle_route.orchestration_pattern.value,
                "requires_rag": lifecycle_route.requires_rag,
                "requires_tools": lifecycle_route.requires_tools,
                "requires_confirmation": lifecycle_route.requires_confirmation,
                "write_operation_level": lifecycle_route.write_operation_level.value,
                "target_scope": lifecycle_route.target_scope,
                "confidence": lifecycle_route.confidence,
                "reason": lifecycle_route.reason,
            },
        )
        lifecycle_request.add_trace(
            AgentLifecycleStage.CAPABILITY,
            "capability resolved for chat request",
            {
                "requested_capability_id": lifecycle_route.capability_id,
                "active_capability_id": capability_resolution.capability_id,
                "enabled": capability_resolution.enabled,
                "reason": capability_resolution.reason,
                "domain_package": capability_resolution.declaration.domain_package
                if capability_resolution.declaration
                else "",
            },
        )
        skill_selection = self.skill_selector.select(
            prompt=request.content,
            command=request.command,
            intent_hint=request.intent_hint,
            selected_document_count=len(selected_document_ids or request.selected_document_ids),
            attachments=request.attachments,
            available_skills=self.skill_registry.list_enabled(),
            route=lifecycle_route.route.value,
            capability_id=capability_resolution.capability_id,
        )
        self.skill_lifecycle_service.attach_active_skill(lifecycle_request, skill_selection)
        lifecycle_context = self.context_service.build_context(
            selected_document_ids=selected_document_ids or [],
            selected_file_ids=selected_file_ids or [],
            pending_action=lifecycle_request.context.pending_action,
        )
        self.context_service.attach_context(lifecycle_request, lifecycle_context)
        lifecycle_request.tool_policy = self.tool_policy_resolver.resolve(
            route=lifecycle_request.route,
            active_skill=lifecycle_request.active_skill,
            capability_id=capability_resolution.capability_id,
            confirmation_granted=confirmation_received,
        )
        lifecycle_request.add_trace(
            AgentLifecycleStage.TOOL_POLICY,
            "tool policy resolved for chat adapter",
            {
                "allowed_tool_count": len(lifecycle_request.tool_policy.allowed_tools),
                "filtered_tool_count": len(lifecycle_request.tool_policy.filtered_tools),
                "confirmation_required": lifecycle_request.tool_policy.confirmation_required,
                "capability_id": lifecycle_request.tool_policy.capability_id,
            },
        )
        return AgentLifecycleResult(
            request=lifecycle_request,
            dispatch_result=self.runtime_dispatcher.dispatch(lifecycle_request),
        )


class AgentLifecycleAdapterService(AgentLifecycleService):
    """Backward-compatible name for the lifecycle entry service."""
