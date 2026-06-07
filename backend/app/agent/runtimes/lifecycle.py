"""Route-specific lifecycle runtimes for PaperDesk.

These runtimes are migration adapters: they provide stable route ownership
while existing business logic is moved out of the large chat/runtime modules.
Each runtime can be connected to a callable implementation route by route.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.models import AgentLifecycleStage, PaperDeskRuntimeKind, RuntimeMetricsEnvelope, RuntimeRequest, RuntimeResult

RuntimeImplementation = Callable[[RuntimeRequest], RuntimeResult]


class LifecycleRuntime:
    """Base lifecycle runtime with optional implementation callback."""

    runtime_kind: PaperDeskRuntimeKind = PaperDeskRuntimeKind.EXPERIMENTAL
    default_status = "deferred"

    def __init__(self, implementation: RuntimeImplementation | None = None) -> None:
        self.implementation = implementation

    def handle(self, request: RuntimeRequest) -> RuntimeResult:
        request.add_trace(
            AgentLifecycleStage.RUNTIME,
            f"{self.runtime_kind.value} selected",
            {
                "runtime": self.runtime_kind.value,
                "route": request.route.route.value,
                "capability_id": request.route.capability_id,
                "orchestration_pattern": request.route.orchestration_pattern.value,
                "execution_policy": self._execution_policy(request),
                "active_skill": self._active_skill_payload(request),
                "adapter": self.implementation is None,
            },
        )
        if self.implementation is not None:
            result = self.implementation(request)
            result.trace = [*request.trace, *result.trace]
            return result
        return RuntimeResult(
            route=request.route.route,
            runtime=self.runtime_kind,
            capability_id=request.route.capability_id,
            status=self.default_status,
            data={
                "message": "runtime implementation not connected yet",
                "runtime": self.runtime_kind.value,
                "capability_id": request.route.capability_id,
                "orchestration_pattern": request.route.orchestration_pattern.value,
                "execution_policy": self._execution_policy(request),
                "target_scope": request.route.target_scope,
                "active_skill": self._active_skill_payload(request),
            },
            metrics=self._metrics(request).model_dump(mode="json"),
            trace=list(request.trace),
        )

    def _execution_policy(self, request: RuntimeRequest) -> dict[str, Any]:
        pattern = request.route.orchestration_pattern.value
        if self.runtime_kind == PaperDeskRuntimeKind.TOOL_ACTION:
            return {
                "pattern": pattern,
                "max_steps": 4,
                "stop_reason": "bounded_tool_steps_or_final_answer",
                "observations": "structured_tool_results",
            }
        if self.runtime_kind in {
            PaperDeskRuntimeKind.CONFIRMED_WRITE,
            PaperDeskRuntimeKind.WORKSPACE_ACTION,
        } and request.route.requires_confirmation:
            return {
                "pattern": pattern,
                "max_steps": 3,
                "stop_reason": "preview_confirm_execute_verify_complete",
                "requires_explicit_scope": True,
            }
        if self.runtime_kind == PaperDeskRuntimeKind.PAPER_RAG:
            return {
                "pattern": pattern,
                "max_steps": 1,
                "stop_reason": "retrieve_then_synthesize_complete",
                "planner_enabled": False,
            }
        if self.runtime_kind == PaperDeskRuntimeKind.DIRECT_CHAT:
            return {
                "pattern": pattern,
                "max_steps": 1,
                "stop_reason": "single_turn_answer_complete",
                "rag_enabled": False,
                "tools_enabled": False,
            }
        if self.runtime_kind == PaperDeskRuntimeKind.EXPERIMENTAL:
            return {
                "pattern": pattern,
                "max_steps": 6,
                "stop_reason": "experimental_policy_limit",
                "feature_flag_required": True,
            }
        return {
            "pattern": pattern,
            "max_steps": 2,
            "stop_reason": "deterministic_service_workflow_complete",
        }

    @staticmethod
    def _active_skill_payload(request: RuntimeRequest) -> dict[str, Any] | None:
        if request.active_skill is None:
            return None
        return {
            "skill_id": request.active_skill.skill_id,
            "name": request.active_skill.name,
            "source": request.active_skill.source,
            "confidence": request.active_skill.confidence,
            "allowed_tool_count": len(request.active_skill.allowed_tool_ids),
            "capability_ids": list(request.active_skill.capability_ids),
        }

    def _metrics(self, request: RuntimeRequest) -> RuntimeMetricsEnvelope:
        return RuntimeMetricsEnvelope.unavailable_tokens(
            route=request.route.route.value,
            runtime=self.runtime_kind.value,
            capability_id=request.route.capability_id,
            status=self.default_status,
            evidence_count=len(request.context.evidence),
            tool_call_count=0,
            allowed_tool_count=len(request.tool_policy.allowed_tools),
            filtered_tool_count=len(request.tool_policy.filtered_tools),
            selected_document_count=len(request.context.selected_document_ids),
            selected_file_count=len(request.context.selected_file_ids),
            metadata={
                "selected_document_count": len(request.context.selected_document_ids),
                "selected_file_count": len(request.context.selected_file_ids),
            },
        )


class DirectChatRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.DIRECT_CHAT


class PaperRagRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.PAPER_RAG


class ToolActionRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.TOOL_ACTION


class ConfirmedWriteRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.CONFIRMED_WRITE


class ReportActionRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.REPORT_ACTION


class WorkspaceActionRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.WORKSPACE_ACTION


class ExperimentalRuntime(LifecycleRuntime):
    runtime_kind = PaperDeskRuntimeKind.EXPERIMENTAL
