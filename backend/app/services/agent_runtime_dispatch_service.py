"""Service-layer lifecycle dispatcher used by ChatService.

This avoids importing the broad `app.runtime` package from services while the
legacy runtime package still imports service helpers.
"""

from __future__ import annotations

from typing import Any

from app.models import AgentLifecycleStage, PaperDeskRuntimeKind, RuntimeRequest, RuntimeResult


class AgentRuntimeDispatchService:
    """Dispatch lifecycle requests to route-specific adapter runtimes."""

    def dispatch(self, request: RuntimeRequest) -> RuntimeResult:
        runtime = request.route.target_runtime
        request.add_trace(
            AgentLifecycleStage.RUNTIME,
            "service-layer runtime dispatch",
            {
                "runtime": runtime.value,
                "route": request.route.route.value,
                "orchestration_pattern": request.route.orchestration_pattern.value,
                "execution_policy": self._execution_policy(request),
                "active_skill": self._active_skill_payload(request),
            },
        )
        return RuntimeResult(
            route=request.route.route,
            runtime=runtime,
            status="deferred",
            data={
                "runtime": runtime.value,
                "adapter": self._adapter_name(runtime),
                "orchestration_pattern": request.route.orchestration_pattern.value,
                "execution_policy": self._execution_policy(request),
                "target_scope": request.route.target_scope,
                "active_skill": self._active_skill_payload(request),
                "message": "legacy chat flow remains response owner during migration",
            },
            metrics={
                "allowed_tool_count": len(request.tool_policy.allowed_tools),
                "filtered_tool_count": len(request.tool_policy.filtered_tools),
                "selected_document_count": len(request.context.selected_document_ids),
                "selected_file_count": len(request.context.selected_file_ids),
                "evidence_count": len(request.context.evidence),
                "tool_call_count": 0,
            },
            trace=list(request.trace),
        )

    @staticmethod
    def _adapter_name(runtime: PaperDeskRuntimeKind) -> str:
        return {
            PaperDeskRuntimeKind.DIRECT_CHAT: "DirectChatRuntime",
            PaperDeskRuntimeKind.PAPER_RAG: "PaperRagRuntime",
            PaperDeskRuntimeKind.TOOL_ACTION: "ToolActionRuntime",
            PaperDeskRuntimeKind.CONFIRMED_WRITE: "ConfirmedWriteRuntime",
            PaperDeskRuntimeKind.REPORT_ACTION: "ReportActionRuntime",
            PaperDeskRuntimeKind.WORKSPACE_ACTION: "WorkspaceActionRuntime",
            PaperDeskRuntimeKind.EXPERIMENTAL: "ExperimentalRuntime",
        }[runtime]

    @staticmethod
    def _execution_policy(request: RuntimeRequest) -> dict[str, Any]:
        route = request.route.route.value
        pattern = request.route.orchestration_pattern.value
        runtime = request.route.target_runtime.value
        if runtime == PaperDeskRuntimeKind.TOOL_ACTION.value:
            return {
                "pattern": pattern,
                "max_steps": 4,
                "stop_reason": "bounded_tool_steps_or_final_answer",
                "observations": "structured_tool_results",
            }
        if runtime in {
            PaperDeskRuntimeKind.CONFIRMED_WRITE.value,
            PaperDeskRuntimeKind.WORKSPACE_ACTION.value,
        } and request.route.requires_confirmation:
            return {
                "pattern": pattern,
                "max_steps": 3,
                "stop_reason": "preview_confirm_execute_verify_complete",
                "requires_explicit_scope": True,
            }
        if runtime == PaperDeskRuntimeKind.PAPER_RAG.value:
            return {
                "pattern": pattern,
                "max_steps": 1,
                "stop_reason": "retrieve_then_synthesize_complete",
                "planner_enabled": False,
            }
        if runtime == PaperDeskRuntimeKind.DIRECT_CHAT.value:
            return {
                "pattern": pattern,
                "max_steps": 1,
                "stop_reason": "single_turn_answer_complete",
                "rag_enabled": False,
                "tools_enabled": False,
            }
        if runtime == PaperDeskRuntimeKind.EXPERIMENTAL.value:
            return {
                "pattern": pattern,
                "max_steps": 6,
                "stop_reason": "experimental_policy_limit",
                "feature_flag_required": True,
            }
        return {
            "pattern": pattern,
            "route": route,
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
        }
