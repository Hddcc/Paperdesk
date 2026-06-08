"""Lifecycle dispatcher and adapters for the PaperDesk Agent refactor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from app.agent.runner import default_runner_policy_for_runtime
from app.models import (
    AgentLifecycleStage,
    PaperDeskRoute,
    PaperDeskRuntimeKind,
    RuntimeRequest,
    RuntimeResult,
    RuntimeMetricsEnvelope,
)
from .lifecycle import (
    ConfirmedWriteRuntime,
    DirectChatRuntime,
    ExperimentalRuntime,
    PaperRagRuntime,
    ReportActionRuntime,
    ToolActionRuntime,
    WorkspaceActionRuntime,
)


class RuntimeHandler(Protocol):
    """Runtime handler contract used by the lifecycle dispatcher."""

    def handle(self, request: RuntimeRequest) -> RuntimeResult:
        """Execute one lifecycle request."""


RuntimeCallable = Callable[[RuntimeRequest], RuntimeResult]


DEFAULT_ROUTE_RUNTIME_MAP: dict[PaperDeskRoute, PaperDeskRuntimeKind] = {
    PaperDeskRoute.DIRECT_CHAT: PaperDeskRuntimeKind.DIRECT_CHAT,
    PaperDeskRoute.PAPER_RAG: PaperDeskRuntimeKind.PAPER_RAG,
    PaperDeskRoute.LIBRARY_READ: PaperDeskRuntimeKind.TOOL_ACTION,
    PaperDeskRoute.TOOL_ACTION: PaperDeskRuntimeKind.TOOL_ACTION,
    PaperDeskRoute.WRITE_PENDING: PaperDeskRuntimeKind.TOOL_ACTION,
    PaperDeskRoute.WRITE_CONFIRMED: PaperDeskRuntimeKind.CONFIRMED_WRITE,
    PaperDeskRoute.REPORT_ACTION: PaperDeskRuntimeKind.REPORT_ACTION,
    PaperDeskRoute.WORKSPACE_READ: PaperDeskRuntimeKind.WORKSPACE_ACTION,
    PaperDeskRoute.WORKSPACE_WRITE: PaperDeskRuntimeKind.WORKSPACE_ACTION,
    PaperDeskRoute.EXPERIMENTAL_RESEARCH: PaperDeskRuntimeKind.EXPERIMENTAL,
}


class RuntimeDispatcher:
    """Dispatch lifecycle requests to route-compatible runtimes."""

    def __init__(
        self,
        handlers: dict[PaperDeskRuntimeKind, RuntimeHandler | RuntimeCallable] | None = None,
        route_runtime_map: dict[PaperDeskRoute, PaperDeskRuntimeKind] | None = None,
    ) -> None:
        self.route_runtime_map = dict(route_runtime_map or DEFAULT_ROUTE_RUNTIME_MAP)
        self._handlers: dict[PaperDeskRuntimeKind, RuntimeHandler | RuntimeCallable] = {}
        for runtime, handler in (handlers or {}).items():
            self.register(runtime, handler)

    def register(self, runtime: PaperDeskRuntimeKind, handler: RuntimeHandler | RuntimeCallable) -> None:
        self._handlers[runtime] = handler

    def runtime_for(self, route: PaperDeskRoute) -> PaperDeskRuntimeKind:
        return self.route_runtime_map.get(route, PaperDeskRuntimeKind.EXPERIMENTAL)

    def dispatch(self, request: RuntimeRequest) -> RuntimeResult:
        runtime = request.route.target_runtime or self.runtime_for(request.route.route)
        request.route.target_runtime = runtime
        request.add_trace(
            AgentLifecycleStage.RUNTIME,
            "runtime dispatch",
            {
                "runtime": runtime.value,
                "route": request.route.route.value,
                "capability_id": request.route.capability_id,
                "orchestration_pattern": request.route.orchestration_pattern.value,
                "execution_policy": self._execution_policy(request, runtime),
                "active_skill": self._active_skill_payload(request),
            },
        )
        handler = self._handlers.get(runtime)
        if handler is None:
            status, message = self._fallback_status_and_message(request, runtime)
            return RuntimeResult(
                route=request.route.route,
                runtime=runtime,
                capability_id=request.route.capability_id,
                status=status,
                response_text="",
                data={
                    "message": message,
                    "runtime": runtime.value,
                    "capability_id": request.route.capability_id,
                    "orchestration_pattern": request.route.orchestration_pattern.value,
                    "execution_policy": self._execution_policy(request, runtime),
                    "target_scope": request.route.target_scope,
                    "active_skill": self._active_skill_payload(request),
                },
                metrics=self._metrics(request, runtime, status=status).model_dump(mode="json"),
                trace=list(request.trace),
            )
        if callable(handler) and not hasattr(handler, "handle"):
            result = handler(request)
        else:
            result = handler.handle(request)  # type: ignore[union-attr]
        result.trace = [*request.trace, *result.trace]
        return result

    @staticmethod
    def _execution_policy(request: RuntimeRequest, runtime: PaperDeskRuntimeKind) -> dict[str, object]:
        return default_runner_policy_for_runtime(runtime, request.route.orchestration_pattern).as_trace_payload()

    @staticmethod
    def _fallback_status_and_message(
        request: RuntimeRequest,
        runtime: PaperDeskRuntimeKind,
    ) -> tuple[str, str]:
        if request.route.requires_confirmation:
            return "pending_confirmation", "runtime requires explicit confirmation before execution"
        if runtime == PaperDeskRuntimeKind.EXPERIMENTAL:
            return "safely_blocked", "experimental runtime is gated by feature policy"
        return "completed", "runtime policy recorded; chat service owns final response persistence"

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

    @staticmethod
    def _metrics(
        request: RuntimeRequest,
        runtime: PaperDeskRuntimeKind,
        *,
        status: str,
    ) -> RuntimeMetricsEnvelope:
        return RuntimeMetricsEnvelope.unavailable_tokens(
            route=request.route.route.value,
            runtime=runtime.value,
            capability_id=request.route.capability_id,
            status=status,
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


def default_runtime_dispatcher() -> RuntimeDispatcher:
    """Create a dispatcher with route/runtime mapping and deferred runtimes."""

    return RuntimeDispatcher(
        handlers={
            PaperDeskRuntimeKind.DIRECT_CHAT: DirectChatRuntime(),
            PaperDeskRuntimeKind.PAPER_RAG: PaperRagRuntime(),
            PaperDeskRuntimeKind.TOOL_ACTION: ToolActionRuntime(),
            PaperDeskRuntimeKind.CONFIRMED_WRITE: ConfirmedWriteRuntime(),
            PaperDeskRuntimeKind.REPORT_ACTION: ReportActionRuntime(),
            PaperDeskRuntimeKind.WORKSPACE_ACTION: WorkspaceActionRuntime(),
            PaperDeskRuntimeKind.EXPERIMENTAL: ExperimentalRuntime(),
        }
    )
