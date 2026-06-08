"""Bounded runner policies for route-aware orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.models import AgentOrchestrationPattern, PaperDeskRuntimeKind


@dataclass(frozen=True, slots=True)
class RunnerPolicy:
    """Execution guardrails for one primary runtime path."""

    runtime: PaperDeskRuntimeKind
    pattern: AgentOrchestrationPattern
    max_steps: int
    stop_reason: str
    tools_enabled: bool = False
    rag_enabled: bool = False
    planner_enabled: bool = False
    requires_explicit_scope: bool = False
    feature_flag_required: bool = False
    observations: str | None = None

    def as_trace_payload(self) -> dict[str, object]:
        """Return the trace-safe policy shape used by dispatchers and runtimes."""

        payload: dict[str, object] = {
            "pattern": self.pattern.value,
            "max_steps": self.max_steps,
            "stop_reason": self.stop_reason,
            "tools_enabled": self.tools_enabled,
            "rag_enabled": self.rag_enabled,
            "planner_enabled": self.planner_enabled,
        }
        if self.requires_explicit_scope:
            payload["requires_explicit_scope"] = True
        if self.feature_flag_required:
            payload["feature_flag_required"] = True
        if self.observations:
            payload["observations"] = self.observations
        return payload


def default_runner_policy_for_runtime(
    runtime: PaperDeskRuntimeKind,
    pattern: AgentOrchestrationPattern,
) -> RunnerPolicy:
    """Return the bounded execution policy used by runtime adapters."""

    if runtime == PaperDeskRuntimeKind.DIRECT_CHAT:
        return RunnerPolicy(runtime, pattern, 1, "single_turn_answer_complete")
    if runtime == PaperDeskRuntimeKind.PAPER_RAG:
        return RunnerPolicy(runtime, pattern, 1, "retrieve_then_synthesize_complete", rag_enabled=True)
    if runtime == PaperDeskRuntimeKind.TOOL_ACTION:
        return RunnerPolicy(
            runtime,
            pattern,
            4,
            "bounded_tool_steps_or_final_answer",
            tools_enabled=True,
            observations="structured_tool_results",
        )
    if runtime == PaperDeskRuntimeKind.REPORT_ACTION:
        return RunnerPolicy(runtime, pattern, 2, "deterministic_service_workflow_complete", tools_enabled=True)
    if runtime == PaperDeskRuntimeKind.EXPERIMENTAL:
        return RunnerPolicy(
            runtime,
            pattern,
            6,
            "experimental_policy_limit",
            tools_enabled=True,
            planner_enabled=True,
            feature_flag_required=True,
        )
    if pattern == AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY:
        return RunnerPolicy(
            runtime,
            pattern,
            3,
            "preview_confirm_execute_verify_complete",
            tools_enabled=True,
            requires_explicit_scope=True,
        )
    return RunnerPolicy(runtime, pattern, 2, "deterministic_service_workflow_complete", tools_enabled=True)
