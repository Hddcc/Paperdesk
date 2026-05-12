"""Decision helpers for the phase-13 single-main-agent research runtime."""

from __future__ import annotations

from app.models import (
    ResearchActionType,
    ResearchEvidenceBufferItem,
    ResearchPlanItem,
    ResearchRequest,
    ResearchRuntimePhase,
    ResearchRuntimeState,
)


class MainAgentRuntime:
    """Choose the next action for the single-main-agent research loop."""

    max_step_budget: int = 64

    @staticmethod
    def should_use_direct_task(request: ResearchRequest) -> bool:
        compact_topic = "".join(request.topic.split())
        return len(compact_topic) <= 8 and not request.notes

    def next_action(self, state: ResearchRuntimeState) -> tuple[ResearchActionType, str | None]:
        if state.failure_count >= 3:
            return ResearchActionType.FAIL, None
        if state.step_count >= self.max_step_budget:
            return ResearchActionType.FAIL, None
        if not state.plan_items:
            return ResearchActionType.PLAN, None

        active_item = self._next_pending_item(state)
        if active_item is None:
            if state.report_id:
                return ResearchActionType.FINISH, None
            return ResearchActionType.FINALIZE_REPORT, None

        evidence = self._find_evidence(state, active_item.task_id)
        if not evidence.online_completed:
            return ResearchActionType.SEARCH_ONLINE, active_item.task_id
        if not evidence.local_completed:
            return ResearchActionType.SEARCH_LOCAL, active_item.task_id
        if self._has_sufficient_evidence(evidence):
            return ResearchActionType.SUMMARIZE_EVIDENCE, active_item.task_id
        if active_item.revise_count == 0:
            return ResearchActionType.REVISE_PLAN, active_item.task_id
        return ResearchActionType.SUMMARIZE_EVIDENCE, active_item.task_id

    @staticmethod
    def initial_phase() -> ResearchRuntimePhase:
        return ResearchRuntimePhase.PLANNING

    @staticmethod
    def step_phase(action: ResearchActionType) -> ResearchRuntimePhase:
        if action == ResearchActionType.PLAN:
            return ResearchRuntimePhase.PLANNING
        if action in {ResearchActionType.SUMMARIZE_EVIDENCE, ResearchActionType.REVISE_PLAN}:
            return ResearchRuntimePhase.SUMMARIZING
        if action == ResearchActionType.FINALIZE_REPORT:
            return ResearchRuntimePhase.WRITING_REPORT
        if action == ResearchActionType.FINISH:
            return ResearchRuntimePhase.COMPLETED
        if action == ResearchActionType.FAIL:
            return ResearchRuntimePhase.FAILED
        return ResearchRuntimePhase.EXECUTING

    @staticmethod
    def should_degrade(task: ResearchPlanItem, evidence: ResearchEvidenceBufferItem) -> bool:
        return task.revise_count > 0 and not MainAgentRuntime._has_sufficient_evidence(evidence)

    @staticmethod
    def summarize_working_notes(state: ResearchRuntimeState) -> str:
        completed = [item.title for item in state.plan_items if item.task_id in state.completed_items]
        if not completed:
            return "尚未完成任务总结。"
        return "已完成任务：" + "；".join(completed)

    @staticmethod
    def _next_pending_item(state: ResearchRuntimeState) -> ResearchPlanItem | None:
        for item in state.plan_items:
            if item.task_id not in state.completed_items:
                return item
        return None

    @staticmethod
    def _find_evidence(state: ResearchRuntimeState, task_id: str) -> ResearchEvidenceBufferItem:
        for item in state.evidence_buffer:
            if item.task_id == task_id:
                return item
        evidence = ResearchEvidenceBufferItem(task_id=task_id)
        state.evidence_buffer.append(evidence)
        return evidence

    @staticmethod
    def _has_sufficient_evidence(evidence: ResearchEvidenceBufferItem) -> bool:
        return bool(evidence.paper_records or evidence.evidence_items)
