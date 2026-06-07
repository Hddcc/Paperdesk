"""Planner candidate providers for controlled research plan revisions."""

from __future__ import annotations

from uuid import uuid4

from app.models import (
    PlannerProviderType,
    ResearchActionDecision,
    ResearchActionType,
    ResearchEvidenceBufferItem,
    ResearchPlanItem,
    ResearchPlanOperation,
    ResearchPlanOperationType,
    ResearchPlannerCandidate,
    ResearchRuntimeState,
)


class RuleBasedPlannerCandidateProvider:
    """Produce bounded plan revision candidates without taking runtime control."""

    provider_type = PlannerProviderType.RULE_BASED

    def propose(
        self,
        state: ResearchRuntimeState,
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem | None,
    ) -> ResearchPlannerCandidate:
        if decision.action_type != ResearchActionType.REVISE_PLAN or plan_item is None:
            return ResearchPlannerCandidate(
                candidate_action=decision.action_type,
                candidate_tool=decision.selected_tool or decision.action_type.value,
                confidence=1.0,
                reason="当前动作不需要计划结构调整。",
                provider=self.provider_type,
            )

        strategy_id = decision.selected_tool or "revise_plan/rewrite_query"
        evidence = self._find_evidence(state, plan_item.task_id)
        if strategy_id == "revise_plan/reorder_priority":
            return self._reorder_candidate(state, decision, plan_item)
        if strategy_id == "revise_plan/split_task" and self._can_split(plan_item, evidence):
            return self._split_candidate(decision, plan_item)
        return self._rewrite_candidate(decision, plan_item)

    def _rewrite_candidate(
        self,
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem,
    ) -> ResearchPlannerCandidate:
        return ResearchPlannerCandidate(
            candidate_action=decision.action_type,
            candidate_tool="revise_plan/rewrite_query",
            candidate_plan_ops=[
                ResearchPlanOperation(
                    operation_type=ResearchPlanOperationType.REWRITE_QUERY,
                    target_task_id=plan_item.task_id,
                    query=f"{plan_item.query} {plan_item.intent}".strip(),
                    reason="仅改写 query，不调整任务结构。",
                )
            ],
            confidence=0.6,
            reason="任务仍可通过收窄 query 继续尝试。",
            provider=self.provider_type,
        )

    def _split_candidate(
        self,
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem,
    ) -> ResearchPlannerCandidate:
        new_title = f"{plan_item.title}：补充证据线索"
        new_intent = f"围绕“{plan_item.intent}”补充更窄的证据线索。"
        return ResearchPlannerCandidate(
            candidate_action=decision.action_type,
            candidate_tool="revise_plan/split_task",
            candidate_plan_ops=[
                ResearchPlanOperation(
                    operation_type=ResearchPlanOperationType.SPLIT_ITEM,
                    target_task_id=plan_item.task_id,
                    new_task_id=f"{plan_item.task_id}-split-{uuid4().hex[:8]}",
                    title=new_title,
                    intent=new_intent,
                    query=f"{plan_item.query} evidence detail",
                    priority=plan_item.priority + 1,
                    reason="当前任务过宽且证据质量不足，插入更窄的补充任务。",
                )
            ],
            confidence=0.72,
            reason="建议拆分当前任务以补足证据缺口。",
            provider=self.provider_type,
        )

    def _reorder_candidate(
        self,
        state: ResearchRuntimeState,
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem,
    ) -> ResearchPlannerCandidate:
        pending = [item for item in state.plan_items if item.task_id not in state.completed_items]
        other_pending = [item for item in pending if item.task_id != plan_item.task_id]
        ordered_ids = [item.task_id for item in other_pending] + [plan_item.task_id]
        return ResearchPlannerCandidate(
            candidate_action=decision.action_type,
            candidate_tool="revise_plan/reorder_priority",
            candidate_plan_ops=[
                ResearchPlanOperation(
                    operation_type=ResearchPlanOperationType.REORDER_ITEMS,
                    target_task_id=plan_item.task_id,
                    ordered_task_ids=ordered_ids,
                    reason="当前任务连续无增量，先推进其他待办任务。",
                )
            ],
            confidence=0.7 if other_pending else 0.4,
            reason="建议重排未完成任务优先级，减少同一任务重复无增量调用。",
            provider=self.provider_type,
        )

    @staticmethod
    def _can_split(plan_item: ResearchPlanItem, evidence: ResearchEvidenceBufferItem | None) -> bool:
        if plan_item.revise_count > 0:
            return False
        if evidence is None:
            return True
        assessment = evidence.evidence_assessment
        if assessment.total_item_count == 0:
            return True
        return assessment.sufficiency_score < 0.45 or not assessment.has_relevant_evidence

    @staticmethod
    def _find_evidence(
        state: ResearchRuntimeState,
        task_id: str,
    ) -> ResearchEvidenceBufferItem | None:
        for item in state.evidence_buffer:
            if item.task_id == task_id:
                return item
        return None
