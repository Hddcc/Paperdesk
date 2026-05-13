"""Decision helpers for the phase-13 single-main-agent research runtime."""

from __future__ import annotations

from app.models import (
    ResearchActionDecision,
    ResearchActionType,
    ResearchEvidenceBufferItem,
    ResearchPlanItem,
    ResearchRequest,
    ResearchRuntimePhase,
    ResearchRuntimeState,
    ResearchToolStrategy,
)


class MainAgentRuntime:
    """Choose the next action for the single-main-agent research loop."""

    max_step_budget: int = 64
    max_no_progress_count: int = 3
    max_same_tool_streak: int = 2
    max_replan_count: int = 2

    @staticmethod
    def should_use_direct_task(request: ResearchRequest) -> bool:
        compact_topic = "".join(request.topic.split())
        return len(compact_topic) <= 8 and not request.notes

    def next_action(self, state: ResearchRuntimeState, request: ResearchRequest | None = None) -> ResearchActionDecision:
        if state.failure_count >= 3:
            return self._decision(
                ResearchActionType.FAIL,
                None,
                "连续工具失败达到上限，停止本轮研究。",
                request=request,
            )
        if state.step_count >= self.max_step_budget:
            return self._decision(
                ResearchActionType.FAIL,
                None,
                "达到全局最大 step 预算，停止以避免失控循环。",
                request=request,
            )
        if state.no_progress_count >= self.max_no_progress_count:
            return self._decision(
                ResearchActionType.FAIL,
                None,
                "连续无增量达到上限，当前没有合理下一步。",
                request=request,
            )
        if not state.plan_items:
            return self._decision(
                ResearchActionType.PLAN,
                None,
                "尚未形成运行时计划，需要先生成候选计划。",
                request=request,
            )

        active_item = self._next_pending_item(state)
        if active_item is None:
            if state.report_id:
                return self._decision(
                    ResearchActionType.FINISH,
                    None,
                    "报告已生成，研究流程可以结束。",
                    request=request,
                )
            return self._decision(
                ResearchActionType.FINALIZE_REPORT,
                None,
                "所有计划项已收束，进入最终报告生成。",
                request=request,
            )

        evidence = self._find_evidence(state, active_item.task_id)
        if self._same_tool_is_stale(state):
            if active_item.revise_count < 1 and state.replan_count < self.max_replan_count:
                return self._decision(
                    ResearchActionType.REVISE_PLAN,
                    active_item.task_id,
                    "同一工具连续调用且没有新增信息，先修正当前计划项。",
                    strategy_id="revise_plan/reorder_priority",
                    request=request,
                )
            return self._decision(
                ResearchActionType.SUMMARIZE_EVIDENCE,
                active_item.task_id,
                "同一工具重复无增量，按当前材料降级收束任务。",
                strategy_id="summarize_evidence/degraded_closeout",
                request=request,
            )
        if not evidence.online_completed:
            return self._decision(
                ResearchActionType.SEARCH_ONLINE,
                active_item.task_id,
                "当前任务缺少在线论文候选，优先补充外部证据。",
                request=request,
            )
        if not evidence.local_completed:
            return self._decision(
                ResearchActionType.SEARCH_LOCAL,
                active_item.task_id,
                "当前任务缺少本地论文库证据，继续检索本地材料。",
                request=request,
            )
        if self._has_sufficient_evidence(evidence):
            return self._decision(
                ResearchActionType.SUMMARIZE_EVIDENCE,
                active_item.task_id,
                "证据相关性和覆盖度已达到任务总结条件。",
                request=request,
            )
        if active_item.revise_count == 0 and state.replan_count < self.max_replan_count:
            return self._decision(
                ResearchActionType.REVISE_PLAN,
                active_item.task_id,
                "检索已完成但证据不足，需要修正 query 和证据需求。",
                strategy_id=self._revise_strategy_for_task(active_item),
                request=request,
            )
        return self._decision(
            ResearchActionType.SUMMARIZE_EVIDENCE,
            active_item.task_id,
            "证据不足但已完成修正尝试，按当前材料降级总结。",
            strategy_id="summarize_evidence/degraded_closeout",
            request=request,
        )

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
    def peek_next_pending_item(state: ResearchRuntimeState) -> ResearchPlanItem | None:
        return MainAgentRuntime._next_pending_item(state)

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
        assessment = evidence.evidence_assessment
        if assessment.total_item_count:
            if assessment.conflict_detected and assessment.sufficiency_score < 0.7:
                return False
            return (
                assessment.has_relevant_evidence
                and assessment.sufficiency_score >= 0.55
                and assessment.relevance_score >= 0.25
            )
        return bool(evidence.paper_records or evidence.evidence_items)

    @staticmethod
    def _decision(
        action: ResearchActionType,
        task_id: str | None,
        reason: str,
        *,
        strategy_id: str | None = None,
        request: ResearchRequest | None = None,
    ) -> ResearchActionDecision:
        strategy = MainAgentRuntime._tool_strategy(action, strategy_id=strategy_id, request=request)
        return ResearchActionDecision(
            action_type=action,
            selected_tool=strategy.strategy_id,
            tool_strategy=strategy,
            reason=reason,
            target_task_id=task_id,
        )

    @staticmethod
    def _tool_strategy(
        action: ResearchActionType,
        *,
        strategy_id: str | None = None,
        request: ResearchRequest | None = None,
    ) -> ResearchToolStrategy:
        selected_strategy = strategy_id or MainAgentRuntime._default_strategy_id(action, request)
        labels = {
            "plan/rule_based_initial": "规则初始规划",
            "search_online/openalex_primary": "OpenAlex 优先检索",
            "search_online/arxiv_primary": "arXiv 优先检索",
            "search_online/mixed_broad_recall": "混合宽召回检索",
            "search_local/vector_recall_default": "默认向量召回",
            "summarize_evidence/task_level_merge": "任务级证据合并",
            "summarize_evidence/degraded_closeout": "证据不足降级收束",
            "revise_plan/rewrite_query": "改写检索 query",
            "revise_plan/split_task": "拆分过宽任务",
            "revise_plan/reorder_priority": "重排待办优先级",
            "finalize_report/report_writer_default": "默认报告生成",
            "finish/runtime_complete": "运行完成",
            "fail/runtime_stop": "运行停止",
        }
        parameters: dict[str, object] = {}
        if action == ResearchActionType.SEARCH_ONLINE and request is not None:
            parameters = {
                "search_provider": request.search_provider,
                "top_k_online": request.top_k_online,
            }
        if action == ResearchActionType.SEARCH_LOCAL and request is not None:
            parameters = {"top_k_local": request.top_k_local}
        return ResearchToolStrategy(
            strategy_id=selected_strategy,
            action_type=action,
            label=labels.get(selected_strategy, selected_strategy),
            parameters=parameters,
            rationale="由主 Agent 在高层动作下选择的具体工具策略。",
        )

    @staticmethod
    def _default_strategy_id(action: ResearchActionType, request: ResearchRequest | None) -> str:
        if action == ResearchActionType.PLAN:
            return "plan/rule_based_initial"
        if action == ResearchActionType.SEARCH_ONLINE:
            provider = (request.search_provider if request is not None else None) or ""
            provider = provider.casefold()
            if provider == "openalex":
                return "search_online/openalex_primary"
            if provider == "arxiv":
                return "search_online/arxiv_primary"
            return "search_online/mixed_broad_recall"
        if action == ResearchActionType.SEARCH_LOCAL:
            return "search_local/vector_recall_default"
        if action == ResearchActionType.SUMMARIZE_EVIDENCE:
            return "summarize_evidence/task_level_merge"
        if action == ResearchActionType.REVISE_PLAN:
            return "revise_plan/rewrite_query"
        if action == ResearchActionType.FINALIZE_REPORT:
            return "finalize_report/report_writer_default"
        if action == ResearchActionType.FINISH:
            return "finish/runtime_complete"
        return "fail/runtime_stop"

    @staticmethod
    def _revise_strategy_for_task(task: ResearchPlanItem) -> str:
        broad_markers = ("同时", "多个", "综合")
        combined_text = f"{task.title} {task.intent} {task.query}"
        if any(marker in combined_text for marker in broad_markers):
            return "revise_plan/split_task"
        return "revise_plan/rewrite_query"

    def _same_tool_is_stale(self, state: ResearchRuntimeState) -> bool:
        if state.same_tool_streak < self.max_same_tool_streak:
            return False
        if state.no_progress_count <= 0:
            return False
        return bool(state.last_tool_signature)
