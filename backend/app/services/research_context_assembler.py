"""Research-specific context assembly and evidence governance."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from app.config import Settings
from app.models import (
    EvidenceItem,
    PaperRecord,
    ResearchCompactedEvidenceItem,
    ResearchContextStage,
    ResearchContextState,
    ResearchEvidenceAssessment,
    ResearchEvidenceBufferItem,
    ResearchPlanItem,
    ResearchRuntimeState,
)

from .context_budget_service import ContextBudgetService


class ResearchContextAssembler:
    """Build the decision context consumed by the research main-agent loop."""

    def __init__(self, *, budget_service: ContextBudgetService, settings: Settings) -> None:
        self.budget_service = budget_service
        self.settings = settings
        self.max_visible_steps = 8

    def refresh(
        self,
        state: ResearchRuntimeState,
        *,
        active_task: ResearchPlanItem | None = None,
    ) -> ResearchContextState:
        for task in state.plan_items:
            evidence = self._find_evidence(state, task.task_id)
            if evidence is not None:
                self._govern_evidence(task, evidence)

        context_text = self.assemble_text(state, active_task=active_task)
        estimated_tokens = self.budget_service.estimate_text(context_text)
        compacted_count = sum(
            item.evidence_assessment.compacted_item_count for item in state.evidence_buffer
        )
        history_compacted = len(state.tool_history) > self.max_visible_steps
        sources = self._sources(state, active_task=active_task)
        stage = self._stage(
            estimated_tokens=estimated_tokens,
            evidence_items_compacted=compacted_count,
            history_compacted=history_compacted,
        )
        last_compacted_at = state.context_state.last_compacted_at
        if stage != ResearchContextStage.NORMAL:
            last_compacted_at = datetime.now(timezone.utc)

        context_state = ResearchContextState(
            stage=stage,
            estimated_tokens=estimated_tokens,
            budget_tokens=self.budget_service.budget_tokens,
            sources=sources,
            last_compacted_at=last_compacted_at,
            active_task_id=active_task.task_id if active_task is not None else None,
            visible_step_count=min(len(state.tool_history), self.max_visible_steps),
            evidence_items_compacted=compacted_count,
            history_compacted=history_compacted,
        )
        state.context_state = context_state
        state.working_summary = self.build_working_summary(state, active_task=active_task)
        return context_state

    def assemble_text(
        self,
        state: ResearchRuntimeState,
        *,
        active_task: ResearchPlanItem | None,
    ) -> str:
        lines = [
            "系统级研究指令：围绕用户研究目标推进，优先使用可引用证据，证据不足时先补材料或降级说明。",
            "项目级规则：保持单主 Agent step loop；保留原始执行日志，但决策上下文只使用摘要和 compact evidence。",
            f"当前 run 目标：{state.goal}",
            f"当前 working summary：{state.working_summary or '尚未形成工作记忆。'}",
        ]
        completed = [item for item in state.plan_items if item.task_id in state.completed_items]
        if completed:
            lines.append("已完成任务摘要：")
            lines.extend(
                f"- {item.title}：{self._clip(item.summary or item.summary_markdown or '已完成。', 220)}"
                for item in completed
            )
        if active_task is not None:
            lines.extend(
                [
                    "当前 active task：",
                    f"- 标题：{active_task.title}",
                    f"- 意图：{active_task.intent}",
                    f"- Query：{active_task.query}",
                    f"- Revise 次数：{active_task.revise_count}",
                ]
            )
        recent_steps = state.tool_history[-self.max_visible_steps :]
        if recent_steps:
            lines.append("近期 step 观察：")
            lines.extend(
                f"- {record.action.value}/{record.status.value}: {self._clip(record.summary, 180)}"
                for record in recent_steps
            )
        active_buffers = (
            [self._find_evidence(state, active_task.task_id)]
            if active_task is not None
            else state.evidence_buffer
        )
        for buffer in [item for item in active_buffers if item is not None]:
            assessment = buffer.evidence_assessment
            lines.append(
                f"证据评估 {buffer.task_id}：sufficiency={assessment.sufficiency_score:.2f}, "
                f"relevance={assessment.relevance_score:.2f}, coverage={','.join(assessment.coverage) or 'none'}, "
                f"conflict={assessment.conflict_detected}"
            )
            visible_evidence = [item for item in buffer.compacted_evidence if item.visible]
            if visible_evidence:
                lines.append("当前 compact evidence：")
                lines.extend(
                    f"- {item.citation} | {item.title} | {item.excerpt}"
                    for item in visible_evidence
                )
        return "\n".join(lines)

    def build_working_summary(
        self,
        state: ResearchRuntimeState,
        *,
        active_task: ResearchPlanItem | None = None,
    ) -> str:
        completed_titles = [
            item.title for item in state.plan_items if item.task_id in state.completed_items
        ]
        pending_titles = [
            item.title for item in state.plan_items if item.task_id not in state.completed_items
        ]
        evidence_notes = []
        gaps = []
        conflicts = []
        for task in state.plan_items:
            evidence = self._find_evidence(state, task.task_id)
            if evidence is None:
                continue
            assessment = evidence.evidence_assessment
            if assessment.total_item_count:
                evidence_notes.append(
                    f"{task.title}：{assessment.relevant_item_count}/{assessment.total_item_count} 条相关证据，"
                    f"覆盖 {','.join(assessment.coverage) or '不足'}"
                )
            if task.task_id not in state.completed_items and assessment.sufficiency_score < 0.55:
                gaps.append(f"{task.title} 仍需补充更相关或更多来源的证据")
            if assessment.conflict_detected:
                conflicts.append(f"{task.title} 存在潜在冲突表述")

        lines = [
            f"已完成 {len(completed_titles)}/{len(state.plan_items)} 个研究任务。",
            f"已完成任务：{'；'.join(completed_titles) if completed_titles else '暂无'}。",
            f"待推进任务：{'；'.join(pending_titles) if pending_titles else '暂无'}。",
        ]
        if active_task is not None:
            lines.append(f"当前决策焦点：{active_task.title}，query={active_task.query}。")
        lines.append(f"证据进展：{'；'.join(evidence_notes) if evidence_notes else '尚无可用证据摘要'}。")
        lines.append(f"证据缺口：{'；'.join(gaps) if gaps else '暂无明显缺口'}。")
        lines.append(f"待核验点：{'；'.join(conflicts) if conflicts else '暂无明显冲突'}。")
        return "\n".join(lines)

    def _govern_evidence(
        self,
        task: ResearchPlanItem,
        evidence: ResearchEvidenceBufferItem,
    ) -> None:
        compacted: list[ResearchCompactedEvidenceItem] = []
        seen: set[str] = set()

        for paper in evidence.paper_records:
            source_key = self._paper_key(paper)
            if source_key in seen:
                continue
            seen.add(source_key)
            excerpt = self._clip(paper.abstract or paper.venue or "在线论文候选，暂无摘要。", 240)
            relevance = self._relevance(task, " ".join([paper.title, paper.abstract or ""]))
            compacted.append(
                ResearchCompactedEvidenceItem(
                    task_id=task.task_id,
                    source_key=source_key,
                    source_type="online_paper",
                    citation=paper.doi or paper.url or paper.title,
                    title=paper.title,
                    excerpt=excerpt,
                    relevance=relevance,
                    coverage=self._coverage(task, " ".join([paper.title, paper.abstract or ""])),
                    potential_conflict=self._has_conflict_signal(excerpt),
                )
            )

        for item in evidence.evidence_items:
            source_key = self._local_key(item)
            if source_key in seen:
                continue
            seen.add(source_key)
            excerpt = self._clip(item.quote or item.snippet or "本地证据命中，暂无摘录。", 240)
            body = " ".join([item.title, item.citation_label, excerpt])
            relevance = self._relevance(task, body, score=item.score)
            compacted.append(
                ResearchCompactedEvidenceItem(
                    task_id=task.task_id,
                    source_key=source_key,
                    source_type="local_document",
                    citation=item.citation_label,
                    title=item.title,
                    page_number=item.page_number,
                    excerpt=excerpt,
                    relevance=relevance,
                    coverage=self._coverage(task, body),
                    potential_conflict=self._has_conflict_signal(excerpt),
                )
            )

        visible_limit = max(self.settings.max_evidence_items, 1)
        compacted.sort(key=self._evidence_rank, reverse=True)
        for index, item in enumerate(compacted):
            item.visible = index < visible_limit

        evidence.compacted_evidence = compacted
        evidence.evidence_assessment = self._assess_evidence(evidence)

    def _assess_evidence(self, evidence: ResearchEvidenceBufferItem) -> ResearchEvidenceAssessment:
        total = len(evidence.compacted_evidence)
        relevant = [item for item in evidence.compacted_evidence if item.relevance == "high"]
        visible = [item for item in evidence.compacted_evidence if item.visible]
        source_types = {item.source_type for item in evidence.compacted_evidence}
        coverage = sorted({label for item in evidence.compacted_evidence for label in item.coverage})
        conflict = any(item.potential_conflict for item in evidence.compacted_evidence)
        paper_count = sum(1 for item in evidence.compacted_evidence if item.source_type == "online_paper")
        local_count = sum(1 for item in evidence.compacted_evidence if item.source_type == "local_document")
        relevance_score = len(relevant) / total if total else 0.0
        diversity_score = min(len(source_types) / 2, 1.0)
        coverage_score = min(len(coverage) / 3, 1.0)
        sufficiency_score = min((len(relevant) / 2) * 0.45 + diversity_score * 0.25 + coverage_score * 0.30, 1.0)
        if total == 0:
            rationale = "尚未检索到可评估证据。"
        elif sufficiency_score >= 0.55:
            rationale = "当前证据已具备基本相关性和覆盖度，可进入总结。"
        else:
            rationale = "当前证据相关性或覆盖度不足，建议补充或修订检索。"
        return ResearchEvidenceAssessment(
            total_item_count=total,
            paper_count=paper_count,
            local_evidence_count=local_count,
            relevant_item_count=len(relevant),
            visible_item_count=len(visible),
            compacted_item_count=max(total - len(visible), 0),
            sufficiency_score=round(sufficiency_score, 3),
            relevance_score=round(relevance_score, 3),
            diversity_score=round(diversity_score, 3),
            coverage=coverage,
            conflict_detected=conflict,
            has_relevant_evidence=bool(relevant),
            rationale=rationale,
        )

    def _stage(
        self,
        *,
        estimated_tokens: int,
        evidence_items_compacted: int,
        history_compacted: bool,
    ) -> ResearchContextStage:
        if estimated_tokens >= self.budget_service.force_tokens:
            return ResearchContextStage.TRUNCATED
        if history_compacted:
            return ResearchContextStage.HISTORY_COMPACTED
        if estimated_tokens >= self.budget_service.warn_tokens or evidence_items_compacted > 0:
            return ResearchContextStage.EVIDENCE_COMPACTED
        return ResearchContextStage.NORMAL

    def _sources(
        self,
        state: ResearchRuntimeState,
        *,
        active_task: ResearchPlanItem | None,
    ) -> list[str]:
        sources = ["research_rules", "run_goal", "working_summary"]
        if active_task is not None:
            sources.append("active_task")
        if state.tool_history:
            sources.append("recent_steps")
        if any(buffer.compacted_evidence for buffer in state.evidence_buffer):
            sources.append("compacted_evidence")
        if state.completed_items:
            sources.append("completed_task_summaries")
        return sources

    @staticmethod
    def _find_evidence(
        state: ResearchRuntimeState,
        task_id: str,
    ) -> ResearchEvidenceBufferItem | None:
        return next((item for item in state.evidence_buffer if item.task_id == task_id), None)

    @staticmethod
    def _paper_key(paper: PaperRecord) -> str:
        if paper.doi:
            return f"doi:{paper.doi.casefold()}"
        if paper.url:
            return f"url:{paper.url.casefold()}"
        return f"title:{ResearchContextAssembler._normalize_text(paper.title)}"

    @staticmethod
    def _local_key(item: EvidenceItem) -> str:
        quote_key = ResearchContextAssembler._normalize_text(item.quote or item.snippet)[:80]
        return ":".join(
            [
                "local",
                item.document_id or item.source_id,
                str(item.page_number or ""),
                quote_key,
            ]
        )

    @staticmethod
    def _relevance(task: ResearchPlanItem, text: str, score: float | None = None) -> str:
        if score is not None and score < 0.25:
            return "low"
        task_terms = ResearchContextAssembler._terms(" ".join([task.title, task.intent, task.query]))
        text_terms = ResearchContextAssembler._terms(text)
        if not task_terms or not text_terms:
            return "unknown"
        overlap = len(task_terms & text_terms) / max(len(task_terms), 1)
        if overlap >= 0.18 or (score is not None and score >= 0.55):
            return "high"
        if overlap >= 0.08 or (score is not None and score >= 0.35):
            return "medium"
        return "low"

    @staticmethod
    def _coverage(task: ResearchPlanItem, text: str) -> list[str]:
        haystack = ResearchContextAssembler._normalize_text(" ".join([task.title, task.intent, task.query, text]))
        labels = []
        mapping = {
            "method": ("方法", "method", "approach", "framework", "architecture", "模型"),
            "evaluation": ("评估", "评价", "evaluation", "benchmark", "metric", "指标"),
            "application": ("应用", "application", "case", "场景", "domain"),
            "limitation": ("局限", "不足", "limitation", "risk", "challenge", "问题"),
            "dataset": ("数据", "dataset", "corpus", "样本", "基准"),
        }
        for label, keywords in mapping.items():
            if any(keyword in haystack for keyword in keywords):
                labels.append(label)
        if not labels:
            labels.append("general")
        return labels

    @staticmethod
    def _has_conflict_signal(text: str) -> bool:
        normalized = ResearchContextAssembler._normalize_text(text)
        markers = ("however", "but", "conflict", "contradict", "局限", "相反", "但是", "然而", "不足")
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _evidence_rank(item: ResearchCompactedEvidenceItem) -> tuple[int, int, int]:
        relevance_rank = {"high": 3, "medium": 2, "unknown": 1, "low": 0}.get(item.relevance, 0)
        source_rank = 1 if item.source_type == "local_document" else 0
        return relevance_rank, len(item.coverage), source_rank

    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = ResearchContextAssembler._normalize_text(text)
        ascii_terms = {term for term in re.findall(r"[a-z0-9]{3,}", normalized) if len(term) >= 3}
        cjk_terms = {term for term in re.findall(r"[\u4e00-\u9fff]{2,}", normalized)}
        return ascii_terms | cjk_terms

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 1]}…"
