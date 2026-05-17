"""Minimal plan-and-execute runtime for chat-side knowledge tasks."""

from __future__ import annotations

from typing import Any

from app.models import (
    AgentModeDecision,
    ChatAttachment,
    ChatMessageRequest,
    ChatSession,
    TraceEventType,
)
from app.repositories import RuntimeRepository

from .knowledge_agent_runtime import KnowledgeAgentResult, KnowledgeAgentRuntime
from .message_bus import MessageBus


class KnowledgePlannerRuntime:
    """Create a visible plan trace, then execute through the existing knowledge tools."""

    def __init__(
        self,
        *,
        knowledge_agent_runtime: KnowledgeAgentRuntime,
        runtime_repository: RuntimeRepository,
    ) -> None:
        self.knowledge_agent_runtime = knowledge_agent_runtime
        self.message_bus = MessageBus(runtime_repository)

    def handle(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        decision: AgentModeDecision,
    ) -> KnowledgeAgentResult:
        plan = self._build_plan(request.content)
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="planner_plan_created",
            message="Knowledge planner created a structured plan.",
            payload={
                "mode": decision.mode.value,
                "plan": plan,
                "reason": decision.reason,
            },
        )
        result = self.knowledge_agent_runtime.run_react(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=decision.trace_id,
            runtime_label="planner_react_execution",
        )
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="planner_execution_finished",
            message="Knowledge planner finished execution.",
            payload={
                "action_status": result.action_status,
                "retrieval_status": result.retrieval_status,
                "used_document_count": len(result.used_document_ids),
                "evidence_count": len(result.evidence_items),
            },
        )
        result.agent_trace_id = decision.trace_id
        return result

    @staticmethod
    def _build_plan(content: str) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = [
            {
                "step": 1,
                "goal": "识别任务涉及的论文、标签或范围。",
                "tool": "library.explorer.*",
                "depends_on": [],
                "done_criteria": "形成可执行的论文集合或标签统计。",
            }
        ]
        if any(marker in content for marker in ("无标签", "没有标签", "补标签", "打标签", "添加标签", "归类")):
            plan.append(
                {
                    "step": len(plan) + 1,
                    "goal": "对明确范围内的论文执行非破坏性标签补全。",
                    "tool": "library.operator.assign_category",
                    "depends_on": [1],
                    "done_criteria": "只对后端校验通过的论文追加标签，不覆盖已有标签。",
                }
            )
        if any(marker in content for marker in ("总结", "综述", "概述", "对比", "比较", "分别", "按标签", "按分类")):
            plan.append(
                {
                    "step": len(plan) + 1,
                    "goal": "检索本地证据并生成基于观察结果的回答。",
                    "tool": "evidence.retriever.* -> report.drafter.*",
                    "depends_on": list(range(1, len(plan) + 1)),
                    "done_criteria": "输出带边界说明的总结或分组总结。",
                }
            )
        if len(plan) == 1:
            plan.append(
                {
                    "step": 2,
                    "goal": "按观察结果回答用户问题。",
                    "tool": "final.answer",
                    "depends_on": [1],
                    "done_criteria": "最终回答只引用可用观察结果。",
                }
            )
        return plan

