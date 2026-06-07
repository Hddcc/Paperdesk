"""Structured self-reflection runtime for chat-side answer correction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

from openai import OpenAI

from app.models import (
    AgentModeDecision,
    AgentRunMode,
    ChatAttachment,
    ChatMessage,
    ChatMessageRequest,
    ChatSession,
    ReflectionImprovementAction,
    ReflectionResult,
    TraceEventType,
)
from app.repositories import RuntimeRepository
from app.services.chat_memory_service import ChatMemoryService

from .knowledge_agent_runtime import KnowledgeAgentResult, KnowledgeAgentRuntime
from .message_bus import MessageBus


_REFLECTION_META: dict[int, dict[str, Any]] = {}


class ReflectionRuntime:
    """Review answers, score quality, and coordinate a bounded improvement turn."""

    _DESTRUCTIVE_MARKERS = ("删除", "移除", "清空", "删掉", "delete", "remove", "clear")
    _DESTRUCTIVE_TARGETS = ("论文", "文档", "分类", "标签", "会话", "记录")
    _WRITE_MARKERS = (
        "新建",
        "新增",
        "创建",
        "补标签",
        "打标签",
        "添加标签",
        "设置分类",
        "归类",
        "加标签",
        "加个",
        "加一个",
        "加上",
        "补上",
        "换成",
        "换为",
        "替换为",
        "替换成",
        "改成",
        "改为",
        "改名为",
        "重命名",
        "重命名为",
        "rename",
        "replace",
    )
    _CATEGORY_MARKERS = ("标签", "分类", "有标签", "无标签", "未分类", "归类")
    _RAG_MARKERS = ("总结", "综述", "概述", "对比", "比较", "引用", "证据", "结合论文", "基于论文")
    _LIBRARY_MARKERS = ("论文库", "文献库", "本地论文", "库里", "库内", "论文", "这几篇", "这些论文")
    _FAILED_STATUSES = {"failed", "degraded", "validation_failed", "retryable_error", "non_retryable_error"}
    _NON_RETRY_STATUSES = {"confirmation_required", "needs_clarification"}
    _REFLECTION_TOOL_TYPES = {
        "none",
        "library_stats",
        "metadata",
        "category",
        "tag_operator",
        "rag",
        "document_search",
        "operator_verify",
    }

    def __init__(
        self,
        *,
        knowledge_agent_runtime: KnowledgeAgentRuntime,
        runtime_repository: RuntimeRepository,
        memory_service: ChatMemoryService | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        persist_lessons_to_memory: bool = False,
    ) -> None:
        self.knowledge_agent_runtime = knowledge_agent_runtime
        self.runtime_repository = runtime_repository
        self.memory_service = memory_service
        self.model = model or ""
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.persist_lessons_to_memory = persist_lessons_to_memory
        self.message_bus = MessageBus(runtime_repository)

    def handle(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        history: list[ChatMessage],
        decision: AgentModeDecision,
    ) -> KnowledgeAgentResult:
        previous = self._last_assistant(history)
        previous_user = self._last_user_before_last_assistant(history)
        previous_traces = (
            self.runtime_repository.list_traces(previous.agent_trace_id)
            if previous is not None and previous.agent_trace_id
            else []
        )
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="reflection_started",
            message="Reflection runtime started.",
            payload={
                "previous_message_id": previous.id if previous else None,
                "previous_trace_id": previous.agent_trace_id if previous else None,
                "previous_trace_event_count": len(previous_traces),
                "reason": decision.reason,
            },
        )

        reviewed = KnowledgeAgentResult(
            content=previous.content if previous else "",
            retrieval_status=previous.retrieval_status or "skipped" if previous else "skipped",
            citations=previous.citations if previous else [],
            used_document_ids=previous.used_document_ids if previous else [],
            action_status=previous.action_status if previous else None,
            agent_trace_id=previous.agent_trace_id if previous else None,
            library_mutated=False,
        )
        original_goal = previous_user.content if previous_user is not None else request.content
        reflection = self._evaluate(
            user_goal=original_goal,
            final_answer=reviewed.content,
            traces=previous_traces,
            result=reviewed,
            mode="REFLECTION",
            user_feedback=request.content,
        )
        self._record_reflection_outputs(
            session=session,
            trace_id=decision.trace_id,
            source_trace_id=previous.agent_trace_id if previous else None,
            user_goal=original_goal,
            reviewed_action_status=reviewed.action_status,
            reflection=reflection,
        )

        inherited_document_ids = selected_document_ids or (previous.used_document_ids if previous else [])
        should_recheck = self._should_recheck_with_tools(request.content, inherited_document_ids, previous)
        if (
            should_recheck or reflection.should_retry
        ) and not self._is_destructive_intent(original_goal + "\n" + request.content):
            result = self._run_improvement(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=inherited_document_ids,
                decision=decision,
                original_result=reviewed,
                reflection=reflection,
                explicit_feedback=request.content,
            )
        else:
            result = KnowledgeAgentResult(
                content=self._direct_reflection_answer(request.content, previous, reflection),
                retrieval_status="skipped",
                action_status="reflection_completed",
                agent_trace_id=decision.trace_id,
                library_mutated=False,
            )

        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="reflection_finished",
            message="Reflection runtime finished.",
            payload={
                "action_status": result.action_status,
                "retrieval_status": result.retrieval_status,
                "used_document_count": len(result.used_document_ids),
                "evidence_count": len(result.evidence_items),
                "overall_score": reflection.overall_score,
                "retry_attempted": result is not reviewed and result.action_status != "reflection_completed",
            },
        )
        result.agent_trace_id = decision.trace_id
        return result

    def review_agent_result(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        decision: AgentModeDecision,
        result: KnowledgeAgentResult,
    ) -> KnowledgeAgentResult:
        """Score a REACT/PLANNER result and run at most one improvement pass."""

        if decision.mode not in {AgentRunMode.REACT, AgentRunMode.PLANNER}:
            return result
        try:
            traces = self.runtime_repository.list_traces(decision.trace_id)
            self.message_bus.append_trace(
                run_id=decision.trace_id,
                task_id=None,
                trace_type=TraceEventType.CONTROL,
                status="reflection_review_started",
                message="Self-reflection review started.",
                payload={
                    "mode": decision.mode.value,
                    "reviewed_action_status": result.action_status,
                    "trace_event_count": len(traces),
                },
            )
            reflection = self._evaluate(
                user_goal=request.content,
                final_answer=result.content,
                traces=traces,
                result=result,
                mode=decision.mode.value,
            )
            self._record_reflection_outputs(
                session=session,
                trace_id=decision.trace_id,
                source_trace_id=result.agent_trace_id,
                user_goal=request.content,
                reviewed_action_status=result.action_status,
                reflection=reflection,
            )
            if self._should_run_improvement(
                user_goal=request.content,
                result=result,
                reflection=reflection,
            ):
                return self._run_improvement(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    decision=decision,
                    original_result=result,
                    reflection=reflection,
                )
        except Exception as exc:
            self.message_bus.append_trace(
                run_id=decision.trace_id,
                task_id=None,
                trace_type=TraceEventType.CONTROL,
                status="reflection_review_failed",
                message="Self-reflection review failed.",
                payload={"error": str(exc)},
            )
        return result

    @staticmethod
    def _last_assistant(history: list[ChatMessage]) -> ChatMessage | None:
        for message in reversed(history):
            if message.role == "assistant":
                return message
        return None

    @staticmethod
    def _last_user_before_last_assistant(history: list[ChatMessage]) -> ChatMessage | None:
        seen_assistant = False
        for message in reversed(history):
            if message.role == "assistant" and not seen_assistant:
                seen_assistant = True
                continue
            if seen_assistant and message.role == "user":
                return message
        return None

    @staticmethod
    def _should_recheck_with_tools(
        content: str,
        selected_document_ids: list[str],
        previous: ChatMessage | None,
    ) -> bool:
        """Guardrail/fallback: explicit document grounding must re-enter tools."""

        if selected_document_ids:
            return True
        if previous is not None and previous.used_document_ids:
            return True
        return any(marker in content for marker in ("论文库", "库里", "标签", "分类", "论文", "为什么没查", "为什么没有查"))

    def _evaluate(
        self,
        *,
        user_goal: str,
        final_answer: str,
        traces: list[Any],
        result: KnowledgeAgentResult,
        mode: str,
        user_feedback: str | None = None,
    ) -> ReflectionResult:
        llm_result = self._evaluate_with_llm(
            user_goal=user_goal,
            final_answer=final_answer,
            traces=traces,
            result=result,
            mode=mode,
            user_feedback=user_feedback,
        )
        if llm_result is not None:
            return llm_result
        return self._fallback_evaluate(
            user_goal=user_goal,
            final_answer=final_answer,
            traces=traces,
            result=result,
            mode=mode,
            user_feedback=user_feedback,
        )

    def _evaluate_with_llm(
        self,
        *,
        user_goal: str,
        final_answer: str,
        traces: list[Any],
        result: KnowledgeAgentResult,
        mode: str,
        user_feedback: str | None,
    ) -> ReflectionResult | None:
        if not self.api_key:
            return None
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 Self-Reflection 评分器，只输出 JSON。"
                            "你评估可审计结果，不输出隐藏推理。分数为 1-10。"
                            "如果 overall_score < 6，should_retry 必须为 true；否则为 false。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "mode": mode,
                                "user_goal": user_goal,
                                "user_feedback": user_feedback,
                                "final_answer": final_answer[:3000],
                                "result_status": {
                                    "retrieval_status": result.retrieval_status,
                                    "action_status": result.action_status,
                                    "citation_count": len(result.citations),
                                    "used_document_count": len(result.used_document_ids),
                                    "evidence_count": len(result.evidence_items),
                                },
                                "trace_digest": self._trace_digest(traces),
                                "available_tool_types": sorted(self._REFLECTION_TOOL_TYPES),
                                "evaluation_contract": {
                                    "llm_role": "judge whether the current answer satisfied the user's actual intent and whether another tool observation is needed",
                                    "database_truth_source": "trace observations and result fields only",
                                    "fallback_note": "keyword fallback is used only when this JSON evaluation is unavailable or invalid",
                                },
                                "output_schema": {
                                    "overall_score": 8,
                                    "intent_score": 8,
                                    "tool_score": 8,
                                    "evidence_score": 8,
                                    "answer_score": 8,
                                    "completion_score": 8,
                                    "risk_level": "safe|read_only|write|destructive|critical",
                                    "detected_issue_type": "wrong_answer|missing_evidence|wrong_tool|incomplete_write|format_pollution|status_only_answer|step_missing|parameter_pollution|operation_level_mismatch|none",
                                    "needs_tool_recheck": False,
                                    "needed_tool_type": "none|library_stats|metadata|category|tag_operator|rag|document_search|operator_verify",
                                    "issues": ["visible issue"],
                                    "improvement_actions": [
                                        {
                                            "type": "call_tool|rewrite_answer|replan|record_lesson",
                                            "tool": "optional tool name",
                                            "args": {},
                                            "reason": "visible reason",
                                        }
                                    ],
                                    "should_retry": False,
                                    "memory_lessons": ["reusable lesson only"],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
        except Exception:
            return None
        text = self._extract_message_text(response)
        if not text:
            return None
        payload = self._extract_json_payload(text)
        if not isinstance(payload, dict):
            return None
        try:
            return self._reflection_result_from_payload(payload, source="llm")
        except Exception:
            return None

    def _fallback_evaluate(
        self,
        *,
        user_goal: str,
        final_answer: str,
        traces: list[Any],
        result: KnowledgeAgentResult,
        mode: str,
        user_feedback: str | None,
    ) -> ReflectionResult:
        """Conservative fallback only; normal semantic retry decisions come from the LLM evaluator."""

        scores = {
            "intent": 8,
            "tool": 8,
            "evidence": 8,
            "answer": 8,
            "completion": 8,
        }
        issues: list[str] = []
        actions: list[ReflectionImprovementAction] = []
        lessons: list[str] = []
        tools = self._tool_sequence(traces)
        statuses = [str(getattr(trace, "status", "") or "") for trace in traces]
        has_observation = any(status == "react_observation" for status in statuses)
        has_report_observation = any(tool in {"report.drafter.write", "report.drafter.write_by_category"} for tool in tools)
        has_failure = (
            result.retrieval_status == "degraded"
            or (result.action_status or "") in self._FAILED_STATUSES
            or any(status in self._FAILED_STATUSES for status in statuses)
        )
        waiting_for_user = result.action_status in self._NON_RETRY_STATUSES
        needs_observation = self._requires_tool_observation(user_goal, result, mode)
        needs_rag = self._requires_rag_evidence(user_goal)
        has_category_entity_observation = self._has_category_entity_observation(traces)
        needs_category = (
            any(marker in user_goal for marker in self._CATEGORY_MARKERS)
            and not self._has_report_observation_with_answer(traces)
            and not has_category_entity_observation
        )
        has_write = any(tool.startswith("library.operator.") for tool in tools) or any(
            marker in user_goal for marker in self._WRITE_MARKERS
        )

        if user_feedback:
            scores["intent"] = min(scores["intent"], 6)
            issues.append("用户显式指出上一轮回答需要重新检查。")

        if needs_observation and not has_observation and not waiting_for_user:
            scores["tool"] = min(scores["tool"], 4)
            scores["evidence"] = min(scores["evidence"], 4)
            scores["answer"] = min(scores["answer"], 5)
            scores["completion"] = min(scores["completion"], 5)
            issues.append("需要知识库或工具 Observation 支撑，但 trace 中没有可用观察结果。")
            actions.append(
                ReflectionImprovementAction(
                    type="call_tool",
                    tool="library.explorer.stats",
                    args={},
                    reason="先读取真实数据库状态，再修正回答。",
                )
            )
            lessons.append("需要回答论文库事实时，必须先调用真实数据库或RAG工具取得 Observation。")

        if has_failure:
            scores["tool"] = min(scores["tool"], 4)
            scores["evidence"] = min(scores["evidence"], 4)
            scores["answer"] = min(scores["answer"], 5)
            scores["completion"] = min(scores["completion"], 5)
            issues.append("工具链失败或检索降级，回答可靠性不足。")
            actions.append(
                ReflectionImprovementAction(
                    type="call_tool",
                    tool="evidence.retriever.search",
                    args={"question": user_goal},
                    reason="补充一次只读检索以恢复证据覆盖。",
                )
            )
            lessons.append("工具链失败或RAG降级时，应明确证据边界，并优先补充只读检索。")

        if result.action_status in self._NON_RETRY_STATUSES:
            scores["completion"] = min(scores["completion"], 6)
            issues.append("当前任务需要用户补充信息或确认，不能自动重试。")

        if needs_category and not waiting_for_user and not any(
            tool in {"library.explorer.category_stats", "library.explorer.document_categories"}
            for tool in tools
        ):
            scores["tool"] = min(scores["tool"], 5)
            scores["evidence"] = min(scores["evidence"], 5)
            issues.append("标签/分类问题缺少分类关联读取工具支撑。")
            actions.append(
                ReflectionImprovementAction(
                    type="call_tool",
                    tool="library.explorer.category_stats",
                    args={},
                    reason="读取真实标签/分类关联后再回答。",
                )
            )
            lessons.append("用户询问标签或分类时，必须读取真实分类关联表，不能只依赖论文总数或文件名。")

        if needs_rag and not result.evidence_items and not waiting_for_user and not has_report_observation:
            scores["evidence"] = min(scores["evidence"], 5 if has_observation else 4)
            scores["answer"] = min(scores["answer"], 6)
            issues.append("总结、综述或引用类任务缺少 RAG 证据覆盖。")
            actions.append(
                ReflectionImprovementAction(
                    type="call_tool",
                    tool="evidence.retriever.search",
                    args={"question": user_goal, "document_ids": result.used_document_ids},
                    reason="补充正文证据后再生成可靠回答。",
                )
            )
            lessons.append("复杂综述或引用任务应优先检索RAG证据，并在证据不足时说明边界。")

        if has_write and not self._has_post_write_read(tools):
            scores["evidence"] = min(scores["evidence"], 6)
            scores["completion"] = min(scores["completion"], 6)
            issues.append("写操作后缺少二次读取校验。")
            lessons.append("写操作后必须二次读取数据库确认结果，避免成功数量与实际状态不一致。")

        if mode == "PLANNER" and self._looks_complex(user_goal) and len([tool for tool in tools if tool != "final.answer"]) <= 1:
            scores["completion"] = min(scores["completion"], 5)
            scores["tool"] = min(scores["tool"], 5)
            issues.append("复杂多步骤任务执行步骤过少，可能没有完整完成。")
            actions.append(
                ReflectionImprovementAction(
                    type="replan",
                    args={"question": user_goal},
                    reason="重新规划后补齐缺失步骤。",
                )
            )
            lessons.append("复杂复合任务应优先走 Plan-and-Execute，并在执行后核对每个子目标。")

        if not final_answer.strip():
            scores["answer"] = min(scores["answer"], 4)
            scores["completion"] = min(scores["completion"], 4)
            issues.append("最终回答为空。")

        overall = round(sum(scores.values()) / len(scores))
        reflection = ReflectionResult(
            overall_score=self._clamp_score(overall),
            intent_score=self._clamp_score(scores["intent"]),
            tool_score=self._clamp_score(scores["tool"]),
            evidence_score=self._clamp_score(scores["evidence"]),
            answer_score=self._clamp_score(scores["answer"]),
            completion_score=self._clamp_score(scores["completion"]),
            issues=issues,
            improvement_actions=self._dedupe_actions(actions),
            should_retry=False,
            memory_lessons=self._dedupe_strings(lessons),
        )
        return self._with_reflection_meta(
            self._normalize_reflection_result(reflection),
            {
                "source": "fallback_rule",
                "risk_level": self._fallback_risk_level(user_goal, result),
                "needs_tool_recheck": any(action.type == "call_tool" for action in actions),
                "needed_tool_type": self._fallback_needed_tool_type(actions, user_goal),
                "schema_valid": True,
            },
        )

    def _reflection_result_from_payload(self, payload: dict[str, Any], *, source: str) -> ReflectionResult:
        normalized_payload = dict(payload)
        if "overall_score" not in normalized_payload:
            raise ValueError("reflection overall_score is required")
        overall_score = self._clamp_score(normalized_payload.get("overall_score"))
        for field_name in (
            "overall_score",
            "intent_score",
            "tool_score",
            "evidence_score",
            "answer_score",
            "completion_score",
        ):
            normalized_payload[field_name] = self._clamp_score(normalized_payload.get(field_name, overall_score))
        needed_tool_type = self._normalize_needed_tool_type(normalized_payload.get("needed_tool_type"))
        risk_level = self._normalize_risk_level(normalized_payload.get("risk_level"))
        detected_issue_type = self._normalize_issue_type(normalized_payload.get("detected_issue_type"))
        needs_tool_recheck = self._coerce_bool(normalized_payload.get("needs_tool_recheck"))
        reflection = self._normalize_reflection_result(ReflectionResult.model_validate(normalized_payload))
        return self._with_reflection_meta(
            reflection,
            {
                "source": source,
                "risk_level": risk_level,
                "detected_issue_type": detected_issue_type,
                "needs_tool_recheck": needs_tool_recheck,
                "needed_tool_type": needed_tool_type,
                "schema_valid": True,
            },
        )

    def _should_run_improvement(
        self,
        *,
        user_goal: str,
        result: KnowledgeAgentResult,
        reflection: ReflectionResult,
    ) -> bool:
        meta = self._reflection_meta(reflection)
        semantic_retry = bool(reflection.should_retry or meta.get("needs_tool_recheck"))
        if not semantic_retry:
            return False
        if self._is_destructive_intent(user_goal):
            return False
        if meta.get("risk_level") == "destructive":
            return False
        if result.library_mutated:
            return False
        if result.action_status in self._NON_RETRY_STATUSES:
            return False
        return True

    def _run_improvement(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        decision: AgentModeDecision,
        original_result: KnowledgeAgentResult,
        reflection: ReflectionResult,
        explicit_feedback: str | None = None,
    ) -> KnowledgeAgentResult:
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="reflection_improvement_started",
            message="Self-reflection improvement started.",
            payload={
                "overall_score": reflection.overall_score,
                "issue_count": len(reflection.issues),
                "action_count": len(reflection.improvement_actions),
            },
        )
        try:
            reflected_request = ChatMessageRequest(
                content=self._improvement_prompt(request.content, original_result, reflection, explicit_feedback),
                attachments=request.attachments,
                selected_document_ids=selected_document_ids,
            )
            improved = self.knowledge_agent_runtime.run_react(
                session=session,
                request=reflected_request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
                trace_id=decision.trace_id,
                runtime_label="reflection_improvement_execution",
            )
            if not self._is_improved_result(original_result, improved):
                self.message_bus.append_trace(
                    run_id=decision.trace_id,
                    task_id=None,
                    trace_type=TraceEventType.MERGE,
                    status="reflection_improvement_discarded",
                    message="Self-reflection improvement did not improve the result.",
                    payload={
                        "original_retrieval_status": original_result.retrieval_status,
                        "improved_retrieval_status": improved.retrieval_status,
                        "original_action_status": original_result.action_status,
                        "improved_action_status": improved.action_status,
                    },
                )
                if explicit_feedback:
                    return self._fallback_improvement_result(
                        decision=decision,
                        original_result=original_result,
                        reflection=reflection,
                        explicit_feedback=explicit_feedback,
                    )
                return original_result
            improved.agent_trace_id = decision.trace_id
            self.message_bus.append_trace(
                run_id=decision.trace_id,
                task_id=None,
                trace_type=TraceEventType.MERGE,
                status="reflection_improvement_finished",
                message="Self-reflection improvement finished.",
                payload={
                    "action_status": improved.action_status,
                    "retrieval_status": improved.retrieval_status,
                    "used_document_count": len(improved.used_document_ids),
                    "evidence_count": len(improved.evidence_items),
                },
            )
            return improved
        except Exception as exc:
            self.message_bus.append_trace(
                run_id=decision.trace_id,
                task_id=None,
                trace_type=TraceEventType.CONTROL,
                status="reflection_improvement_failed",
                message="Self-reflection improvement failed.",
                payload={"error": str(exc)},
            )
            if explicit_feedback:
                return self._fallback_improvement_result(
                    decision=decision,
                    original_result=original_result,
                    reflection=reflection,
                    explicit_feedback=explicit_feedback,
                )
            return original_result

    def _is_improved_result(
        self,
        original: KnowledgeAgentResult,
        improved: KnowledgeAgentResult,
    ) -> bool:
        if (
            improved.evidence_items
            and self.knowledge_agent_runtime is not None
            and self.knowledge_agent_runtime.is_status_only_answer(improved.content)
        ):
            return False
        if improved.retrieval_status == "ready" and original.retrieval_status != "ready":
            return True
        if len(improved.evidence_items) > len(original.evidence_items):
            return True
        if original.action_status in self._FAILED_STATUSES and improved.action_status not in (
            self._FAILED_STATUSES | self._NON_RETRY_STATUSES
        ):
            return True
        if not original.content.strip() and bool(improved.content.strip()):
            return True
        if original.retrieval_status == "skipped" and improved.used_document_ids:
            return True
        if improved.library_mutated and not original.library_mutated:
            return True
        return False

    @staticmethod
    def _fallback_improvement_result(
        *,
        decision: AgentModeDecision,
        original_result: KnowledgeAgentResult,
        reflection: ReflectionResult,
        explicit_feedback: str,
    ) -> KnowledgeAgentResult:
        issue_text = "；".join(reflection.issues[:2]) if reflection.issues else "补充核对没有得到更可靠证据"
        return KnowledgeAgentResult(
            content="\n".join(
                [
                    "我检查了上一轮回答，但这次补充核对没有得到更可靠的新证据。",
                    f"你的反馈是：{explicit_feedback.strip()}",
                    f"自检发现：{issue_text}。",
                    "我不会把没有证据支撑的改写覆盖原回答；请给出具体论文、标签或问题范围后，我会重新查证。",
                ]
            ),
            retrieval_status=original_result.retrieval_status,
            warning=original_result.warning,
            citations=original_result.citations,
            used_document_ids=original_result.used_document_ids,
            evidence_items=original_result.evidence_items,
            action_status="reflection_completed",
            agent_trace_id=decision.trace_id,
            library_mutated=original_result.library_mutated,
        )

    def _record_reflection_outputs(
        self,
        *,
        session: ChatSession,
        trace_id: str,
        source_trace_id: str | None,
        user_goal: str,
        reviewed_action_status: str | None,
        reflection: ReflectionResult,
    ) -> None:
        payload = {
            "source_trace_id": source_trace_id,
            "reviewed_action_status": reviewed_action_status,
            "reflection_result": reflection.model_dump(mode="json"),
            "evaluator": self._reflection_meta(reflection),
        }
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="reflection_result_created",
            message="Structured self-reflection result created.",
            payload=payload,
        )
        self._append_reflection_record(
            session=session,
            trace_id=trace_id,
            source_trace_id=source_trace_id,
            user_goal=user_goal,
            reflection=reflection,
        )
        if self.memory_service is not None:
            self.memory_service.record_reflection_lessons(
                session_id=session.id,
                trace_id=trace_id,
                lessons=reflection.memory_lessons,
                persist_long_term=self.persist_lessons_to_memory,
            )

    def _append_reflection_record(
        self,
        *,
        session: ChatSession,
        trace_id: str,
        source_trace_id: str | None,
        user_goal: str,
        reflection: ReflectionResult,
    ) -> None:
        try:
            self.knowledge_agent_runtime.file_store.initialize_session(session.id, session.title)
            path = self.knowledge_agent_runtime.file_store.get_session_dir(session.id) / "react_reflections.jsonl"
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": "self_reflection",
                "trace_id": trace_id,
                "source_trace_id": source_trace_id,
                "user_goal": user_goal[:240],
                "overall_score": reflection.overall_score,
                "issues": reflection.issues,
                "should_retry": reflection.should_retry,
                "memory_lessons": reflection.memory_lessons,
                "reflection_result": reflection.model_dump(mode="json"),
                "evaluator": self._reflection_meta(reflection),
            }
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _improvement_prompt(
        self,
        content: str,
        original_result: KnowledgeAgentResult,
        reflection: ReflectionResult,
        explicit_feedback: str | None,
    ) -> str:
        issue_lines = "\n".join(f"- {item}" for item in reflection.issues) or "- 未给出具体问题。"
        action_lines = "\n".join(
            f"- {action.type}: {action.tool or 'n/a'} {json.dumps(action.args, ensure_ascii=False)}"
            for action in reflection.improvement_actions
        ) or "- 重新核对工具和证据后修正回答。"
        return "\n".join(
            [
                content.strip(),
                "",
                "系统自检发现上一版回答质量不足，请只通过现有知识库工具补充核对并修正回答。",
                "不得执行删除、清空、移除等破坏性操作；如需要破坏性操作，必须要求用户确认。",
                "自检问题：",
                issue_lines,
                "建议动作：",
                action_lines,
                "上一版回答摘录：",
                original_result.content[:1200],
                f"用户纠错反馈：{explicit_feedback.strip()}" if explicit_feedback else "",
            ]
        ).strip()

    @staticmethod
    def _direct_reflection_answer(
        content: str,
        previous: ChatMessage | None,
        reflection: ReflectionResult,
    ) -> str:
        if previous is None:
            return "我没有找到上一轮可反思的回答。本轮我会先按你的最新要求重新组织回答。"
        issue_text = "；".join(reflection.issues[:2]) if reflection.issues else "没有发现明确的工具或证据缺口"
        return "\n".join(
            [
                "我重新看了上一轮回答。",
                f"你的反馈是：{content.strip()}",
                f"自检结论：{issue_text}。",
                "如果问题需要库内证据，我会优先重新查论文库、标签或 RAG 证据后再回答。",
            ]
        )

    def _requires_tool_observation(self, user_goal: str, result: KnowledgeAgentResult, mode: str) -> bool:
        if mode in {"REACT", "PLANNER"}:
            return True
        if result.used_document_ids or result.evidence_items:
            return True
        return any(marker in user_goal for marker in self._LIBRARY_MARKERS + self._CATEGORY_MARKERS)

    def _requires_rag_evidence(self, user_goal: str) -> bool:
        return any(marker in user_goal for marker in self._RAG_MARKERS)

    @staticmethod
    def _looks_complex(user_goal: str) -> bool:
        markers = ("先", "再", "然后", "之后", "最后", "分别", "按标签", "按分类", "每类")
        return sum(1 for marker in markers if marker in user_goal) >= 2

    @staticmethod
    def _has_post_write_read(tools: list[str]) -> bool:
        read_tools = {
            "library.explorer.stats",
            "library.explorer.category_stats",
            "library.explorer.document_categories",
            "library.explorer.find_documents",
        }
        write_seen = False
        for tool in tools:
            if tool.startswith("library.operator."):
                write_seen = True
                continue
            if write_seen and tool in read_tools:
                return True
        return False

    def _is_destructive_intent(self, content: str) -> bool:
        lowered = content.casefold()
        return any(marker in lowered for marker in self._DESTRUCTIVE_MARKERS) and any(
            target in content for target in self._DESTRUCTIVE_TARGETS
        )

    @staticmethod
    def _tool_sequence(traces: list[Any]) -> list[str]:
        tools: list[str] = []
        for trace in traces:
            payload = getattr(trace, "payload", {}) or {}
            tool = payload.get("tool") if isinstance(payload, dict) else None
            if isinstance(tool, str) and tool:
                tools.append(tool)
        return tools

    @staticmethod
    def _has_report_observation_with_answer(traces: list[Any]) -> bool:
        for trace in traces:
            payload = getattr(trace, "payload", {}) or {}
            if not isinstance(payload, dict):
                continue
            tool = payload.get("tool")
            if tool not in {"report.drafter.write", "report.drafter.write_by_category"}:
                continue
            observation_payload = payload.get("payload")
            if isinstance(observation_payload, dict) and observation_payload.get("answer"):
                return True
        return False

    @staticmethod
    def _has_category_entity_observation(traces: list[Any]) -> bool:
        """Return true when a tool already resolved a tag/category entity from the real library."""

        for trace in traces:
            payload = getattr(trace, "payload", {}) or {}
            if not isinstance(payload, dict):
                continue
            if payload.get("tool") != "library.explorer.find_documents":
                continue
            observation_payload = payload.get("payload")
            if not isinstance(observation_payload, dict):
                continue
            if (
                observation_payload.get("category_lookup")
                or observation_payload.get("category_names")
                or observation_payload.get("category_name")
            ):
                return True
        return False

    @staticmethod
    def _trace_digest(traces: list[Any]) -> list[dict[str, Any]]:
        digest: list[dict[str, Any]] = []
        for trace in traces[-20:]:
            payload = getattr(trace, "payload", {}) or {}
            item: dict[str, Any] = {
                "status": getattr(trace, "status", ""),
                "message": getattr(trace, "message", ""),
            }
            if isinstance(payload, dict):
                item["tool"] = payload.get("tool")
                item["observation_status"] = payload.get("status")
                item["summary"] = payload.get("summary")
                item["mode"] = payload.get("mode")
            digest.append(item)
        return digest

    def _normalize_reflection_result(self, reflection: ReflectionResult) -> ReflectionResult:
        overall_score = self._clamp_score(reflection.overall_score)
        updates = {
            "overall_score": overall_score,
            "intent_score": self._clamp_score(reflection.intent_score),
            "tool_score": self._clamp_score(reflection.tool_score),
            "evidence_score": self._clamp_score(reflection.evidence_score),
            "answer_score": self._clamp_score(reflection.answer_score),
            "completion_score": self._clamp_score(reflection.completion_score),
            "should_retry": bool(reflection.should_retry or overall_score < 6),
            "issues": self._dedupe_strings(reflection.issues)[:8],
            "memory_lessons": self._dedupe_strings(reflection.memory_lessons)[:5],
            "improvement_actions": self._dedupe_actions(reflection.improvement_actions)[:5],
        }
        return reflection.model_copy(update=updates)

    @staticmethod
    def _with_reflection_meta(reflection: ReflectionResult, meta: dict[str, Any]) -> ReflectionResult:
        _REFLECTION_META[id(reflection)] = meta
        return reflection

    @staticmethod
    def _reflection_meta(reflection: ReflectionResult) -> dict[str, Any]:
        return _REFLECTION_META.get(
            id(reflection),
            {
                "source": "unknown",
                "risk_level": "safe",
                "detected_issue_type": "none",
                "needs_tool_recheck": False,
                "needed_tool_type": "none",
                "schema_valid": True,
            },
        )

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        risk = str(value or "").casefold().strip()
        if risk in {"safe", "read_only", "write", "destructive"}:
            return risk
        return "safe"

    def _normalize_needed_tool_type(self, value: Any) -> str:
        tool_type = str(value or "").casefold().strip()
        if tool_type in self._REFLECTION_TOOL_TYPES:
            return tool_type
        return "none"

    @staticmethod
    def _normalize_issue_type(value: Any) -> str:
        issue_type = str(value or "").casefold().strip()
        allowed = {
            "wrong_answer",
            "missing_evidence",
            "wrong_tool",
            "incomplete_write",
            "format_pollution",
            "status_only_answer",
            "none",
        }
        return issue_type if issue_type in allowed else "none"

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.casefold().strip() in {"true", "1", "yes", "y"}
        return False

    def _fallback_risk_level(self, user_goal: str, result: KnowledgeAgentResult) -> str:
        if self._is_destructive_intent(user_goal):
            return "destructive"
        if result.library_mutated or any(marker in user_goal for marker in self._WRITE_MARKERS):
            return "write"
        if (
            result.used_document_ids
            or result.evidence_items
            or any(marker in user_goal for marker in self._LIBRARY_MARKERS + self._CATEGORY_MARKERS)
        ):
            return "read_only"
        return "safe"

    def _fallback_needed_tool_type(self, actions: list[ReflectionImprovementAction], user_goal: str) -> str:
        tool_names = " ".join(str(action.tool or "") for action in actions)
        if "document_metadata" in tool_names:
            return "metadata"
        if any("library.operator" in str(action.tool or "") for action in actions):
            return "tag_operator"
        if "category" in tool_names or any(marker in user_goal for marker in self._CATEGORY_MARKERS):
            return "category"
        if "evidence.retriever" in tool_names or self._requires_rag_evidence(user_goal):
            return "rag"
        if "library.explorer.stats" in tool_names:
            return "library_stats"
        if any("operator" in str(action.tool or "") for action in actions):
            return "operator_verify"
        if any("find_documents" in str(action.tool or "") for action in actions):
            return "document_search"
        return "none"

    @staticmethod
    def _clamp_score(value: Any) -> int:
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            return 1
        return max(1, min(10, score))

    @staticmethod
    def _dedupe_strings(items: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = re.sub(r"\s+", " ", str(item)).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results

    @staticmethod
    def _dedupe_actions(actions: list[ReflectionImprovementAction]) -> list[ReflectionImprovementAction]:
        results: list[ReflectionImprovementAction] = []
        seen: set[str] = set()
        for action in actions:
            key = json.dumps(action.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            results.append(action)
        return results

    @staticmethod
    def _extract_message_text(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip() if parts else None
        return None

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
