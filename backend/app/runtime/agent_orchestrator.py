"""Mode router for the chat-side PaperDesk agent entrypoint."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openai import OpenAI

from app.models import (
    AgentModeDecision,
    AgentOrchestratorInput,
    AgentRunMode,
    KnowledgeIntent,
    KnowledgeRiskLevel,
    KnowledgeRoute,
    KnowledgeTargetObject,
    ResearchRunStatus,
    TraceEventType,
)
from app.repositories import ResearchRepository, RuntimeRepository

from .message_bus import MessageBus
from .skill_registry import SkillRegistry
from app.services.skill_context_builder import SkillContextBuilder
from app.services.skill_selector import SkillSelector
from .tool_registry import ToolRegistry


@dataclass(slots=True)
class _ModeCandidate:
    mode: AgentRunMode
    reason: str
    confidence: float
    target_runtime: str
    required_capabilities: list[str] = field(default_factory=list)
    initial_context: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    source: str = "rule"
    error: str | None = None
    risk_level: str = "safe"
    needs_tool: bool = False
    needs_document_grounding: bool = False
    needs_plan: bool = False
    needs_reflection: bool = False
    allowed_modes: set[AgentRunMode] | None = None
    task_type: str = "other"
    operation_level: str = "none"
    requested_fields: list[str] = field(default_factory=list)
    entity_mentions: list[dict[str, Any]] = field(default_factory=list)
    requested_output: str = "other"
    user_intent: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    needs_verification: bool = False
    action_plan: list[dict[str, Any]] = field(default_factory=list)
    requested_route: KnowledgeRoute | None = None


class AgentOrchestrator:
    """Select DIRECT / REACT / PLANNER / REFLECTION before execution."""

    # Route/mode compatibility contract. KnowledgeRoute is the product routing
    # vocabulary; AgentRunMode and runtime names are retained for dispatching
    # existing runtime implementations without changing behavior.
    _RUN_MODE_BY_ROUTE = {
        KnowledgeRoute.DIRECT_ANSWER: AgentRunMode.DIRECT,
        KnowledgeRoute.TOOL_ACTION: AgentRunMode.REACT,
        KnowledgeRoute.CONFIRMED_WRITE: AgentRunMode.REACT,
        KnowledgeRoute.OPTIONAL_PLANNER: AgentRunMode.PLANNER,
        KnowledgeRoute.OPTIONAL_REFLECTION: AgentRunMode.REFLECTION,
    }
    _DEFAULT_ROUTE_BY_RUN_MODE = {
        AgentRunMode.DIRECT: KnowledgeRoute.DIRECT_ANSWER,
        AgentRunMode.REACT: KnowledgeRoute.TOOL_ACTION,
        AgentRunMode.PLANNER: KnowledgeRoute.OPTIONAL_PLANNER,
        AgentRunMode.REFLECTION: KnowledgeRoute.OPTIONAL_REFLECTION,
    }
    _RUNTIME_BY_RUN_MODE = {
        AgentRunMode.DIRECT: "DirectChatRuntime",
        AgentRunMode.REACT: "KnowledgeAgentRuntime",
        AgentRunMode.PLANNER: "KnowledgePlannerRuntime",
        AgentRunMode.REFLECTION: "ReflectionRuntime",
    }

    _CONFIRM_MARKERS = ("确认", "是的", "继续", "执行", "同意", "可以", "confirm", "yes")
    _DESTRUCTIVE_MARKERS = ("删除", "移除", "清空", "删掉", "delete", "remove", "clear")
    _DESTRUCTIVE_TARGETS = ("论文", "文档", "分类", "标签", "会话", "记录")
    _REFLECTION_MARKERS = (
        "刚才错",
        "刚才答错",
        "回答不对",
        "重新检查",
        "再检查",
        "反思",
        "为什么没查",
        "为什么没有查",
        "不对",
        "答错",
    )
    _PLANNER_MARKERS = ("先", "再", "然后", "之后", "最后", "分别", "按标签", "按分类", "分组", "每类")
    _LIBRARY_MARKERS = (
        "论文库",
        "文献库",
        "本地论文",
        "库里",
        "库内",
        "标签",
        "分类",
        "打标签",
        "归类",
        "无标签",
        "没有标签",
        "已选论文",
        "库内论文",
    )
    _PAPER_TASK_MARKERS = (
        "几篇",
        "多少",
        "哪些",
        "什么标签",
        "总结",
        "综述",
        "概述",
        "对比",
        "比较",
        "检索",
        "引用",
        "证据",
        "写一篇",
    )
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
    _DRAFT_MARKERS = ("总结", "综述", "概述", "对比", "比较", "brief", "review", "summary", "compare")

    def __init__(
        self,
        *,
        research_repository: ResearchRepository,
        runtime_repository: RuntimeRepository,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        enable_optional_planner: bool = True,
        enable_auto_reflection: bool = False,
        enable_mcp_in_knowledge: bool = False,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 30.0,
    ) -> None:
        self.research_repository = research_repository
        self.runtime_repository = runtime_repository
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.enable_optional_planner = enable_optional_planner
        self.enable_auto_reflection = enable_auto_reflection
        self.enable_mcp_in_knowledge = enable_mcp_in_knowledge
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.message_bus = MessageBus(runtime_repository)
        self.skill_selector = SkillSelector()
        self.skill_context_builder = SkillContextBuilder(self.skill_registry)

    def available_tools(self):
        return self.tool_registry.list_default_candidates(scope="knowledge")

    def available_skills(self):
        return self.skill_registry.list_enabled()

    def select_skills_for_trace(self, payload: AgentOrchestratorInput):
        return self.skill_selector.select(
            prompt=payload.user_prompt,
            command=payload.runtime_context.get("command"),
            intent_hint=payload.runtime_context.get("intent_hint"),
            selected_document_count=len(payload.selected_document_ids),
            attachments=payload.attachments,
            available_skills=payload.available_skills,
            task_type=payload.runtime_context.get("task_type"),
            route=payload.runtime_context.get("route"),
        )

    def select_mode(self, payload: AgentOrchestratorInput) -> AgentModeDecision:
        """Create a trace and return the final mode decision."""

        trace_id = self._begin_trace(payload)
        skill_selection = self.select_skills_for_trace(payload)
        guardrail_candidate = self._hard_guardrail_candidate(payload)
        fallback_candidate = self._fallback_rule_candidate(payload)
        llm_candidate = self._llm_candidate(payload)
        final_candidate = self._adjudicate(payload, guardrail_candidate, fallback_candidate, llm_candidate)
        route = self._knowledge_route_for(final_candidate)
        intent = self._knowledge_intent_for(final_candidate)
        risk_level = self._knowledge_risk_level_for(final_candidate)
        target_objects = self._target_objects_for(payload, final_candidate)
        requires_confirmation = self._requires_confirmation(final_candidate, route)
        decision = AgentModeDecision(
            mode=final_candidate.mode,
            route=route,
            intent=intent,
            reason=final_candidate.reason,
            confidence=final_candidate.confidence,
            target_runtime=final_candidate.target_runtime,
            requires_tools=final_candidate.needs_tool or final_candidate.mode in {AgentRunMode.REACT, AgentRunMode.PLANNER},
            requires_rag=final_candidate.needs_document_grounding or self._intent_requires_rag(intent),
            requires_confirmation=requires_confirmation,
            risk_level=risk_level,
            target_objects=target_objects,
            initial_context=final_candidate.initial_context,
            required_capabilities=final_candidate.required_capabilities,
            trace_id=trace_id,
            fallback_used=final_candidate.fallback_used,
        )
        self._append_decision_trace(payload, decision, guardrail_candidate, fallback_candidate, llm_candidate, skill_selection)
        return decision

    def append_trace(self, trace_id: str, *, status: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status=status,
            message=message,
            payload=payload or {},
        )

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: ResearchRunStatus = ResearchRunStatus.COMPLETED,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="agent_orchestrator_finished",
            message="Agent orchestrator finished.",
            payload=payload or {},
        )
        self.research_repository.update_run_status(trace_id, status)

    def _begin_trace(self, payload: AgentOrchestratorInput) -> str:
        trace_id = f"chat-orch-{uuid4().hex}"
        self.research_repository.create_run(trace_id, f"Chat Orchestrator: {payload.user_prompt[:80]}")
        self.research_repository.update_run_status(trace_id, ResearchRunStatus.RUNNING_TASK)
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="agent_orchestrator_started",
            message="Agent orchestrator started.",
            payload={
                "session_id": payload.session_id,
                "message_id": payload.message_id,
                "selected_document_count": len(payload.selected_document_ids),
                "attachment_count": len(payload.attachments),
            },
        )
        return trace_id

    def _append_decision_trace(
        self,
        payload: AgentOrchestratorInput,
        decision: AgentModeDecision,
        guardrail_candidate: _ModeCandidate | None,
        fallback_candidate: _ModeCandidate,
        llm_candidate: _ModeCandidate | None,
        skill_selection=None,
    ) -> None:
        guardrail_payload = self._candidate_trace_payload(guardrail_candidate) if guardrail_candidate else None
        fallback_payload = self._candidate_trace_payload(fallback_candidate)
        llm_payload = self._candidate_trace_payload(llm_candidate) if llm_candidate else None
        skill_context_summary = self.skill_context_builder.build(skill_selection)
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="agent_mode_selected",
            message=f"Agent mode selected: {decision.mode.value}.",
            payload={
                "mode": decision.mode.value,
                "route": decision.route.value,
                "intent": decision.intent.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "target_runtime": decision.target_runtime,
                "requires_tools": decision.requires_tools,
                "requires_rag": decision.requires_rag,
                "requires_confirmation": decision.requires_confirmation,
                "risk_level": decision.risk_level.value,
                "target_objects": [item.model_dump(mode="json") for item in decision.target_objects],
                "required_capabilities": decision.required_capabilities,
                "fallback_used": decision.fallback_used,
                "decision_source": self._decision_source(decision, guardrail_candidate, fallback_candidate, llm_candidate),
                "guardrail_candidate": guardrail_payload,
                "rule_candidate": fallback_payload,
                "fallback_candidate": fallback_payload,
                "llm_candidate": llm_payload,
                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                "available_skill_ids": [skill.skill_id for skill in payload.available_skills[:20]],
                "primary_skill_id": (
                    skill_selection.primary_skill.skill_id
                    if skill_selection is not None and skill_selection.primary_skill is not None
                    else None
                ),
                "used_skill_ids": (
                    [skill.skill_id for skill in skill_selection.used_skills]
                    if skill_selection is not None
                    else []
                ),
                "used_skills": (
                    [skill.model_dump(mode="json") for skill in skill_selection.used_skills]
                    if skill_selection is not None
                    else []
                ),
                "skill_context_summary": (
                    skill_context_summary.model_dump(mode="json") if skill_context_summary is not None else None
                ),
                "has_conversation_referents": bool(payload.conversation_referents),
                "memory_hit_count": len(payload.memory_snapshot.items),
            },
        )

    @staticmethod
    def _candidate_trace_payload(candidate: _ModeCandidate | None) -> dict[str, Any] | None:
        if candidate is None:
            return None
        return {
            "mode": candidate.mode.value,
            "reason": candidate.reason,
            "confidence": candidate.confidence,
            "source": candidate.source,
            "risk_level": candidate.risk_level,
            "needs_tool": candidate.needs_tool,
            "needs_document_grounding": candidate.needs_document_grounding,
            "needs_plan": candidate.needs_plan,
            "needs_reflection": candidate.needs_reflection,
            "task_type": candidate.task_type,
            "operation_level": candidate.operation_level,
            "requested_fields": candidate.requested_fields,
            "entity_mentions": candidate.entity_mentions,
            "requested_output": candidate.requested_output,
            "user_intent": candidate.user_intent,
            "entities": candidate.entities,
            "needs_verification": candidate.needs_verification,
            "action_plan": candidate.action_plan,
            "required_capabilities": candidate.required_capabilities,
            "allowed_modes": [mode.value for mode in candidate.allowed_modes] if candidate.allowed_modes else None,
            "error": candidate.error,
        }

    @staticmethod
    def _decision_source(
        decision: AgentModeDecision,
        guardrail_candidate: _ModeCandidate | None,
        fallback_candidate: _ModeCandidate,
        llm_candidate: _ModeCandidate | None,
    ) -> str:
        if guardrail_candidate is not None and decision.reason == guardrail_candidate.reason:
            return guardrail_candidate.source
        if llm_candidate is not None and decision.mode == llm_candidate.mode and decision.confidence == llm_candidate.confidence:
            return llm_candidate.source
        if decision.mode == fallback_candidate.mode and decision.reason == fallback_candidate.reason:
            return fallback_candidate.source
        if decision.fallback_used:
            return "fallback_rule"
        return "adjudicator"

    def _hard_guardrail_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate | None:
        content = payload.user_prompt.strip()

        if self._has_pending_confirmation(payload):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="用户正在确认上一轮待执行操作，需要交给知识库工具链执行确认后的动作。",
                confidence=0.96,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["pending_action", "library_operator"],
                initial_context={"pending_action": True},
                source="guardrail",
                risk_level="write",
                needs_tool=True,
            )

        if self._is_destructive_intent(content):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="请求包含删除、清空或移除等破坏性动作，必须进入确认保护流程。",
                confidence=0.99,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["confirmation_required"],
                initial_context={"permission_policy": "confirmation_required"},
                source="guardrail",
                risk_level="destructive",
                needs_tool=True,
            )

        if self._is_reflection_feedback(content):
            return _ModeCandidate(
                mode=AgentRunMode.REFLECTION,
                reason="用户在纠错或要求重新检查上一轮回答，需要进入反思修正流程。",
                confidence=0.94,
                target_runtime="ReflectionRuntime",
                required_capabilities=["trace_review", "answer_revision"],
                source="guardrail",
                risk_level="read_only",
                needs_reflection=True,
            )

        if self._has_selected_document_context(payload):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason=(
                    "Selected library documents require the knowledge runtime so answers are grounded in "
                    "document observations instead of a direct model guess."
                ),
                confidence=0.93,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["document_grounding", "knowledge_tools"],
                source="guardrail",
                risk_level="read_only",
                needs_tool=True,
                needs_document_grounding=True,
                allowed_modes={AgentRunMode.REACT, AgentRunMode.PLANNER},
            )

        if self._is_library_write_intent(content):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason=(
                    "LLM intention decision was checked, but library write requests must enter "
                    "the tool runtime so the database mutation can be verified."
                ),
                confidence=0.91,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["library_operator", "write_verification"],
                source="guardrail",
                risk_level="write",
                needs_tool=True,
                allowed_modes={AgentRunMode.REACT, AgentRunMode.PLANNER},
            )

        if self._is_library_read_intent(content):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason=(
                    "LLM intention decision was checked, but library state questions must enter "
                    "the tool runtime so tags, counts, and document links come from database observations."
                ),
                confidence=0.89,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["knowledge_tools"],
                source="guardrail",
                risk_level="read_only",
                needs_tool=True,
                task_type="labeled_document_query"
                if self._is_labeled_document_collection_intent(content)
                else "library_stats",
                needs_document_grounding=self._is_labeled_document_collection_intent(content),
                allowed_modes={AgentRunMode.REACT, AgentRunMode.PLANNER},
            )

        return None

    def _fallback_rule_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate:
        content = payload.user_prompt.strip()

        if self._is_planner_intent(content):
            if not self.enable_optional_planner:
                return _ModeCandidate(
                    mode=AgentRunMode.REACT,
                    reason="Optional planner is disabled; explicit long Knowledge tasks stay on the stable tool-action path.",
                    confidence=0.88,
                    target_runtime="KnowledgeAgentRuntime",
                    required_capabilities=["knowledge_tools"],
                    source="fallback_rule",
                    risk_level="read_only",
                    needs_tool=True,
                )
            return _ModeCandidate(
                mode=AgentRunMode.PLANNER,
                reason="用户请求包含多阶段任务或查询、写操作、检索、总结组合，需要先规划再执行。",
                confidence=0.9,
                target_runtime="KnowledgePlannerRuntime",
                required_capabilities=["structured_plan", "knowledge_tools"],
                source="fallback_rule",
                risk_level="read_only",
                needs_tool=True,
                needs_plan=True,
            )

        if self._is_react_intent(content, payload):
            task_type = "labeled_document_query" if self._is_labeled_document_collection_intent(content) else "other"
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="用户请求需要访问论文库、标签、库内论文或本地证据，适合 ReAct 工具链。",
                confidence=0.86,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["knowledge_tools"],
                source="fallback_rule",
                risk_level="read_only",
                needs_tool=True,
                task_type=task_type,
                needs_document_grounding=task_type == "labeled_document_query" or self._has_selected_document_context(payload),
            )

        return _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="普通对话或解释类问题，无需访问论文库、向量库或工具链。",
            confidence=0.82,
            target_runtime="DirectChatRuntime",
            required_capabilities=[],
            source="fallback_rule",
        )

    def _rule_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate:
        """Backward-compatible conservative candidate used by existing tests and callers."""

        return self._hard_guardrail_candidate(payload) or self._fallback_rule_candidate(payload)

    def _llm_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate | None:
        if not self.api_key:
            return None
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            handler_descriptions = self._router_handler_descriptions()
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 Knowledge Chat 轻量路由器，只输出 JSON。"
                            "你的任务是根据用户当前输入、历史上下文、可用处理器和工具描述，"
                            "选择最合适的主路由。默认可选 route 只有 DirectAnswer、ToolAction、ConfirmedWrite。"
                            "只能选择 available_handlers 中存在的处理器；工具计划只能引用 available_tools 中存在的 tool_id。"
                            "没有合适工具需求时选择 DirectAnswer。你只判断路由，不执行工具，不输出隐藏推理。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_prompt": payload.user_prompt,
                                "history_hint": payload.conversation_referents,
                                "memory_items": self._router_memory_items(payload),
                                "runtime_context": payload.runtime_context,
                                "selected_document_count": len(payload.selected_document_ids),
                                "selected_document_ids": payload.selected_document_ids[:20],
                                "attachment_kinds": [item.kind for item in payload.attachments],
                                "attachment_document_ids": [
                                    item.document_id for item in payload.attachments if item.document_id
                                ],
                                "has_conversation_referents": bool(payload.conversation_referents),
                                "available_handlers": handler_descriptions,
                                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                                "available_tools": [
                                    {
                                        "tool_id": tool.tool_id,
                                        "name": tool.name,
                                        "description": tool.description,
                                        "read_only": tool.read_only,
                                        "input_schema": tool.input_schema,
                                    }
                                    for tool in payload.available_tools[:40]
                                ],
                                "output_schema": {
                                    "route": "DirectAnswer|ToolAction|ConfirmedWrite",
                                    "reason": "brief visible reason",
                                    "confidence": 0.0,
                                    "intent": "chat|paper_qa|paper_compare|tag_query|tag_write|report_query|report_save|correction|long_research_task",
                                    "risk_level": "none|low|medium|high",
                                    "operation_level": "query-level|entity-level|relation-level|content-level",
                                    "operation": "ask|summarize|compare|assign_label|remove_label|clear_labels|rename_label|delete_empty_labels|save_report|other",
                                    "target_type": "paper|paper_label_relation|label_entity|report|evidence|general_chat|unknown",
                                    "scope_hint": "current_selection|recent_selection|explicit_documents|explicit_label|all_library|unknown",
                                    "referenced_documents": ["document id, title, or filename mentioned by user"],
                                    "label_or_category": "explicit label/category name if any",
                                    "suggested_tool": "available tool id if a tool is needed",
                                    "requires_confirmation": False,
                                    "clarification_needed": False,
                                    "user_intent": "one sentence summary of the user's real goal",
                                    "needs_tool": True,
                                    "needs_document_grounding": False,
                                    "needs_plan": False,
                                    "needs_reflection": False,
                                    "needs_verification": False,
                                    "requested_fields": ["journal", "publish_time"],
                                    "entity_mentions": [
                                        {"text": "标签名", "entity_type": "tag|category|collection|unknown", "confidence": 0.0}
                                    ],
                                    "entities": [
                                        {
                                            "text": "entity text",
                                            "type": "paper|tag|category|report|chunk|index|conversation|unknown",
                                            "role": "source|target|filter|object",
                                            "confidence": 0.0,
                                        }
                                    ],
                                    "requested_output": "list|count|summary|comparison|analysis_report|other",
                                    "action_plan": [
                                        {
                                            "tool": "available tool id if a tool is needed",
                                            "purpose": "short visible purpose",
                                            "arguments": {},
                                            "operation_level": "query-level|entity-level|relation-level|content-level",
                                            "requires_verification_after": False,
                                        }
                                    ],
                                    "required_capabilities": ["capability"],
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
        payload_json = self._extract_json_payload(text)
        if not isinstance(payload_json, dict):
            return None
        candidate = self._llm_candidate_from_payload(payload_json)
        if candidate is None:
            return None
        return self._validate_llm_candidate(payload, candidate)

    def _llm_candidate_from_payload(self, payload_json: dict[str, Any]) -> _ModeCandidate | None:
        try:
            mode = self._mode_from_route_payload(payload_json)
        except ValueError:
            return None
        requested_route = self._route_from_payload(payload_json)
        capabilities = payload_json.get("required_capabilities")
        risk_level = self._normalize_risk_level(payload_json.get("risk_level"))
        needs_tool = self._coerce_bool(payload_json.get("needs_tool"))
        needs_document_grounding = self._coerce_bool(payload_json.get("needs_document_grounding"))
        needs_plan = self._coerce_bool(payload_json.get("needs_plan"))
        needs_reflection = self._coerce_bool(payload_json.get("needs_reflection"))
        task_type = self._normalize_task_type(payload_json.get("task_type"))
        operation_level = self._normalize_operation_level(payload_json.get("operation_level"))
        requested_fields = payload_json.get("requested_fields")
        normalized_fields = [
            str(item).strip()
            for item in requested_fields
            if str(item).strip()
        ] if isinstance(requested_fields, list) else []
        entity_mentions = payload_json.get("entity_mentions")
        normalized_mentions = [
            item
            for item in entity_mentions
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ] if isinstance(entity_mentions, list) else []
        entities = payload_json.get("entities")
        normalized_entities = [
            item
            for item in entities
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ] if isinstance(entities, list) else []
        requested_output = self._normalize_requested_output(payload_json.get("requested_output"))
        operation = str(payload_json.get("operation") or "").strip()
        target_type = str(payload_json.get("target_type") or "").strip()
        scope_hint = str(payload_json.get("scope_hint") or "").strip()
        referenced_documents = [
            str(item).strip()
            for item in payload_json.get("referenced_documents") or []
            if str(item).strip()
        ] if isinstance(payload_json.get("referenced_documents"), list) else []
        label_or_category = str(payload_json.get("label_or_category") or "").strip()
        suggested_tool = str(payload_json.get("suggested_tool") or "").strip()
        requires_confirmation = self._coerce_bool(payload_json.get("requires_confirmation"))
        clarification_needed = self._coerce_bool(payload_json.get("clarification_needed"))
        action_plan = payload_json.get("action_plan")
        normalized_plan = [
            item
            for item in action_plan
            if isinstance(item, dict) and str(item.get("tool") or "").strip()
        ] if isinstance(action_plan, list) else []
        user_intent = str(payload_json.get("user_intent") or "").strip()
        needs_verification = self._coerce_bool(payload_json.get("needs_verification"))
        return _ModeCandidate(
            mode=mode,
            reason=str(payload_json.get("reason") or "LLM mode candidate."),
            confidence=self._clamp_confidence(payload_json.get("confidence")),
            target_runtime=self._target_runtime_for(mode),
            required_capabilities=[str(item) for item in capabilities] if isinstance(capabilities, list) else [],
            initial_context={
                "task_type": task_type,
                "operation_level": operation_level,
                "requested_fields": normalized_fields,
                "entity_mentions": normalized_mentions,
                "entities": normalized_entities,
                "requested_output": requested_output,
                "operation": operation,
                "target_type": target_type,
                "scope_hint": scope_hint,
                "referenced_documents": referenced_documents,
                "label_or_category": label_or_category,
                "suggested_tool": suggested_tool,
                "requires_confirmation": requires_confirmation,
                "clarification_needed": clarification_needed,
                "user_intent": user_intent,
                "needs_verification": needs_verification,
                "action_plan": normalized_plan,
            },
            source="llm",
            risk_level=risk_level,
            needs_tool=needs_tool,
            needs_document_grounding=needs_document_grounding,
            needs_plan=needs_plan,
            needs_reflection=needs_reflection,
            task_type=task_type,
            operation_level=operation_level,
            requested_fields=normalized_fields,
            entity_mentions=normalized_mentions,
            requested_output=requested_output,
            user_intent=user_intent,
            entities=normalized_entities,
            needs_verification=needs_verification,
            action_plan=normalized_plan,
            requested_route=requested_route,
        )

    @staticmethod
    def _mode_from_route_payload(payload_json: dict[str, Any]) -> AgentRunMode:
        # Compatibility layer: the router speaks KnowledgeRoute names, while
        # ChatService still dispatches existing runtime implementations by
        # AgentRunMode. Keep this mapping centralized and behavior-preserving.
        requested_route = AgentOrchestrator._route_from_payload(payload_json)
        if requested_route is not None:
            return AgentOrchestrator._run_mode_for_route(requested_route)
        return AgentRunMode(str(payload_json.get("mode") or "").upper())

    @staticmethod
    def _route_from_payload(payload_json: dict[str, Any]) -> KnowledgeRoute | None:
        raw_route = str(payload_json.get("route") or "").strip()
        for route in KnowledgeRoute:
            if route.value.casefold() == raw_route.casefold():
                return route
        return None

    def _adjudicate(
        self,
        payload: AgentOrchestratorInput,
        guardrail_candidate: _ModeCandidate | None,
        fallback_candidate: _ModeCandidate,
        llm_candidate: _ModeCandidate | None,
    ) -> _ModeCandidate:
        content = payload.user_prompt.strip()
        if guardrail_candidate is not None and guardrail_candidate.allowed_modes is None:
            return guardrail_candidate
        if self._is_general_question_bundle(content, payload):
            if (
                llm_candidate is not None
                and llm_candidate.confidence >= 0.65
                and self._llm_candidate_has_runtime_need(llm_candidate)
            ):
                llm_candidate.reason = f"LLM intention decision: {llm_candidate.reason}"
                return llm_candidate
            direct_candidate = _ModeCandidate(
                mode=AgentRunMode.DIRECT,
                reason=(
                    "同一条消息包含多个普通知识问题，且没有选中文档、论文库或写库意图；"
                    "应作为直接问答逐项回答，避免误入只检索不作答的工具链。"
                ),
                confidence=0.9,
                target_runtime="DirectChatRuntime",
                required_capabilities=[],
                fallback_used=bool(llm_candidate is not None and llm_candidate.mode != AgentRunMode.DIRECT),
                source="guardrail",
                risk_level="safe",
            )
            return direct_candidate

        if guardrail_candidate is not None:
            if (
                llm_candidate is not None
                and llm_candidate.confidence >= 0.65
                and guardrail_candidate.allowed_modes is not None
                and llm_candidate.mode in guardrail_candidate.allowed_modes
                and (
                    llm_candidate.mode != AgentRunMode.PLANNER
                    or (self.enable_optional_planner and self._is_optional_planner_intent(content, payload))
                )
            ):
                llm_candidate.reason = f"LLM intention decision within guardrail boundary: {llm_candidate.reason}"
                return self._merge_guardrail_context(llm_candidate, guardrail_candidate)
            if (
                guardrail_candidate.allowed_modes is not None
                and fallback_candidate.mode in guardrail_candidate.allowed_modes
            ):
                merged = self._merge_guardrail_context(fallback_candidate, guardrail_candidate)
                merged.reason = (
                    guardrail_candidate.reason
                    if fallback_candidate.reason == guardrail_candidate.reason
                    else f"{guardrail_candidate.reason} {fallback_candidate.reason}"
                )
                merged.fallback_used = bool(self.api_key)
                if self.api_key:
                    merged.error = "llm_unavailable_invalid_or_low_confidence"
                return merged
            guardrail_candidate.fallback_used = bool(llm_candidate is None and self.api_key)
            return guardrail_candidate

        if llm_candidate is None:
            fallback_candidate.fallback_used = bool(self.api_key)
            if self.api_key:
                fallback_candidate.error = "llm_unavailable_or_invalid"
            return fallback_candidate
        if llm_candidate.confidence < 0.65:
            fallback_candidate.fallback_used = True
            fallback_candidate.error = "llm_low_confidence"
            return fallback_candidate
        if llm_candidate.mode == AgentRunMode.PLANNER and (
            not self.enable_optional_planner or not self._is_optional_planner_intent(content, payload)
        ):
            fallback_candidate.fallback_used = True
            fallback_candidate.error = (
                "llm_planner_disabled"
                if not self.enable_optional_planner
                else "llm_planner_without_explicit_long_task"
            )
            return fallback_candidate
        if llm_candidate.mode == AgentRunMode.REFLECTION and not self.enable_auto_reflection:
            fallback_candidate.fallback_used = True
            fallback_candidate.error = "llm_reflection_disabled_for_router"
            return fallback_candidate
        llm_candidate.reason = f"LLM intention decision: {llm_candidate.reason}"
        return llm_candidate

    @staticmethod
    def _router_handler_descriptions() -> list[dict[str, Any]]:
        return [
            {
                "name": KnowledgeRoute.DIRECT_ANSWER.value,
                "runtime": "DirectChatRuntime",
                "description": "直接回答路径。用于解释概念、回答常识问题、系统使用说明和不需要读取论文库的内容。",
                "use_when": "没有选中论文、没有论文库/标签/报告对象、没有写操作意图。",
            },
            {
                "name": KnowledgeRoute.TOOL_ACTION.value,
                "runtime": "KnowledgeAgentRuntime",
                "description": "工具动作路径。用于只读访问论文库、标签、报告或 RAG 证据。",
                "use_when": "用户询问库内论文内容、选中论文后的总结/对比/创新点、标签分类查询或报告查询。",
            },
            {
                "name": KnowledgeRoute.CONFIRMED_WRITE.value,
                "runtime": "KnowledgeAgentRuntime",
                "description": "确认写路径。用于删除、清空、覆盖、批量修改、重建索引或保存/覆盖报告等写操作。",
                "use_when": "规则识别到写操作或破坏性动作时必须进入该路径，确认前只说明影响范围。",
            },
        ]

    @staticmethod
    def _optional_handler_descriptions() -> list[dict[str, Any]]:
        return [
            {
                "name": KnowledgeRoute.OPTIONAL_PLANNER.value,
                "runtime": "KnowledgePlannerRuntime",
                "description": "可选规划路径。仅用于明确长任务、研究计划、大批量论文或多阶段产物。",
                "use_when": "用户明确要求分步骤研究、制定研究计划、综述大纲，或从 Research 入口发起任务。",
            },
            {
                "name": KnowledgeRoute.OPTIONAL_REFLECTION.value,
                "runtime": "ReflectionRuntime",
                "description": "可选反思路径。仅用于显式纠错、重新检查、工具结果为空但需要证据或写后校验失败。",
                "use_when": "用户指出上一轮回答错误、要求重新检查、质疑没有查论文库或引用来源不对。",
            },
        ]

    @staticmethod
    def _router_memory_items(payload: AgentOrchestratorInput) -> list[dict[str, str | None]]:
        return [
            {
                "type": item.memory_type,
                "summary": item.summary,
                "source": item.source_kind,
            }
            for item in payload.memory_snapshot.items[:8]
        ]

    def _validate_llm_candidate(
        self,
        payload: AgentOrchestratorInput,
        candidate: _ModeCandidate,
    ) -> _ModeCandidate | None:
        if candidate.mode not in set(AgentRunMode):
            return None
        available_tool_ids = {tool.tool_id for tool in payload.available_tools}
        suggested_tool = str(candidate.initial_context.get("suggested_tool") or "").strip()
        if suggested_tool and suggested_tool not in available_tool_ids:
            candidate.initial_context["suggested_tool"] = ""
            candidate.error = "llm_unknown_suggested_tool_removed"
        if candidate.action_plan:
            valid_plan = [
                action
                for action in candidate.action_plan
                if str(action.get("tool") or "").strip() in available_tool_ids
            ]
            if len(valid_plan) != len(candidate.action_plan):
                candidate.error = "llm_unknown_tool_removed"
            candidate.action_plan = valid_plan
            candidate.initial_context["action_plan"] = valid_plan
        if candidate.mode == AgentRunMode.DIRECT and candidate.needs_tool:
            candidate.error = "llm_direct_with_tool_need"
            return None
        if candidate.mode == AgentRunMode.PLANNER:
            candidate.needs_plan = True
        if candidate.mode == AgentRunMode.REFLECTION:
            candidate.needs_reflection = True
        if candidate.mode in {AgentRunMode.REACT, AgentRunMode.PLANNER}:
            candidate.needs_tool = True
        return candidate

    @staticmethod
    def _llm_candidate_has_runtime_need(candidate: _ModeCandidate) -> bool:
        return bool(
            candidate.mode != AgentRunMode.DIRECT
            and (
                candidate.needs_tool
                or candidate.needs_document_grounding
                or candidate.needs_plan
                or candidate.needs_reflection
                or candidate.required_capabilities
                or candidate.action_plan
            )
        )

    @staticmethod
    def _merge_guardrail_context(llm_candidate: _ModeCandidate, guardrail_candidate: _ModeCandidate) -> _ModeCandidate:
        required_capabilities = list(dict.fromkeys([
            *guardrail_candidate.required_capabilities,
            *llm_candidate.required_capabilities,
        ]))
        initial_context = {
            **guardrail_candidate.initial_context,
            **llm_candidate.initial_context,
        }
        return llm_candidate.__class__(
            mode=llm_candidate.mode,
            reason=llm_candidate.reason,
            confidence=llm_candidate.confidence,
            target_runtime=llm_candidate.target_runtime,
            required_capabilities=required_capabilities,
            initial_context=initial_context,
            fallback_used=False,
            source=llm_candidate.source,
            error=llm_candidate.error,
            risk_level=guardrail_candidate.risk_level
            if guardrail_candidate.risk_level in {"write", "destructive"}
            else llm_candidate.risk_level,
            needs_tool=guardrail_candidate.needs_tool or llm_candidate.needs_tool,
            needs_document_grounding=(
                guardrail_candidate.needs_document_grounding or llm_candidate.needs_document_grounding
            ),
            needs_plan=guardrail_candidate.needs_plan or llm_candidate.needs_plan,
            needs_reflection=guardrail_candidate.needs_reflection or llm_candidate.needs_reflection,
            allowed_modes=guardrail_candidate.allowed_modes,
            task_type=llm_candidate.task_type,
            operation_level=llm_candidate.operation_level if llm_candidate.operation_level != "none" else guardrail_candidate.operation_level,
            requested_fields=llm_candidate.requested_fields,
            entity_mentions=llm_candidate.entity_mentions,
            requested_output=llm_candidate.requested_output,
            user_intent=llm_candidate.user_intent,
            entities=llm_candidate.entities,
            needs_verification=guardrail_candidate.needs_tool or llm_candidate.needs_verification,
            action_plan=llm_candidate.action_plan,
            requested_route=llm_candidate.requested_route,
        )

    def _knowledge_route_for(self, candidate: _ModeCandidate) -> KnowledgeRoute:
        # Product route derivation stays separate from AgentRunMode so write
        # protection can be described as ConfirmedWrite even when execution
        # still uses the KnowledgeAgentRuntime/ReAct compatibility path.
        if candidate.requested_route is not None:
            return candidate.requested_route
        if self._candidate_requires_write_path(candidate):
            return KnowledgeRoute.CONFIRMED_WRITE
        return self._default_route_for_run_mode(candidate.mode)

    def _knowledge_intent_for(self, candidate: _ModeCandidate) -> KnowledgeIntent:
        if candidate.mode == AgentRunMode.REFLECTION or candidate.needs_reflection:
            return KnowledgeIntent.CORRECTION
        if candidate.mode == AgentRunMode.PLANNER or candidate.needs_plan:
            return KnowledgeIntent.LONG_RESEARCH_TASK
        if self._candidate_requires_write_path(candidate):
            if candidate.task_type in {"report_generation"} or any("report" in str(item) for item in candidate.required_capabilities):
                return KnowledgeIntent.REPORT_SAVE
            return KnowledgeIntent.TAG_WRITE
        if candidate.task_type in {"tag_query", "category_query", "library_stats", "labeled_document_query"}:
            return KnowledgeIntent.TAG_QUERY if "tag" in candidate.task_type or "category" in candidate.task_type else KnowledgeIntent.PAPER_QA
        if candidate.task_type in {"metadata_query", "document_qa", "rag_summary", "labeled_document_analysis"}:
            return KnowledgeIntent.PAPER_QA
        if candidate.requested_output == "comparison":
            return KnowledgeIntent.PAPER_COMPARE
        if candidate.task_type in {"collection_report", "report_generation"}:
            return KnowledgeIntent.REPORT_QUERY
        if candidate.needs_document_grounding or "document_grounding" in candidate.required_capabilities:
            return KnowledgeIntent.PAPER_QA
        if candidate.needs_tool:
            return KnowledgeIntent.PAPER_QA
        return KnowledgeIntent.CHAT

    @staticmethod
    def _knowledge_risk_level_for(candidate: _ModeCandidate) -> KnowledgeRiskLevel:
        if candidate.risk_level in {"destructive", "critical"}:
            return KnowledgeRiskLevel.HIGH
        if candidate.risk_level in {"write", "scoped_write"}:
            return KnowledgeRiskLevel.MEDIUM
        if candidate.risk_level in {"read_only", "safe_write"} or candidate.needs_tool:
            return KnowledgeRiskLevel.LOW
        return KnowledgeRiskLevel.NONE

    @staticmethod
    def _intent_requires_rag(intent: KnowledgeIntent) -> bool:
        return intent in {KnowledgeIntent.PAPER_QA, KnowledgeIntent.PAPER_COMPARE}

    @staticmethod
    def _requires_confirmation(candidate: _ModeCandidate, route: KnowledgeRoute) -> bool:
        return route == KnowledgeRoute.CONFIRMED_WRITE and (
            candidate.risk_level in {"destructive", "critical"}
            or "confirmation_required" in candidate.required_capabilities
            or candidate.initial_context.get("permission_policy") == "confirmation_required"
        )

    @staticmethod
    def _candidate_requires_write_path(candidate: _ModeCandidate) -> bool:
        if candidate.risk_level in {"write", "safe_write", "scoped_write", "destructive", "critical"}:
            return True
        if AgentOrchestrator._normalize_operation_level(candidate.operation_level) in {"entity-level", "relation-level", "content-level"} and (
            candidate.required_capabilities
            or candidate.action_plan
            or candidate.needs_verification
        ):
            return True
        if any(capability in candidate.required_capabilities for capability in ("confirmation_required", "library_operator", "write_verification")):
            return True
        return candidate.initial_context.get("permission_policy") == "confirmation_required"

    def _target_objects_for(
        self,
        payload: AgentOrchestratorInput,
        candidate: _ModeCandidate,
    ) -> list[KnowledgeTargetObject]:
        grouped: dict[str, KnowledgeTargetObject] = {}

        def add(object_type: str, *, ids: list[str] | None = None, names: list[str] | None = None) -> None:
            if object_type == "document":
                object_type = "paper"
            if object_type not in {"paper", "category", "tag", "report", "chunk", "index", "conversation"}:
                return
            target = grouped.setdefault(object_type, KnowledgeTargetObject(type=object_type))
            for item in ids or []:
                if item and item not in target.ids:
                    target.ids.append(item)
            for item in names or []:
                if item and item not in target.names:
                    target.names.append(item)

        add("paper", ids=payload.selected_document_ids[:20])
        for attachment in payload.attachments:
            if attachment.document_id:
                add("paper", ids=[attachment.document_id])
        for entity in [*candidate.entities, *candidate.entity_mentions]:
            object_type = str(entity.get("type") or entity.get("entity_type") or "").casefold().strip()
            text = str(entity.get("text") or "").strip()
            add(object_type, names=[text] if text else [])

        if self._has_library_marker(payload.user_prompt) or "论文" in payload.user_prompt:
            add("paper")
        if any(marker in payload.user_prompt for marker in ("标签", "分类", "tag", "category")):
            add("tag")
            add("category")
        if "报告" in payload.user_prompt or "report" in payload.user_prompt.casefold():
            add("report")
        if any(marker in payload.user_prompt.casefold() for marker in ("chunk", "索引", "index")):
            add("chunk")
            add("index")
        if "会话" in payload.user_prompt:
            add("conversation")
        return list(grouped.values())

    def _is_planner_intent(self, content: str) -> bool:
        if not self._is_optional_planner_intent(content, None):
            return False
        stage_count = sum(1 for marker in self._PLANNER_MARKERS if marker in content)
        has_library_task = self._has_library_marker(content) or ("论文" in content and any(marker in content for marker in self._PAPER_TASK_MARKERS))
        has_write = any(marker in content for marker in self._WRITE_MARKERS)
        has_draft = any(marker in content.casefold() for marker in self._DRAFT_MARKERS)
        has_query = any(marker in content for marker in ("查", "统计", "哪些", "几篇", "多少", "检索"))
        if re.search(r"先.+再|先.+然后|再.+然后", content):
            return has_library_task or has_write or has_draft
        if stage_count >= 2 and (has_library_task or has_write or has_draft):
            return True
        return has_library_task and has_query and has_write and has_draft

    def _is_optional_planner_intent(self, content: str, payload: AgentOrchestratorInput | None) -> bool:
        normalized = content.casefold()
        if payload is not None and payload.runtime_context.get("entrypoint") == "research":
            return True
        explicit_long_task = any(
            marker in content
            for marker in (
                "研究计划",
                "分步骤研究",
                "分步研究",
                "综述大纲",
                "研究大纲",
                "研究脉络",
                "多阶段",
                "先筛选",
                "再分类",
                "生成报告",
            )
        )
        batch_papers = bool(re.search(r"\d+\s*篇论文", content)) or any(marker in content for marker in ("这些论文", "所有论文", "全部论文"))
        staged_product = bool(re.search(r"先.+再.+(?:然后|最后)", content)) and any(
            marker in content for marker in ("报告", "综述", "大纲", "研究", "筛选", "分类")
        )
        return explicit_long_task or (batch_papers and staged_product) or (
            batch_papers and any(marker in normalized for marker in ("outline", "review", "report", "plan"))
        )

    def _is_general_question_bundle(self, content: str, payload: AgentOrchestratorInput) -> bool:
        if self._question_count(content) < 2:
            return False
        if payload.selected_document_ids or any(attachment.document_id for attachment in payload.attachments):
            return False
        if self._is_library_write_intent(content) or self._is_library_read_intent(content):
            return False
        if self._has_library_marker(content):
            return False
        if "论文" in content and any(marker in content for marker in self._PAPER_TASK_MARKERS):
            return False
        return True

    @staticmethod
    def _has_selected_document_context(payload: AgentOrchestratorInput) -> bool:
        if payload.selected_document_ids:
            return True
        return any(attachment.document_id for attachment in payload.attachments)

    def _has_pending_confirmation(self, payload: AgentOrchestratorInput) -> bool:
        """Guardrail: confirmations for protected pending actions must re-enter the tool runtime."""

        if not payload.runtime_context.get("has_pending_action"):
            return False
        content = payload.user_prompt.casefold()
        return any(marker in content for marker in self._CONFIRM_MARKERS) or (
            "确认" in content and any(marker in content for marker in self._DESTRUCTIVE_MARKERS)
        )

    def _is_reflection_feedback(self, content: str) -> bool:
        """Guardrail: explicit correction of the previous answer must go through reflection."""

        strong_markers = (
            "刚才错",
            "刚才答错",
            "回答不对",
            "重新检查",
            "再检查",
            "反思",
            "为什么没查",
            "为什么没有查",
        )
        if any(marker in content for marker in strong_markers):
            return True
        broad_correction = any(marker in content for marker in ("不对", "答错"))
        previous_answer_reference = any(
            marker in content
            for marker in ("刚才", "刚刚", "上一轮", "上次", "前面", "你说", "你的回答", "回答")
        )
        return broad_correction and previous_answer_reference

    @staticmethod
    def _question_count(content: str) -> int:
        explicit = len(re.findall(r"[？?]", content))
        if explicit:
            return explicit
        question_markers = (
            "谁",
            "什么",
            "哪些",
            "哪个",
            "多少",
            "几",
            "怎么",
            "如何",
            "为什么",
            "为何",
            "是否",
            "who",
            "what",
            "which",
            "when",
            "where",
            "why",
            "how",
        )
        parts = [part.strip() for part in re.split(r"[；;\n。]+", content) if part.strip()]
        return sum(1 for part in parts if any(marker in part.casefold() for marker in question_markers))

    def _is_react_intent(self, content: str, payload: AgentOrchestratorInput) -> bool:
        if payload.selected_document_ids:
            return True
        if any(attachment.document_id for attachment in payload.attachments):
            return True
        if self._has_library_marker(content):
            return True
        return "论文" in content and any(marker in content for marker in self._PAPER_TASK_MARKERS)

    def _has_library_marker(self, content: str) -> bool:
        return any(marker in content for marker in self._LIBRARY_MARKERS)

    @staticmethod
    def _is_labeled_document_collection_intent(content: str) -> bool:
        lowered = content.casefold()
        if any(marker in lowered for marker in ("有标签", "带标签", "已打标签", "已分类", "tagged")) and not re.search(
            r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,40}\s*(?:标签|分类|分组|集合)\s*(?:下|下面|里的|中的|内|对应|关联)",
            content,
        ):
            return False
        has_entity_marker = any(marker in lowered for marker in ("标签", "分类", "分组", "集合", "tag", "category", "group"))
        has_document_scope = any(
            marker in lowered
            for marker in ("论文", "文章", "文档", "paper", "papers", "document", "documents", "下", "下面", "里的", "中的")
        )
        return has_entity_marker and has_document_scope

    def _is_library_write_intent(self, content: str) -> bool:
        has_write = any(marker in content for marker in self._WRITE_MARKERS)
        has_target = self._has_library_marker(content) or any(marker in content for marker in ("论文", "文章", "文档"))
        return has_write and has_target

    def _is_library_read_intent(self, content: str) -> bool:
        if self._is_library_write_intent(content):
            return False
        has_target = self._has_library_marker(content) or any(marker in content for marker in ("论文", "文章", "文档"))
        has_read = any(
            marker in content
            for marker in (
                "几篇",
                "多少",
                "哪些",
                "什么",
                "对应",
                "每篇",
                "每一篇",
                "分别",
                "查看",
                "列出",
                "统计",
                "有没有",
                "有几",
            )
        )
        return has_target and has_read

    def _is_destructive_intent(self, content: str) -> bool:
        normalized = self._strip_quoted_text(content).casefold()
        if self._is_non_destructive_category_write(normalized):
            return False
        if not any(marker in normalized for marker in self._DESTRUCTIVE_MARKERS):
            return False
        targets = tuple(target for target in self._DESTRUCTIVE_TARGETS if target not in {"标签", "分类"})
        if any(target in normalized for target in targets):
            return True
        if any(marker in normalized for marker in ("标签", "分类", "tag", "category")):
            return True
        return bool(re.search(r"(删除|移除|清空|删掉|delete|remove|clear)\s*(?:这个|该|此)?\s*(?:标签|分类)", normalized))

    @staticmethod
    def _strip_quoted_text(content: str) -> str:
        stripped = content
        for pattern in (
            r"“[^”]*”",
            r"‘[^’]*’",
            r"「[^」]*」",
            r"『[^』]*』",
            r"'[^']*'",
            r'"[^"]*"',
        ):
            stripped = re.sub(pattern, "", stripped)
        return stripped

    @staticmethod
    def _is_non_destructive_category_write(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        if not has_category_target:
            return False
        return any(
            marker in content
            for marker in (
                "新建",
                "新增",
                "创建",
                "建立",
                "加个",
                "加一个",
                "添加",
                "打上",
                "打个",
                "打一个",
                "补上",
                "归类",
                "换成",
                "换为",
                "替换为",
                "替换成",
                "改成",
                "改为",
                "改名为",
                "重命名",
                "重命名为",
                "assign",
                "create",
                "add",
                "rename",
                "replace",
            )
        )

    @staticmethod
    def _target_runtime_for(mode: AgentRunMode) -> str:
        return AgentOrchestrator._RUNTIME_BY_RUN_MODE[mode]

    @staticmethod
    def _run_mode_for_route(route: KnowledgeRoute) -> AgentRunMode:
        return AgentOrchestrator._RUN_MODE_BY_ROUTE[route]

    @staticmethod
    def _default_route_for_run_mode(mode: AgentRunMode) -> KnowledgeRoute:
        return AgentOrchestrator._DEFAULT_ROUTE_BY_RUN_MODE[mode]

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        risk = str(value or "").casefold().strip()
        new_route_risks = {
            "none": "safe",
            "low": "read_only",
            "medium": "write",
            "high": "destructive",
        }
        if risk in new_route_risks:
            return new_route_risks[risk]
        if risk in {"safe", "read_only", "safe_write", "scoped_write", "write", "destructive", "critical"}:
            return risk
        return "safe"

    @staticmethod
    def _normalize_operation_level(value: Any) -> str:
        operation_level = str(value or "").casefold().strip()
        aliases = {
            "none": "query-level",
            "read": "query-level",
            "query": "query-level",
            "entity": "entity-level",
            "relation": "relation-level",
            "document": "entity-level",
            "global": "relation-level",
            "content": "content-level",
        }
        operation_level = aliases.get(operation_level, operation_level)
        allowed = {"query-level", "entity-level", "relation-level", "content-level"}
        return operation_level if operation_level in allowed else "query-level"

    @staticmethod
    def _normalize_task_type(value: Any) -> str:
        task_type = str(value or "").casefold().strip()
        allowed = {
            "general_chat",
            "metadata_query",
            "document_qa",
            "rag_summary",
            "tag_query",
            "tag_write",
            "tag_rename",
            "delete_unused_categories",
            "category_entity_cleanup",
            "category_write",
            "library_stats",
            "collection_analysis",
            "labeled_document_query",
            "labeled_document_analysis",
            "category_query",
            "collection_report",
            "reflection",
            "report_generation",
            "other",
        }
        return task_type if task_type in allowed else "other"

    @staticmethod
    def _normalize_requested_output(value: Any) -> str:
        requested_output = str(value or "").casefold().strip()
        allowed = {"answer", "list", "count", "summary", "comparison", "analysis_report", "operation_result", "other"}
        return requested_output if requested_output in allowed else "other"

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.casefold().strip() in {"true", "1", "yes", "y"}
        return False

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
