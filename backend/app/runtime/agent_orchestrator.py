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
    ResearchRunStatus,
    TraceEventType,
)
from app.repositories import ResearchRepository, RuntimeRepository

from .message_bus import MessageBus
from .skill_registry import SkillRegistry
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


class AgentOrchestrator:
    """Select DIRECT / REACT / PLANNER / REFLECTION before execution."""

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
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 30.0,
    ) -> None:
        self.research_repository = research_repository
        self.runtime_repository = runtime_repository
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.message_bus = MessageBus(runtime_repository)

    def available_tools(self):
        return self.tool_registry.list_enabled()

    def available_skills(self):
        return self.skill_registry.list_enabled()

    def select_mode(self, payload: AgentOrchestratorInput) -> AgentModeDecision:
        """Create a trace and return the final mode decision."""

        trace_id = self._begin_trace(payload)
        rule_candidate = self._rule_candidate(payload)
        llm_candidate = self._llm_candidate(payload)
        final_candidate = self._adjudicate(payload, rule_candidate, llm_candidate)
        decision = AgentModeDecision(
            mode=final_candidate.mode,
            reason=final_candidate.reason,
            confidence=final_candidate.confidence,
            target_runtime=final_candidate.target_runtime,
            initial_context=final_candidate.initial_context,
            required_capabilities=final_candidate.required_capabilities,
            trace_id=trace_id,
            fallback_used=final_candidate.fallback_used,
        )
        self._append_decision_trace(payload, decision, rule_candidate, llm_candidate)
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
        rule_candidate: _ModeCandidate,
        llm_candidate: _ModeCandidate | None,
    ) -> None:
        self.message_bus.append_trace(
            run_id=decision.trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="agent_mode_selected",
            message=f"Agent mode selected: {decision.mode.value}.",
            payload={
                "mode": decision.mode.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "target_runtime": decision.target_runtime,
                "required_capabilities": decision.required_capabilities,
                "fallback_used": decision.fallback_used,
                "rule_candidate": {
                    "mode": rule_candidate.mode.value,
                    "reason": rule_candidate.reason,
                    "confidence": rule_candidate.confidence,
                },
                "llm_candidate": (
                    {
                        "mode": llm_candidate.mode.value,
                        "reason": llm_candidate.reason,
                        "confidence": llm_candidate.confidence,
                    }
                    if llm_candidate is not None
                    else None
                ),
                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                "available_skill_ids": [skill.skill_id for skill in payload.available_skills[:20]],
                "has_conversation_referents": bool(payload.conversation_referents),
                "memory_hit_count": len(payload.memory_snapshot.items),
            },
        )

    def _rule_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate:
        content = payload.user_prompt.strip()
        lowered = content.casefold()
        has_pending_action = bool(payload.runtime_context.get("has_pending_action"))

        if has_pending_action and any(marker in lowered for marker in self._CONFIRM_MARKERS):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="用户正在确认上一轮待执行操作，需要交给知识库工具链执行确认后的动作。",
                confidence=0.96,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["pending_action", "library_operator"],
                initial_context={"pending_action": True},
            )

        if self._is_destructive_intent(content):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="请求包含删除、清空或移除等破坏性动作，必须进入确认保护流程。",
                confidence=0.99,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["confirmation_required"],
                initial_context={"permission_policy": "confirmation_required"},
            )

        if any(marker in content for marker in self._REFLECTION_MARKERS):
            return _ModeCandidate(
                mode=AgentRunMode.REFLECTION,
                reason="用户在纠错或要求重新检查上一轮回答，需要进入反思修正流程。",
                confidence=0.94,
                target_runtime="ReflectionRuntime",
                required_capabilities=["trace_review", "answer_revision"],
            )

        if self._is_planner_intent(content):
            return _ModeCandidate(
                mode=AgentRunMode.PLANNER,
                reason="用户请求包含多阶段任务或查询、写操作、检索、总结组合，需要先规划再执行。",
                confidence=0.9,
                target_runtime="KnowledgePlannerRuntime",
                required_capabilities=["structured_plan", "knowledge_tools"],
            )

        if self._is_react_intent(content, payload):
            return _ModeCandidate(
                mode=AgentRunMode.REACT,
                reason="用户请求需要访问论文库、标签、库内论文或本地证据，适合 ReAct 工具链。",
                confidence=0.86,
                target_runtime="KnowledgeAgentRuntime",
                required_capabilities=["knowledge_tools"],
            )

        return _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="普通对话或解释类问题，无需访问论文库、向量库或工具链。",
            confidence=0.82,
            target_runtime="DirectChatRuntime",
            required_capabilities=[],
        )

    def _llm_candidate(self, payload: AgentOrchestratorInput) -> _ModeCandidate | None:
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
                            "你是 PaperDesk 的模式路由器，只输出 JSON。"
                            "可选 mode 只有 DIRECT、REACT、PLANNER、REFLECTION。"
                            "你只判断模式，不执行工具，不输出隐藏推理。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_prompt": payload.user_prompt,
                                "selected_document_count": len(payload.selected_document_ids),
                                "attachment_kinds": [item.kind for item in payload.attachments],
                                "has_conversation_referents": bool(payload.conversation_referents),
                                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                                "available_skill_ids": [skill.skill_id for skill in payload.available_skills[:20]],
                                "output_schema": {
                                    "mode": "DIRECT|REACT|PLANNER|REFLECTION",
                                    "reason": "brief visible reason",
                                    "confidence": 0.0,
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
        try:
            mode = AgentRunMode(str(payload_json.get("mode") or "").upper())
        except ValueError:
            return None
        capabilities = payload_json.get("required_capabilities")
        return _ModeCandidate(
            mode=mode,
            reason=str(payload_json.get("reason") or "LLM mode candidate."),
            confidence=self._clamp_confidence(payload_json.get("confidence")),
            target_runtime=self._target_runtime_for(mode),
            required_capabilities=[str(item) for item in capabilities] if isinstance(capabilities, list) else [],
        )

    def _adjudicate(
        self,
        payload: AgentOrchestratorInput,
        rule_candidate: _ModeCandidate,
        llm_candidate: _ModeCandidate | None,
    ) -> _ModeCandidate:
        content = payload.user_prompt.strip()
        if self._is_destructive_intent(content):
            return rule_candidate
        if payload.runtime_context.get("has_pending_action") and any(
            marker in content.casefold() for marker in self._CONFIRM_MARKERS
        ):
            return rule_candidate
        if any(marker in content for marker in self._REFLECTION_MARKERS):
            return rule_candidate
        if self._has_selected_document_context(payload) and (
            llm_candidate is None or llm_candidate.mode == AgentRunMode.DIRECT
        ):
            rule_candidate.fallback_used = bool(llm_candidate is None and self.api_key)
            rule_candidate.reason = (
                "Selected library documents require the knowledge runtime so answers are grounded in "
                "document observations instead of a direct model guess."
            )
            return rule_candidate
        if self._is_general_question_bundle(content, payload):
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
            )
            return direct_candidate
        if self._is_library_write_intent(content) and (
            llm_candidate is None or llm_candidate.mode not in {AgentRunMode.REACT, AgentRunMode.PLANNER}
        ):
            rule_candidate.fallback_used = bool(llm_candidate is None and self.api_key)
            rule_candidate.reason = (
                "LLM intention decision was checked, but library write requests must enter "
                "the tool runtime so the database mutation can be verified."
            )
            return rule_candidate
        if self._is_library_read_intent(content) and (
            llm_candidate is None or llm_candidate.mode not in {AgentRunMode.REACT, AgentRunMode.PLANNER}
        ):
            rule_candidate.fallback_used = bool(llm_candidate is None and self.api_key)
            rule_candidate.reason = (
                "LLM intention decision was checked, but library state questions must enter "
                "the tool runtime so tags, counts, and document links come from database observations."
            )
            return rule_candidate

        if llm_candidate is None:
            rule_candidate.fallback_used = bool(self.api_key)
            return rule_candidate
        if llm_candidate.confidence < 0.65:
            rule_candidate.fallback_used = True
            return rule_candidate
        llm_candidate.reason = f"LLM intention decision: {llm_candidate.reason}"
        return llm_candidate

    def _is_planner_intent(self, content: str) -> bool:
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
        return {
            AgentRunMode.DIRECT: "DirectChatRuntime",
            AgentRunMode.REACT: "KnowledgeAgentRuntime",
            AgentRunMode.PLANNER: "KnowledgePlannerRuntime",
            AgentRunMode.REFLECTION: "ReflectionRuntime",
        }[mode]

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

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
