"""Claude Code-style runtime for the chat knowledge agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import json
import re
from typing import Any
from uuid import uuid4

from openai import OpenAI

from app.models import (
    AgentTask,
    AgentTaskStatus,
    ChatAttachment,
    ChatMessageRequest,
    ChatSession,
    EvidenceItem,
    LibraryDocument,
    ResearchRunStatus,
    SubagentProfile,
    TaskNotification,
    ToolPolicy,
    TraceEventType,
)
from app.repositories import CategoryRepository, ResearchRepository, RuntimeRepository
from app.services.context_file_store import ContextFileStore
from app.services.document_library_service import DocumentLibraryService
from app.services.rag_service import RagService
from app.vectorstores import AbstractVectorStore

from .message_bus import MessageBus


@dataclass(slots=True)
class KnowledgeAgentResult:
    """Result returned to the chat service after a knowledge-agent run."""

    content: str
    retrieval_status: str = "skipped"
    warning: str | None = None
    citations: list[str] = field(default_factory=list)
    used_document_ids: list[str] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    action_status: str | None = None
    agent_trace_id: str | None = None
    library_mutated: bool = False


@dataclass(slots=True)
class _TaskOutcome:
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ReactAction:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(slots=True)
class _ReactObservation:
    tool: str
    status: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _AnswerObligation:
    key: str
    description: str
    required_tools: tuple[str, ...]
    target: dict[str, Any] = field(default_factory=dict)


class KnowledgeAgentRuntime:
    """Coordinate library subagents for chat-side PaperDesk operations."""

    _CONFIRM_MARKERS = ("确认", "是的", "继续", "执行", "同意", "可以", "confirm", "yes")
    _DELETE_MARKERS = ("删除", "移除", "清空", "删掉", "delete", "remove", "clear")
    _SUMMARY_MARKERS = ("总结", "综述", "概述", "对比", "比较", "review", "summary", "summarize", "compare")
    _INTERNAL_DEGRADED_TOOL = "agent.intent.degraded"
    _INTERNAL_CATEGORY_CONFLICT_TOOL = "agent.category_semantics_conflict"

    def __init__(
        self,
        *,
        document_library_service: DocumentLibraryService,
        category_repository: CategoryRepository,
        research_repository: ResearchRepository,
        runtime_repository: RuntimeRepository,
        rag_service: RagService,
        vectorstore: AbstractVectorStore,
        file_store: ContextFileStore,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 30.0,
    ) -> None:
        self.document_library_service = document_library_service
        self.category_repository = category_repository
        self.research_repository = research_repository
        self.runtime_repository = runtime_repository
        self.rag_service = rag_service
        self.vectorstore = vectorstore
        self.file_store = file_store
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.message_bus = MessageBus(runtime_repository)

    def build_context_lines(self) -> list[str]:
        """Build a compact library capability summary for the chat prompt."""

        documents = self.document_library_service.list_documents()
        ready_documents = [document for document in documents if document.status == "ready"]
        category_stats = self._category_stats_payload(limit_documents=0)
        preview = documents[:8]
        lines = [
            f"论文库：共 {len(documents)} 篇，已可用 {len(ready_documents)} 篇；"
            f"有标签 {category_stats['tagged_document_count']} 篇，无标签 {category_stats['untagged_document_count']} 篇。",
            "知识库聊天采用 ReAct 主 Agent：先选择工具、读取观察结果，再继续行动或回答。",
            "可用工具：library.explorer.*、library.operator.*、evidence.retriever.search、report.drafter.write、memory.read/write。",
            "权限策略：只读自动允许；新建分类、追加标签等非破坏写操作经校验后执行；删除论文/分类必须先确认。",
            "重要经验：标签统计必须读取真实分类关联；复合命令要拆成多步，不能把后半句当标签名。",
        ]
        if category_stats["categories"]:
            lines.append("现有分类：" + "、".join(category["name"] for category in category_stats["categories"][:12]))
        if preview:
            lines.append(
                "论文预览："
                + "；".join(
                    f"{document.display_name or document.filename}({document.status})"
                    for document in preview
                )
            )
        return lines

    def conversation_referents(self, session_id: str) -> dict[str, Any]:
        """Expose current conversation referents for the orchestrator."""

        return self._read_react_state(session_id)

    def has_pending_action(self, session_id: str) -> bool:
        """Return whether a protected action is awaiting user confirmation."""

        return self._read_pending_action(session_id) is not None

    def handle(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        content = request.content.strip()
        if not content:
            return None

        pending_result = self._maybe_execute_pending_action(session, content, trace_id=trace_id)
        if pending_result is not None:
            return pending_result

        if self._is_destructive_intent(content) and not self._should_plan_before_destructive_confirmation(
            content,
            selected_document_ids,
            attachments,
        ):
            return self._request_confirmation(session, content, trace_id=trace_id)

        if not self._should_handle_with_react(content, selected_document_ids, attachments):
            return None

        return self._run_react_agent(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
        )

    def run_react(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
        runtime_label: str = "knowledge_react_execution",
    ) -> KnowledgeAgentResult:
        """Run the ReAct executor after an upstream orchestrator selected it."""

        content = request.content.strip()
        pending_result = self._maybe_execute_pending_action(session, content, trace_id=trace_id)
        if pending_result is not None:
            return pending_result
        if self._is_destructive_intent(content) and not self._should_plan_before_destructive_confirmation(
            content,
            selected_document_ids,
            attachments,
        ):
            return self._request_confirmation(session, content, trace_id=trace_id)
        return self._run_react_agent(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
            runtime_label=runtime_label,
        )

    def _should_handle_with_react(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return True

        lowered = content.casefold()
        library_markers = (
            "论文库",
            "文献库",
            "本地论文",
            "库里",
            "标签",
            "分类",
            "打标签",
            "归类",
            "无标签",
            "没有标签",
        )
        drafting_markers = ("总结", "综述", "概述", "对比", "比较", "brief", "review", "summary", "summarize", "compare")
        if any(marker in content for marker in library_markers):
            return True
        if any(marker in lowered for marker in drafting_markers) and self._extract_document_tokens(content):
            return True
        if "论文" in content and any(marker in content for marker in ("几篇", "多少", "哪些", "什么", "写一篇")):
            return True
        return False

    def _should_plan_before_destructive_confirmation(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        """Allow ReAct to handle safe subgoals before confirming a destructive subgoal."""

        if not self._is_destructive_intent(content):
            return False
        safe_obligations = [
            obligation
            for obligation in self._answer_obligations(content, selected_document_ids, attachments)
            if not obligation.key.endswith("_verified")
        ]
        if safe_obligations:
            return True
        return self._has_final_literal_output_request(content)

    def _run_react_agent(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
        runtime_label: str = "knowledge_react_execution",
    ) -> KnowledgeAgentResult:
        content = request.content.strip()
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        if not owns_run:
            self._append_react_trace(
                run_id=run_id,
                status=f"{runtime_label}_started",
                payload={"session_id": session.id, "topic": content},
            )
        observations: list[_ReactObservation] = []
        final_text = ""
        final_status = "completed"
        used_document_ids: list[str] = []
        evidence_items: list[EvidenceItem] = []
        retrieval_status = "skipped"
        warning: str | None = None
        library_mutated = False

        try:
            for step in range(12):
                obligations = self._answer_obligations(content, selected_document_ids, attachments)
                action = self._next_react_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
                self._append_react_trace(
                    run_id=run_id,
                    status="react_action_planned",
                    payload={
                        "step": step + 1,
                        "tool": action.tool,
                        "arguments": self._safe_trace_payload(action.arguments),
                        "rationale": action.rationale,
                        "answer_obligations": self._obligations_payload(obligations, observations),
                    },
                )

                if action.tool == "final.answer":
                    final_text = self._user_visible_final_answer(
                        content,
                        observations,
                        str(action.arguments.get("content") or ""),
                    )
                    break

                validation_error = self._validate_react_action(action, observations)
                if validation_error:
                    observation = _ReactObservation(
                        tool=action.tool,
                        status="validation_failed",
                        summary=validation_error,
                        payload={"tool": action.tool},
                    )
                else:
                    observation = self._execute_react_action(
                        run_id=run_id,
                        session=session,
                        content=content,
                        action=action,
                        observations=observations,
                    )

                observations.append(observation)
                self._append_react_trace(
                    run_id=run_id,
                    status="react_observation",
                    payload={
                        "step": step + 1,
                        "tool": observation.tool,
                        "status": observation.status,
                        "summary": observation.summary,
                        "payload": self._safe_trace_payload(observation.payload),
                    },
                )

                used_document_ids = self._merge_document_ids(used_document_ids, observation.payload)
                evidence_items = self._merge_evidence_items(evidence_items, observation.payload)
                if evidence_items:
                    retrieval_status = "ready"
                evidence_quality = observation.payload.get("evidence_quality")
                if isinstance(evidence_quality, dict) and "vector_unavailable" in (evidence_quality.get("warnings") or []):
                    warning = (
                        "Milvus 向量服务当前不可用，本轮已改用已入库论文正文的关键词检索；"
                        "结论仍基于检索到的正文片段，但语义召回能力会弱于向量检索。"
                    )
                if observation.payload.get("library_mutated"):
                    library_mutated = True
                if observation.status == "degraded":
                    retrieval_status = "degraded"
                    warning = observation.summary
                if observation.status == "failed" and observation.tool.startswith("evidence.retriever."):
                    retrieval_status = "degraded"
                if observation.status in {"needs_clarification", "confirmation_required", "validation_failed", "failed", "degraded"}:
                    final_status = observation.status
                    final_text = self._synthesize_react_answer(content, observations)
                    break
                if observation.tool in {"report.drafter.write", "report.drafter.write_by_category"}:
                    final_text = str(observation.payload.get("answer") or "").strip()
                    break
            else:
                final_text = self._synthesize_react_answer(content, observations)

            if not final_text:
                final_text = self._synthesize_react_answer(content, observations)
            deferred_destructive = self._defer_destructive_part_after_safe_plan(
                session=session,
                content=content,
                observations=observations,
                current_answer=final_text,
                run_id=run_id,
            )
            if deferred_destructive is not None:
                final_text = deferred_destructive
                final_status = "confirmation_required"
            self._record_react_reflection(
                session=session,
                user_goal=content,
                observations=observations,
                status=final_status,
            )
            self._update_react_state(
                session=session,
                user_goal=content,
                observations=observations,
                status=final_status,
            )
            if owns_run:
                self._finish_run(run_id)
            else:
                self._append_react_trace(
                    run_id=run_id,
                    status=f"{runtime_label}_finished",
                    payload={"final_status": final_status, "observation_count": len(observations)},
                )
            return KnowledgeAgentResult(
                content=final_text,
                retrieval_status=retrieval_status,
                warning=warning,
                citations=self._collect_citations(evidence_items),
                used_document_ids=used_document_ids,
                evidence_items=evidence_items,
                action_status=final_status,
                agent_trace_id=run_id,
                library_mutated=library_mutated,
            )
        except Exception as exc:
            self._append_react_trace(
                run_id=run_id,
                status="react_failed",
                payload={"error": str(exc)},
            )
            if owns_run:
                self._finish_run(run_id, ResearchRunStatus.FAILED)
            return KnowledgeAgentResult(
                content="知识库 Agent 执行时遇到异常，已经停止本轮工具调用，避免对论文库产生不可靠修改。",
                retrieval_status="degraded",
                warning=str(exc),
                action_status="failed",
                agent_trace_id=run_id,
            )

    def _next_react_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction:
        if self._is_tag_category_semantics_conflict(content):
            return _ReactAction(
                self._INTERNAL_CATEGORY_CONFLICT_TOOL,
                {},
                "用户请求把标签和分类当作两套字段处理，但当前系统中二者是同一字段。",
            )
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        unmet_obligations = self._unmet_obligations(obligations, observations)
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if observations and not unmet_obligations:
            return _ReactAction(
                "final.answer",
                {"content": self._synthesize_react_answer(content, observations)},
                "All answer obligations are satisfied by tool observations; synthesize the final answer.",
            )
        if self.api_key:
            llm_action = self._next_react_action_with_llm(
                session=session,
                content=content,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
            )
            if llm_action is not None:
                if (
                    llm_action.tool == "final.answer"
                    and unmet_obligations
                ):
                    repair_action = self._action_for_unmet_obligation(
                        session=session,
                        content=content,
                        selected_document_ids=selected_document_ids,
                        attachments=attachments,
                        observations=observations,
                        obligations=unmet_obligations,
                    )
                    if repair_action is not None:
                        return repair_action
                    return _ReactAction(
                        self._INTERNAL_DEGRADED_TOOL,
                        {
                            "reason": (
                                "Configured LLM returned final before satisfying required answer obligations: "
                                + ", ".join(obligation.key for obligation in unmet_obligations)
                            )
                        },
                        "LLM returned a premature final answer; no safe next tool could be inferred.",
                    )
                if llm_action.tool == "final.answer":
                    return llm_action
                if observations and llm_action.tool in completed_tools:
                    return _ReactAction(
                        "final.answer",
                        {"content": self._synthesize_react_answer(content, observations)},
                        "The model repeated an already completed tool after obligations were satisfied; answer from the first successful observation.",
                    )
                if llm_action.tool != "final.answer" or observations or not self._requires_tool_observation(
                    content,
                    selected_document_ids,
                    attachments,
                ):
                    return llm_action
                return _ReactAction(
                    self._INTERNAL_DEGRADED_TOOL,
                    {
                        "reason": (
                            "Configured LLM returned a final answer before reading required "
                            "library/tool observations."
                        )
                        },
                    "LLM tool planning returned a premature final answer; degrading explicitly.",
                )
            if unmet_obligations:
                repair_action = self._action_for_unmet_obligation(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                    obligations=unmet_obligations,
                )
                if repair_action is not None:
                    return repair_action
            return _ReactAction(
                self._INTERNAL_DEGRADED_TOOL,
                {"reason": "Configured LLM did not return a valid JSON tool plan."},
                "LLM tool planning failed; a write request cannot be completed without a tool call.",
            )
        return self._fallback_next_react_action(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            observations=observations,
        )

    def _next_react_action_with_llm(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 ReAct 主 Agent。你只能输出一个 JSON 对象，不能输出 Markdown。"
                            "你的工作方式是 Reason -> Tool Action -> Observation -> Next Action -> Final。"
                            "如果需要知道论文库、标签、可用工具或论文正文信息，必须先选择工具；不要凭记忆猜数据库内容。"
                            "复杂需求必须拆成多步工具链，例如先查标签覆盖，再补标签，再RAG检索，再写总结。"
                            "你会收到 task_state.answer_obligations：它是本轮必须回答或执行的任务清单。"
                            "只有当 task_state.unmet_obligations 为空时，才允许输出 final；否则必须选择一个能推进未满足任务的工具。"
                            "不要把 task_state 当成固定流程；你仍要按用户意图、当前 observations 和工具能力选择下一步。"
                            "用户说把某个标签删掉/替换成另一个标签时，如果语义是换名或保留论文关联，必须用 library.operator.rename_category；"
                            "用户说清除标签或分类时，标签和分类是同一字段，必须用 library.operator.clear_categories；"
                            "如果用户要求清除标签但保留分类，必须停止并说明当前系统中标签和分类是同一字段，不能这样执行。"
                            "用户说用所有某标签论文写报告/对比/总结时，必须先用 category_name 精确读取该标签下真实论文。"
                            "用户问每篇论文对应什么标签、每篇文章的标签、所有论文分别是什么标签时，"
                            "必须调用 library.explorer.document_categories，不能只用 category_stats 或运行态摘要。"
                            "会话上下文可以、也应该用于理解“另外几篇”“刚才那些论文”等指代；"
                            "但当前库状态、计数和写入结果必须来自工具 observation。"
                            "写操作只能通过 library.operator.* 工具提出，后端会再次校验；删除论文或分类必须交给 confirmation_required 流程。"
                            "最终回答要直接给用户结论，不要输出“根据对话记录/根据上下文/根据运行态摘要”等过程话术。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_goal": content,
                                "context_snapshot": self._build_react_snapshot(
                                    session.id,
                                    selected_document_ids,
                                    attachments,
                                ),
                                "task_state": self._react_task_state(
                                    content,
                                    selected_document_ids,
                                    attachments,
                                    observations,
                                ),
                                "available_tools": self._react_tool_specs(),
                                "observations": [
                                    {
                                        "tool": observation.tool,
                                        "status": observation.status,
                                        "summary": observation.summary,
                                        "payload": self._safe_trace_payload(observation.payload),
                                    }
                                    for observation in observations
                                ],
                                "output_schema": {
                                    "tool_call": {
                                        "type": "tool",
                                        "tool": "one available tool name",
                                        "arguments": {},
                                        "rationale": "brief visible planning summary",
                                    },
                                    "final": {"type": "final", "content": "answer based only on observations"},
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
        except Exception:
            return None
        message = self._extract_message_text(response)
        if not message:
            return None
        payload = self._extract_json_payload(message)
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("tool_call"), dict):
            payload = payload["tool_call"]
        elif isinstance(payload.get("final"), dict):
            payload = payload["final"]
        if payload.get("type") == "final":
            return _ReactAction(
                tool="final.answer",
                arguments={"content": str(payload.get("content") or "")},
                rationale=str(payload.get("rationale") or "final"),
            )
        if payload.get("type") == "tool":
            arguments = payload.get("arguments")
            return _ReactAction(
                tool=str(payload.get("tool") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                rationale=str(payload.get("rationale") or ""),
            )
        return None

    def _force_llm_write_tool_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        """Ask the configured LLM again for a write tool call, never a natural-language final."""

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
                            "你是 PaperDesk 的工具调用规划器，只输出 JSON。"
                            "当前用户请求涉及修改论文库标签/分类，不能输出 final，必须选择一个写库工具。"
                            "只能从 available_write_tools 里选择工具。"
                            "如果用户要求给没有标签的论文加标签，应使用 library.operator.assign_category，"
                            "arguments 里使用 scope=untagged 和用户指定的 category_name。"
                            "如果用户要求清空带有某标签的论文的标签，应使用 library.operator.clear_categories，"
                            "arguments 里使用 category_name=该标签名。"
                            "如果已经有成功写库 observation，可以输出 final。"
                            "除此之外，不要解释、不要说正在执行、不要输出 Markdown。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_goal": content,
                                "context_snapshot": self._build_react_snapshot(
                                    session.id,
                                    selected_document_ids,
                                    attachments,
                                ),
                                "available_write_tools": [
                                    tool
                                    for tool in self._react_tool_specs()
                                    if str(tool.get("name", "")).startswith("library.operator.")
                                ],
                                "observations": [
                                    {
                                        "tool": observation.tool,
                                        "status": observation.status,
                                        "summary": observation.summary,
                                        "payload": self._safe_trace_payload(observation.payload),
                                    }
                                    for observation in observations
                                ],
                                "output_schema": {
                                    "type": "tool",
                                    "tool": "library.operator.assign_category",
                                    "arguments": {"category_name": "标签名", "scope": "untagged"},
                                    "rationale": "brief tool-selection reason",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
        except Exception:
            return None
        message = self._extract_message_text(response)
        if not message:
            return None
        payload = self._extract_json_payload(message)
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("tool_call"), dict):
            payload = payload["tool_call"]
        if payload.get("type") != "tool":
            return None
        tool = str(payload.get("tool") or "")
        if not tool.startswith("library.operator."):
            return None
        arguments = payload.get("arguments")
        return _ReactAction(
            tool=tool,
            arguments=arguments if isinstance(arguments, dict) else {},
            rationale=str(payload.get("rationale") or "forced write tool planning"),
        )

    def _answer_obligations(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> list[_AnswerObligation]:
        obligations: list[_AnswerObligation] = []

        def add(key: str, description: str, tools: tuple[str, ...], **target: Any) -> None:
            if any(item.key == key for item in obligations):
                return
            obligations.append(_AnswerObligation(key, description, tools, target))

        if self._needs_library_stats(content):
            add(
                "library_stats",
                "回答论文库总量、可用数量或处理状态。",
                ("library.explorer.stats",),
            )
        if self._needs_category_stats(content):
            add(
                "category_stats",
                "回答标签/分类数量、有标签论文数、无标签论文数或标签覆盖统计。",
                ("library.explorer.category_stats",),
            )
        if self._requires_document_category_observation(content, selected_document_ids):
            add(
                "document_category_details",
                "回答逐篇论文对应的真实标签/分类明细。",
                ("library.explorer.document_categories",),
                scope="selected" if selected_document_ids else "all_or_tagged",
            )
        if self._is_clear_categories_intent(content):
            add(
                "clear_categories_verified",
                "执行并校验清空论文标签/分类关系。",
                ("library.operator.clear_categories",),
            )
        rename_pair = self._extract_category_rename_request(content)
        if self._is_rename_category_intent(content) or rename_pair:
            add(
                "rename_category_verified",
                "执行并校验标签/分类重命名或合并。",
                ("library.operator.rename_category",),
            )
        if self._is_assignment_intent(content):
            if not (
                self._needs_untagged_assignment(content)
                or selected_document_ids
                or self._mentions_previous_referent(content)
                or self._extract_document_tokens(content)
            ):
                add(
                    "resolved_document_set",
                    "先定位用户提到的论文集合。",
                    ("library.explorer.find_documents",),
                )
            add(
                "assign_category_verified",
                "执行并校验给目标论文追加标签/分类。",
                ("library.operator.assign_category",),
            )
        if self._is_create_category_intent(content):
            add(
                "create_category_verified",
                "执行并校验新建标签/分类。",
                ("library.operator.create_category",),
                content=content,
            )
        if self._is_document_category_query(content) and not self._requires_document_category_observation(content, selected_document_ids):
            add(
                "resolved_document_set",
                "先定位用户提到的论文集合。",
                ("library.explorer.find_documents",),
            )
            add(
                "document_category_details",
                "回答已定位论文对应的真实标签/分类明细。",
                ("library.explorer.document_categories",),
                scope="resolved",
            )
        if self._is_summary_request(content, selected_document_ids):
            if self._is_grouped_category_summary_request(content):
                add(
                    "category_evidence",
                    "按标签/分类分组检索论文证据。",
                    ("evidence.retriever.search_by_category",),
                )
                add(
                    "category_report",
                    "基于分组证据生成总结或对比。",
                    ("report.drafter.write_by_category",),
                )
            else:
                if not selected_document_ids and not attachments and self._extract_document_tokens(content):
                    add(
                        "resolved_document_set",
                        "先定位用户提到的论文集合。",
                        ("library.explorer.find_documents",),
                    )
                add(
                    "evidence_retrieval",
                    "检索论文正文或元数据证据。",
                    ("evidence.retriever.search",),
                )
                add(
                    "report",
                    "基于真实检索证据生成总结或对比回答。",
                    ("report.drafter.write",),
                )
        if self._is_compound_question_request(content) and self._should_answer_compound_with_evidence(
            content,
            selected_document_ids,
            attachments,
        ):
            if not selected_document_ids and not attachments and self._extract_document_tokens(content):
                add(
                    "resolved_document_set",
                    "先定位用户提到的论文集合。",
                    ("library.explorer.find_documents",),
                )
            add(
                "evidence_retrieval",
                "为同一条消息中的多个问题检索可回答的论文证据。",
                ("evidence.retriever.search",),
            )
            add(
                "report",
                "按用户问题顺序逐项生成回答，而不是只返回检索状态。",
                ("report.drafter.write",),
            )
        return obligations

    def _react_task_state(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> dict[str, Any]:
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        return {
            "answer_obligations": self._obligations_payload(obligations, observations),
            "satisfied_obligations": [
                item.key for item in obligations if self._obligation_satisfied(item, observations)
            ],
            "unmet_obligations": [
                item.key for item in obligations if not self._obligation_satisfied(item, observations)
            ],
            "planning_rule": "Use ReAct: choose tools until every unmet obligation is satisfied, then final.",
        }

    def _obligations_payload(
        self,
        obligations: list[_AnswerObligation],
        observations: list[_ReactObservation],
    ) -> list[dict[str, Any]]:
        return [
            {
                "key": obligation.key,
                "description": obligation.description,
                "required_tools": list(obligation.required_tools),
                "target": obligation.target,
                "status": "satisfied" if self._obligation_satisfied(obligation, observations) else "pending",
            }
            for obligation in obligations
        ]

    def _unmet_obligations(
        self,
        obligations: list[_AnswerObligation],
        observations: list[_ReactObservation],
    ) -> list[_AnswerObligation]:
        return [
            obligation
            for obligation in obligations
            if not self._obligation_satisfied(obligation, observations)
        ]

    def _obligation_satisfied(
        self,
        obligation: _AnswerObligation,
        observations: list[_ReactObservation],
    ) -> bool:
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if obligation.key == "document_category_details":
            return self._has_document_category_observation(observations)
        if obligation.key == "create_category_verified":
            category_name = self._extract_category_name_from_request(
                str(obligation.target.get("content") or "")
            )
            for observation in observations:
                if observation.status != "completed":
                    continue
                if observation.tool == "library.operator.create_category":
                    return True
                if observation.tool == "library.operator.assign_category" and observation.payload.get("category_name"):
                    if not category_name or observation.payload.get("category_name") == category_name:
                        return True
        if obligation.key.endswith("_verified"):
            return any(tool in completed_tools for tool in obligation.required_tools)
        if obligation.key == "report":
            report = self._latest_payload(observations, "report.drafter.write")
            return bool(report and report.get("answer"))
        if obligation.key == "category_report":
            report = self._latest_payload(observations, "report.drafter.write_by_category")
            return bool(report and report.get("answer"))
        return any(tool in completed_tools for tool in obligation.required_tools)

    def _action_for_unmet_obligation(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
        obligations: list[_AnswerObligation],
    ) -> _ReactAction | None:
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        for obligation in obligations:
            if obligation.key == "library_stats":
                return _ReactAction("library.explorer.stats", {}, "Read library statistics for an unmet answer obligation.")
            if obligation.key == "category_stats":
                return _ReactAction("library.explorer.category_stats", {}, "Read category statistics for an unmet answer obligation.")
            if obligation.key == "document_category_details":
                document_ids = selected_document_ids or self._document_ids_from_observations(observations)
                if not document_ids:
                    document_ids = self._document_category_target_ids(content, observations)
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": document_ids},
                    "Read per-document category links for an unmet answer obligation.",
                )
            if obligation.key == "resolved_document_set":
                category_name = self._category_name_from_request_or_observations(content, observations)
                arguments: dict[str, Any] = {"query": content, "expected": "many"}
                if category_name and self._category_exists(category_name):
                    arguments["category_name"] = category_name
                elif self._is_assignment_intent(content) and not self._extract_document_tokens(content):
                    return None
                return _ReactAction(
                    "library.explorer.find_documents",
                    arguments,
                    "Resolve the paper set required by the remaining task.",
                )
            if obligation.key == "create_category_verified":
                category_name = self._extract_category_name_from_request(content) or ""
                return _ReactAction(
                    "library.operator.create_category",
                    {"category_name": category_name},
                    "Create the requested category before answering.",
                )
            if obligation.key == "assign_category_verified":
                if (
                    not self._needs_untagged_assignment(content)
                    and not selected_document_ids
                    and not self._mentions_previous_referent(content)
                ):
                    document_ids = self._document_ids_from_observations(observations)
                    if not document_ids:
                        return None
                if "library.operator.create_category" in completed_tools and self._needs_untagged_assignment(content):
                    category_name = self._extract_category_name_from_request(content) or self._category_name_from_observations(observations) or ""
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": category_name, "scope": "untagged"},
                        "Assign the created category to the requested untagged documents.",
                    )
                can_target_untagged_from_context = self._should_target_untagged_from_context(
                    content,
                    self._read_react_state(session.id),
                )
                if (
                    not self._needs_untagged_assignment(content)
                    and not can_target_untagged_from_context
                    and not selected_document_ids
                    and not self._document_ids_from_observations(observations)
                ):
                    return None
                forced_action = self._force_llm_write_tool_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
                if forced_action is not None:
                    return forced_action
                return self._fallback_next_react_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
            if obligation.key == "rename_category_verified":
                rename_pair = self._extract_category_rename_request(content)
                return _ReactAction(
                    "library.operator.rename_category",
                    {
                        "source_category_name": rename_pair[0] if rename_pair else "",
                        "target_category_name": rename_pair[1] if rename_pair else "",
                    },
                    "Execute the requested category rename/merge before answering.",
                )
            if obligation.key == "clear_categories_verified":
                category_name = self._category_name_from_request_or_observations(content, observations)
                arguments = {"category_name": category_name} if category_name else {
                    "scope": "all" if self._mentions_all_library(content) else "documents",
                    "document_ids": selected_document_ids or self._document_ids_from_observations(observations),
                }
                return _ReactAction(
                    "library.operator.clear_categories",
                    arguments,
                    "Execute the requested category clearing before answering.",
                )
            if obligation.key == "category_evidence":
                category_name = self._category_name_from_request_or_observations(content, observations)
                return _ReactAction(
                    "evidence.retriever.search_by_category",
                    {"question": content, "category_names": [category_name] if category_name else []},
                    "Retrieve grouped evidence for the requested category summary.",
                )
            if obligation.key == "category_report":
                return _ReactAction(
                    "report.drafter.write_by_category",
                    {"question": content},
                    "Draft the category-grouped answer from retrieved observations.",
                )
            if obligation.key == "evidence_retrieval":
                document_ids = selected_document_ids or self._document_ids_from_observations(observations)
                if not document_ids:
                    category_name = self._category_name_from_request_or_observations(content, observations)
                    if category_name and self._category_exists(category_name):
                        return _ReactAction(
                            "library.explorer.find_documents",
                            {"category_name": category_name, "expected": "many"},
                            "Resolve category documents before evidence retrieval.",
                        )
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"query": content, "expected": "many"},
                        "Resolve documents before evidence retrieval.",
                    )
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": document_ids},
                    "Retrieve evidence for the requested report.",
                )
            if obligation.key == "report":
                return _ReactAction(
                    "report.drafter.write",
                    {"question": content, "document_ids": self._document_ids_from_observations(observations)},
                    "Draft the final answer from retrieved evidence.",
                )
        return None

    def _fallback_next_react_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction:
        completed_tools = [observation.tool for observation in observations if observation.status == "completed"]
        document_ids = self._document_ids_from_observations(observations) or selected_document_ids
        category_name = self._category_name_from_request_or_observations(content, observations)
        assignment_category_name = self._extract_category_name_from_request(content) or category_name
        rename_pair = self._extract_category_rename_request(content)
        referent_state = self._read_react_state(session.id)
        referent_ids = self._state_document_ids(referent_state)

        if not observations:
            if self._is_clear_categories_intent(content):
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.operator.clear_categories",
                        {"category_name": category_name},
                        "按用户指定标签定位论文，再清空这些论文的分类/标签关系。",
                    )
                return _ReactAction(
                    "library.operator.clear_categories",
                    {
                        "scope": "all" if self._mentions_all_library(content) else "documents",
                        "document_ids": selected_document_ids,
                    },
                    "清除论文的分类/标签关系；标签和分类在系统中是同一字段。",
                )
            if rename_pair and not self._is_assignment_intent(content):
                return _ReactAction(
                    "library.operator.rename_category",
                    {
                        "source_category_name": rename_pair[0],
                        "target_category_name": rename_pair[1],
                    },
                    "将用户的标签替换意图解析为保留关联的安全重命名/合并。",
                )
            if self._needs_library_stats(content):
                return _ReactAction("library.explorer.stats", {}, "读取论文库统计。")
            if self._needs_category_stats(content):
                return _ReactAction("library.explorer.category_stats", {}, "读取标签与分类统计。")
            if self._requires_document_category_observation(content, selected_document_ids):
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": selected_document_ids},
                    "读取每篇论文和真实标签/分类的关联。",
                )
            if self._is_create_category_intent(content):
                return _ReactAction(
                    "library.operator.create_category",
                    {"category_name": assignment_category_name or ""},
                    "先创建用户明确指定的标签/分类。",
                )
            if category_name and self._category_exists(category_name) and self._is_summary_request(content, selected_document_ids):
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"category_name": category_name, "expected": "many"},
                    "按用户指定的标签读取真实论文集合。",
                )
            if self._is_assignment_intent(content):
                if self._needs_untagged_assignment(content):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": assignment_category_name or "", "scope": "untagged"},
                        "给当前无标签论文追加指定标签。",
                    )
                if self._should_target_untagged_from_context(content, referent_state):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": assignment_category_name or "", "scope": "untagged"},
                        "根据上下文指代定位到剩余未打标签论文，并追加指定标签。",
                    )
                if self._mentions_previous_referent(content) and referent_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": assignment_category_name or "", "scope": "last_referenced"},
                        "把上一轮对话指代的论文集合解析为真实 document ids 后追加标签。",
                    )
                if selected_document_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": assignment_category_name or "", "document_ids": selected_document_ids},
                        "给当前选中论文追加指定标签。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "many"},
                    "先定位需要打标签的论文。",
                )
            if self._is_document_category_query(content):
                if selected_document_ids:
                    return _ReactAction(
                        "library.explorer.document_categories",
                        {"document_ids": selected_document_ids},
                        "读取选中论文的标签。",
                    )
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"category_name": category_name, "expected": "many"},
                        "按用户指定的标签读取真实论文集合。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "one"},
                    "先根据标题片段定位论文。",
                )
            if self._is_grouped_category_summary_request(content):
                return _ReactAction(
                    "evidence.retriever.search_by_category",
                    {"question": content, "category_names": [category_name] if category_name else []},
                    "按标签分别检索正文证据并评估每组覆盖情况。",
                )
            if selected_document_ids:
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": selected_document_ids},
                    "用户已选中论文，先检索本地证据。",
                )
            if self._is_summary_request(content, selected_document_ids):
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"category_name": category_name, "expected": "many"},
                        "按用户指定标签读取真实论文集合。",
                    )
                if selected_document_ids:
                    return _ReactAction(
                        "evidence.retriever.search",
                        {"question": content, "document_ids": selected_document_ids},
                        "先检索选中论文的证据。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "many"},
                    "先模糊匹配用户提到的论文。",
                )
            return _ReactAction(
                "final.answer",
                {"content": "我需要先知道你希望我查论文库、改标签，还是基于论文生成总结。"},
                "无法安全选择工具。",
            )

        if "library.explorer.stats" in completed_tools and self._needs_category_stats(content) and "library.explorer.category_stats" not in completed_tools:
            return _ReactAction("library.explorer.category_stats", {}, "继续读取标签覆盖情况。")
        if (
            "library.explorer.category_stats" in completed_tools
            and self._requires_document_category_observation(content, selected_document_ids)
            and "library.explorer.document_categories" not in completed_tools
        ):
            return _ReactAction(
                "library.explorer.document_categories",
                {"document_ids": self._document_category_target_ids(content, observations)},
                "继续读取每篇论文和真实标签/分类的关联，补全复合查询的明细部分。",
            )
        if "library.operator.create_category" in completed_tools and self._needs_untagged_assignment(content) and "library.operator.assign_category" not in completed_tools:
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": assignment_category_name or "", "scope": "untagged"},
                "把刚创建的标签追加给无标签论文。",
            )
        if "library.operator.assign_category" in completed_tools and rename_pair and "library.operator.rename_category" not in completed_tools:
            return _ReactAction(
                "library.operator.rename_category",
                {"source_category_name": rename_pair[0], "target_category_name": rename_pair[1]},
                "继续执行同一复合命令里的标签重命名/合并。",
            )
        if "library.operator.rename_category" in completed_tools:
            return _ReactAction(
                "final.answer",
                {"content": self._synthesize_react_answer(content, observations)},
                "标签重命名/合并已完成，生成最终回答。",
            )
        if "library.explorer.find_documents" in completed_tools:
            if self._is_document_category_query(content) and "library.explorer.document_categories" not in completed_tools:
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": document_ids},
                    "读取已定位论文的标签。",
                )
            if self._is_assignment_intent(content) and "library.operator.assign_category" not in completed_tools:
                return _ReactAction(
                    "library.operator.assign_category",
                    {"category_name": assignment_category_name or "", "document_ids": document_ids},
                    "给已定位论文追加标签。",
                )
            if self._is_clear_categories_intent(content) and "library.operator.clear_categories" not in completed_tools:
                return _ReactAction(
                    "library.operator.clear_categories",
                    {"document_ids": document_ids},
                    "清空已定位论文的分类/标签关系。",
                )
            if self._is_summary_request(content, selected_document_ids) and "evidence.retriever.search" not in completed_tools:
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": document_ids},
                    "检索已定位论文的证据。",
                )
        if "evidence.retriever.search" in completed_tools and "report.drafter.write" not in completed_tools:
            return _ReactAction(
                "report.drafter.write",
                {"question": content, "document_ids": document_ids},
                "基于检索观察生成回答。",
            )
        if "evidence.retriever.search_by_category" in completed_tools and "report.drafter.write_by_category" not in completed_tools:
            return _ReactAction(
                "report.drafter.write_by_category",
                {"question": content},
                "基于每个标签的证据覆盖情况生成分组总结。",
            )
        return _ReactAction(
            "final.answer",
            {"content": self._synthesize_react_answer(content, observations)},
            "已有足够观察，生成最终回答。",
        )

    def _execute_react_action(
        self,
        *,
        run_id: str,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        tool_map = {
            self._INTERNAL_DEGRADED_TOOL: self._tool_intent_degraded,
            self._INTERNAL_CATEGORY_CONFLICT_TOOL: self._tool_category_semantics_conflict,
            "tool.registry.list": self._tool_registry_list,
            "library.explorer.stats": self._tool_library_stats,
            "library.explorer.category_stats": self._tool_category_stats,
            "library.explorer.find_documents": self._tool_find_documents,
            "library.explorer.document_categories": self._tool_document_categories,
            "library.operator.create_category": self._tool_create_category,
            "library.operator.assign_category": self._tool_assign_category,
            "library.operator.rename_category": self._tool_rename_category,
            "library.operator.clear_categories": self._tool_clear_categories,
            "evidence.retriever.search": self._tool_retrieve_evidence,
            "evidence.retriever.search_by_category": self._tool_retrieve_evidence_by_category,
            "report.drafter.write": self._tool_draft_report,
            "report.drafter.write_by_category": self._tool_draft_report_by_category,
            "memory.read": self._tool_memory_read,
            "memory.write": self._tool_memory_write,
        }
        executor = tool_map.get(action.tool)
        if executor is None:
            return _ReactObservation(
                tool=action.tool,
                status="validation_failed",
                summary=f"未知工具：{action.tool}",
            )
        observation = executor(run_id, session, content, action.arguments, observations)
        if observation.payload.get("library_mutated"):
            observation.payload.setdefault("verified_state", self._category_stats_payload())
            observation.payload.setdefault(
                "affected_document_ids",
                [item for item in observation.payload.get("document_ids") or [] if isinstance(item, str)],
            )
            observation.payload.setdefault("operation_summary", observation.summary)
        return observation

    def _tool_intent_degraded(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        reason = str(arguments.get("reason") or "Configured LLM did not produce a usable tool plan.")
        return _ReactObservation(
            tool=self._INTERNAL_DEGRADED_TOOL,
            status="degraded",
            summary=(
                "模型意图识别/工具规划没有返回可执行的结构化结果，本轮已停止自动工具调用；"
                "没有用规则分支冒充模型理解。"
            ),
            payload={"reason": reason, "llm_configured": bool(self.api_key), "fallback_used": False},
        )

    def _tool_category_semantics_conflict(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        return _ReactObservation(
            tool=self._INTERNAL_CATEGORY_CONFLICT_TOOL,
            status="needs_clarification",
            summary="当前系统中标签和分类是同一字段，无法同时清除标签又保留分类；请确认是否要清空分类。",
            payload={
                "library_mutated": False,
                "operation_summary": "tag_category_same_field_conflict",
                "verified_state": self._category_stats_payload(),
            },
        )

    def _tool_registry_list(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        tools = self._react_tool_specs()
        return _ReactObservation(
            tool="tool.registry.list",
            status="completed",
            summary=f"已读取 {len(tools)} 个可用工具及其权限说明。",
            payload={"tools": tools},
        )

    def _tool_library_stats(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        def worker() -> _TaskOutcome:
            payload = self._library_stats_payload()
            return _TaskOutcome(
                summary=f"Found {payload['total']} library documents.",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read PaperDesk library statistics from SQLite.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "tool": "library.explorer.stats"},
            worker=worker,
        )
        payload = outcome.payload
        return _ReactObservation(
            tool="library.explorer.stats",
            status="completed",
            summary=f"SQLite 中共有 {payload['total']} 篇论文，{payload['ready']} 篇可用。",
            payload=payload,
        )

    def _tool_category_stats(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        def worker() -> _TaskOutcome:
            payload = self._category_stats_payload()
            return _TaskOutcome(
                summary=(
                    f"Found {payload['category_count']} categories, "
                    f"{payload['tagged_document_count']} tagged documents."
                ),
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read PaperDesk category and tag coverage statistics.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "tool": "library.explorer.category_stats"},
            worker=worker,
        )
        payload = outcome.payload
        return _ReactObservation(
            tool="library.explorer.category_stats",
            status="completed",
            summary=(
                f"SQLite 中有 {payload['tagged_document_count']} 篇有标签论文，"
                f"{payload['untagged_document_count']} 篇无标签论文，"
                f"{payload['category_count']} 类标签。"
            ),
            payload=payload,
        )

    def _tool_find_documents(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        query = str(arguments.get("query") or content)
        expected = str(arguments.get("expected") or "many")
        selected_ids = [str(item) for item in arguments.get("selected_document_ids") or [] if item]
        allow_all = bool(arguments.get("allow_all")) or self._mentions_all_library(content)
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        if not category_name and self._is_summary_request(content, selected_ids):
            mentioned_category = self._extract_existing_category_mention(content)
            if mentioned_category:
                category_name = mentioned_category

        def worker() -> _TaskOutcome:
            if category_name:
                documents = self._documents_for_category(category_name)
            else:
                documents = self._resolve_documents(query, selected_ids, allow_all=allow_all)
            payload = {
                "documents": [self._document_payload(document) for document in documents],
                "document_ids": [document.id for document in documents],
                "candidates": self._candidate_titles(),
                "category_name": category_name or None,
            }
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s).",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Fuzzy match library documents from the user request.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "query": query, "tool": "library.explorer.find_documents"},
            worker=worker,
        )
        documents = outcome.payload.get("documents") or []
        if not documents:
            return _ReactObservation(
                tool="library.explorer.find_documents",
                status="needs_clarification",
                summary="没有在论文库中唯一定位到用户提到的论文。",
                payload=outcome.payload,
            )
        if expected == "one" and len(documents) > 1:
            return _ReactObservation(
                tool="library.explorer.find_documents",
                status="needs_clarification",
                summary="匹配到多篇候选论文，需要用户确认具体是哪一篇。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="library.explorer.find_documents",
            status="completed",
            summary=f"已定位 {len(documents)} 篇论文。",
            payload=outcome.payload,
        )

    def _tool_document_categories(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            document_ids = self._document_ids_from_observations(observations)
        if not document_ids and self._requests_all_document_categories(content):
            document_ids = self._document_category_target_ids(content, observations)

        def worker() -> _TaskOutcome:
            documents = [document for document in self.document_library_service.list_documents() if document.id in set(document_ids)]
            payload = {
                "documents": [
                    {
                        **self._document_payload(document),
                        "categories": [category.name for category in document.categories],
                    }
                    for document in documents
                ],
                "document_ids": [document.id for document in documents],
            }
            return _TaskOutcome(
                summary=f"Read categories for {len(documents)} document(s).",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read selected document categories as tags.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "document_ids": document_ids},
            worker=worker,
        )
        documents = outcome.payload.get("documents") or []
        if not documents:
            return _ReactObservation(
                tool="library.explorer.document_categories",
                status="needs_clarification",
                summary="没有找到可读取标签的论文。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="library.explorer.document_categories",
            status="completed",
            summary=f"已读取 {len(documents)} 篇论文的标签。",
            payload=outcome.payload,
        )

    def _tool_create_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        if not category_name:
            category_name = self._extract_category_name_from_request(content) or ""
        validation_error = self._category_name_validation_error(category_name)
        if validation_error:
            return _ReactObservation(
                tool="library.operator.create_category",
                status="validation_failed",
                summary=validation_error,
                payload={"category_name": category_name},
            )

        def worker() -> _TaskOutcome:
            category, created = self._get_or_create_category(category_name)
            return _TaskOutcome(
                summary=f"Category ready: {category.name}.",
                payload={
                    "category": category.model_dump(mode="json"),
                    "category_name": category.name,
                    "created": created,
                    "library_mutated": created,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Create a non-destructive category/tag if it does not already exist.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name},
            worker=worker,
        )
        created = bool(outcome.payload.get("created"))
        return _ReactObservation(
            tool="library.operator.create_category",
            status="completed",
            summary=f"{'已新建' if created else '已存在'}标签/分类「{outcome.payload['category_name']}」。",
            payload=outcome.payload,
        )

    def _tool_assign_category(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        if not category_name or category_name in {"这个标签", "该标签", "此标签"}:
            if self._needs_untagged_assignment(content):
                category_name = self._extract_category_name_from_request(content) or ""
            else:
                category_name = self._category_name_from_observations(observations) or self._extract_category_name_from_request(content) or ""
        validation_error = self._category_name_validation_error(category_name)
        if validation_error:
            return _ReactObservation(
                tool="library.operator.assign_category",
                status="validation_failed",
                summary=validation_error,
                payload={"category_name": category_name},
            )

        scope = str(arguments.get("scope") or "")
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids and self._should_target_untagged_from_context(content, self._read_react_state(session.id)):
            document_ids = []
            scope = "untagged"
        if scope == "last_referenced" or (
            not document_ids and scope != "untagged" and self._mentions_previous_referent(content)
        ):
            document_ids = self._state_document_ids(self._read_react_state(session.id))
            scope = "last_referenced"
        if not document_ids and scope != "untagged":
            document_ids = self._document_ids_from_observations(observations)

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            if scope == "untagged":
                target_documents = [document for document in documents if not document.categories]
            else:
                selected = set(document_ids)
                target_documents = [document for document in documents if document.id in selected]
            category, _created = self._get_or_create_category(category_name)
            updated = []
            for document in target_documents:
                existing_ids = [item.id for item in document.categories]
                next_ids = list(dict.fromkeys([*existing_ids, category.id]))
                categories = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [item.name for item in categories],
                    }
                )
            verified_documents = self.document_library_service.list_documents()
            verified_tagged = [
                document.id
                for document in verified_documents
                if any(category_item.name == category.name for category_item in document.categories)
            ]
            return _TaskOutcome(
                summary=f"Assigned {category.name} to {len(updated)} document(s).",
                payload={
                    "category": category.model_dump(mode="json"),
                    "category_name": category.name,
                    "scope": scope or "documents",
                    "updated_count": len(updated),
                    "documents": updated,
                    "document_ids": [item["id"] for item in updated],
                    "verified_tagged_document_ids": verified_tagged,
                    "library_mutated": bool(updated),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Append a category/tag to matched library documents.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name, "scope": scope, "document_ids": document_ids},
            worker=worker,
        )
        payload = outcome.payload
        if payload["updated_count"] == 0 and scope != "untagged":
            return _ReactObservation(
                tool="library.operator.assign_category",
                status="needs_clarification",
                summary="没有定位到需要打标签的论文。",
                payload=payload,
            )
        return _ReactObservation(
            tool="library.operator.assign_category",
            status="completed",
            summary=f"已把标签/分类「{payload['category_name']}」追加到 {payload['updated_count']} 篇论文。",
            payload=payload,
        )

    def _tool_rename_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        source_name = self._clean_category_name(str(arguments.get("source_category_name") or ""))
        target_name = self._clean_category_name(str(arguments.get("target_category_name") or ""))
        if not source_name or not target_name:
            extracted = self._extract_category_rename_request(content)
            if extracted:
                source_name = source_name or extracted[0]
                target_name = target_name or extracted[1]
        source_error = self._category_name_validation_error(source_name)
        target_error = self._category_name_validation_error(target_name)
        if source_error or target_error:
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="validation_failed",
                summary=source_error or target_error or "标签名称无效，本轮没有改动论文库。",
                payload={"source_category_name": source_name, "target_category_name": target_name},
            )
        if source_name == target_name:
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="completed",
                summary=f"源标签和目标标签同为「{source_name}」，无需改动。",
                payload={
                    "source_category_name": source_name,
                    "target_category_name": target_name,
                    "updated_count": 0,
                    "document_ids": [],
                    "merged": False,
                    "library_mutated": False,
                },
            )

        def worker() -> _TaskOutcome:
            categories = self.category_repository.list_categories()
            source = next((category for category in categories if category.name == source_name), None)
            if source is None:
                return _TaskOutcome(
                    summary=f"Source category not found: {source_name}.",
                    payload={
                        "source_category_name": source_name,
                        "target_category_name": target_name,
                        "source_found": False,
                        "updated_count": 0,
                        "document_ids": [],
                        "library_mutated": False,
                    },
                )
            target = next((category for category in categories if category.name == target_name), None)
            if target is None:
                target = self.category_repository.create_category(target_name, source.color)
                merged = False
            else:
                merged = True

            documents = self.document_library_service.list_documents()
            updated_documents = []
            for document in documents:
                if not any(category.id == source.id for category in document.categories):
                    continue
                next_ids = [
                    category.id
                    for category in document.categories
                    if category.id != source.id and category.id != target.id
                ]
                next_ids.append(target.id)
                categories_after = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated_documents.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [category.name for category in categories_after],
                    }
                )
            self.category_repository.delete_category(source.id)
            return _TaskOutcome(
                summary=f"Renamed category {source_name} to {target_name} for {len(updated_documents)} documents.",
                payload={
                    "source_category_name": source_name,
                    "target_category_name": target_name,
                    "source_category_id": source.id,
                    "target_category_id": target.id,
                    "source_found": True,
                    "merged": merged,
                    "updated_count": len(updated_documents),
                    "documents": updated_documents,
                    "document_ids": [document["id"] for document in updated_documents],
                    "library_mutated": True,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Safely rename or merge a category while preserving document-category links.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"source_category_name": source_name, "target_category_name": target_name},
            worker=worker,
        )
        payload = outcome.payload
        if not payload.get("source_found", True):
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="needs_clarification",
                summary=f"没有找到名为「{source_name}」的源标签，因此没有改动论文库。",
                payload=payload,
            )
        verb = "合并到已有标签" if payload.get("merged") else "重命名为"
        return _ReactObservation(
            tool="library.operator.rename_category",
            status="completed",
            summary=(
                f"已将标签/分类「{source_name}」{verb}「{target_name}」，"
                f"保留并迁移了 {payload.get('updated_count', 0)} 篇论文的标签关联。"
            ),
            payload=payload,
        )

    def _tool_clear_categories(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        if self._is_tag_category_semantics_conflict(content):
            return self._tool_category_semantics_conflict(run_id, session, content, arguments, observations)
        scope = str(arguments.get("scope") or "")
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        if not category_name:
            category_name = self._category_name_from_request_or_observations(content, observations) or ""
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids and category_name and self._category_exists(category_name):
            document_ids = [document.id for document in self._documents_for_category(category_name)]
        if not document_ids and scope == "last_referenced":
            document_ids = self._state_document_ids(self._read_react_state(session.id))
        if not document_ids and scope not in {"all", "tagged"}:
            document_ids = self._document_ids_from_observations(observations)
        if not scope:
            scope = "all" if self._mentions_all_library(content) or self._mentions_all_categories(content) else "documents"

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            if scope == "all":
                target_documents = documents
            elif scope == "tagged":
                target_documents = [document for document in documents if document.categories]
            else:
                selected = set(document_ids)
                target_documents = [document for document in documents if document.id in selected]
            updated_documents = []
            for document in target_documents:
                if not document.categories:
                    continue
                self.category_repository.replace_document_categories(document.id, [])
                updated_documents.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [],
                    }
                )
            verified_state = self._category_stats_payload()
            return _TaskOutcome(
                summary=f"Cleared categories for {len(updated_documents)} document(s).",
                payload={
                    "scope": scope,
                    "category_name": category_name or None,
                    "updated_count": len(updated_documents),
                    "documents": updated_documents,
                    "document_ids": [document["id"] for document in updated_documents],
                    "affected_document_ids": [document["id"] for document in updated_documents],
                    "library_mutated": bool(updated_documents),
                    "verified_state": verified_state,
                    "operation_summary": "cleared_document_categories",
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Clear document category/tag links. Tags and categories are the same field in PaperDesk.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={
                "scope": scope,
                "category_name": category_name or None,
                "document_ids": document_ids,
                "tag_category_same_field": True,
            },
            worker=worker,
        )
        payload = outcome.payload
        return _ReactObservation(
            tool="library.operator.clear_categories",
            status="completed",
            summary=f"已清空 {payload.get('updated_count', 0)} 篇论文的分类/标签关系。",
            payload=payload,
        )

    def _tool_retrieve_evidence(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            document_ids = self._document_ids_from_observations(observations)
        if not document_ids:
            category_name = self._category_name_from_request_or_observations(content, observations)
            if category_name and self._category_exists(category_name):
                document_ids = [document.id for document in self._documents_for_category(category_name)]
        documents = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids) and document.status == "ready"
        ]

        def worker() -> _TaskOutcome:
            evidence = self.rag_service.retrieve_evidence(
                question=str(arguments.get("question") or content),
                documents=documents,
                top_k=min(10, max(4, len(documents) * 3)),
            )
            return _TaskOutcome(
                summary=f"Retrieved {len(evidence)} evidence item(s).",
                payload={
                    "document_ids": [document.id for document in documents],
                    "evidence_items": [item.model_dump(mode="json") for item in evidence],
                    "evidence_quality": self.rag_service.last_evidence_quality.model_dump(mode="json"),
                    "cache_hit": self.rag_service.last_retrieval_cache_hit,
                    "retrieval_strategy": self.rag_service.last_retrieval_strategy,
                },
            )

        if not documents:
            return _ReactObservation(
                tool="evidence.retriever.search",
                status="needs_clarification",
                summary=self._no_ready_documents_message(document_ids),
                payload={"document_ids": document_ids},
            )
        try:
            outcome = self._run_subagent(
                run_id=run_id,
                role="evidence-retriever",
                profile=SubagentProfile.EXPLORE,
                goal="Retrieve local evidence for the ReAct answer.",
                tool_policy=ToolPolicy(read_only=True),
                context_bundle={"question": content, "document_ids": document_ids},
                worker=worker,
            )
        except Exception as exc:
            return _ReactObservation(
                tool="evidence.retriever.search",
                status="failed",
                summary=self._retrieval_service_unavailable_message(),
                payload={
                    "document_ids": document_ids,
                    "error_type": exc.__class__.__name__,
                    "requires_service": "milvus",
                },
            )
        retrieval_strategy = str(outcome.payload.get("retrieval_strategy") or "")
        evidence_count = len(outcome.payload.get("evidence_items") or [])
        summary = (
            f"Milvus 向量服务不可用，已通过论文正文关键词检索到 {evidence_count} 条证据。"
            if retrieval_strategy == "keyword_only_vector_unavailable"
            else f"已检索到 {evidence_count} 条证据。"
        )
        return _ReactObservation(
            tool="evidence.retriever.search",
            status="completed",
            summary=summary,
            payload=outcome.payload,
        )

    def _tool_retrieve_evidence_by_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        question = str(arguments.get("question") or content)
        requested_categories = {
            str(item).strip()
            for item in arguments.get("category_names") or []
            if str(item).strip()
        }

        def worker() -> _TaskOutcome:
            documents = [document for document in self.document_library_service.list_documents() if document.status == "ready"]
            groups: dict[str, list[LibraryDocument]] = {}
            for document in documents:
                for category in document.categories:
                    if requested_categories and category.name not in requested_categories:
                        continue
                    groups.setdefault(category.name, []).append(document)

            category_groups = []
            flat_evidence = []
            for category_name, group_documents in groups.items():
                retrieval = self.rag_service.retrieve_evidence_with_quality(
                    question=f"{question}\n请优先检索与「{category_name}」标签相关的证据。",
                    documents=group_documents,
                    top_k=min(12, max(4, len(group_documents) * 4)),
                )
                evidence = retrieval.evidence_items
                evidence_payload = [item.model_dump(mode="json") for item in evidence]
                flat_evidence.extend(evidence_payload)
                boundary = None
                if not evidence_payload:
                    boundary = "该标签下论文存在，但正文证据不足。"
                category_groups.append(
                    {
                        "category_name": category_name,
                        "document_ids": [document.id for document in group_documents],
                        "documents": [self._document_payload(document) for document in group_documents],
                        "evidence_items": evidence_payload,
                        "evidence_quality": retrieval.evidence_quality.model_dump(mode="json"),
                        "cache_hit": retrieval.cache_hit,
                        "retrieval_strategy": retrieval.retrieval_strategy,
                        "evidence_boundary": boundary,
                    }
                )
            return _TaskOutcome(
                summary=f"Retrieved grouped evidence for {len(category_groups)} category group(s).",
                payload={
                    "category_groups": category_groups,
                    "evidence_items": flat_evidence,
                    "document_ids": list(
                        dict.fromkeys(
                            document_id
                            for group in category_groups
                            for document_id in group["document_ids"]
                        )
                    ),
                },
            )

        try:
            outcome = self._run_subagent(
                run_id=run_id,
                role="evidence-retriever",
                profile=SubagentProfile.EXPLORE,
                goal="Retrieve local evidence grouped by category/tag.",
                tool_policy=ToolPolicy(read_only=True),
                context_bundle={
                    "question": content,
                    "category_names": sorted(requested_categories),
                    "tool": "evidence.retriever.search_by_category",
                },
                worker=worker,
            )
        except Exception as exc:
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="failed",
                summary=self._retrieval_service_unavailable_message(grouped=True),
                payload={
                    "error_type": exc.__class__.__name__,
                    "requires_service": "milvus",
                },
            )
        groups = outcome.payload.get("category_groups") or []
        if not groups:
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="needs_clarification",
                summary="没有找到带标签且可用于检索的 ready 论文分组。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="evidence.retriever.search_by_category",
            status="completed",
            summary=f"已按 {len(groups)} 个标签分组检索证据。",
            payload=outcome.payload,
        )

    @staticmethod
    def _retrieval_service_unavailable_message(*, grouped: bool = False) -> str:
        scope = "按标签分组的论文正文检索" if grouped else "论文正文向量检索"
        return (
            f"{scope}服务当前不可用，无法完成基于所选论文正文的分析。"
            "请先启动 Milvus 向量服务并确认论文状态为 ready，然后重新发送问题。"
            "本轮不会改用普通聊天或仅凭论文元数据生成分析结论。"
        )

    def _no_ready_documents_message(self, document_ids: list[str]) -> str:
        selected = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids)
        ]
        if not selected:
            return "没有可用于检索的 ready 论文。"
        status_text = "、".join(
            f"{document.display_name or document.filename}（{document.status}）"
            for document in selected[:6]
        )
        suffix = "等" if len(selected) > 6 else ""
        return f"没有可用于检索的 ready 论文；当前选中的 {status_text}{suffix} 尚不能用于正文分析。"

    def _tool_draft_report(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            document_ids = self._document_ids_from_observations(observations)
        documents = [document for document in self.document_library_service.list_documents() if document.id in set(document_ids)]
        evidence_items = self._evidence_items_from_observations(observations)

        def worker() -> _TaskOutcome:
            answer = self._draft_with_llm(str(arguments.get("question") or content), documents, evidence_items)
            return _TaskOutcome(
                summary="Drafted final ReAct answer from observations.",
                payload={
                    "answer": answer,
                    "document_ids": [document.id for document in documents],
                    "evidence_items": [item.model_dump(mode="json") for item in evidence_items],
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Write a Markdown answer grounded in retrieved observations.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "document_ids": document_ids, "evidence_count": len(evidence_items)},
            worker=worker,
        )
        return _ReactObservation(
            tool="report.drafter.write",
            status="completed",
            summary="已生成基于观察结果的回答。",
            payload=outcome.payload,
        )

    def _tool_draft_report_by_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        target_chars = int(arguments.get("target_chars") or arguments.get("target_words") or 600)
        category_groups = self._category_groups_from_observations(observations)
        if not category_groups:
            category_name = self._category_name_from_request_or_observations(content, observations)
            category_groups = self._category_groups_from_library([category_name] if category_name else None)

        def worker() -> _TaskOutcome:
            answer = self._draft_category_summaries_with_llm(
                question=str(arguments.get("question") or content),
                category_groups=category_groups,
                target_chars=target_chars,
            )
            return _TaskOutcome(
                summary=f"Drafted summaries for {len(category_groups)} category group(s).",
                payload={
                    "answer": answer,
                    "category_groups": category_groups,
                    "document_ids": list(
                        dict.fromkeys(
                            document_id
                            for group in category_groups
                            for document_id in group.get("document_ids", [])
                        )
                    ),
                    "target_chars": target_chars,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Write grouped Markdown summaries by category/tag.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={
                "question": content,
                "category_count": len(category_groups),
                "target_chars": target_chars,
            },
            worker=worker,
        )
        return _ReactObservation(
            tool="report.drafter.write_by_category",
            status="completed",
            summary=f"已按标签生成 {len(category_groups)} 份总结。",
            payload=outcome.payload,
        )

    def _tool_memory_read(
        self,
        _run_id: str,
        session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        payload = self._react_memory_payload(session.id)
        return _ReactObservation(
            tool="memory.read",
            status="completed",
            summary="已读取项目规则、用户偏好和历史反思。",
            payload=payload,
        )

    def _tool_memory_write(
        self,
        _run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        summary = str(arguments.get("summary") or "").strip()
        if not summary:
            summary = self._reflection_summary(content, observations, "completed")
        if summary:
            self.file_store.add_user_preference(summary[:160])
        return _ReactObservation(
            tool="memory.write",
            status="completed",
            summary="已写入一条轻量反思记忆。",
            payload={"summary": summary[:160]},
        )

    def _validate_react_action(self, action: _ReactAction, observations: list[_ReactObservation]) -> str | None:
        tool_names = {item["name"] for item in self._react_tool_specs()} | {"final.answer"}
        if action.tool in {self._INTERNAL_DEGRADED_TOOL, self._INTERNAL_CATEGORY_CONFLICT_TOOL}:
            return None
        if action.tool not in tool_names:
            return f"模型请求了未注册工具「{action.tool}」，本轮已拦截。"
        if action.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            category_name = self._clean_category_name(str(action.arguments.get("category_name") or ""))
            if not category_name or category_name in {"这个标签", "该标签", "此标签"}:
                category_name = self._category_name_from_observations(observations) or category_name
            if not category_name:
                return "我没有可靠识别出要使用的标签/分类名称，因此没有改动论文库。"
            error = self._category_name_validation_error(category_name)
            if error:
                return error
            if action.tool == "library.operator.assign_category":
                scope = str(action.arguments.get("scope") or "")
                document_ids = action.arguments.get("document_ids") or []
                if (
                    scope not in {"untagged", "last_referenced"}
                    and not document_ids
                    and not self._document_ids_from_observations(observations)
                ):
                    return "我没有可靠定位到要打标签的论文，因此没有改动论文库。"
                action.arguments["category_name"] = category_name
        if action.tool == "library.operator.rename_category":
            source_name = self._clean_category_name(str(action.arguments.get("source_category_name") or ""))
            target_name = self._clean_category_name(str(action.arguments.get("target_category_name") or ""))
            if not source_name or not target_name:
                return "我没有可靠识别出源标签和目标标签，因此没有改动论文库。"
            source_error = self._category_name_validation_error(source_name)
            target_error = self._category_name_validation_error(target_name)
            if source_error or target_error:
                return source_error or target_error
        if action.tool == "library.operator.clear_categories":
            scope = str(action.arguments.get("scope") or "")
            document_ids = action.arguments.get("document_ids") or []
            category_name = self._clean_category_name(str(action.arguments.get("category_name") or ""))
            if scope not in {"all", "tagged", "last_referenced", "documents", ""}:
                return "清空分类/标签的范围无效，因此没有改动论文库。"
            if category_name:
                if not self._category_exists(category_name):
                    return f"没有找到名为「{category_name}」的标签/分类，因此没有改动论文库。"
            elif scope in {"documents", ""} and not document_ids and not self._document_ids_from_observations(observations):
                return "我没有可靠定位到要清空分类/标签的论文，因此没有改动论文库。"
        if action.tool == "report.drafter.write":
            document_ids = action.arguments.get("document_ids") or self._document_ids_from_observations(observations)
            if not document_ids:
                return "我没有可靠定位到要总结的论文，因此无法生成报告。"
        return None

    def _react_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "tool.registry.list",
                "description": "List available PaperDesk tools, input expectations, and permission level.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.stats",
                "description": "Read paper counts and processing status from SQLite.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.category_stats",
                "description": "Read category/tag counts, tagged papers, untagged papers, and per-category counts.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.find_documents",
                "description": "Fuzzy match papers by title fragment, filename, abbreviation, selected IDs, or exact category_name.",
                "read_only": True,
                "arguments": {"query": "string", "expected": "one|many", "allow_all": "boolean", "category_name": "string optional"},
            },
            {
                "name": "library.explorer.document_categories",
                "description": "Read tags/categories for resolved document IDs.",
                "read_only": True,
                "arguments": {"document_ids": "list[string]"},
            },
            {
                "name": "library.operator.create_category",
                "description": "Create a category/tag. Non-destructive write. Requires a short explicit category_name.",
                "read_only": False,
                "arguments": {"category_name": "string"},
            },
            {
                "name": "library.operator.assign_category",
                "description": "Append a category/tag to document_ids or to scope=untagged. Does not overwrite existing tags.",
                "read_only": False,
                "arguments": {"category_name": "string", "scope": "untagged|last_referenced|documents", "document_ids": "list[string]"},
            },
            {
                "name": "library.operator.rename_category",
                "description": "Safely rename or merge a category/tag while preserving all paper links. Use for replace/rename semantics, not destructive delete.",
                "read_only": False,
                "arguments": {"source_category_name": "string", "target_category_name": "string"},
            },
            {
                "name": "library.operator.clear_categories",
                "description": "Clear category/tag links from document_ids, exact category_name, scope=all, scope=tagged, or scope=last_referenced. Use category_name when the user says to clear papers with a named tag. Tags and categories are the same field.",
                "read_only": False,
                "arguments": {"scope": "all|tagged|last_referenced|documents", "category_name": "string optional", "document_ids": "list[string]"},
            },
            {
                "name": "evidence.retriever.search",
                "description": "Retrieve local evidence from ready papers.",
                "read_only": True,
                "arguments": {"question": "string", "document_ids": "list[string]"},
            },
            {
                "name": "evidence.retriever.search_by_category",
                "description": "Retrieve RAG evidence grouped by category/tag for all tagged ready papers or selected category_names.",
                "read_only": True,
                "arguments": {"question": "string", "category_names": "list[string] optional"},
            },
            {
                "name": "report.drafter.write",
                "description": "Write a Markdown answer from resolved documents and retrieved evidence.",
                "read_only": True,
                "arguments": {"question": "string", "document_ids": "list[string]"},
            },
            {
                "name": "report.drafter.write_by_category",
                "description": "Write one Markdown summary per category/tag from grouped RAG observations.",
                "read_only": True,
                "arguments": {"question": "string", "target_chars": "integer, default 600"},
            },
            {
                "name": "memory.read",
                "description": "Read project rules, user preferences, and prior reflection notes.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "memory.write",
                "description": "Write a short reflection note after a task.",
                "read_only": False,
                "arguments": {"summary": "string"},
            },
        ]

    def _build_react_snapshot(
        self,
        session_id: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> dict[str, Any]:
        library_stats = self._library_stats_payload()
        category_stats = self._category_stats_payload(limit_documents=6)
        documents = self.document_library_service.list_documents()[:8]
        return {
            "library": library_stats,
            "categories": {
                "category_count": category_stats["category_count"],
                "tagged_document_count": category_stats["tagged_document_count"],
                "untagged_document_count": category_stats["untagged_document_count"],
                "category_names": [item["name"] for item in category_stats["categories"]],
            },
            "document_preview": [self._document_payload(document) for document in documents],
            "selected_document_ids": selected_document_ids,
            "attachment_document_ids": [attachment.document_id for attachment in attachments if attachment.document_id],
            "conversation_referents": self._read_react_state(session_id),
            "memory": self._react_memory_payload(session_id),
            "guardrails": [
                "有几篇有标签/几类标签必须调用 category_stats，不要只回答论文总数。",
                "新增 hdcc 标签并给无标签论文打这个标签必须拆成 create_category + assign_category(scope=untagged)。",
                "标签名只能来自引号、冒号后的短名称，或“新增一个X标签”中的 X。",
                "删除A标签并换成B标签属于保留关联的重命名/合并，必须用 rename_category。",
                "用所有X标签论文写报告时，先按 category_name 精确读取 X 标签下的论文。",
                "标签和分类是同一字段；不能声称清除标签但保留分类。",
                "会话上下文可用于指代解析，但当前库状态、计数和写入结果必须以最新 observation 为准。",
                "最终回答不要出现“根据对话记录/根据上下文/根据运行态摘要”等过程话术，直接回答用户。",
            ],
        }

    def _library_stats_payload(self) -> dict[str, Any]:
        documents = self.document_library_service.list_documents()
        return {
            "total": len(documents),
            "ready": sum(1 for document in documents if document.status == "ready"),
            "processing": sum(1 for document in documents if document.status == "processing"),
            "failed": sum(1 for document in documents if document.status == "failed"),
            "referent_document_ids": [document.id for document in documents],
            "referent_label": "全部论文",
        }

    def _category_stats_payload(self, *, limit_documents: int | None = None) -> dict[str, Any]:
        documents = self.document_library_service.list_documents()
        categories = self.category_repository.list_categories()
        counts_by_category_id = {category.id: 0 for category in categories}
        tagged_documents = []
        untagged_documents = []
        for document in documents:
            if document.categories:
                tagged_documents.append(
                    {
                        **self._document_payload(document),
                        "categories": [category.name for category in document.categories],
                    }
                )
                for category in document.categories:
                    counts_by_category_id[category.id] = counts_by_category_id.get(category.id, 0) + 1
            else:
                untagged_documents.append(self._document_payload(document))

        def maybe_limit(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if limit_documents is None:
                return items
            return items[:limit_documents]

        return {
            "category_count": len(categories),
            "categories": [
                {
                    "id": category.id,
                    "name": category.name,
                    "color": category.color,
                    "document_count": counts_by_category_id.get(category.id, 0),
                }
                for category in categories
            ],
            "total_documents": len(documents),
            "tagged_document_count": len(tagged_documents),
            "untagged_document_count": len(untagged_documents),
            "tagged_document_ids": [document["id"] for document in tagged_documents],
            "untagged_document_ids": [document["id"] for document in untagged_documents],
            "tagged_documents": maybe_limit(tagged_documents),
            "untagged_documents": maybe_limit(untagged_documents),
        }

    @staticmethod
    def _document_payload(document: LibraryDocument) -> dict[str, Any]:
        return {
            "id": document.id,
            "name": document.display_name or document.filename,
            "filename": document.filename,
            "title": document.title,
            "status": document.status,
            "page_count": document.page_count,
        }

    def _get_or_create_category(self, category_name: str):
        existing = next(
            (category for category in self.category_repository.list_categories() if category.name == category_name),
            None,
        )
        if existing is not None:
            return existing, False
        palette = ["#0f5fb8", "#047c71", "#6957d8", "#b76a00", "#b42318"]
        color = palette[len(self.category_repository.list_categories()) % len(palette)]
        category = self.category_repository.create_category(category_name, color)
        return category, True

    @staticmethod
    def _has_completed_library_write(observations: list[_ReactObservation]) -> bool:
        return any(
            observation.status == "completed" and bool(observation.payload.get("library_mutated"))
            for observation in observations
        )

    def _write_goal_fulfilled_by_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> bool:
        """Decide whether the current write request should stop planning and answer from observations."""

        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if self._is_clear_categories_intent(content):
            return "library.operator.clear_categories" in completed_tools
        if self._is_rename_category_intent(content):
            return "library.operator.rename_category" in completed_tools
        rename_pair = self._extract_category_rename_request(content)
        if rename_pair and "library.operator.rename_category" not in completed_tools:
            return False
        if self._is_assignment_intent(content):
            return "library.operator.assign_category" in completed_tools
        if rename_pair:
            return "library.operator.rename_category" in completed_tools
        if self._is_create_category_intent(content):
            return "library.operator.create_category" in completed_tools
        return False

    def _repair_premature_write_final(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        """Convert a premature LLM final answer into a validated write action."""

        if not self._is_assignment_intent(content):
            return None
        state = self._read_react_state(session.id)
        category_name = (
            self._extract_category_name_from_request(content)
            or self._category_name_from_observations(observations)
            or state.get("last_category_name")
            or ""
        )
        if not isinstance(category_name, str) or self._category_name_validation_error(category_name):
            return None
        if self._needs_untagged_assignment(content) or self._should_target_untagged_from_context(content, state):
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "scope": "untagged"},
                "模型提前给出完成回答；后端改为执行已校验的未分类论文打标签工具。",
            )
        document_ids = self._document_ids_from_observations(observations) or selected_document_ids
        if document_ids:
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "document_ids": document_ids},
                "模型提前给出完成回答；后端改为执行已校验的指定论文打标签工具。",
            )
        if self._mentions_previous_referent(content) and self._state_document_ids(state):
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "scope": "last_referenced"},
                "模型提前给出完成回答；后端改为执行已校验的上下文指代打标签工具。",
            )
        return None

    @staticmethod
    def _requires_library_write_observation(content: str) -> bool:
        write_markers = (
            "新建",
            "创建",
            "新增",
            "添加",
            "添加标签",
            "加标签",
            "加个",
            "加一个",
            "加上",
            "打上",
            "打标签",
            "打一个",
            "补上",
            "归类",
            "设置",
            "设为",
            "设成",
            "换成",
            "换为",
            "替换为",
            "替换成",
            "重命名",
            "重命名为",
            "改成",
            "改为",
            "改名为",
            "删除",
            "移除",
            "delete",
            "remove",
            "rename",
            "replace",
            "assign",
                "清除",
                "清空",
                "清楚",
                "去掉",
        )
        target_markers = ("标签", "分类", "论文", "文档", "tag", "category", "paper", "document")
        lowered = content.casefold()
        return any(marker in lowered for marker in write_markers) and any(
            marker in lowered for marker in target_markers
        )

    def _requires_tool_observation(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        return self._should_handle_with_react(content, selected_document_ids, attachments)

    def _requires_document_category_observation(
        self,
        content: str,
        selected_document_ids: list[str],
    ) -> bool:
        return self._requests_all_document_categories(content) or (
            bool(selected_document_ids) and self._is_document_category_query(content)
        )

    def _document_category_target_ids(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> list[str]:
        if self._requests_tagged_document_categories(content):
            category_stats = self._latest_payload(observations, "library.explorer.category_stats")
            if isinstance(category_stats, dict):
                return [str(item) for item in category_stats.get("tagged_document_ids") or [] if item]
            return [
                document.id
                for document in self.document_library_service.list_documents()
                if document.categories
            ]
        return [document.id for document in self.document_library_service.list_documents()]

    @staticmethod
    def _requests_all_document_categories(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        has_document_target = any(marker in content for marker in ("每篇", "每一篇", "每个", "全部", "所有", "论文", "文章", "文档"))
        has_mapping = any(marker in content for marker in ("对应", "分别", "各自", "是什么", "哪些", "列出", "查看"))
        return has_category_target and has_document_target and has_mapping

    @staticmethod
    def _requests_tagged_document_categories(content: str) -> bool:
        return any(marker in content for marker in ("带标签", "有标签", "已打标签", "已分类")) and any(
            marker in content for marker in ("分别", "对应", "是什么", "哪些", "列出", "查看")
        )

    @staticmethod
    def _has_document_category_observation(observations: list[_ReactObservation]) -> bool:
        return any(
            observation.tool == "library.explorer.document_categories" and observation.status == "completed"
            for observation in observations
        )

    @staticmethod
    def _needs_library_stats(content: str) -> bool:
        return "论文" in content and any(marker in content for marker in ("几篇", "多少", "数量", "总共", "共有"))

    @staticmethod
    def _needs_category_stats(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类", "无标签", "没有标签")) and any(
            marker in content
            for marker in (
                "几篇",
                "几类",
                "多少",
                "分别",
                "哪些",
                "统计",
                "列出",
                "最多",
                "最少",
                "最大",
                "最小",
                "排名",
                "分布",
                "占比",
            )
        )

    @staticmethod
    def _is_tagged_count_question(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("几篇", "多少", "数量", "有几")
        )

    @staticmethod
    def _is_clear_categories_intent(content: str) -> bool:
        if any(marker in content for marker in KnowledgeAgentRuntime._rename_category_markers()):
            return False
        if any(marker in content for marker in ("加个", "加一个", "添加", "新增", "打上", "补上", "归类", "assign", "add")):
            return False
        lowered = content.casefold()
        has_category_target = any(marker in lowered for marker in ("标签", "分类", "tag", "category"))
        if not has_category_target:
            return False
        if any(marker in lowered for marker in ("清除", "清空", "清楚", "去掉", "clear")):
            return True
        if any(marker in lowered for marker in ("移除", "删掉", "删除", "remove")):
            return any(
                marker in lowered
                for marker in (
                    "所有论文",
                    "全部论文",
                    "所有标签",
                    "全部标签",
                    "所有分类",
                    "全部分类",
                    "标签都",
                    "分类都",
                )
            )
        return False

    @staticmethod
    def _is_rename_category_intent(content: str) -> bool:
        if KnowledgeAgentRuntime._is_assignment_intent(content):
            return False
        return any(marker in content for marker in KnowledgeAgentRuntime._rename_category_markers()) and any(
            marker in content for marker in ("标签", "分类", "tag", "category")
        )

    @staticmethod
    def _rename_category_markers() -> tuple[str, ...]:
        return ("换成", "换为", "替换为", "替换成", "改成", "改为", "改名为", "重命名为", "重命名", "rename", "replace")

    @staticmethod
    def _is_tag_category_semantics_conflict(content: str) -> bool:
        normalized = content.casefold()
        clear_tag = (
            any(marker in normalized for marker in ("清除标签", "清空标签", "清楚标签", "删除标签", "移除标签", "去掉标签", "clear tags", "remove tags"))
            or ("标签" in normalized and any(marker in normalized for marker in ("清除", "清空", "清楚", "删除", "移除", "去掉")))
        )
        keep_category = any(
            marker in normalized
            for marker in (
                "保留分类",
                "分类保留",
                "保留对应的分类",
                "保留所属的分类",
                "keep categories",
                "preserve categories",
            )
        )
        return clear_tag and keep_category

    @staticmethod
    def _is_create_category_intent(content: str) -> bool:
        return any(marker in content for marker in ("新建", "创建", "新增", "建立", "加一个", "添加一个")) and any(
            marker in content for marker in ("分类", "标签")
        )

    @staticmethod
    def _is_assignment_intent(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "打标签",
                "打上",
                "打一个",
                "加标签",
                "加个",
                "加一个",
                "添加标签",
                "加上",
                "补上",
                "归类",
                "设置分类",
                "添加分类",
                "设置成",
                "设为",
                "设成",
                "改成",
                "标为",
                "新增",
            )
        )

    @staticmethod
    def _needs_untagged_assignment(content: str) -> bool:
        return any(marker in content for marker in ("没有标签", "无标签", "未打标签", "未分类", "没标签")) and any(
            marker in content for marker in ("加上", "补上", "添加", "打上", "都", "所有", "新增")
        )

    def _should_target_untagged_from_context(self, content: str, state: dict[str, Any]) -> bool:
        if not any(marker in content for marker in ("另外", "剩下", "剩余", "其余", "其他")):
            return False
        if not self._is_assignment_intent(content):
            return False
        current_untagged_count = self._category_stats_payload(limit_documents=0).get("untagged_document_count", 0)
        if current_untagged_count <= 0:
            return False
        last_goal = str(state.get("last_user_goal") or "")
        if not last_goal:
            return True
        if not any(marker in last_goal for marker in ("标签", "分类", "有标签", "无标签", "未分类")):
            return False
        if not any(marker in last_goal for marker in ("几篇", "多少", "数量", "统计", "哪些", "列出")):
            return False
        return True

    @staticmethod
    def _mentions_all_categories(content: str) -> bool:
        return any(marker in content for marker in ("所有标签", "全部标签", "所有分类", "全部分类", "所有论文的标签", "所有论文的分类"))

    @staticmethod
    def _is_document_category_query(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("什么", "哪些", "分别", "属于", "查看", "列出")
        )

    def _category_name_from_request_or_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> str | None:
        return (
            self._extract_existing_category_mention(content)
            or self._extract_category_name_from_request(content)
            or self._category_name_from_observations(observations)
        )

    @staticmethod
    def _category_name_from_observations(observations: list[_ReactObservation]) -> str | None:
        for observation in reversed(observations):
            category_name = observation.payload.get("category_name")
            if isinstance(category_name, str) and category_name.strip():
                return category_name.strip()
            category = observation.payload.get("category")
            if isinstance(category, dict) and isinstance(category.get("name"), str):
                return category["name"].strip()
        return None

    def _extract_category_name_from_request(self, content: str) -> str | None:
        quote_match = re.search(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content)
        if quote_match:
            candidate = self._clean_category_name(quote_match.group(1))
            if candidate and not self._category_name_validation_error(candidate):
                return candidate

        patterns = [
            r"(?:标签|分类)\s*[：:]\s*([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
            r"(?:名为|叫做|叫|命名为)\s*[\"'“”‘’「」『』《》]?([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
            r"(?:新增|新建|创建|建立|添加|加一个)\s*(?:一个|1个)?\s*[\"'“”‘’「」『』《》]?([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})[\"'“”‘’「」『』《》]?\s*(?:标签|分类)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            candidate = self._clean_category_name(match.group(1))
            if candidate and not self._category_name_validation_error(candidate):
                return candidate
        existing = self._extract_existing_category_mention(content)
        if existing:
            return existing
        return None

    def _extract_category_rename_request(self, content: str) -> tuple[str, str] | None:
        rename_markers = self._rename_category_markers()
        if not any(marker in content for marker in rename_markers):
            return None
        categories = sorted(self.category_repository.list_categories(), key=lambda item: len(item.name), reverse=True)
        marker_positions = [
            position
            for marker in rename_markers
            if (position := content.casefold().find(marker.casefold())) >= 0
        ]
        replace_position = min(marker_positions) if marker_positions else len(content)
        source_candidates: list[tuple[int, str]] = []
        for category in categories:
            if not category.name:
                continue
            position = content.rfind(category.name, 0, replace_position)
            if position >= 0:
                source_candidates.append((position, category.name))
        source_name = None
        if source_candidates:
            source_candidates.sort(key=lambda item: item[0], reverse=True)
            source_name = source_candidates[0][1]
        else:
            for category in categories:
                if category.name and category.name in content:
                    source_name = category.name
                    break
        if source_name is None:
            return None
        source_index = content.find(source_name)
        trailing = content[source_index + len(source_name) :] if source_index >= 0 else content
        target_patterns = [
            r"(?:换成|换为|替换为|替换成|改成|改为|改名为|重命名为|重命名)\s*[\"'“”‘’「」『』《》]?([^\"'“”‘’「」『』《》，,。；;\s]{1,40})",
            r"(?:rename|replace).{0,20}?(?:to|as)\s*[\"']?([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
        ]
        target_name = None
        for pattern in target_patterns:
            match = re.search(pattern, trailing, flags=re.IGNORECASE)
            if match:
                target_name = self._clean_category_name(match.group(1))
                break
        if not target_name:
            quoted = re.findall(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content)
            for candidate in quoted:
                cleaned = self._clean_category_name(candidate)
                if cleaned and cleaned != source_name:
                    target_name = cleaned
                    break
        if not target_name or self._category_name_validation_error(target_name):
            return None
        return source_name, target_name

    def _extract_existing_category_mention(self, content: str) -> str | None:
        categories = sorted(self.category_repository.list_categories(), key=lambda item: len(item.name), reverse=True)
        for category in categories:
            name = category.name.strip()
            if not name or name not in content:
                continue
            if any(
                marker in content
                for marker in (
                    f"{name}标签",
                    f"{name} 标签",
                    f"{name}分类",
                    f"{name} 分类",
                    f"带着{name}",
                    f"带有{name}",
                    f"属于{name}",
                )
            ):
                return name
            if any(marker in content for marker in ("标签", "分类")):
                return name
        return None

    @staticmethod
    def _clean_category_name(value: str) -> str:
        cleaned = value.strip().strip("：:，,。.;；、 \t\r\n\"'“”‘’「」『』《》")
        cleaned = re.split(r"(?:并且|然后|同时|再|并|且|都|给|把|目前|所有)", cleaned, maxsplit=1)[0]
        return cleaned.strip().strip("：:，,。.;；、 \t\r\n\"'“”‘’「」『』《》")

    @staticmethod
    def _category_name_validation_error(category_name: str) -> str | None:
        name = category_name.strip()
        if not name:
            return "我没有可靠识别出要使用的标签/分类名称，因此没有改动论文库。"
        if len(name) > 40:
            return "识别到的标签/分类名称过长，像是把整句话当成了标签名；本轮没有改动论文库。"
        command_markers = ("并且", "然后", "所有", "目前", "没有标签", "无标签", "加上", "补上", "这个标签")
        if any(marker in name for marker in command_markers):
            return "识别到的标签/分类名称包含后续命令，已拦截，避免创建错误标签。"
        return None

    def _category_exists(self, category_name: str) -> bool:
        return any(category.name == category_name for category in self.category_repository.list_categories())

    def _documents_for_category(self, category_name: str) -> list[LibraryDocument]:
        return [
            document
            for document in self.document_library_service.list_documents()
            if document.status == "ready" and any(category.name == category_name for category in document.categories)
        ]

    @staticmethod
    def _document_ids_from_observations(observations: list[_ReactObservation]) -> list[str]:
        document_ids: list[str] = []
        for observation in observations:
            for item in observation.payload.get("document_ids") or []:
                if isinstance(item, str) and item not in document_ids:
                    document_ids.append(item)
            for document in observation.payload.get("documents") or []:
                if isinstance(document, dict):
                    document_id = document.get("id")
                    if isinstance(document_id, str) and document_id not in document_ids:
                        document_ids.append(document_id)
        return document_ids

    @staticmethod
    def _merge_document_ids(existing: list[str], payload: dict[str, Any]) -> list[str]:
        merged = list(existing)
        for item in payload.get("document_ids") or []:
            if isinstance(item, str) and item not in merged:
                merged.append(item)
        for document in payload.get("documents") or []:
            if isinstance(document, dict):
                document_id = document.get("id")
                if isinstance(document_id, str) and document_id not in merged:
                    merged.append(document_id)
        for group in payload.get("category_groups") or []:
            if not isinstance(group, dict):
                continue
            for document_id in group.get("document_ids") or []:
                if isinstance(document_id, str) and document_id not in merged:
                    merged.append(document_id)
        return merged

    @staticmethod
    def _merge_evidence_items(existing: list[EvidenceItem], payload: dict[str, Any]) -> list[EvidenceItem]:
        merged = list(existing)
        seen = {item.id for item in merged}
        for item in payload.get("evidence_items") or []:
            try:
                evidence = item if isinstance(item, EvidenceItem) else EvidenceItem(**item)
            except Exception:
                continue
            if evidence.id not in seen:
                seen.add(evidence.id)
                merged.append(evidence)
        return merged

    def _evidence_items_from_observations(self, observations: list[_ReactObservation]) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        for observation in observations:
            evidence_items = self._merge_evidence_items(evidence_items, observation.payload)
        return evidence_items

    @staticmethod
    def _category_groups_from_observations(observations: list[_ReactObservation]) -> list[dict[str, Any]]:
        for observation in reversed(observations):
            groups = observation.payload.get("category_groups")
            if isinstance(groups, list) and groups:
                return [group for group in groups if isinstance(group, dict)]
        return []

    def _category_groups_from_library(self, category_names: list[str] | None = None) -> list[dict[str, Any]]:
        requested = {name for name in (category_names or []) if name}
        groups: dict[str, dict[str, Any]] = {}
        for document in self.document_library_service.list_documents():
            if document.status != "ready":
                continue
            for category in document.categories:
                if requested and category.name not in requested:
                    continue
                group = groups.setdefault(
                    category.name,
                    {
                        "category_name": category.name,
                        "document_ids": [],
                        "documents": [],
                        "evidence_items": [],
                    },
                )
                group["document_ids"].append(document.id)
                group["documents"].append(self._document_payload(document))
        return list(groups.values())

    def _compose_read_observation_answer(
        self,
        content: str,
        stats: dict[str, Any] | None,
        category_stats: dict[str, Any] | None,
        document_categories: dict[str, Any] | None,
    ) -> str:
        parts: list[str] = []
        if stats and self._needs_library_stats(content):
            parts.append(
                f"当前本地论文库共有 {stats['total']} 篇论文，其中 {stats['ready']} 篇可用、"
                f"{stats['processing']} 篇处理中、{stats['failed']} 篇处理失败。"
            )
        include_full_category_stats = bool(
            category_stats
            and self._needs_category_stats(content)
            and not document_categories
            and not self._asks_category_extreme(content)
            and self._asks_category_overview(content)
        )
        if category_stats and self._is_tagged_count_question(content) and not include_full_category_stats:
            parts.append(f"当前带分类/标签的论文有 {category_stats['tagged_document_count']} 篇。")
        if document_categories and self._should_include_document_category_detail(content):
            lines = []
            for document in document_categories.get("documents", []):
                categories = document.get("categories") or ["暂无标签/分类"]
                lines.append(f"- {document.get('name')}：{'、'.join(categories)}")
            if lines:
                parts.append("这些论文当前的标签/分类是：\n" + "\n".join(lines))
        if category_stats and self._asks_category_extreme(content):
            extreme_answer, category_name = self._category_extreme_answer(content, category_stats)
            if extreme_answer:
                parts.append(extreme_answer)
                repeated = self._repeat_text_answer(content, category_name)
                if repeated:
                    parts.append(repeated)
        elif category_stats:
            repeated = self._repeat_text_answer(content, self._category_name_from_request_or_observations(content, []))
            if repeated:
                parts.append(repeated)
        if (not parts or include_full_category_stats) and category_stats and self._needs_category_stats(content):
            category_names = [
                f"{item['name']}（{item['document_count']} 篇）"
                for item in category_stats.get("categories", [])
            ]
            labels = "、".join(category_names) if category_names else "暂无标签/分类"
            parts.append(
                f"有标签的论文 {category_stats['tagged_document_count']} 篇，"
                f"没有标签的论文 {category_stats['untagged_document_count']} 篇；"
                f"共有 {category_stats['category_count']} 类标签/分类：{labels}。"
            )
        if not parts and document_categories:
            lines = []
            for document in document_categories.get("documents", []):
                categories = document.get("categories") or ["暂无标签/分类"]
                lines.append(f"- {document.get('name')}：{'、'.join(categories)}")
            if lines:
                parts.append("这些论文当前的标签/分类是：\n" + "\n".join(lines))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _should_include_document_category_detail(content: str) -> bool:
        has_detail_marker = any(marker in content for marker in ("分别", "对应", "各自", "是什么", "哪些", "列出", "查看"))
        has_document_marker = any(marker in content for marker in ("论文", "文章", "文档", "每篇", "每个"))
        has_category_marker = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        return has_detail_marker and has_document_marker and has_category_marker

    @staticmethod
    def _asks_category_overview(content: str) -> bool:
        return any(
            marker in content
            for marker in ("几类", "无标签", "没有标签", "统计", "分布", "占比", "有哪些标签", "哪些标签", "列出标签")
        )

    @staticmethod
    def _asks_category_extreme(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        has_extreme = any(marker in content for marker in ("最多", "最少", "最大", "最小", "top", "most", "least"))
        return has_category_target and has_extreme

    @staticmethod
    def _category_extreme_answer(content: str, category_stats: dict[str, Any]) -> tuple[str, str | None]:
        categories = [
            item
            for item in category_stats.get("categories", [])
            if isinstance(item, dict) and item.get("name") and isinstance(item.get("document_count"), int)
        ]
        if not categories:
            return "当前还没有可统计的标签/分类。", None
        wants_min = any(marker in content for marker in ("最少", "最小", "least"))
        target_count = (min if wants_min else max)(item["document_count"] for item in categories)
        winners = [item for item in categories if item["document_count"] == target_count]
        names = "、".join(str(item["name"]) for item in winners)
        label = "最少" if wants_min else "最多"
        return f"论文数量{label}的标签/分类是：{names}（{target_count} 篇）。", str(winners[0]["name"])

    @staticmethod
    def _repeat_text_answer(content: str, value: str | None) -> str:
        if not value:
            return ""
        match = re.search(r"(?:重复输出|重复|输出).{0,12}?([0-9一二三四五六七八九十两]+)\s*次", content)
        if not match:
            return ""
        count = KnowledgeAgentRuntime._parse_small_count(match.group(1))
        if count <= 0:
            return ""
        count = min(count, 20)
        return "\n".join(str(value) for _ in range(count))

    @staticmethod
    def _parse_small_count(value: str) -> int:
        if value.isdigit():
            return int(value)
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if value == "十":
            return 10
        if "十" in value:
            left, _, right = value.partition("十")
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        return digits.get(value, 0)

    @staticmethod
    def _has_final_literal_output_request(content: str) -> bool:
        return KnowledgeAgentRuntime._final_literal_output(content) is not None

    @classmethod
    def _is_compound_question_request(cls, content: str) -> bool:
        return len(cls._split_questions(content)) >= 2

    def _should_answer_compound_with_evidence(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return True
        return bool(self._extract_document_tokens(content))

    @staticmethod
    def _split_questions(content: str) -> list[str]:
        questions: list[str] = []
        start = 0
        for match in re.finditer(r"[？?]", content):
            question = content[start : match.end()].strip(" \t\r\n；;。")
            if question:
                questions.append(question)
            start = match.end()
        if questions:
            return questions

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
        return [part for part in parts if any(marker in part.casefold() for marker in question_markers)]

    @staticmethod
    def _final_literal_output(content: str) -> str | None:
        patterns = [
            r"(?:最后|最终|末尾).{0,12}?输出(?:一个|一句|文本)?[「“\"']([^」”\"']{1,80})[」”\"']",
            r"(?:最后|最终|末尾).{0,12}?回复[「“\"']([^」”\"']{1,80})[」”\"']",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return None

    def _synthesize_react_answer(self, content: str, observations: list[_ReactObservation]) -> str:
        if not observations:
            return "我还没有拿到论文库观察结果，无法可靠回答。"
        last = observations[-1]
        if last.status in {"needs_clarification", "validation_failed", "failed", "confirmation_required"}:
            if last.tool == "library.explorer.find_documents" and last.payload.get("candidates"):
                candidates = "、".join(str(item) for item in last.payload["candidates"][:6])
                return f"{last.summary} 可以从这些候选里确认一下：{candidates}"
            return last.summary

        stats = self._latest_payload(observations, "library.explorer.stats")
        category_stats = self._latest_payload(observations, "library.explorer.category_stats")
        document_categories = self._latest_payload(observations, "library.explorer.document_categories")
        created = self._latest_payload(observations, "library.operator.create_category")
        assigned = self._latest_payload(observations, "library.operator.assign_category")
        renamed = self._latest_payload(observations, "library.operator.rename_category")
        cleared = self._latest_payload(observations, "library.operator.clear_categories")
        report = self._latest_payload(observations, "report.drafter.write")
        category_report = self._latest_payload(observations, "report.drafter.write_by_category")

        if category_report and category_report.get("answer"):
            return str(category_report["answer"])
        if report and report.get("answer"):
            return str(report["answer"])
        if cleared:
            verified_state = cleared.get("verified_state")
            if isinstance(verified_state, dict):
                return (
                    f"已清空 {cleared.get('updated_count', 0)} 篇论文的分类/标签关系。"
                    f"当前带分类/标签的论文有 {verified_state.get('tagged_document_count', 0)} 篇，"
                    f"未分类/无标签的论文有 {verified_state.get('untagged_document_count', 0)} 篇。"
                )
            return f"已清空 {cleared.get('updated_count', 0)} 篇论文的分类/标签关系。"
        if renamed:
            source_name = renamed.get("source_category_name", "源标签")
            target_name = renamed.get("target_category_name", "目标标签")
            updated_count = renamed.get("updated_count", 0)
            rename_text = (
                f"已将标签/分类「{source_name}」"
                f"{'合并到已有标签' if renamed.get('merged') else '重命名为'}「{target_name}」，"
                f"迁移了 {updated_count} 篇论文的关联。"
            )
            if assigned:
                category_name = assigned.get("category_name", "该标签")
                return (
                    f"已为 {assigned.get('updated_count', 0)} 篇论文追加「{category_name}」标签；"
                    + rename_text
                )
            return rename_text
        if assigned:
            category_name = assigned.get("category_name", "该标签")
            updated_count = assigned.get("updated_count", 0)
            prefix = ""
            if created:
                prefix = "已新建" if created.get("created") else "已确认已有"
                prefix = f"{prefix}标签/分类「{category_name}」；"
            scope = assigned.get("scope")
            if scope == "untagged":
                return f"{prefix}已为 {updated_count} 篇原本没有标签的论文补上「{category_name}」标签。"
            names = "、".join(item.get("name", "") for item in assigned.get("documents", [])[:6] if item.get("name"))
            suffix = f"：{names}" if names else ""
            return f"{prefix}已把标签/分类「{category_name}」追加到 {updated_count} 篇论文{suffix}。"
        if created:
            verb = "已新建" if created.get("created") else "已存在"
            return f"{verb}标签/分类「{created.get('category_name')}」。"
        read_answer = self._compose_read_observation_answer(content, stats, category_stats, document_categories)
        if read_answer:
            return read_answer
        return last.summary

    def _user_visible_final_answer(
        self,
        content: str,
        observations: list[_ReactObservation],
        candidate: str,
    ) -> str:
        text = candidate.strip()
        if not text:
            return self._synthesize_react_answer(content, observations)
        if self._must_answer_from_observations(observations) or self._contains_internal_process_wording(text):
            return self._synthesize_react_answer(content, observations)
        return self._strip_internal_process_lines(text) or self._synthesize_react_answer(content, observations)

    @staticmethod
    def _must_answer_from_observations(observations: list[_ReactObservation]) -> bool:
        if not observations:
            return False
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if any(tool.startswith("library.operator.") for tool in completed_tools):
            return True
        return bool(
            completed_tools
            & {
                "library.explorer.stats",
                "library.explorer.category_stats",
                "library.explorer.document_categories",
                "report.drafter.write",
                "report.drafter.write_by_category",
            }
        )

    @classmethod
    def _contains_internal_process_wording(cls, text: str) -> bool:
        return cls._strip_internal_process_lines(text) != text.strip()

    @staticmethod
    def _strip_internal_process_lines(text: str) -> str:
        process_markers = (
            "运行态摘要",
            "根据刚刚",
            "根据上下文",
            "根据对话记录",
            "根据我们的对话",
            "我将使用",
            "我需要确认",
            "首先，我需要",
            "正在执行",
            "工具",
            "tool",
            "library.",
        )
        kept = [
            line
            for line in text.strip().splitlines()
            if not any(marker in line for marker in process_markers)
        ]
        return "\n".join(line for line in kept if line.strip()).strip()

    @staticmethod
    def _latest_payload(observations: list[_ReactObservation], tool: str) -> dict[str, Any] | None:
        for observation in reversed(observations):
            if observation.tool == tool and observation.status == "completed":
                return observation.payload
        return None

    def _react_memory_payload(self, session_id: str) -> dict[str, Any]:
        session_summary = self.file_store.read_session_summary(session_id)
        reflection_path = self.file_store.get_session_dir(session_id) / "react_reflections.jsonl"
        reflections: list[dict[str, Any]] = []
        if reflection_path.exists():
            for line in reflection_path.read_text(encoding="utf-8").splitlines()[-5:]:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reflections.append(payload)
        return {
            "project_rules": self.file_store.read_project_rules()[:1200],
            "user_preferences": self.file_store.read_user_preferences()[:8],
            "session_summary": session_summary[:1200],
            "recent_reflections": reflections,
            "fixed_lessons": [
                "标签/分类统计要读取 document_categories 与文档关联，不能只报论文总数。",
                "复合写命令要拆步执行，标签名不能取后半句。",
            ],
        }

    def _react_state_path(self, session_id: str):
        self.file_store.initialize_session(session_id, "新对话")
        return self.file_store.get_session_dir(session_id) / "react_state.json"

    def _read_react_state(self, session_id: str) -> dict[str, Any]:
        path = self._react_state_path(session_id)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_react_state(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._react_state_path(session_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_react_state(
        self,
        *,
        session: ChatSession,
        user_goal: str,
        observations: list[_ReactObservation],
        status: str,
    ) -> None:
        if status not in {"completed", "degraded"}:
            return
        state = self._read_react_state(session.id)
        referent = self._derive_referent_from_observations(user_goal, observations)
        if referent is not None:
            state["last_document_set"] = referent
        category_name = self._category_name_from_observations(observations)
        if category_name:
            state["last_category_name"] = category_name
        state["last_user_goal"] = user_goal[:240]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_react_state(session.id, state)

    def _derive_referent_from_observations(
        self,
        user_goal: str,
        observations: list[_ReactObservation],
    ) -> dict[str, Any] | None:
        for observation in reversed(observations):
            payload = observation.payload
            if observation.tool == "library.operator.assign_category":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才打标签的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.find_documents":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才定位到的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.document_categories":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才查询标签的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.category_stats":
                if any(marker in user_goal for marker in ("无标签", "没有标签", "未分类", "未打标签")):
                    ids = [str(item) for item in payload.get("untagged_document_ids") or [] if item]
                    label = f"刚才统计到的 {len(ids)} 篇无标签论文"
                elif any(marker in user_goal for marker in ("有标签", "已打标签", "已分类")):
                    ids = [str(item) for item in payload.get("tagged_document_ids") or [] if item]
                    label = f"刚才统计到的 {len(ids)} 篇有标签论文"
                else:
                    ids = []
                    label = ""
                if ids:
                    return {
                        "label": label,
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.stats":
                ids = [str(item) for item in payload.get("referent_document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才统计到的全部 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
        return None

    @staticmethod
    def _state_document_ids(state: dict[str, Any]) -> list[str]:
        document_set = state.get("last_document_set")
        if not isinstance(document_set, dict):
            return []
        ids = document_set.get("document_ids")
        if not isinstance(ids, list):
            return []
        return [str(item) for item in ids if isinstance(item, str) and item]

    @staticmethod
    def _mentions_previous_referent(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "这几篇",
                "这四篇",
                "这些",
                "这几个",
                "上述",
                "上面",
                "刚才",
                "刚刚",
                "另外",
                "剩下",
                "剩余",
                "其余",
                "其他",
                "它们",
                "全部这些",
                "这批",
            )
        )

    def _record_react_reflection(
        self,
        *,
        session: ChatSession,
        user_goal: str,
        observations: list[_ReactObservation],
        status: str,
    ) -> None:
        self.file_store.initialize_session(session.id, session.title)
        path = self.file_store.get_session_dir(session.id) / "react_reflections.jsonl"
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_goal": user_goal[:240],
            "status": status,
            "tools": [observation.tool for observation in observations],
            "summary": self._reflection_summary(user_goal, observations, status),
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _reflection_summary(user_goal: str, observations: list[_ReactObservation], status: str) -> str:
        tools = " -> ".join(observation.tool for observation in observations) or "no tools"
        return f"ReAct task {status}: {user_goal[:80]} | tools: {tools}"

    def _append_react_trace(self, *, run_id: str, status: str, payload: dict[str, Any]) -> None:
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status=status,
            message=status.replace("_", " "),
            payload=payload,
        )

    @staticmethod
    def _safe_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= 4000:
            return payload
        return {"truncated": True, "preview": text[:4000]}

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

    def _begin_run(self, session: ChatSession, topic: str) -> str:
        run_id = f"chat-{uuid4().hex}"
        self.research_repository.create_run(run_id, f"Chat: {topic[:80]}")
        self.research_repository.update_run_status(run_id, ResearchRunStatus.RUNNING_TASK)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="knowledge_agent_started",
            message="Knowledge main agent started.",
            payload={"session_id": session.id, "topic": topic},
        )
        return run_id

    def _finish_run(self, run_id: str, status: ResearchRunStatus = ResearchRunStatus.COMPLETED) -> None:
        self.research_repository.update_run_status(run_id, status)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="knowledge_agent_finished",
            message="Knowledge main agent finished.",
            payload={},
        )

    def _run_subagent(
        self,
        *,
        run_id: str,
        role: str,
        profile: SubagentProfile,
        goal: str,
        tool_policy: ToolPolicy,
        context_bundle: dict[str, Any],
        worker: Callable[[], _TaskOutcome],
    ) -> _TaskOutcome:
        task = AgentTask(
            id=f"{role}-{uuid4().hex[:8]}",
            run_id=run_id,
            profile=profile,
            goal=goal,
            context_bundle={"agent_role": role, **context_bundle},
            done_criteria="Return a compact result payload for the knowledge main agent.",
            tool_policy=tool_policy,
            artifact_dir=str(self.file_store.runtime_context_path if hasattr(self.file_store, "runtime_context_path") else self.file_store.root),
        )
        self.runtime_repository.create_task(task)
        self.runtime_repository.update_task_status(task.id, AgentTaskStatus.RUNNING)
        self.message_bus.publish_control(
            run_id=run_id,
            task_id=task.id,
            event_type="knowledge_subagent_started",
            message=f"{role} started.",
            payload={"agent_role": role, "tool_policy": tool_policy.model_dump(mode="json")},
        )
        try:
            outcome = worker()
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=profile,
                status=AgentTaskStatus.COMPLETED,
                summary=outcome.summary,
                result_payload={"agent_role": role, **outcome.payload},
                created_at=datetime.now(timezone.utc),
            )
            self.runtime_repository.update_task_status(task.id, AgentTaskStatus.COMPLETED)
            self.message_bus.publish_notification(
                run_id=run_id,
                notification=notification,
                event_type="knowledge_subagent_completed",
            )
            return outcome
        except Exception as exc:
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=profile,
                status=AgentTaskStatus.FAILED,
                summary=f"{role} failed.",
                result_payload={"agent_role": role},
                error=str(exc),
                created_at=datetime.now(timezone.utc),
            )
            self.runtime_repository.update_task_status(task.id, AgentTaskStatus.FAILED)
            self.message_bus.publish_notification(
                run_id=run_id,
                notification=notification,
                event_type="knowledge_subagent_failed",
            )
            raise

    def _answer_library_count(self, session: ChatSession, content: str) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            ready_count = sum(1 for document in documents if document.status == "ready")
            processing_count = sum(1 for document in documents if document.status == "processing")
            failed_count = sum(1 for document in documents if document.status == "failed")
            return _TaskOutcome(
                summary=f"Found {len(documents)} library documents.",
                payload={
                    "total": len(documents),
                    "ready": ready_count,
                    "processing": processing_count,
                    "failed": failed_count,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Count local library documents.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content},
            worker=worker,
        )
        self._finish_run(run_id)
        payload = outcome.payload
        answer = (
            f"当前本地论文库共有 {payload['total']} 篇论文，其中 {payload['ready']} 篇已可用、"
            f"{payload['processing']} 篇处理中、{payload['failed']} 篇处理失败。"
        )
        return KnowledgeAgentResult(content=answer, action_status="completed", agent_trace_id=run_id)

    def _answer_document_categories(
        self,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, selected_document_ids, allow_all=False)
            if not documents:
                return _TaskOutcome(
                    summary="No document matched the category query.",
                    payload={"documents": [], "candidates": self._candidate_titles()},
                )
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s) for category lookup.",
                payload={
                    "documents": [
                        {
                            "id": document.id,
                            "name": document.display_name or document.filename,
                            "title": document.title,
                            "categories": [category.name for category in document.categories],
                        }
                        for document in documents
                    ]
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read document categories.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content},
            worker=worker,
        )
        self._finish_run(run_id)
        documents = outcome.payload.get("documents") or []
        if not documents:
            return KnowledgeAgentResult(
                content="我没有在论文库里唯一定位到你说的那篇论文。可以把标题多给几个字，或先在下方选中论文后再问。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )
        lines = []
        for document in documents:
            categories = document["categories"] or ["暂无分类/标签"]
            lines.append(f"- {document['name']}：{'、'.join(categories)}")
        return KnowledgeAgentResult(
            content="这些论文当前的分类/标签是：\n" + "\n".join(lines),
            action_status="completed",
            used_document_ids=[item["id"] for item in documents],
            agent_trace_id=run_id,
        )

    def _create_category(self, session: ChatSession, content: str, category_name: str) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            existing = next(
                (category for category in self.category_repository.list_categories() if category.name == category_name),
                None,
            )
            if existing is not None:
                return _TaskOutcome(
                    summary=f"Category already exists: {category_name}.",
                    payload={"category": existing.model_dump(mode="json"), "created": False},
                )
            palette = ["#0f5fb8", "#047c71", "#6957d8", "#b76a00", "#b42318"]
            color = palette[len(self.category_repository.list_categories()) % len(palette)]
            category = self.category_repository.create_category(category_name, color)
            return _TaskOutcome(
                summary=f"Created category: {category.name}.",
                payload={"category": category.model_dump(mode="json"), "created": True},
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Create a non-destructive library category.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name},
            worker=worker,
        )
        self._finish_run(run_id)
        category = outcome.payload["category"]
        verb = "已新建" if outcome.payload.get("created") else "已经存在"
        return KnowledgeAgentResult(
            content=f"{verb}分类「{category['name']}」。",
            action_status="completed",
            agent_trace_id=run_id,
            library_mutated=bool(outcome.payload.get("created")),
        )

    def _assign_category(
        self,
        session: ChatSession,
        content: str,
        assignment: dict[str, str],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)
        category_name = assignment["category_name"]

        def worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, [], allow_all=False)
            if not documents:
                return _TaskOutcome(
                    summary="No document matched category assignment.",
                    payload={"documents": [], "category_name": category_name},
                )
            category = next(
                (item for item in self.category_repository.list_categories() if item.name == category_name),
                None,
            )
            if category is None:
                category = self.category_repository.create_category(category_name, "#0f5fb8")
            updated = []
            for document in documents:
                existing_ids = [item.id for item in document.categories]
                next_ids = list(dict.fromkeys([*existing_ids, category.id]))
                categories = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "categories": [item.name for item in categories],
                    }
                )
            return _TaskOutcome(
                summary=f"Assigned category {category_name} to {len(updated)} document(s).",
                payload={"documents": updated, "category": category.model_dump(mode="json")},
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Assign a category to matched library documents.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name},
            worker=worker,
        )
        self._finish_run(run_id)
        documents = outcome.payload.get("documents") or []
        if not documents:
            return KnowledgeAgentResult(
                content="我没有唯一定位到要打标签的论文。请把论文标题多给几个字，或先在下方选中论文。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )
        names = "、".join(item["name"] for item in documents)
        return KnowledgeAgentResult(
            content=f"已把分类/标签「{category_name}」添加到：{names}。",
            action_status="completed",
            used_document_ids=[item["id"] for item in documents],
            agent_trace_id=run_id,
            library_mutated=True,
        )

    def _draft_report_like_answer(
        self,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)
        attachment_ids = [item.document_id for item in attachments if item.document_id]
        seed_ids = list(dict.fromkeys([*selected_document_ids, *[item for item in attachment_ids if item]]))

        def resolve_worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, seed_ids, allow_all=self._mentions_all_library(content))
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s) for drafting.",
                payload={
                    "documents": [
                        {
                            "id": document.id,
                            "name": document.display_name or document.filename,
                            "title": document.title,
                            "status": document.status,
                        }
                        for document in documents
                    ]
                },
            )

        resolved = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Resolve documents for summary/review request.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "selected_document_ids": seed_ids},
            worker=resolve_worker,
        )
        document_ids = [item["id"] for item in resolved.payload.get("documents", [])]
        documents = [document for document in self.document_library_service.list_documents() if document.id in document_ids]
        if not documents:
            self._finish_run(run_id)
            return KnowledgeAgentResult(
                content="我还没有定位到可用于总结的论文。你可以先在论文库选择论文，或告诉我更完整的标题。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )

        def retrieve_worker() -> _TaskOutcome:
            evidence = self.rag_service.retrieve_evidence(
                question=content,
                documents=documents,
                top_k=min(10, max(4, len(documents) * 3)),
            )
            return _TaskOutcome(
                summary=f"Retrieved {len(evidence)} evidence item(s).",
                payload={"evidence_items": [item.model_dump(mode="json") for item in evidence]},
            )

        retrieved = self._run_subagent(
            run_id=run_id,
            role="evidence-retriever",
            profile=SubagentProfile.EXPLORE,
            goal="Retrieve local evidence for drafting.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"document_ids": document_ids},
            worker=retrieve_worker,
        )
        evidence_items = [EvidenceItem(**item) for item in retrieved.payload.get("evidence_items", [])]

        def draft_worker() -> _TaskOutcome:
            answer = self._draft_with_llm(content, documents, evidence_items)
            return _TaskOutcome(
                summary="Drafted chat-side Markdown answer.",
                payload={"answer": answer},
            )

        drafted = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Draft the final chat-visible Markdown answer.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"document_ids": document_ids, "evidence_count": len(evidence_items)},
            worker=draft_worker,
        )
        self._finish_run(run_id)
        citations = self._collect_citations(evidence_items)
        return KnowledgeAgentResult(
            content=drafted.payload["answer"],
            retrieval_status="ready" if evidence_items else "skipped",
            warning=None if evidence_items else "没有检索到足够片段，本轮先基于论文元数据生成概要。",
            citations=citations,
            used_document_ids=document_ids,
            evidence_items=evidence_items,
            action_status="completed",
            agent_trace_id=run_id,
        )

    def _request_confirmation(
        self,
        session: ChatSession,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult:
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        pending = self._build_destructive_action(content)
        if pending is not None:
            self._write_pending_action(session.id, pending)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="confirmation_required",
            message="Destructive action requires confirmation.",
            payload={"pending_action": pending or {"raw_request": content}},
        )
        if owns_run:
            self._finish_run(run_id)
        target = pending.get("label") if pending else content
        return KnowledgeAgentResult(
            content=f"这个操作会删除「{target}」，我需要你明确回复“确认删除”后才会执行。",
            action_status="confirmation_required",
            agent_trace_id=run_id,
        )

    def _defer_destructive_part_after_safe_plan(
        self,
        *,
        session: ChatSession,
        content: str,
        observations: list[_ReactObservation],
        current_answer: str,
        run_id: str,
    ) -> str | None:
        if not self._is_destructive_intent(content):
            return None
        if not self._should_plan_before_destructive_confirmation(content, [], []):
            return None
        pending = self._build_destructive_action_from_observations(content, observations)
        if pending is None:
            return (
                f"{current_answer}\n\n"
                "删除部分还没有可靠定位到具体论文、标签或分类；请明确要删除的对象后我再继续。"
            ).strip()
        literal = self._final_literal_output(content)
        if literal:
            pending["post_confirm_literal"] = literal
        pending["source_goal"] = content[:500]
        self._write_pending_action(session.id, pending)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="confirmation_required",
            message="Destructive subtask requires confirmation after safe subtasks.",
            payload={"pending_action": pending},
        )
        target = pending.get("label") or "该对象"
        suffix = f"确认后我会继续输出「{literal}」。" if literal else "确认后我会执行删除。"
        return (
            f"{current_answer}\n\n"
            f"删除部分会删除「{target}」，需要你明确回复“确认删除”后才会执行。{suffix}"
        ).strip()

    def _build_destructive_action_from_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> dict[str, Any] | None:
        explicit = self._build_destructive_action(content)
        if explicit is not None:
            return explicit
        category_name = self._destructive_category_referent(content, observations)
        if category_name:
            category = next(
                (item for item in self.category_repository.list_categories() if item.name == category_name),
                None,
            )
            if category is not None:
                return {"type": "delete_category", "category_id": category.id, "label": category.name}
        return None

    def _destructive_category_referent(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> str | None:
        if not any(marker in content for marker in ("这个标签", "该标签", "此标签", "这个分类", "该分类", "此分类")):
            return None
        category_stats = self._latest_payload(observations, "library.explorer.category_stats")
        if category_stats and self._asks_category_extreme(content):
            _answer, category_name = self._category_extreme_answer(content, category_stats)
            if category_name:
                return category_name
        state_category = self._category_name_from_observations(observations)
        if state_category:
            return state_category
        return None

    def _maybe_execute_pending_action(
        self,
        session: ChatSession,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        if not any(marker in content.casefold() for marker in self._CONFIRM_MARKERS):
            return None
        pending = self._read_pending_action(session.id)
        if pending is None:
            return None
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None

        def worker() -> _TaskOutcome:
            if pending["type"] == "delete_category":
                category_id = str(pending["category_id"])
                label = str(pending["label"])
                deleted = self.category_repository.delete_category(pending["category_id"])
                category_still_exists = self.category_repository.get_category(category_id) is not None
                verified_state = self._category_stats_payload()
                linked_document_ids = [
                    document["id"]
                    for document in verified_state.get("tagged_documents", [])
                    if label in (document.get("categories") or [])
                ]
                verified_deleted = bool(deleted) and not category_still_exists and not linked_document_ids
                return _TaskOutcome(
                    summary=(
                        f"Deleted category {label}."
                        if verified_deleted
                        else f"Category deletion was not verified: {label}."
                    ),
                    payload={
                        "deleted": verified_deleted,
                        "delete_attempted": bool(deleted),
                        "label": label,
                        "category_id": category_id,
                        "category_still_exists": category_still_exists,
                        "linked_document_ids": linked_document_ids,
                        "verified_state": verified_state,
                        "library_mutated": verified_deleted,
                    },
                )
            if pending["type"] == "delete_document":
                deleted = self.document_library_service.delete_document(pending["document_id"])
                return _TaskOutcome(
                    summary=f"Deleted document {pending['label']}.",
                    payload={"deleted": bool(deleted), "label": pending["label"], "library_mutated": bool(deleted)},
                )
            return _TaskOutcome(summary="No supported pending action.", payload={"deleted": False})

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Execute confirmed destructive library action.",
            tool_policy=ToolPolicy(read_only=False, db_write=True, workspace_write=True),
            context_bundle={"pending_action": pending},
            worker=worker,
        )
        self._clear_pending_action(session.id)
        if owns_run:
            self._finish_run(run_id)
        if outcome.payload.get("deleted"):
            verified_state = outcome.payload.get("verified_state")
            state_text = ""
            if isinstance(verified_state, dict):
                category_names = [
                    f"{item['name']}（{item['document_count']} 篇）"
                    for item in verified_state.get("categories", [])
                    if isinstance(item, dict)
                ]
                if category_names:
                    state_text = "当前分类：" + "、".join(category_names) + "。"
            post_confirm_literal = str(pending.get("post_confirm_literal") or "").strip()
            content_parts = [f"已删除「{outcome.payload['label']}」。"]
            if state_text:
                content_parts.append(state_text)
            if post_confirm_literal:
                content_parts.append(post_confirm_literal)
            return KnowledgeAgentResult(
                content="\n".join(content_parts),
                action_status="completed",
                agent_trace_id=run_id,
                library_mutated=bool(outcome.payload.get("library_mutated")),
            )
        return KnowledgeAgentResult(
            content=(
                f"我没有能验证「{outcome.payload.get('label', '该对象')}」已从论文库中删除，"
                "因此没有把这次操作标记为成功。"
            ),
            action_status="failed",
            agent_trace_id=run_id,
        )

    def _build_destructive_action(self, content: str) -> dict[str, Any] | None:
        categories = self.category_repository.list_categories()
        for category in categories:
            if category.name and category.name in content:
                return {"type": "delete_category", "category_id": category.id, "label": category.name}
        documents = self._resolve_documents(content, [], allow_all=False)
        if len(documents) == 1:
            document = documents[0]
            return {
                "type": "delete_document",
                "document_id": document.id,
                "label": document.display_name or document.filename,
            }
        return None

    def _draft_with_llm(
        self,
        question: str,
        documents: list[LibraryDocument],
        evidence_items: list[EvidenceItem],
    ) -> str:
        if self._requests_per_paper_abstract_format(question):
            return self._draft_per_paper_abstract_template(question, documents)
        document_block = "\n".join(
            (
                f"- {document.display_name or document.filename} | 标题：{document.title or '未知'} "
                f"| 页数：{document.page_count} | 摘要：{self._extract_document_abstract(document)}"
            )
            for document in documents
        )
        evidence_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"来源：{item.citation_label}",
                        f"标题：{item.title}",
                        f"页码：{item.page_number if item.page_number is not None else '未知'}",
                        f"证据：{item.quote or item.snippet}",
                    ]
                )
                for item in evidence_items[:10]
            ]
        )
        fallback = self._draft_template(question, documents, evidence_items)
        if not evidence_items:
            return fallback
        if not self.api_key:
            return fallback
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 report-drafter 子代理。请用中文输出 Markdown，"
                            "只基于给定论文元数据和证据撰写，不要编造引用。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"用户请求：{question}",
                                "论文列表：\n" + document_block,
                                "证据列表：\n" + (evidence_block or "暂无可引用证据。"),
                                "请严格保留用户指定的结构；如果用户要求论文一/论文名称/论文完整摘要/论文页数，"
                                "每篇论文都必须包含这些字段，摘要只能使用给定摘要文本，缺失时写“未提取到真实摘要”。"
                                "如果证据不足，要明确说明边界。",
                            ]
                        ),
                    },
                ],
            )
        except Exception:
            return fallback
        return self._extract_message_text(response) or fallback

    def _draft_category_summaries_with_llm(
        self,
        *,
        question: str,
        category_groups: list[dict[str, Any]],
        target_chars: int,
    ) -> str:
        fallback = self._draft_category_summaries_template(
            question=question,
            category_groups=category_groups,
            target_chars=target_chars,
        )
        if not self.api_key:
            return fallback

        group_blocks = []
        for group in category_groups:
            documents = group.get("documents") or []
            evidence_items = group.get("evidence_items") or []
            document_lines = [
                f"- {item.get('name') or item.get('filename')} | 标题：{item.get('title') or '未知'}"
                for item in documents
                if isinstance(item, dict)
            ]
            evidence_lines = []
            for item in evidence_items[:8]:
                if not isinstance(item, dict):
                    continue
                evidence_lines.append(
                    "\n".join(
                        [
                            f"来源：{item.get('citation_label')}",
                            f"标题：{item.get('title')}",
                            f"页码：{item.get('page_number')}",
                            f"证据：{item.get('quote') or item.get('snippet')}",
                        ]
                    )
                )
            group_blocks.append(
                "\n".join(
                    [
                        f"## 标签：{group.get('category_name')}",
                        "论文：",
                        *document_lines,
                        "证据：",
                        "\n\n".join(evidence_lines) or "暂无可引用证据。",
                    ]
                )
            )

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 report-drafter 子代理。请用中文输出 Markdown。"
                            "任务是按标签分别写总结，每个标签一节，尽量接近用户要求的字数。"
                            "只能基于给定论文元数据和证据，不要编造引用。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"用户目标：{question}",
                                f"每个标签目标长度：约 {target_chars} 字。",
                                "按标签分组的论文与证据：",
                                "\n\n".join(group_blocks),
                            ]
                        ),
                    },
                ],
            )
        except Exception:
            return fallback
        return self._extract_message_text(response) or fallback

    @staticmethod
    def _draft_category_summaries_template(
        *,
        question: str,
        category_groups: list[dict[str, Any]],
        target_chars: int,
    ) -> str:
        lines = [
            f"# {question}",
            "",
            f"以下按标签分别整理，每部分目标约 {target_chars} 字；如果证据不足，会明确说明边界。",
            "",
        ]
        if not category_groups:
            lines.append("当前没有可按标签分组的 ready 论文。")
            return "\n".join(lines)
        for group in category_groups:
            category_name = group.get("category_name") or "未命名标签"
            documents = [item for item in group.get("documents", []) if isinstance(item, dict)]
            evidence_items = [item for item in group.get("evidence_items", []) if isinstance(item, dict)]
            lines.extend(
                [
                    f"## {category_name}",
                    "",
                    "涉及论文："
                    + ("、".join(item.get("name") or item.get("filename") or "未命名论文" for item in documents) or "暂无论文"),
                    "",
                ]
            )
            if evidence_items:
                snippets = [
                    str(item.get("quote") or item.get("snippet") or "").strip()
                    for item in evidence_items[:4]
                    if str(item.get("quote") or item.get("snippet") or "").strip()
                ]
                lines.append(
                    "该标签下的论文目前检索到的证据主要集中在："
                    + "；".join(snippets)
                    + "。基于这些片段，可以先形成一份初步总结，但仍建议回到原 PDF 核对关键结论和页码。"
                )
            else:
                boundary = str(group.get("evidence_boundary") or "").strip()
                lines.append(boundary or "该标签下论文存在，但正文证据不足。")
                lines.append("因此这里先基于题名和元数据占位总结，强结论需要后续补充 RAG 证据。")
            quality = group.get("evidence_quality")
            if isinstance(quality, dict) and quality.get("warnings"):
                lines.append("证据覆盖边界：" + "、".join(str(item) for item in quality.get("warnings", [])))
            lines.append("")
        return "\n".join(lines).strip()

    @classmethod
    def _requests_per_paper_abstract_format(cls, question: str) -> bool:
        return any(marker in question for marker in ("论文一", "论文名称", "论文完整摘要", "完整摘要")) and any(
            marker in question for marker in ("页数", "多少页", "page")
        )

    def _draft_per_paper_abstract_template(
        self,
        question: str,
        documents: list[LibraryDocument],
    ) -> str:
        lines: list[str] = []
        chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        for index, document in enumerate(documents):
            ordinal = chinese_numbers[index] if index < len(chinese_numbers) else str(index + 1)
            lines.extend(
                [
                    f"论文{ordinal}：",
                    f"论文名称：{document.display_name or document.filename}",
                    f"论文完整摘要：{self._extract_document_abstract(document)}",
                    f"论文页数：{document.page_count or 0}页",
                    "",
                ]
            )
        if documents:
            names = "、".join(document.display_name or document.filename for document in documents)
            lines.append(
                f"这{len(documents)}篇论文都来自用户指定标签下的真实论文集合，分别是{names}；"
                "相同点和不同点需要以已解析正文和摘要为边界进行判断，当前模板优先保证论文名称、真实摘要提取结果和页数不被编造。"
            )
        else:
            lines.append("没有找到符合该标签范围的论文，因此无法生成该格式报告。")
        return "\n".join(lines).strip()

    def _extract_document_abstract(self, document: LibraryDocument) -> str:
        try:
            chunks = self.rag_service.chunk_repository.list_chunks(document_ids=[document.id])
        except Exception:
            chunks = []
        if not chunks:
            return "未提取到真实摘要"
        for chunk in chunks:
            section = f"{chunk.section or ''} {chunk.title or ''}".casefold()
            if "abstract" in section or "摘要" in section:
                candidate = self._clean_abstract_text(chunk.content or chunk.text or "")
                if candidate:
                    return candidate
        first_pages = "\n".join(
            chunk.content or chunk.text or ""
            for chunk in chunks
            if (chunk.page_number or 0) <= 2
        )
        patterns = [
            r"(?is)\babstract\b\s*[:：]?\s*(.{80,2200}?)(?:\b(?:keywords|index terms|introduction)\b|1\s+introduction|\n\s*1[.\s])",
            r"(?s)摘要\s*[:：]?\s*(.{40,1600}?)(?:关键词|关键字|引言|一、|1\s*[\.、]\s*)",
        ]
        for pattern in patterns:
            try:
                match = re.search(pattern, first_pages)
            except re.error:
                continue
            if match:
                candidate = self._clean_abstract_text(match.group(1))
                if candidate:
                    return candidate
        return "未提取到真实摘要"

    @staticmethod
    def _clean_abstract_text(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" ：:;；")
        if not cleaned:
            return ""
        if len(cleaned) < 30:
            return ""
        return cleaned[:2200]

    @staticmethod
    def _draft_template(
        question: str,
        documents: list[LibraryDocument],
        evidence_items: list[EvidenceItem],
    ) -> str:
        names = "、".join(document.display_name or document.filename for document in documents)
        lines = [
            f"# {question}",
            "",
            "## 涉及论文",
            "",
            *[
                f"- {document.display_name or document.filename}：{document.title or '未提取标题'}"
                for document in documents
            ],
            "",
            "## 初步总结",
            "",
            f"本次围绕 {names} 进行整理。",
        ]
        if evidence_items:
            lines.extend(
                [
                    "当前检索到的证据主要覆盖以下片段：",
                    "",
                    *[
                        f"- {item.quote or item.snippet}（{item.citation_label}）"
                        for item in evidence_items[:5]
                    ],
                ]
            )
        else:
            lines.append("当前没有检索到足够正文片段，因此无法生成基于论文正文的分析结论。")
        lines.extend(
            [
                "",
                "## 结论边界",
                "",
                "强结论应回到原 PDF 与引用页码继续核验；未被证据直接支持的内容不应作为最终学术结论。",
            ]
        )
        return "\n".join(lines)

    def _resolve_documents(
        self,
        content: str,
        selected_document_ids: list[str],
        *,
        allow_all: bool,
    ) -> list[LibraryDocument]:
        documents = self.document_library_service.list_documents()
        ready_documents = [document for document in documents if document.status == "ready"]
        if selected_document_ids:
            selected = [document for document in ready_documents if document.id in set(selected_document_ids)]
            if selected:
                return selected
        if allow_all:
            return ready_documents

        tokens = self._extract_document_tokens(content)
        matches: list[LibraryDocument] = []
        for token in tokens:
            token_matches = self._match_documents(token, ready_documents)
            for document in token_matches:
                if document.id not in {item.id for item in matches}:
                    matches.append(document)
        return matches

    @staticmethod
    def _match_documents(token: str, documents: list[LibraryDocument]) -> list[LibraryDocument]:
        normalized_token = KnowledgeAgentRuntime._normalize_match_text(token)
        if not normalized_token:
            return []
        exact: list[LibraryDocument] = []
        scored: list[tuple[float, LibraryDocument]] = []
        for document in documents:
            haystacks = [
                document.display_name or "",
                document.filename or "",
                document.title or "",
            ]
            normalized_haystacks = [KnowledgeAgentRuntime._normalize_match_text(item) for item in haystacks]
            if any(normalized_token in item for item in normalized_haystacks):
                exact.append(document)
                continue
            score = max(
                (difflib.SequenceMatcher(None, normalized_token, item).ratio() for item in normalized_haystacks),
                default=0.0,
            )
            if score >= 0.55:
                scored.append((score, document))
        if exact:
            return exact
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _score, document in scored[:2]]

    @staticmethod
    def _extract_document_tokens(content: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"《([^》]{2,120})》", content))
        tokens.extend(re.findall(r"[A-Za-z]+\d+[A-Za-z0-9_-]*|\d+[A-Za-z]+[A-Za-z0-9_-]*", content))
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", content):
            if raw.casefold() not in {"pdf", "paper", "review", "summary"}:
                tokens.append(raw)
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            cleaned = token.strip("：:，,。？?！!、 ")
            key = cleaned.casefold()
            if len(cleaned) < 2 or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _candidate_titles(self) -> list[str]:
        return [
            document.display_name or document.filename
            for document in self.document_library_service.list_documents()[:8]
        ]

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).casefold()

    @staticmethod
    def _extract_create_category_name(content: str) -> str | None:
        if not any(marker in content for marker in ("新建", "创建", "新增", "建立")):
            return None
        if not any(marker in content for marker in ("分类", "标签")):
            return None
        patterns = [
            r"(?:分类|标签)[：:\s]+([^，。！？\n]+)",
            r"(?:叫|名为|命名为)[「“\"]?([^」”\"，。！？\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                name = match.group(1).strip(" 「」“”\"'")
                return name[:40] if name else None
        tail = re.split(r"分类|标签", content, maxsplit=1)[-1].strip(" ：:，,。 ")
        return tail[:40] if tail else None

    @staticmethod
    def _extract_category_assignment(content: str) -> dict[str, str] | None:
        if not any(marker in content for marker in ("打标签", "添加标签", "加标签", "归类", "设置分类", "添加分类")):
            return None
        match = re.search(r"(?:标签|分类)[：:\s为到成]*[「“\"]?([^」”\"，。！？\n]+)", content)
        if not match:
            return None
        category_name = match.group(1).strip(" 「」“”\"'")
        if not category_name:
            return None
        return {"category_name": category_name[:40]}

    @staticmethod
    def _is_library_count_question(content: str) -> bool:
        return "论文" in content and any(marker in content for marker in ("几篇", "多少篇", "数量", "count"))

    @staticmethod
    def _is_category_question(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("什么", "哪些", "多少", "列出", "查看")
        )

    @classmethod
    def _is_destructive_intent(cls, content: str) -> bool:
        normalized = cls._strip_quoted_text(content).casefold()
        if cls._is_clear_categories_intent(normalized):
            return False
        if cls._is_non_destructive_category_write(normalized):
            return False
        if any(marker in normalized for marker in cls._rename_category_markers()) and any(
            marker in normalized for marker in ("标签", "分类", "tag", "category")
        ):
            return False
        if not any(marker in normalized for marker in cls._DELETE_MARKERS):
            return False
        if any(marker in normalized for marker in ("论文", "文档", "paper", "document")):
            return True
        if any(marker in normalized for marker in ("标签", "分类", "tag", "category")):
            return True
        return bool(re.search(r"(删除|移除|清空|删掉|delete|remove|clear)\s*(?:这个|该|此)?\s*(?:分类|标签)", normalized))

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
                "重命名",
                "重命名为",
                "改成",
                "改为",
                "改名为",
                "assign",
                "create",
                "add",
                "rename",
                "replace",
            )
        )

    @classmethod
    def _is_summary_request(cls, content: str, selected_document_ids: list[str]) -> bool:
        if any(marker in content.casefold() for marker in cls._SUMMARY_MARKERS):
            return True
        return bool(selected_document_ids) and any(marker in content for marker in ("写", "生成", "分析"))

    @classmethod
    def _is_grouped_category_summary_request(cls, content: str) -> bool:
        return cls._is_summary_request(content, []) and any(
            marker in content for marker in ("按标签", "按分类", "每个标签", "每个分类", "分组")
        )

    @staticmethod
    def _mentions_all_library(content: str) -> bool:
        return any(marker in content for marker in ("所有论文", "全部论文", "论文库", "全库", "所有文献"))

    def _pending_path(self, session_id: str):
        self.file_store.initialize_session(session_id, "知识库对话")
        return self.file_store.get_session_dir(session_id) / "pending_knowledge_action.json"

    def _write_pending_action(self, session_id: str, payload: dict[str, Any]) -> None:
        self._pending_path(session_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_pending_action(self, session_id: str) -> dict[str, Any] | None:
        path = self._pending_path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _clear_pending_action(self, session_id: str) -> None:
        path = self._pending_path(session_id)
        path.unlink(missing_ok=True)

    @staticmethod
    def _collect_citations(evidence_items: list[EvidenceItem]) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for item in evidence_items:
            if item.citation_label in seen:
                continue
            seen.add(item.citation_label)
            citations.append(item.citation_label)
        return citations

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
