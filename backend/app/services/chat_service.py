"""Chat-first knowledge agent service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import shutil
from uuid import uuid4

from openai import OpenAI

from app.models import (
    AgentModeDecision,
    AgentOrchestratorInput,
    AgentRunMode,
    ChatAttachment,
    ChatContextState,
    ChatMessage,
    ChatMessageRequest,
    ChatSendResponse,
    ChatSession,
    ChatSessionDetail,
    MemorySnapshot,
    ResearchRunStatus,
)
from app.repositories import ChatRepository, LibraryRepository

from .chat_memory_service import ChatMemoryService
from .context_assembler import ContextAssembler
from .rag_service import RagService


@dataclass(slots=True)
class _LLMDiagnostic:
    status: str
    error_type: str | None = None
    error: str | None = None
    model: str | None = None
    base_url_configured: bool = False
    api_key_configured: bool = False
    stream: bool = False

    def as_trace_payload(self) -> dict:
        return {
            "status": self.status,
            "error_type": self.error_type,
            "error": self.error,
            "model": self.model,
            "base_url_configured": self.base_url_configured,
            "api_key_configured": self.api_key_configured,
            "stream": self.stream,
        }


class ChatService:
    """Persist chat sessions while mixing model replies with optional RAG context."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        library_repository: LibraryRepository,
        rag_service: RagService,
        memory_service: ChatMemoryService,
        context_assembler: ContextAssembler,
        agent_orchestrator=None,
        knowledge_agent_runtime=None,
        knowledge_planner_runtime=None,
        reflection_runtime=None,
        enable_research_from_knowledge: bool = False,
        enable_auto_reflection: bool = False,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 30.0,
    ) -> None:
        self.chat_repository = chat_repository
        self.library_repository = library_repository
        self.rag_service = rag_service
        self.memory_service = memory_service
        self.context_assembler = context_assembler
        self.agent_orchestrator = agent_orchestrator
        self.knowledge_agent_runtime = knowledge_agent_runtime
        self.knowledge_planner_runtime = knowledge_planner_runtime
        self.reflection_runtime = reflection_runtime
        self.enable_research_from_knowledge = enable_research_from_knowledge
        self.enable_auto_reflection = enable_auto_reflection
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.last_llm_diagnostic = _LLMDiagnostic(
            status="not_called",
            model=self.model,
            base_url_configured=bool(self.base_url),
            api_key_configured=bool(self.api_key),
        )

    def list_sessions(self) -> list[ChatSession]:
        return self.chat_repository.list_sessions()

    def create_session(self, title: str | None = None) -> ChatSession:
        session = self.chat_repository.create_session(title or "新对话")
        self.memory_service.record_project_context(session_id=session.id, title=session.title)
        return session

    def delete_session(self, session_id: str) -> ChatSession:
        session = self._require_session(session_id)
        deleted = self.chat_repository.delete_session(session_id)
        session_dir = self.context_assembler.file_store.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
        return deleted or session

    def get_session_detail(self, session_id: str) -> ChatSessionDetail:
        session = self._require_session(session_id)
        messages = self.chat_repository.list_messages(session_id)
        document_ids = self._collect_document_ids_from_messages(messages)
        memory_snapshot = self.memory_service.build_snapshot(
            session_id=session_id,
            selected_document_ids=document_ids,
        )
        context_state = self.context_assembler.read_context_state(session_id)
        return ChatSessionDetail(
            session=session,
            messages=messages,
            memory_snapshot=memory_snapshot,
            context_state=context_state,
        )

    def get_memory_snapshot(self, session_id: str) -> MemorySnapshot:
        session = self._require_session(session_id)
        messages = self.chat_repository.list_messages(session_id)
        document_ids = self._collect_document_ids_from_messages(messages)
        return self.memory_service.build_snapshot(
            session_id=session.id,
            selected_document_ids=document_ids,
        )

    def get_context_state(self, session_id: str) -> ChatContextState:
        self._require_session(session_id)
        return self.context_assembler.read_context_state(session_id)

    def send_message(
        self,
        session_id: str,
        request: ChatMessageRequest,
        *,
        delta_sink: Callable[[str], None] | None = None,
    ) -> ChatSendResponse:
        session = self._require_session(session_id)
        normalized_attachments = self._normalize_attachments(
            request.attachments,
            request.selected_document_ids,
        )

        user_message = ChatMessage(
            id=str(uuid4()),
            session_id=session.id,
            role="user",
            content=request.content.strip(),
            attachments=[],
            created_at=datetime.now(timezone.utc),
        )
        self.chat_repository.create_message(user_message)
        self.chat_repository.save_attachments(user_message.id, normalized_attachments)
        user_message.attachments = normalized_attachments

        if session.title == "新对话":
            next_title = self._derive_session_title(request.content)
            session = self.chat_repository.update_session_title(session.id, next_title) or session
            self.memory_service.record_project_context(session_id=session.id, title=next_title)

        self.memory_service.record_user_preferences(
            session_id=session.id,
            message_id=user_message.id,
            content=request.content,
        )
        self.memory_service.record_reference_memories(
            session_id=session.id,
            message_id=user_message.id,
            attachments=normalized_attachments,
        )

        document_ids = self._collect_document_ids(normalized_attachments, request.selected_document_ids)
        research_redirect = self._research_task_redirect_message(request.content)
        warning: str | None = None
        retrieval_failure_message: str | None = None
        retrieval_status = "skipped"
        evidence_items = []
        ready_documents = []
        citations: list[str] = []
        agent_trace_id: str | None = None
        action_status: str | None = None
        library_mutated = False

        message_history = self.chat_repository.list_messages(session.id)
        history_before_current = (
            message_history[:-1]
            if message_history and message_history[-1].id == user_message.id
            else message_history
        )
        memory_snapshot = self.memory_service.build_snapshot(
            session_id=session.id,
            selected_document_ids=document_ids,
        )
        mode_decision = None
        if research_redirect is None:
            mode_decision = self._select_agent_mode(
                session=session,
                user_message=user_message,
                request=request,
                attachments=normalized_attachments,
                selected_document_ids=document_ids,
                memory_snapshot=memory_snapshot,
            )
        agent_trace_id = mode_decision.trace_id if mode_decision is not None else None

        agent_result = None
        if research_redirect is not None:
            assistant_text = research_redirect
            retrieval_status = "skipped"
            action_status = "research_task_redirect"
            context_state = self._assemble_context_state(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                evidence_items=[],
            )
        else:
            agent_result = self._execute_agent_mode(
                decision=mode_decision,
                session=session,
                request=request,
                attachments=normalized_attachments,
                selected_document_ids=document_ids,
                history=history_before_current,
            )

        if agent_result is not None:
            agent_result = self._review_agent_result(
                decision=mode_decision,
                session=session,
                request=request,
                attachments=normalized_attachments,
                selected_document_ids=document_ids,
                result=agent_result,
            )
            assistant_text = agent_result.content
            retrieval_status = agent_result.retrieval_status
            warning = agent_result.warning
            evidence_items = agent_result.evidence_items
            citations = agent_result.citations
            document_ids = agent_result.used_document_ids or document_ids
            agent_trace_id = agent_result.agent_trace_id
            action_status = agent_result.action_status
            library_mutated = bool(getattr(agent_result, "library_mutated", False))
            if (
                self.knowledge_agent_runtime is not None
                and evidence_items
                and self.knowledge_agent_runtime.is_status_only_answer(assistant_text)
                and not library_mutated
                and action_status not in {"needs_clarification", "confirmation_required", "validation_failed", "failed"}
            ):
                assistant_text = self.knowledge_agent_runtime.ensure_final_answer(
                    user_prompt=request.content,
                    original_request=request,
                    runtime_mode=mode_decision.mode.value if mode_decision is not None else "knowledge_agent",
                    evidence_items=evidence_items,
                    citations=citations,
                    used_document_ids=document_ids,
                    tool_observations=[],
                    previous_content=assistant_text,
                    trace_digest={
                        "boundary": "chat_service",
                        "action_status": action_status,
                        "retrieval_status": retrieval_status,
                    },
                    trace_id=agent_trace_id,
                    action_status=action_status,
                )
                agent_result.content = assistant_text
            memory_snapshot = self.memory_service.build_snapshot(
                session_id=session.id,
                selected_document_ids=document_ids,
            )
            context_state = self._assemble_context_state(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                evidence_items=evidence_items,
            )
            self._finish_agent_trace(mode_decision, agent_result)
        elif mode_decision is not None and mode_decision.mode == AgentRunMode.DIRECT:
            assistant_text, context_state = self._generate_answer(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                memory_snapshot=memory_snapshot,
                evidence_items=[],
                delta_sink=delta_sink,
            )
            citations = []
            action_status = "direct_completed"
            self._append_direct_llm_trace(mode_decision)
            self._finish_agent_trace(
                mode_decision,
                action_status=action_status,
                retrieval_status=retrieval_status,
                used_document_count=0,
                evidence_count=0,
            )
        elif document_ids:
            ready_documents = [
                document
                for document in (
                    self.library_repository.get_document(document_id)
                    for document_id in document_ids
                )
                if document is not None and document.status == "ready"
            ]
            if ready_documents:
                try:
                    evidence_items = self.rag_service.retrieve_evidence(
                        question=request.content,
                        documents=ready_documents,
                        top_k=4,
                    )
                    retrieval_status = "ready"
                except Exception:
                    retrieval_status = "degraded"
                    retrieval_failure_message = self._selected_document_retrieval_failed_message()
                    warning = None
            else:
                retrieval_status = "skipped"
                warning = "所选论文仍在入库处理中，本轮先按普通模型回答。"

        if retrieval_failure_message and research_redirect is None and agent_result is None and not (
            mode_decision is not None and mode_decision.mode == AgentRunMode.DIRECT
        ):
            assistant_text = retrieval_failure_message
            context_state = self._assemble_context_state(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                evidence_items=[],
            )
            citations = []
            action_status = "retrieval_failed"
            if mode_decision is not None:
                self._finish_agent_trace(
                    mode_decision,
                    action_status=action_status,
                    retrieval_status=retrieval_status,
                    used_document_count=len(document_ids),
                    evidence_count=0,
                )
        elif research_redirect is None and agent_result is None and not (mode_decision is not None and mode_decision.mode == AgentRunMode.DIRECT):
            memory_snapshot = self.memory_service.build_snapshot(
                session_id=session.id,
                selected_document_ids=document_ids,
            )
            assistant_text, context_state = self._generate_answer(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                memory_snapshot=memory_snapshot,
                evidence_items=evidence_items,
                delta_sink=delta_sink,
            )
            citations = self._collect_citations(evidence_items)
            self._append_direct_llm_trace(mode_decision)
            if mode_decision is not None:
                action_status = "fallback_completed"
                self._finish_agent_trace(
                    mode_decision,
                    action_status=action_status,
                    retrieval_status=retrieval_status,
                    used_document_count=len(document_ids),
                    evidence_count=len(evidence_items),
                )
        assistant_message = ChatMessage(
            id=str(uuid4()),
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            status="completed",
            retrieval_status=retrieval_status,
            warning=warning,
            citations=citations,
            used_document_ids=document_ids,
            memory_hits=memory_snapshot.items,
            agent_trace_id=agent_trace_id,
            action_status=action_status,
            created_at=datetime.now(timezone.utc),
        )
        self.chat_repository.create_message(assistant_message)
        session = self.chat_repository.get_session(session.id) or session

        return ChatSendResponse(
            session=session,
            user_message=user_message,
            assistant_message=assistant_message,
            memory_snapshot=memory_snapshot,
            context_state=context_state,
            library_mutated=library_mutated,
        )

    def _review_agent_result(
        self,
        *,
        decision: AgentModeDecision | None,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        result,
    ):
        if decision is None or self.reflection_runtime is None or not self.enable_auto_reflection:
            return result
        if result.action_status in {"confirmation_required", "needs_clarification", "validation_failed", "failed"}:
            return result
        if decision.mode not in {AgentRunMode.REACT, AgentRunMode.PLANNER}:
            return result
        if not self._should_run_reflection_review(result):
            return result
        try:
            return self.reflection_runtime.review_agent_result(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
                decision=decision,
                result=result,
            )
        except Exception:
            return result

    @staticmethod
    def _should_run_reflection_review(result) -> bool:
        action_status = str(getattr(result, "action_status", "") or "")
        retrieval_status = str(getattr(result, "retrieval_status", "") or "")
        return action_status in {"failed", "validation_failed"} or retrieval_status == "degraded"

    def _generate_answer(
        self,
        *,
        session: ChatSession,
        history: list[ChatMessage],
        current_request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        memory_snapshot: MemorySnapshot,
        evidence_items,
        delta_sink: Callable[[str], None] | None = None,
    ) -> tuple[str, ChatContextState]:
        prompt_messages, context_state = self.context_assembler.assemble(
            session=session,
            history=history,
            current_request=current_request,
            attachments=attachments,
            evidence_items=evidence_items,
            knowledge_context_lines=self._knowledge_context_lines(
                current_request=current_request,
                attachments=attachments,
                evidence_items=evidence_items,
            ),
        )
        try:
            response = self._call_llm(
                prompt_messages,
                has_images=any(item.kind == "image" for item in attachments),
                delta_sink=delta_sink,
            )
        except RuntimeError as exc:
            return str(exc), context_state
        answer = response or self._build_template_answer(
            content=current_request.content,
            attachments=attachments,
            evidence_items=evidence_items,
            model_available=bool(self.api_key),
        )
        return answer, context_state

    def _append_direct_llm_trace(self, decision: AgentModeDecision | None) -> None:
        if decision is None or self.agent_orchestrator is None:
            return
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="direct_llm_call_finished",
            message="Direct answer LLM call finished.",
            payload=self.last_llm_diagnostic.as_trace_payload(),
        )

    def _assemble_context_state(
        self,
        *,
        session: ChatSession,
        history: list[ChatMessage],
        current_request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        evidence_items,
    ) -> ChatContextState:
        _, context_state = self.context_assembler.assemble(
            session=session,
            history=history,
            current_request=current_request,
            attachments=attachments,
            evidence_items=evidence_items,
            knowledge_context_lines=self._knowledge_context_lines(
                current_request=current_request,
                attachments=attachments,
                evidence_items=evidence_items,
            ),
        )
        return context_state

    def _select_agent_mode(
        self,
        *,
        session: ChatSession,
        user_message: ChatMessage,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        memory_snapshot: MemorySnapshot,
    ) -> AgentModeDecision | None:
        # ChatService assembles the chat-side decision packet, then delegates
        # routing to AgentOrchestrator. Product routing rules should live
        # there, not in API handlers or frontend store state.
        if self.agent_orchestrator is None:
            return None
        conversation_referents: dict = {}
        has_pending_action = False
        if self.knowledge_agent_runtime is not None:
            try:
                conversation_referents = self.knowledge_agent_runtime.conversation_referents(session.id)
            except Exception:
                conversation_referents = {}
            try:
                has_pending_action = self.knowledge_agent_runtime.has_pending_action(session.id)
            except Exception:
                has_pending_action = False
        payload = AgentOrchestratorInput(
            session_id=session.id,
            message_id=user_message.id,
            user_prompt=request.content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            conversation_referents=conversation_referents,
            memory_snapshot=memory_snapshot,
            available_tools=self.agent_orchestrator.available_tools(),
            available_skills=[],
            runtime_context={
                "has_pending_action": has_pending_action,
                "session_title": session.title,
                "entrypoint": "knowledge",
            },
        )
        try:
            return self.agent_orchestrator.select_mode(payload)
        except Exception:
            return None

    def _execute_agent_mode(
        self,
        *,
        decision: AgentModeDecision | None,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        history: list[ChatMessage],
    ):
        # Runtime dispatch remains the compatibility boundary: DIRECT falls
        # back to normal answer generation, while REACT/PLANNER/REFLECTION are
        # delegated to their existing runtimes without changing chat behavior.
        if self._has_pending_action(session.id):
            return self._run_knowledge_agent(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
                trace_id=decision.trace_id if decision is not None else None,
            )
        if decision is None:
            return self._run_knowledge_agent(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
            )
        if decision.mode == AgentRunMode.DIRECT:
            return None
        try:
            if decision.mode == AgentRunMode.REACT:
                return self._run_knowledge_agent(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    trace_id=decision.trace_id,
                )
            if decision.mode == AgentRunMode.PLANNER and self.knowledge_planner_runtime is not None:
                return self.knowledge_planner_runtime.handle(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    decision=decision,
                )
            if decision.mode == AgentRunMode.REFLECTION and self.reflection_runtime is not None:
                return self.reflection_runtime.handle(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    history=history,
                    decision=decision,
                )
        except Exception as exc:
            if self.agent_orchestrator is not None:
                self.agent_orchestrator.append_trace(
                    decision.trace_id,
                    status="agent_mode_execution_failed",
                    message="Agent mode execution failed.",
                    payload={"mode": decision.mode.value, "error": str(exc)},
                )
        return None

    def _finish_agent_trace(
        self,
        decision: AgentModeDecision | None,
        result=None,
        *,
        action_status: str | None = None,
        retrieval_status: str | None = None,
        used_document_count: int = 0,
        evidence_count: int = 0,
    ) -> None:
        if decision is None or self.agent_orchestrator is None:
            return
        payload = {
            "mode": decision.mode.value,
            "action_status": action_status or getattr(result, "action_status", None),
            "retrieval_status": retrieval_status or getattr(result, "retrieval_status", None),
            "used_document_count": used_document_count or len(getattr(result, "used_document_ids", []) or []),
            "evidence_count": evidence_count or len(getattr(result, "evidence_items", []) or []),
        }
        self.agent_orchestrator.finish_trace(
            decision.trace_id,
            status=ResearchRunStatus.COMPLETED,
            payload=payload,
        )

    def _run_knowledge_agent(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
    ):
        if self.knowledge_agent_runtime is None:
            return None
        try:
            if trace_id is None:
                return self.knowledge_agent_runtime.handle(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                )
            return self.knowledge_agent_runtime.run_react(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
                trace_id=trace_id,
            )
        except Exception:
            return None

    def _research_task_redirect_message(self, content: str) -> str | None:
        if self.enable_research_from_knowledge:
            return None
        if not self._looks_like_explicit_research_task_request(content):
            return None
        return (
            "当前 Knowledge 页面保持在稳定论文库问答链路中，不会直接启动实验性的 Research Task Agent Loop。"
            "我可以先帮你把这个需求整理成研究计划、检索问题和报告大纲；如果要运行后端研究任务，"
            "请使用明确的 `/api/research/stream` 实验入口，并开启对应配置。"
        )

    @staticmethod
    def _looks_like_explicit_research_task_request(content: str) -> bool:
        normalized = content.casefold()
        markers = (
            "创建研究任务",
            "按研究任务执行",
            "分步骤完成研究计划",
            "分步骤研究",
            "研究任务 agent",
            "research task",
            "research agent",
        )
        return any(marker in normalized for marker in markers)

    def _has_pending_action(self, session_id: str) -> bool:
        if self.knowledge_agent_runtime is None:
            return False
        try:
            return bool(self.knowledge_agent_runtime.has_pending_action(session_id))
        except Exception:
            return False

    def _knowledge_context_lines(
        self,
        *,
        current_request: ChatMessageRequest | None = None,
        attachments: list[ChatAttachment] | None = None,
        evidence_items=None,
    ) -> list[str]:
        if current_request is not None:
            has_library_attachment = any(
                item.kind in {"uploaded_pdf", "library_document"}
                for item in (attachments or [])
            )
            if not has_library_attachment and not evidence_items and not self._explicitly_requests_library_grounding(
                current_request.content
            ):
                return []
        if self.knowledge_agent_runtime is None:
            return []
        try:
            return self.knowledge_agent_runtime.build_context_lines()
        except Exception:
            return []

    @staticmethod
    def _explicitly_requests_library_grounding(content: str) -> bool:
        normalized = content.casefold()
        return any(
            marker in normalized
            for marker in (
                "论文库",
                "文献库",
                "库里",
                "库内",
                "上传论文",
                "所选论文",
                "选中的论文",
                "这篇论文",
                "这几篇论文",
                "这些论文",
                "根据论文",
                "基于论文",
                "根据文献",
                "基于文献",
                "文献证据",
                "paper library",
                "uploaded paper",
                "selected paper",
                "selected papers",
                "based on the paper",
                "based on these papers",
            )
        )

    @staticmethod
    def _selected_document_retrieval_failed_message() -> str:
        return (
            "论文正文向量检索服务当前不可用，无法完成基于所选论文正文的分析。"
            "请先启动 Milvus 向量服务并确认论文状态为 ready，然后重新发送问题。"
            "本轮不会改用普通聊天或仅凭论文元数据生成分析结论。"
        )

    def _call_llm(
        self,
        messages: list[dict],
        *,
        has_images: bool,
        delta_sink: Callable[[str], None] | None = None,
    ) -> str | None:
        if not self.api_key:
            self.last_llm_diagnostic = _LLMDiagnostic(
                status="not_configured",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=False,
                stream=delta_sink is not None,
            )
            return None
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            if delta_sink is not None:
                stream = client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=messages,
                    stream=True,
                )
                parts: list[str] = []
                for item in stream:
                    choices = getattr(item, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                    text = self._content_to_text(content)
                    if not text:
                        continue
                    parts.append(text)
                    delta_sink(text)
                result = "".join(parts).strip() if parts else None
                self.last_llm_diagnostic = _LLMDiagnostic(
                    status="success" if result else "empty_response",
                    model=self.model,
                    base_url_configured=bool(self.base_url),
                    api_key_configured=True,
                    stream=True,
                )
                return result
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=messages,
            )
        except Exception as exc:
            self.last_llm_diagnostic = _LLMDiagnostic(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=delta_sink is not None,
            )
            if has_images:
                raise RuntimeError("当前模型未开启图片理解或图片请求失败，请检查视觉模型配置。") from exc
            return None

        choices = getattr(response, "choices", None)
        if not choices:
            self.last_llm_diagnostic = _LLMDiagnostic(
                status="empty_response",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=False,
            )
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            self.last_llm_diagnostic = _LLMDiagnostic(
                status="empty_response",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=False,
            )
            return None
        content = getattr(message, "content", None)
        result = self._content_to_text(content, strip=True)
        self.last_llm_diagnostic = _LLMDiagnostic(
            status="success" if result else "empty_response",
            model=self.model,
            base_url_configured=bool(self.base_url),
            api_key_configured=True,
            stream=False,
        )
        return result

    @staticmethod
    def _content_to_text(content, *, strip: bool = False) -> str | None:
        if isinstance(content, str):
            return content.strip() if strip else content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if isinstance(text, str):
                    cleaned = text.strip() if strip else text
                    if cleaned:
                        parts.append(cleaned)
            return "\n".join(parts).strip() if parts else None
        return None

    def _build_template_answer(
        self,
        *,
        content: str,
        attachments: list[ChatAttachment],
        evidence_items,
        model_available: bool,
    ) -> str:
        if evidence_items:
            lines = [
                f"针对你的问题“{content.strip()}”，我只能基于本轮已检索到的证据先给出证据层面的回答：",
                "",
            ]
            for index, item in enumerate(evidence_items[:3], start=1):
                lines.append(f"{index}. {item.quote or item.snippet}（{item.citation_label}）")
            lines.append("")
            lines.append("证据边界：以上内容只来自本轮检索片段；更完整的综述需要继续基于正文证据合成。")
            return "\n".join(lines)

        if any(item.kind in {"uploaded_pdf", "library_document"} for item in attachments):
            return (
                "本轮没有检索到可引用的论文正文证据，因此我不能只凭文件名、标题或元数据生成论文总结。"
                "请确认论文已完成入库并且正文索引可用后再试。"
            )

        if any(item.kind == "image" for item in attachments):
            return (
                "我已收到图片附件，但当前模型没有返回可用结果。"
                "请检查视觉模型配置或稍后重试。"
            )

        if not model_available:
            return (
                "当前没有可用的 LLM 配置，无法可靠回答这个通用问题。"
                "请配置 LLM_API_KEY/LLM_BASE_URL 后重试；在模型可用时，普通问题会走直接问答链路。"
            )

        return "当前模型调用没有返回可用内容，请稍后重试。"

    def _normalize_attachments(
        self,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
    ) -> list[ChatAttachment]:
        normalized: list[ChatAttachment] = []
        seen_document_ids: set[str] = set()

        for attachment in attachments:
            if attachment.kind == "image":
                normalized.append(
                    ChatAttachment(
                        id=attachment.id or str(uuid4()),
                        kind="image",
                        display_name=attachment.display_name,
                        mime_type=attachment.mime_type,
                        data_url=attachment.data_url,
                        status=attachment.status or "ready",
                        metadata=attachment.metadata,
                    )
                )
                continue

            if attachment.kind == "uploaded_pdf" and not attachment.document_id:
                normalized.append(
                    ChatAttachment(
                        id=attachment.id or str(uuid4()),
                        kind="uploaded_pdf",
                        display_name=attachment.display_name,
                        mime_type=attachment.mime_type,
                        file_path=attachment.file_path,
                        status=attachment.status or "ready",
                        metadata=attachment.metadata,
                    )
                )
                continue

            if attachment.document_id:
                seen_document_ids.add(attachment.document_id)
                document = self.library_repository.get_document(attachment.document_id)
                display_name = (
                    attachment.display_name
                    or (document.display_name if document is not None else attachment.document_id)
                )
                normalized.append(
                    ChatAttachment(
                        id=attachment.id or str(uuid4()),
                        kind=attachment.kind,
                        display_name=display_name,
                        document_id=attachment.document_id,
                        status=attachment.status or (document.status if document is not None else "unknown"),
                        metadata=attachment.metadata,
                    )
                )

        for document_id in selected_document_ids:
            if document_id in seen_document_ids:
                continue
            document = self.library_repository.get_document(document_id)
            if document is None:
                continue
            normalized.append(
                ChatAttachment(
                    id=str(uuid4()),
                    kind="library_document",
                    display_name=document.display_name or document.filename,
                    document_id=document.id,
                    status=document.status,
                    metadata={"title": document.title, "filename": document.filename},
                )
            )
        return normalized

    @staticmethod
    def _collect_document_ids(
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
    ) -> list[str]:
        document_ids = [item.document_id for item in attachments if item.document_id]
        document_ids.extend(selected_document_ids)
        deduped: list[str] = []
        seen: set[str] = set()
        for document_id in document_ids:
            if document_id and document_id not in seen:
                seen.add(document_id)
                deduped.append(document_id)
        return deduped

    @staticmethod
    def _collect_document_ids_from_messages(messages: list[ChatMessage]) -> list[str]:
        seen: set[str] = set()
        results: list[str] = []
        for message in messages:
            for document_id in message.used_document_ids:
                if document_id not in seen:
                    seen.add(document_id)
                    results.append(document_id)
            for attachment in message.attachments:
                if attachment.document_id and attachment.document_id not in seen:
                    seen.add(attachment.document_id)
                    results.append(attachment.document_id)
        return results

    @staticmethod
    def _collect_citations(evidence_items) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for item in evidence_items:
            if item.citation_label in seen:
                continue
            seen.add(item.citation_label)
            citations.append(item.citation_label)
        return citations

    @staticmethod
    def _derive_session_title(content: str) -> str:
        compact = re.sub(r"\s+", " ", content).strip()
        if not compact:
            return "新对话"
        if len(compact) <= 28:
            return compact
        return f"{compact[:27]}…"

    def _require_session(self, session_id: str) -> ChatSession:
        session = self.chat_repository.get_session(session_id)
        if session is None:
            raise ValueError("Chat session not found")
        return session
