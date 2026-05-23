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
    KnowledgeIntent,
    KnowledgeRiskLevel,
    KnowledgeRoute,
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


@dataclass(slots=True)
class _FastPathResponse:
    content: str
    action_status: str
    intent: KnowledgeIntent
    used_document_ids: list[str]
    reason: str
    tool_name: str
    trace_summary: str
    trace_payload: dict[str, object]


class ChatService:
    """Persist chat sessions while mixing model replies with optional RAG context."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        library_repository: LibraryRepository,
        category_repository=None,
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
        self.category_repository = category_repository
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
        fast_path_response = None
        general_chat_fast_path = False
        if research_redirect is None:
            fast_path_response = self._try_deterministic_read_fast_path(
                content=request.content,
                selected_document_ids=document_ids,
                attachments=normalized_attachments,
            )
            if fast_path_response is not None:
                mode_decision = self._build_fast_path_decision(
                    session=session,
                    user_message=user_message,
                    request=request,
                    attachments=normalized_attachments,
                    selected_document_ids=document_ids,
                    memory_snapshot=memory_snapshot,
                    route=KnowledgeRoute.TOOL_ACTION,
                    intent=fast_path_response.intent,
                    reason=fast_path_response.reason,
                    requires_tools=True,
                    target_runtime="DeterministicReadRuntime",
                )
            elif self._can_skip_router_for_general_chat(
                content=request.content,
                selected_document_ids=document_ids,
                attachments=normalized_attachments,
            ):
                general_chat_fast_path = True
                mode_decision = self._build_fast_path_decision(
                    session=session,
                    user_message=user_message,
                    request=request,
                    attachments=normalized_attachments,
                    selected_document_ids=document_ids,
                    memory_snapshot=memory_snapshot,
                    route=KnowledgeRoute.DIRECT_ANSWER,
                    intent=KnowledgeIntent.CHAT,
                    reason="Conservative local fast path: plain chat without library, grounding, or write intent.",
                    requires_tools=False,
                    target_runtime="DirectChatRuntime",
                )
            else:
                mode_decision = self._select_agent_mode(
                    session=session,
                    user_message=user_message,
                    request=request,
                    attachments=normalized_attachments,
                    selected_document_ids=document_ids,
                    memory_snapshot=memory_snapshot,
                )
        agent_trace_id = mode_decision.trace_id if mode_decision is not None else None
        self._append_workbench_hint_trace(mode_decision, request)
        self._append_slash_command_hint_trace(
            mode_decision,
            request,
            selected_document_ids=document_ids,
            attachments=normalized_attachments,
        )

        agent_result = None
        if fast_path_response is not None:
            assistant_text = fast_path_response.content
            document_ids = fast_path_response.used_document_ids or document_ids
            retrieval_status = "skipped"
            action_status = fast_path_response.action_status
            self._record_deterministic_read_trace(
                mode_decision,
                response=fast_path_response,
            )
            self._record_fast_path_referents(
                session=session,
                content=request.content,
                document_ids=fast_path_response.used_document_ids,
                source=fast_path_response.reason,
            )
            context_state = self._assemble_context_state(
                session=session,
                history=history_before_current,
                current_request=request,
                attachments=normalized_attachments,
                evidence_items=[],
            )
            self._finish_agent_trace(
                mode_decision,
                action_status=action_status,
                retrieval_status=retrieval_status,
                used_document_count=len(document_ids),
                evidence_count=0,
            )
        elif research_redirect is not None:
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
        elif fast_path_response is None and mode_decision is not None and mode_decision.mode == AgentRunMode.DIRECT:
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
            if general_chat_fast_path:
                retrieval_status = "skipped"
        elif fast_path_response is None and document_ids:
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

        if fast_path_response is None and retrieval_failure_message and research_redirect is None and agent_result is None and not (
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
        elif fast_path_response is None and research_redirect is None and agent_result is None and not (mode_decision is not None and mode_decision.mode == AgentRunMode.DIRECT):
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

    def _append_workbench_hint_trace(
        self,
        decision: AgentModeDecision | None,
        request: ChatMessageRequest,
    ) -> None:
        if decision is None or self.agent_orchestrator is None:
            return
        payload = self._workbench_hint_payload(request)
        if not payload:
            return
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="workbench_hint_recorded",
            message="Workbench profile and model hints recorded without changing routing.",
            payload=payload,
        )

    @staticmethod
    def _workbench_hint_payload(request: ChatMessageRequest) -> dict[str, str]:
        payload: dict[str, str] = {}
        if request.agent_profile_id:
            payload["agent_profile_id"] = request.agent_profile_id
        if request.model_id:
            payload["model_id"] = request.model_id
        return payload

    def _append_slash_command_hint_trace(
        self,
        decision: AgentModeDecision | None,
        request: ChatMessageRequest,
        *,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> None:
        if decision is None or self.agent_orchestrator is None:
            return
        if not request.command and not request.intent_hint:
            return
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="slash_command_hint_recorded",
            message="Slash command hint recorded without changing runtime routing.",
            payload={
                "command": request.command,
                "intent_hint": request.intent_hint,
                "selected_document_count": len(selected_document_ids),
                "has_attachments": bool(attachments),
            },
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

    def _build_fast_path_decision(
        self,
        *,
        session: ChatSession,
        user_message: ChatMessage,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        memory_snapshot: MemorySnapshot,
        route: KnowledgeRoute,
        intent: KnowledgeIntent,
        reason: str,
        requires_tools: bool,
        target_runtime: str,
    ) -> AgentModeDecision | None:
        if self.agent_orchestrator is None:
            return None
        payload = AgentOrchestratorInput(
            session_id=session.id,
            message_id=user_message.id,
            user_prompt=request.content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            conversation_referents={},
            memory_snapshot=memory_snapshot,
            available_tools=self.agent_orchestrator.available_tools(),
            available_skills=[],
            runtime_context={
                "fast_path": True,
                "session_title": session.title,
                "entrypoint": "knowledge",
            },
        )
        begin_trace = getattr(self.agent_orchestrator, "_begin_trace", None)
        trace_id = begin_trace(payload) if callable(begin_trace) else f"chat-fast-{uuid4().hex}"
        decision = AgentModeDecision(
            mode=AgentRunMode.DIRECT,
            route=route,
            intent=intent,
            reason=reason,
            confidence=0.93,
            target_runtime=target_runtime,
            requires_tools=requires_tools,
            requires_rag=False,
            requires_confirmation=False,
            risk_level=KnowledgeRiskLevel.LOW if requires_tools else KnowledgeRiskLevel.NONE,
            required_capabilities=["deterministic_read"] if requires_tools else [],
            trace_id=trace_id,
            fallback_used=False,
        )
        self.agent_orchestrator.append_trace(
            trace_id,
            status="agent_mode_selected",
            message=f"Agent mode selected by local fast path: {decision.mode.value}.",
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
                "target_objects": [],
                "required_capabilities": decision.required_capabilities,
                "fallback_used": False,
                "decision_source": "local_fast_path",
                "guardrail_candidate": None,
                "rule_candidate": None,
                "fallback_candidate": None,
                "llm_candidate": None,
                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                "available_skill_ids": [],
                "has_conversation_referents": False,
                "memory_hit_count": len(memory_snapshot.items),
            },
        )
        return decision

    def _record_fast_path_referents(
        self,
        *,
        session: ChatSession,
        content: str,
        document_ids: list[str],
        source: str,
    ) -> None:
        if self.knowledge_agent_runtime is None or not document_ids:
            return
        read_state = getattr(self.knowledge_agent_runtime, "_read_react_state", None)
        write_state = getattr(self.knowledge_agent_runtime, "_write_react_state", None)
        if not callable(read_state) or not callable(write_state):
            return
        ids = list(dict.fromkeys(str(document_id) for document_id in document_ids if document_id))
        if not ids:
            return
        try:
            state = read_state(session.id)
            referent = {
                "label": f"fast-path resolved {len(ids)} documents",
                "document_ids": ids,
                "source_tool": "chat.fast_path",
                "source": source,
                "count": len(ids),
            }
            state["last_document_set"] = referent
            if len(ids) == 1:
                state["last_single_document"] = {
                    **referent,
                    "document_id": ids[0],
                    "count": 1,
                }
            else:
                state["last_multi_document_set"] = referent
            state["last_user_goal"] = content[:240]
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_state(session.id, state)
        except Exception:
            return

    def _record_deterministic_read_trace(
        self,
        decision: AgentModeDecision | None,
        *,
        response: _FastPathResponse,
    ) -> None:
        if self.agent_orchestrator is None or decision is None or not decision.trace_id:
            return
        target_objects = self._deterministic_target_objects(response)
        counts = self._deterministic_counts(response)
        base_payload = {
            "route_id": "deterministic_read_execution",
            "tool_name": response.tool_name,
            "operation_level": "query-level",
            "io_type": "read",
            "write_type": "none",
            "target_objects": target_objects,
            "affected_objects": [],
            "requires_confirmation": False,
            "success": True,
            "verification_success": None,
            "duration_ms": 0,
            "decision_source": "local_fast_path",
        }
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="deterministic_action",
            message=f"Deterministic read selected {response.tool_name}.",
            payload={
                **base_payload,
                "summary": response.trace_summary,
                "counts": counts,
            },
        )
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="tool_call_log",
            message=f"Deterministic read executed {response.tool_name}.",
            payload={
                **base_payload,
                "counts": counts,
            },
        )
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="deterministic_observation",
            message=response.trace_summary,
            payload={
                "tool": response.tool_name,
                "status": response.action_status,
                "summary": response.trace_summary,
                "payload": response.trace_payload,
                "observation": {
                    "tool_name": response.tool_name,
                    "success": True,
                    "operation_level": "query-level",
                    "io_type": "read",
                    "write_type": "none",
                    "target_objects": target_objects,
                    "affected_objects": [],
                    "counts": counts,
                    "data": response.trace_payload,
                    "requires_followup": False,
                    "requires_confirmation": False,
                    "verification": None,
                    "error": None,
                },
            },
        )

    @staticmethod
    def _deterministic_target_objects(response: _FastPathResponse) -> list[dict[str, object]]:
        targets: list[dict[str, object]] = []
        for document_id in response.used_document_ids:
            targets.append({"type": "paper", "id": document_id})
        category_names = response.trace_payload.get("category_names")
        if isinstance(category_names, list):
            for name in category_names:
                if str(name).strip():
                    targets.append({"type": "category", "name": str(name)})
        elif response.trace_payload.get("category_name"):
            targets.append({"type": "category", "name": str(response.trace_payload["category_name"])})
        if not targets and response.tool_name == "library.explorer.category_stats":
            targets.append({"type": "category"})
        if not targets and response.tool_name == "library.explorer.stats":
            targets.append({"type": "library"})
        return targets

    @staticmethod
    def _deterministic_counts(response: _FastPathResponse) -> dict[str, object]:
        payload = response.trace_payload
        counts: dict[str, object] = {}
        for source_key, target_key in (
            ("document_count", "documents"),
            ("total_documents", "total_documents"),
            ("category_count", "categories"),
            ("tagged_document_count", "tagged_documents"),
            ("untagged_document_count", "untagged_documents"),
        ):
            if source_key in payload:
                counts[target_key] = payload[source_key]
        if not counts and response.used_document_ids:
            counts["documents"] = len(response.used_document_ids)
        return counts

    def _can_skip_router_for_general_chat(
        self,
        *,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if selected_document_ids:
            return False
        if any(item.kind == "library_document" for item in attachments):
            return False
        if self._has_read_only_safety_marker(content.casefold()):
            return False
        if self._has_active_write_intent(content):
            return False
        if self._looks_like_read_then_write_request(content):
            return False
        if self._is_correction_or_reflection_request(content):
            return False
        if self._has_library_or_paper_reference(content):
            return False
        return True

    def _try_deterministic_read_fast_path(
        self,
        *,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> _FastPathResponse | None:
        if self._looks_like_read_then_write_request(content):
            return None
        if self._has_active_write_intent(content):
            return None
        if self._has_mixed_or_non_field_read_intent(content):
            return None
        document_ids = self._collect_document_ids(attachments, selected_document_ids)
        if self._is_selected_document_tag_read(content, document_ids):
            return self._fast_path_document_categories(content=content, document_ids=document_ids)
        if self._is_selected_document_metadata_read(content, document_ids):
            return self._fast_path_document_metadata(content=content, document_ids=document_ids)
        if self._is_untagged_documents_read(content):
            return self._fast_path_untagged_documents()
        if self._is_category_documents_read(content):
            return self._fast_path_category_documents(content)
        if self._is_tagged_document_detail_read(content):
            return None
        if self._is_category_stats_read(content):
            return self._fast_path_category_stats()
        if self._is_library_count_read(content):
            documents = self.library_repository.list_documents()
            document_ids = [document.id for document in documents]
            return _FastPathResponse(
                content=f"\u5f53\u524d\u8bba\u6587\u5e93\u5171\u6709 {len(documents)} \u7bc7\u8bba\u6587\u3002",
                action_status="completed",
                intent=KnowledgeIntent.TAG_QUERY,
                used_document_ids=document_ids,
                reason="Pure library metadata count answered from local SQLite metadata.",
                tool_name="library.explorer.stats",
                trace_summary=f"Deterministically read library count: {len(documents)} document(s).",
                trace_payload={
                    "total_documents": len(documents),
                    "document_count": len(documents),
                    "document_ids": document_ids,
                },
            )
        return None

    def _fast_path_document_metadata(self, *, content: str, document_ids: list[str]) -> _FastPathResponse | None:
        documents = [self.library_repository.get_document(document_id) for document_id in document_ids]
        documents = [document for document in documents if document is not None]
        if not documents:
            return None
        wants_title = self._contains_any(content, ("title", "\u6807\u9898"))
        wants_authors = self._contains_any(content, ("author", "authors", "\u4f5c\u8005"))
        wants_pages = self._contains_any(content, ("page", "pages", "\u9875", "\u9875\u6570"))
        wants_status = self._contains_any(content, ("status", "\u72b6\u6001"))
        metadata_by_document_id = (
            self._chunk_metadata_by_document_ids([document.id for document in documents])
            if wants_authors
            else {}
        )
        lines = []
        for document in documents:
            fields = []
            if wants_title or not (wants_authors or wants_pages or wants_status):
                fields.append(f"\u6807\u9898\uff1a{document.title or document.display_name or document.filename}")
            if wants_authors:
                authors = metadata_by_document_id.get(document.id, {}).get("authors") or []
                author_text = "\u3001".join(authors) if authors else "\u672a\u77e5"
                fields.append(f"\u4f5c\u8005\uff1a{author_text}")
            if wants_pages:
                fields.append(f"\u9875\u6570\uff1a{document.page_count}")
            if wants_status:
                fields.append(f"\u72b6\u6001\uff1a{document.status}")
            lines.append("; ".join(fields))
        document_payloads = [
            {
                "id": document.id,
                "title": document.title or document.display_name or document.filename,
                "filename": document.filename,
                "display_name": document.display_name,
                "page_count": document.page_count,
                "status": document.status,
                "authors": metadata_by_document_id.get(document.id, {}).get("authors") or [],
            }
            for document in documents
        ]
        return _FastPathResponse(
            content="\n".join(lines),
            action_status="completed",
            intent=KnowledgeIntent.PAPER_QA,
            used_document_ids=[document.id for document in documents],
            reason="Pure selected-document metadata read answered without ReAct.",
            tool_name="library.explorer.document_metadata",
            trace_summary=f"Deterministically read metadata for {len(documents)} document(s).",
            trace_payload={
                "documents": document_payloads,
                "document_ids": [document.id for document in documents],
                "requested_fields": [
                    field
                    for field, wanted in (
                        ("title", wants_title),
                        ("authors", wants_authors),
                        ("page_count", wants_pages),
                        ("status", wants_status),
                    )
                    if wanted
                ],
                "document_count": len(documents),
            },
        )

    def _chunk_metadata_by_document_ids(self, document_ids: list[str]) -> dict[str, dict[str, list[str]]]:
        chunk_repository = getattr(self.rag_service, "chunk_repository", None)
        if chunk_repository is None:
            return {}
        try:
            chunks = chunk_repository.list_chunks(document_ids=document_ids)
        except Exception:
            return {}
        metadata_by_document_id: dict[str, dict[str, list[str]]] = {}
        for chunk in chunks:
            document_id = str(getattr(chunk, "document_id", "") or "")
            if not document_id or document_id in metadata_by_document_id:
                continue
            metadata = getattr(chunk, "metadata", None) or {}
            authors = self._normalize_author_values(metadata.get("authors") or metadata.get("author"))
            if authors:
                metadata_by_document_id[document_id] = {"authors": authors}
        return metadata_by_document_id

    @staticmethod
    def _normalize_author_values(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            separators = [";", "\uff1b", "\u3001", ","]
            values = [value]
            for separator in separators:
                if separator in value:
                    values = value.split(separator)
                    break
            return [item.strip() for item in values if item.strip()]
        return []

    def _fast_path_document_categories(self, *, content: str, document_ids: list[str]) -> _FastPathResponse | None:
        if self.category_repository is None:
            return None
        documents = [self.library_repository.get_document(document_id) for document_id in document_ids]
        documents = [document for document in documents if document is not None]
        if not documents:
            return None
        categories_by_document_id = self.category_repository.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        lines = []
        document_payloads = []
        for document in documents:
            categories = categories_by_document_id.get(document.id, [])
            names = [category.name for category in categories]
            tag_text = "\u3001".join(names) if names else "\u65e0\u6807\u7b7e"
            lines.append(f"{document.title or document.display_name or document.filename}\uff1a{tag_text}")
            document_payloads.append(
                {
                    "id": document.id,
                    "title": document.title or document.display_name or document.filename,
                    "filename": document.filename,
                    "categories": names,
                }
            )
        return _FastPathResponse(
            content="\n".join(lines),
            action_status="completed",
            intent=KnowledgeIntent.TAG_QUERY,
            used_document_ids=[document.id for document in documents],
            reason="Pure selected-document tag read answered from category links.",
            tool_name="library.explorer.document_categories",
            trace_summary=f"Deterministically read categories for {len(documents)} document(s).",
            trace_payload={
                "documents": document_payloads,
                "document_ids": [document.id for document in documents],
                "document_count": len(documents),
            },
        )

    def _fast_path_category_stats(self) -> _FastPathResponse | None:
        if self.category_repository is None:
            return None
        documents = self.library_repository.list_documents()
        categories = self.category_repository.list_categories()
        categories_by_document_id = self.category_repository.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        counts = {category.name: 0 for category in categories}
        tagged_document_ids = set()
        for document_id, document_categories in categories_by_document_id.items():
            if document_categories:
                tagged_document_ids.add(document_id)
            for category in document_categories:
                counts[category.name] = counts.get(category.name, 0) + 1
        if not categories:
            content = f"\u5f53\u524d\u8bba\u6587\u5e93\u5171\u6709 {len(documents)} \u7bc7\u8bba\u6587\uff0c\u5f53\u524d\u6ca1\u6709\u6807\u7b7e\u6216\u5206\u7c7b\u3002"
        else:
            parts = [f"{name}\uff08{count} \u7bc7\uff09" for name, count in counts.items()]
            joined_parts = "\u3001".join(parts)
            content = (
                f"\u5f53\u524d\u8bba\u6587\u5e93\u5171\u6709 {len(documents)} \u7bc7\u8bba\u6587\uff0c"
                f"\u5171\u6709 {len(categories)} \u4e2a\u6807\u7b7e\uff08\u5171\u6709 {len(categories)} \u7c7b\u6807\u7b7e\uff09\uff1a"
                f"{joined_parts}\u3002"
            )
        untagged_count = len(documents) - len(tagged_document_ids)
        content += (
            f"\u6709\u6807\u7b7e\u7684\u8bba\u6587 {len(tagged_document_ids)} \u7bc7\uff0c"
            f"\u6ca1\u6709\u6807\u7b7e\u7684\u8bba\u6587 {untagged_count} \u7bc7\u3002"
        )
        return _FastPathResponse(
            content=content,
            action_status="completed",
            intent=KnowledgeIntent.TAG_QUERY,
            used_document_ids=[],
            reason="Pure category list/statistics read answered from category links.",
            tool_name="library.explorer.category_stats",
            trace_summary=f"Deterministically read {len(categories)} category/categories.",
            trace_payload={
                "category_count": len(categories),
                "categories": [
                    {
                        "id": category.id,
                        "name": category.name,
                        "document_count": counts.get(category.name, 0),
                    }
                    for category in categories
                ],
                "category_names": [category.name for category in categories],
                "total_documents": len(documents),
                "tagged_document_count": len(tagged_document_ids),
                "untagged_document_count": untagged_count,
            },
        )

    def _fast_path_untagged_documents(self) -> _FastPathResponse | None:
        if self.category_repository is None:
            return None
        documents = self.library_repository.list_documents()
        categories_by_document_id = self.category_repository.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        untagged = [
            document
            for document in documents
            if not categories_by_document_id.get(document.id)
        ]
        if not untagged:
            content = "\u5f53\u524d\u6ca1\u6709\u65e0\u6807\u7b7e\u8bba\u6587\u3002"
        else:
            names = "\u3001".join(document.title or document.display_name or document.filename for document in untagged)
            content = f"\u5f53\u524d\u6709 {len(untagged)} \u7bc7\u6ca1\u6709\u6807\u7b7e\u7684\u8bba\u6587\uff1a{names}\u3002"
        return _FastPathResponse(
            content=content,
            action_status="completed",
            intent=KnowledgeIntent.TAG_QUERY,
            used_document_ids=[document.id for document in untagged],
            reason="Pure untagged-document read answered from category links.",
            tool_name="library.explorer.category_stats",
            trace_summary=f"Deterministically read {len(untagged)} untagged document(s).",
            trace_payload={
                "documents": [
                    {
                        "id": document.id,
                        "title": document.title or document.display_name or document.filename,
                        "filename": document.filename,
                    }
                    for document in untagged
                ],
                "document_ids": [document.id for document in untagged],
                "document_count": len(untagged),
                "untagged_document_count": len(untagged),
            },
        )

    def _fast_path_category_documents(self, content: str) -> _FastPathResponse | None:
        if self.category_repository is None:
            return None
        category = self._resolve_category_mention(content)
        if category is None:
            return None
        documents = self.library_repository.list_documents()
        categories_by_document_id = self.category_repository.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        matched = [
            document
            for document in documents
            if any(item.id == category.id for item in categories_by_document_id.get(document.id, []))
        ]
        if matched:
            names = "\u3001".join(document.display_name or document.title or document.filename for document in matched)
            content_text = f"\u5df2\u8bc6\u522b\u5230\u300c{category.name}\u300d\u662f\u6807\u7b7e/\u5206\u7c7b\uff0c\u4e0b\u9762\u5171\u6709 {len(matched)} \u7bc7\u8bba\u6587\uff1a{names}\u3002"
        else:
            content_text = f"\u5df2\u8bc6\u522b\u5230\u300c{category.name}\u300d\u662f\u6807\u7b7e/\u5206\u7c7b\uff0c\u4e0b\u9762\u6ca1\u6709\u8bba\u6587\u3002"
        return _FastPathResponse(
            content=content_text,
            action_status="completed",
            intent=KnowledgeIntent.TAG_QUERY,
            used_document_ids=[document.id for document in matched],
            reason="Pure category membership read answered from category links.",
            tool_name="library.explorer.find_documents",
            trace_summary=f"Deterministically resolved {len(matched)} document(s) for category {category.name}.",
            trace_payload={
                "documents": [
                    {
                        "id": document.id,
                        "title": document.title or document.display_name or document.filename,
                        "filename": document.filename,
                        "display_name": document.display_name,
                    }
                    for document in matched
                ],
                "document_ids": [document.id for document in matched],
                "document_count": len(matched),
                "category_name": category.name,
                "category_names": [category.name],
            },
        )

    def _resolve_category_mention(self, content: str):
        if self.category_repository is None:
            return None
        normalized = content.casefold()
        categories = self.category_repository.list_categories()
        for category in categories:
            if category.name.casefold() in normalized:
                return category
        match = re.search(r"(?:under|in|with)\s+(?:the\s+)?['\"]?(.+?)['\"]?\s+(?:tag|category)\b", normalized)
        if match:
            candidate = match.group(1).strip()
            for category in categories:
                if category.name.casefold() == candidate:
                    return category
        return None

    def _has_active_write_intent(self, content: str) -> bool:
        normalized = content.casefold()
        read_only_safety = self._has_read_only_safety_marker(normalized)
        destructive_markers = (
            "delete",
            "remove",
            "clear",
            "\u5220\u9664",
            "\u79fb\u9664",
            "\u6e05\u7a7a",
        )
        write_markers = (
            "add tag",
            "assign tag",
            "tag these",
            "tag this",
            "rename",
            "modify",
            "update",
            "save report",
            "\u6253\u6807\u7b7e",
            "\u52a0\u6807\u7b7e",
            "\u6dfb\u52a0\u6807\u7b7e",
            "\u91cd\u547d\u540d",
            "\u4fee\u6539",
            "\u4fdd\u5b58\u62a5\u544a",
        )
        if any(marker in normalized for marker in write_markers):
            return True
        if self._contains_any(
            normalized,
            ("\u6807\u7b7e", "\u5206\u7c7b", "tag", "category"),
        ) and self._contains_any(
            normalized,
            ("\u52a0\u4e0a", "\u90fd\u52a0", "\u52a0\u4e00\u4e2a", "\u6253\u4e0a", "\u65b0\u589e", "\u65b0\u5efa", "\u521b\u5efa", "\u8865\u4e0a", "\u6539\u6210", "\u6362\u6210"),
        ):
            return True
        if any(marker in normalized for marker in destructive_markers):
            return not read_only_safety
        return False

    @staticmethod
    def _has_read_only_safety_marker(normalized_content: str) -> bool:
        return any(
            marker in normalized_content
            for marker in (
                "do not",
                "don't",
                "readonly",
                "read only",
                "only read",
                "\u4e0d\u8981",
                "\u522b",
                "\u53ea\u8bfb",
                "\u53ea\u770b",
                "\u4e0d\u6267\u884c",
                "\u4e0d\u4fee\u6539",
                "\u4e0d\u5220\u9664",
            )
        )

    def _looks_like_read_then_write_request(self, content: str) -> bool:
        return self._has_active_write_intent(content) and self._has_read_intent(content)

    def _has_mixed_or_non_field_read_intent(self, content: str) -> bool:
        normalized = content.casefold()
        if re.search(r"what\s+does\b.+\bmean", normalized):
            return True
        return self._contains_any(
            normalized,
            (
                "explain",
                "analysis",
                "analyze",
                "compare",
                "comparison",
                "evaluate",
                "review",
                "write report",
                "write a report",
                "save report",
                "what does",
                "abstract",
                "\u89e3\u91ca",
                "\u8bf4\u660e\u4e00\u4e0b",
                "\u662f\u4ec0\u4e48\u610f\u601d",
                "\u4ec0\u4e48\u610f\u601d",
                "\u5206\u6790",
                "\u603b\u7ed3",
                "\u6982\u62ec",
                "\u8bc4\u4ef7",
                "\u5bf9\u6bd4",
                "\u6bd4\u8f83",
                "\u62a5\u544a",
                "\u5199\u62a5\u544a",
                "\u751f\u6210\u62a5\u544a",
                "\u5199\u7efc\u8ff0",
                "\u4fdd\u5b58\u62a5\u544a",
                "\u6458\u8981",
                "\u6838\u5fc3\u4efb\u52a1",
                "\u8bc1\u636e\u4f9d\u636e",
                "\u7136\u540e",
                "\u5e76\u4e14",
                "\u987a\u4fbf",
                "\u518d\u8865\u5145",
                "\u8865\u5145\u4e00\u53e5",
                "\u518d\u5e2e\u6211",
                "\u540c\u65f6",
                "\u63a5\u7740",
                "\u7ee7\u7eed\u5904\u7406",
                "\u6253\u6807\u7b7e",
                "\u52a0\u6807\u7b7e",
                "\u5220\u9664\u6807\u7b7e",
                "\u6e05\u7a7a",
                "\u79fb\u9664",
                "\u4fee\u6539\u5206\u7c7b",
                "\u91cd\u547d\u540d",
            ),
        )

    def _is_correction_or_reflection_request(self, content: str) -> bool:
        return self._contains_any(
            content,
            ("wrong", "incorrect", "check again", "\u4e0d\u5bf9", "\u9519", "\u91cd\u65b0\u68c0\u67e5", "\u91cd\u65b0"),
        )

    def _is_analysis_or_report_request(self, content: str) -> bool:
        return self._contains_any(
            content,
            (
                "compare",
                "comparison",
                "summary",
                "summarize",
                "report",
                "review",
                "analyze",
                "\u5bf9\u6bd4",
                "\u6bd4\u8f83",
                "\u603b\u7ed3",
                "\u6458\u8981",
                "\u62a5\u544a",
                "\u7efc\u8ff0",
                "\u5206\u6790",
                "\u8bb2\u4ec0\u4e48",
            ),
        )

    def _has_read_intent(self, content: str) -> bool:
        return any(
            predicate(content)
            for predicate in (
                self._is_library_count_read,
                self._is_category_stats_read,
                self._is_untagged_documents_read,
                self._is_category_documents_read,
            )
        ) or self._contains_any(content, ("what", "which", "how many", "\u591a\u5c11", "\u54ea\u4e9b", "\u6709\u54ea\u4e9b"))

    def _is_tagged_document_detail_read(self, content: str) -> bool:
        return self._contains_any(
            content,
            ("\u5e26\u6807\u7b7e\u7684\u6587\u7ae0", "\u5e26\u6807\u7b7e\u7684\u8bba\u6587", "\u6709\u6807\u7b7e\u7684\u6587\u7ae0", "\u5206\u522b\u662f\u4ec0\u4e48"),
        )

    def _has_library_or_paper_reference(self, content: str) -> bool:
        return self._explicitly_requests_library_grounding(content) or self._contains_any(
            content,
            (
                "paper",
                "papers",
                "literature",
                "document",
                "documents",
                "tag",
                "tags",
                "category",
                "categories",
                "report",
                "review",
                "evidence",
                "library",
                "\u8bba\u6587",
                "\u6587\u732e",
                "\u6807\u7b7e",
                "\u5206\u7c7b",
                "\u62a5\u544a",
                "\u7efc\u8ff0",
                "\u8bc1\u636e",
                "\u8bba\u6587\u5e93",
            ),
        )

    def _is_library_count_read(self, content: str) -> bool:
        return self._contains_any(content, ("how many papers", "paper count", "number of papers", "\u591a\u5c11\u7bc7\u8bba\u6587", "\u51e0\u7bc7\u8bba\u6587", "\u6709\u51e0\u7bc7\u8bba\u6587")) and self._contains_any(
            content,
            ("library", "paperdesk", "\u8bba\u6587\u5e93", "\u5e93\u91cc", "\u5f53\u524d"),
        )

    def _is_selected_document_metadata_read(self, content: str, document_ids: list[str]) -> bool:
        if not document_ids:
            return False
        return self._contains_any(content, ("title", "author", "authors", "page", "pages", "status", "\u6807\u9898", "\u4f5c\u8005", "\u9875\u6570", "\u72b6\u6001"))

    def _is_selected_document_tag_read(self, content: str, document_ids: list[str]) -> bool:
        if not document_ids:
            return False
        return self._contains_any(content, ("tag", "tags", "category", "categories", "\u6807\u7b7e", "\u5206\u7c7b")) and self._contains_any(
            content,
            ("what", "which", "have", "list", "\u6709\u54ea\u4e9b", "\u54ea\u4e9b", "\u5f53\u524d"),
        )

    def _is_category_stats_read(self, content: str) -> bool:
        if not self._contains_any(content, ("tag", "tags", "category", "categories", "\u6807\u7b7e", "\u5206\u7c7b")):
            return False
        return self._contains_any(content, ("available", "list", "statistics", "stats", "counts", "what", "which", "how many", "\u6709\u54ea\u4e9b", "\u54ea\u4e9b", "\u7edf\u8ba1", "\u6570\u91cf", "\u6bcf\u4e2a", "\u6709\u51e0", "\u51e0\u7c7b"))

    def _is_category_documents_read(self, content: str) -> bool:
        return self._contains_any(content, ("under", "in tag", "with tag", "tagged", "\u6807\u7b7e\u4e0b", "\u5c5e\u4e8e", "\u88ab\u6807\u8bb0\u4e3a")) and self._contains_any(
            content,
            ("paper", "papers", "documents", "\u8bba\u6587"),
        )

    def _is_untagged_documents_read(self, content: str) -> bool:
        return self._contains_any(content, ("untagged", "without tags", "no tags", "without categories", "\u6ca1\u6709\u6807\u7b7e", "\u65e0\u6807\u7b7e", "\u4e0d\u5e26\u6807\u7b7e"))

    @staticmethod
    def _contains_any(content: str, markers: tuple[str, ...]) -> bool:
        normalized = content.casefold()
        return any(marker.casefold() in normalized for marker in markers)

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
