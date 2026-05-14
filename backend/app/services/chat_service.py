"""Chat-first knowledge agent service."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import shutil
from uuid import uuid4

from openai import OpenAI

from app.models import (
    ChatAttachment,
    ChatContextState,
    ChatMessage,
    ChatMessageRequest,
    ChatSendResponse,
    ChatSession,
    ChatSessionDetail,
    MemorySnapshot,
)
from app.repositories import ChatRepository, LibraryRepository

from .chat_memory_service import ChatMemoryService
from .context_assembler import ContextAssembler
from .rag_service import RagService


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
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

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

    def send_message(self, session_id: str, request: ChatMessageRequest) -> ChatSendResponse:
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
        warning: str | None = None
        retrieval_status = "skipped"
        evidence_items = []
        ready_documents = []

        if document_ids:
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
                    warning = "知识库检索暂不可用，本轮按普通模型回答。"
            else:
                retrieval_status = "skipped"
                warning = "所选论文仍在入库处理中，本轮先按普通模型回答。"

        message_history = self.chat_repository.list_messages(session.id)
        memory_snapshot = self.memory_service.build_snapshot(
            session_id=session.id,
            selected_document_ids=document_ids,
        )
        assistant_text, context_state = self._generate_answer(
            session=session,
            history=message_history[:-1] if message_history and message_history[-1].id == user_message.id else message_history,
            current_request=request,
            attachments=normalized_attachments,
            memory_snapshot=memory_snapshot,
            evidence_items=evidence_items,
        )
        citations = self._collect_citations(evidence_items)
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
        )

    def _generate_answer(
        self,
        *,
        session: ChatSession,
        history: list[ChatMessage],
        current_request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        memory_snapshot: MemorySnapshot,
        evidence_items,
    ) -> tuple[str, ChatContextState]:
        prompt_messages, context_state = self.context_assembler.assemble(
            session=session,
            history=history,
            current_request=current_request,
            attachments=attachments,
            evidence_items=evidence_items,
        )
        try:
            response = self._call_llm(prompt_messages, has_images=any(item.kind == "image" for item in attachments))
        except RuntimeError as exc:
            return str(exc), context_state
        answer = response or self._build_template_answer(
            content=current_request.content,
            attachments=attachments,
            evidence_items=evidence_items,
        )
        return answer, context_state

    def _call_llm(self, messages: list[dict], *, has_images: bool) -> str | None:
        if not self.api_key:
            return None
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=messages,
            )
        except Exception as exc:
            if has_images:
                raise RuntimeError("当前模型未开启图片理解或图片请求失败，请检查视觉模型配置。") from exc
            return None

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
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip() if parts else None
        return None

    def _build_template_answer(self, *, content: str, attachments: list[ChatAttachment], evidence_items) -> str:
        lines = [f"我先根据你这轮的问题“{content.strip()}”整理一下。"]
        if evidence_items:
            lines.append("我从当前附加论文里提取到这些可直接引用的证据：")
            for index, item in enumerate(evidence_items[:3], start=1):
                lines.append(f"{index}. {item.quote or item.snippet}（{item.citation_label}）")
            lines.append("如果你愿意，我可以继续把这些证据整理成更完整的中文结论。")
        elif any(item.kind == "image" for item in attachments):
            lines.append("我已经收到图片附件。当前环境未返回正式模型结果，但消息和图片都已进入会话历史。")
        elif any(item.kind in {"uploaded_pdf", "library_document"} for item in attachments):
            lines.append("我已经收到论文上下文。如果论文仍在入库处理中，本轮会先按普通聊天理解你的问题。")
        else:
            lines.append("当前这轮没有附加知识库证据，我会按普通聊天方式先回答。")
        return "\n".join(lines)

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
