"""File-backed memory indexing for the chat-style knowledge agent."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from uuid import uuid4

from app.models import ChatAttachment, MemoryHit, MemoryRecord, MemorySnapshot
from app.repositories import ChatRepository, LibraryRepository

from .context_file_store import ContextFileStore


class ChatMemoryService:
    """Expose lightweight memory hits while using runtime context files for summaries."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        library_repository: LibraryRepository,
        file_store: ContextFileStore,
    ) -> None:
        self.chat_repository = chat_repository
        self.library_repository = library_repository
        self.file_store = file_store

    def build_snapshot(
        self,
        *,
        session_id: str,
        selected_document_ids: list[str] | None = None,
    ) -> MemorySnapshot:
        hits: list[MemoryHit] = []
        seen: set[str] = set()

        for index, preference in enumerate(self.file_store.read_user_preferences(), start=1):
            hit = MemoryHit(
                id=f"user-pref-{index}",
                memory_type="user",
                summary=preference,
                detail="来自运行时用户偏好文件",
                source_kind="runtime_user_file",
                source_id="user.md",
                status="active",
                last_verified_at=datetime.now(timezone.utc),
            )
            hits.append(hit)
            seen.add(hit.summary)

        session_summary = self.file_store.read_session_summary(session_id)
        summary_lines = [
            line[2:].strip()
            for line in session_summary.splitlines()
            if line.startswith("- ") and line[2:].strip() != "None yet."
        ]
        for index, line in enumerate(summary_lines[:5], start=1):
            if line in seen:
                continue
            hits.append(
                MemoryHit(
                    id=f"session-summary-{index}",
                    memory_type="project" if index == 1 else "feedback",
                    summary=line,
                    detail="来自运行时会话摘要文件",
                    source_kind="runtime_session_file",
                    source_id=f"{session_id}/session.md",
                    status="active",
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            seen.add(line)

        if selected_document_ids:
            for document_id in selected_document_ids:
                document = self.library_repository.get_document(document_id)
                if document is None:
                    continue
                summary = f"当前选中文档：《{document.display_name or document.filename}》"
                if summary in seen:
                    continue
                hits.append(
                    MemoryHit(
                        id=f"selected-doc-{document.id}",
                        memory_type="reference",
                        summary=summary,
                        detail=f"文档状态：{document.status}",
                        source_kind="library_document",
                        source_id=document.id,
                        status="active",
                        last_verified_at=datetime.now(timezone.utc),
                    )
                )
                seen.add(summary)

        for record in self.chat_repository.list_memories(scope=session_id):
            if record.status != "active" or record.summary in seen:
                continue
            if not self._validate_record(record):
                continue
            hits.append(
                MemoryHit(
                    id=record.id,
                    memory_type=record.memory_type,
                    summary=record.summary,
                    detail=record.detail,
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    status=record.status,
                    last_verified_at=record.last_verified_at,
                )
            )
            seen.add(record.summary)
            if len(hits) >= 8:
                break

        return MemorySnapshot(items=hits[:8], refreshed_at=datetime.now(timezone.utc))

    def record_user_preferences(
        self,
        *,
        session_id: str,
        message_id: str,
        content: str,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return

        preferences: list[tuple[str, str, str, str]] = []
        if "中文" in normalized:
            preferences.append(
                (
                    "默认使用中文回答。",
                    "用户曾明确要求使用中文作答。",
                    "user_preference",
                    "language_zh",
                )
            )

        if any(token in normalized for token in ("引用", "出处", "来源")):
            preferences.append(
                (
                    "回答时优先给出引用或出处。",
                    "用户曾强调需要保留来源、引用或出处说明。",
                    "user_preference",
                    "citations_required",
                )
            )

        if any(token in normalized.casefold() for token in ("markdown", "md 格式", "md格式")):
            preferences.append(
                (
                    "回答时优先使用 Markdown 格式。",
                    "用户曾明确要求使用 Markdown 或 MD 格式。",
                    "user_preference",
                    "markdown_format",
                )
            )

        if any(token in normalized for token in ("先总结再展开", "先概括再展开", "先总结")):
            preferences.append(
                (
                    "回答时优先先总结再展开。",
                    "用户曾明确要求先给总结再展开说明。",
                    "user_preference",
                    "summary_first",
                )
            )

        for summary, detail, source_kind, source_id in preferences:
            self.file_store.add_user_preference(summary)
            self._upsert_memory(
                memory_type="feedback" if source_kind == "chat_message" else "user",
                session_id=session_id,
                summary=summary,
                detail=detail,
                source_kind=source_kind,
                source_id=source_id,
                link_targets=[("chat_session", session_id), ("chat_message", message_id)],
            )

    def record_reference_memories(
        self,
        *,
        session_id: str,
        message_id: str,
        attachments: list[ChatAttachment],
    ) -> None:
        for attachment in attachments:
            if attachment.kind not in {"uploaded_pdf", "library_document"}:
                continue

            if attachment.document_id:
                document = self.library_repository.get_document(attachment.document_id)
                if document is None:
                    continue
                self._upsert_memory(
                    memory_type="reference",
                    session_id=session_id,
                    summary=f"曾在对话中引用论文《{document.display_name or document.filename}》",
                    detail=(
                        "文档状态："
                        f"{document.status}；标题："
                        f"{document.title or document.display_name or document.filename}"
                    ),
                    source_kind="library_document",
                    source_id=document.id,
                    link_targets=[
                        ("chat_session", session_id),
                        ("chat_message", message_id),
                        ("library_document", document.id),
                    ],
                )
                continue

            self._upsert_memory(
                memory_type="reference",
                session_id=session_id,
                summary=f"曾在对话中附加本地 PDF《{attachment.display_name}》",
                detail=f"附件类型：{attachment.kind}；文件名：{attachment.display_name}",
                source_kind="chat_attachment",
                source_id=attachment.id,
                link_targets=[
                    ("chat_session", session_id),
                    ("chat_message", message_id),
                ],
            )

    def record_reflection_lessons(
        self,
        *,
        session_id: str,
        trace_id: str,
        lessons: list[str],
        persist_long_term: bool = False,
    ) -> None:
        """Persist reusable self-reflection lessons into long-term memory."""

        if not persist_long_term:
            return

        for lesson in lessons:
            normalized = self._truncate(lesson, 160)
            if not self._is_high_value_reflection_lesson(normalized):
                continue
            summary = f"反思经验：{normalized}"
            self.file_store.add_user_preference(summary)
            self._upsert_memory(
                memory_type="feedback",
                session_id="global",
                summary=summary,
                detail=normalized,
                source_kind="reflection_lesson",
                source_id=self._reflection_lesson_source_id(normalized),
                link_targets=[
                    ("chat_session", session_id),
                    ("agent_trace", trace_id),
                ],
            )

    def record_project_context(self, *, session_id: str, title: str) -> None:
        if not title.strip():
            return
        self.file_store.initialize_session(session_id, title)
        self.file_store.sync_session_summary(
            session_id,
            title=title.strip(),
            user_preferences=self.file_store.read_user_preferences(),
            references=[],
            pending_topics=[f"当前会话主题：{title.strip()}"],
        )
        self._upsert_memory(
            memory_type="project",
            session_id=session_id,
            summary=f"当前会话主题：{self._truncate(title.strip(), 48)}",
            detail=title.strip(),
            source_kind="chat_session",
            source_id=session_id,
            link_targets=[("chat_session", session_id)],
        )

    def _upsert_memory(
        self,
        *,
        memory_type: str,
        session_id: str,
        summary: str,
        detail: str | None,
        source_kind: str,
        source_id: str,
        link_targets: list[tuple[str, str]],
    ) -> None:
        record = MemoryRecord(
            id=str(uuid4()),
            memory_type=memory_type,
            scope=session_id,
            summary=summary,
            detail=detail,
            source_kind=source_kind,
            source_id=source_id,
            status="active",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_verified_at=datetime.now(timezone.utc),
        )
        saved = self.chat_repository.upsert_memory(record)
        for target_kind, target_id in link_targets:
            self.chat_repository.create_memory_link(saved.id, target_kind, target_id)

    def _validate_record(self, record: MemoryRecord) -> bool:
        if record.source_kind == "library_document" and record.source_id:
            document = self.library_repository.get_document(record.source_id)
            if document is None:
                self.chat_repository.log_memory_refresh(
                    record.id,
                    status="stale",
                    message="Referenced library document no longer exists.",
                    payload={"document_id": record.source_id},
                )
                return False
        return True

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 1]}…"

    @staticmethod
    def _reflection_lesson_source_id(lesson: str) -> str:
        digest = hashlib.sha1(lesson.encode("utf-8")).hexdigest()[:16]
        return f"lesson-{digest}"

    @staticmethod
    def _is_high_value_reflection_lesson(lesson: str) -> bool:
        if len(lesson.strip()) < 8:
            return False
        markers = (
            "工具",
            "数据库",
            "SQLite",
            "RAG",
            "检索",
            "Observation",
            "标签",
            "分类",
            "引用",
            "证据",
            "权限",
            "安全",
            "删除",
            "写操作",
            "二次读取",
            "用户说",
            "这几篇",
            "工作流",
            "复合",
            "复杂",
            "失败",
            "降级",
            "必须",
            "优先",
        )
        return any(marker in lesson for marker in markers)
