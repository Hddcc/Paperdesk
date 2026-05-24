"""SQLite repository for chat sessions, messages, and memory records."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from app.models import (
    ChatAttachment,
    ChatMessage,
    ChatSession,
    MemoryHit,
    MemoryRecord,
)

from .base import BaseRepository


class ChatRepository(BaseRepository):
    """Persist product-side chat state and lightweight memory indexes."""

    def create_session(self, title: str) -> ChatSession:
        session = ChatSession(id=str(uuid4()), title=title)
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.title,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
        return session

    def list_sessions(self) -> list[ChatSession]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    sessions.*,
                    (
                        SELECT SUBSTR(content, 1, 160)
                        FROM chat_messages
                        WHERE session_id = sessions.id
                        ORDER BY created_at DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM chat_sessions AS sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, session_id: str) -> ChatSession | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    sessions.*,
                    (
                        SELECT SUBSTR(content, 1, 160)
                        FROM chat_messages
                        WHERE session_id = sessions.id
                        ORDER BY created_at DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM chat_sessions AS sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> ChatSession | None:
        now = datetime.now(timezone.utc).isoformat()
        with self.database.connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> ChatSession | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        with self.database.connection() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return session

    def touch_session(self, session_id: str) -> None:
        with self.database.connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id),
            )

    def create_message(self, message: ChatMessage) -> ChatMessage:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    id,
                    session_id,
                    role,
                    content,
                    status,
                    retrieval_status,
                    warning,
                    citations_json,
                    used_document_ids_json,
                    used_file_ids_json,
                    memory_hits_json,
                    saved_report_id,
                    agent_trace_id,
                    action_status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.status,
                    message.retrieval_status,
                    message.warning,
                    json.dumps(message.citations, ensure_ascii=False),
                    json.dumps(message.used_document_ids, ensure_ascii=False),
                    json.dumps(message.used_file_ids, ensure_ascii=False),
                    json.dumps([item.model_dump(mode="json") for item in message.memory_hits], ensure_ascii=False),
                    message.saved_report_id,
                    message.agent_trace_id,
                    message.action_status,
                    message.created_at.isoformat(),
                ),
            )
        self.touch_session(message.session_id)
        return message

    def get_message(self, message_id: str) -> ChatMessage | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM chat_messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            attachment_rows = conn.execute(
                """
                SELECT *
                FROM chat_attachments
                WHERE message_id = ?
                ORDER BY sort_order ASC, created_at ASC
                """,
                (message_id,),
            ).fetchall()
        if row is None:
            return None
        message = self._row_to_message(row)
        message.attachments = [self._row_to_attachment(item) for item in attachment_rows]
        return message

    def update_message_report(self, message_id: str, report_id: str) -> ChatMessage | None:
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE chat_messages
                SET saved_report_id = ?, action_status = ?
                WHERE id = ?
                """,
                (report_id, "report_saved", message_id),
            )
        return self.get_message(message_id)

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
            attachment_rows = conn.execute(
                """
                SELECT *
                FROM chat_attachments
                WHERE message_id IN (
                    SELECT id FROM chat_messages WHERE session_id = ?
                )
                ORDER BY sort_order ASC, created_at ASC
                """,
                (session_id,),
            ).fetchall()
        attachments_by_message: dict[str, list[ChatAttachment]] = {}
        for row in attachment_rows:
            attachment = self._row_to_attachment(row)
            attachments_by_message.setdefault(str(row["message_id"]), []).append(attachment)
        messages: list[ChatMessage] = []
        for row in rows:
            message = self._row_to_message(row)
            message.attachments = attachments_by_message.get(message.id, [])
            messages.append(message)
        return messages

    def save_attachments(self, message_id: str, attachments: list[ChatAttachment]) -> None:
        if not attachments:
            return
        with self.database.connection() as conn:
            for index, attachment in enumerate(attachments):
                metadata = dict(attachment.metadata)
                metadata["message_id"] = message_id
                conn.execute(
                    """
                    INSERT INTO chat_attachments (
                        id,
                        message_id,
                        kind,
                        display_name,
                        mime_type,
                        document_id,
                        file_asset_id,
                        data_url,
                        file_path,
                        status,
                        metadata_json,
                        sort_order,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attachment.id,
                        message_id,
                        attachment.kind,
                        attachment.display_name,
                        attachment.mime_type,
                        attachment.document_id,
                        attachment.file_asset_id,
                        attachment.data_url,
                        attachment.file_path,
                        attachment.status,
                        json.dumps(metadata, ensure_ascii=False),
                        index,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    def list_attachments_for_message(self, message_id: str) -> list[ChatAttachment]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_attachments
                WHERE message_id = ?
                ORDER BY sort_order ASC, created_at ASC
                """,
                (message_id,),
            ).fetchall()
        return [self._row_to_attachment(row) for row in rows]

    def get_memory_by_source(self, source_kind: str, source_id: str) -> MemoryRecord | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM memory_records
                WHERE source_kind = ? AND source_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (source_kind, source_id),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        existing = None
        if record.source_kind and record.source_id:
            existing = self.get_memory_by_source(record.source_kind, record.source_id)

        if existing is None:
            with self.database.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_records (
                        id,
                        memory_type,
                        scope,
                        summary,
                        detail,
                        source_kind,
                        source_id,
                        status,
                        created_at,
                        updated_at,
                        last_verified_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.memory_type,
                        record.scope,
                        record.summary,
                        record.detail,
                        record.source_kind,
                        record.source_id,
                        record.status,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.last_verified_at.isoformat() if record.last_verified_at else None,
                    ),
                )
            return record

        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE memory_records
                SET
                    memory_type = ?,
                    scope = ?,
                    summary = ?,
                    detail = ?,
                    status = ?,
                    updated_at = ?,
                    last_verified_at = ?
                WHERE id = ?
                """,
                (
                    record.memory_type,
                    record.scope,
                    record.summary,
                    record.detail,
                    record.status,
                    datetime.now(timezone.utc).isoformat(),
                    record.last_verified_at.isoformat() if record.last_verified_at else None,
                    existing.id,
                ),
            )
        refreshed = self.get_memory(existing.id)
        return refreshed or record

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM memory_records WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(self, *, scope: str | None = None) -> list[MemoryRecord]:
        query = "SELECT * FROM memory_records"
        params: tuple[str, ...] = ()
        if scope is not None:
            query += " WHERE scope = ? OR scope = 'global'"
            params = (scope,)
        query += " ORDER BY updated_at DESC"
        with self.database.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def create_memory_link(self, memory_id: str, target_kind: str, target_id: str) -> None:
        with self.database.connection() as conn:
            existing = conn.execute(
                """
                SELECT 1
                FROM memory_links
                WHERE memory_id = ? AND target_kind = ? AND target_id = ?
                LIMIT 1
                """,
                (memory_id, target_kind, target_id),
            ).fetchone()
            if existing is not None:
                return
            conn.execute(
                """
                INSERT INTO memory_links (memory_id, target_kind, target_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    memory_id,
                    target_kind,
                    target_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_memories_for_target(self, target_kind: str, target_id: str) -> list[MemoryRecord]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT memories.*
                FROM memory_records AS memories
                INNER JOIN memory_links AS links
                    ON links.memory_id = memories.id
                WHERE links.target_kind = ? AND links.target_id = ?
                ORDER BY memories.updated_at DESC
                """,
                (target_kind, target_id),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def log_memory_refresh(
        self,
        memory_id: str,
        *,
        status: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_refresh_log (memory_id, status, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    status,
                    message,
                    json.dumps(payload or {}, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_message_preview=row["last_message_preview"],
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        memory_hits = [MemoryHit.model_validate(item) for item in json.loads(row["memory_hits_json"])]
        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            status=row["status"],
            retrieval_status=row["retrieval_status"],
            warning=row["warning"],
            citations=json.loads(row["citations_json"]),
            used_document_ids=json.loads(row["used_document_ids_json"]),
            used_file_ids=(
                json.loads(row["used_file_ids_json"])
                if "used_file_ids_json" in row.keys() and row["used_file_ids_json"]
                else []
            ),
            memory_hits=memory_hits,
            saved_report_id=row["saved_report_id"] if "saved_report_id" in row.keys() else None,
            agent_trace_id=row["agent_trace_id"] if "agent_trace_id" in row.keys() else None,
            action_status=row["action_status"] if "action_status" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_attachment(row: sqlite3.Row) -> ChatAttachment:
        metadata = json.loads(row["metadata_json"] or "{}")
        return ChatAttachment(
            id=row["id"],
            kind=row["kind"],
            display_name=row["display_name"],
            mime_type=row["mime_type"],
            document_id=row["document_id"],
            file_asset_id=row["file_asset_id"] if "file_asset_id" in row.keys() else None,
            data_url=row["data_url"],
            file_path=row["file_path"],
            status=row["status"],
            metadata=metadata,
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            memory_type=row["memory_type"],
            scope=row["scope"],
            summary=row["summary"],
            detail=row["detail"],
            source_kind=row["source_kind"],
            source_id=row["source_id"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_verified_at=(
                datetime.fromisoformat(row["last_verified_at"])
                if row["last_verified_at"]
                else None
            ),
        )
