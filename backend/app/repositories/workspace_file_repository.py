"""SQLite repository for generated workspace files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from app.models import WorkspaceFile

from .base import BaseRepository


class WorkspaceFileRepository(BaseRepository):
    """Store metadata for generated files without managing file contents."""

    def create(self, workspace_file: WorkspaceFile) -> WorkspaceFile:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO workspace_files (
                    id,
                    session_id,
                    source_message_id,
                    created_by,
                    file_kind,
                    display_name,
                    relative_path,
                    storage_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    status,
                    source_file_ids_json,
                    source_document_ids_json,
                    created_at,
                    updated_at,
                    failure_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_file.id,
                    workspace_file.session_id,
                    workspace_file.source_message_id,
                    workspace_file.created_by,
                    workspace_file.file_kind,
                    workspace_file.display_name,
                    workspace_file.relative_path,
                    workspace_file.storage_path,
                    workspace_file.mime_type,
                    workspace_file.size_bytes,
                    workspace_file.checksum,
                    workspace_file.status,
                    json.dumps(workspace_file.source_file_ids, ensure_ascii=False),
                    json.dumps(workspace_file.source_document_ids, ensure_ascii=False),
                    workspace_file.created_at.isoformat(),
                    workspace_file.updated_at.isoformat(),
                    workspace_file.failure_reason,
                ),
            )
        return workspace_file

    def get(self, file_id: str) -> WorkspaceFile | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM workspace_files WHERE id = ?",
                (file_id,),
            ).fetchone()
        return self._row_to_workspace_file(row) if row else None

    def list_by_session(self, session_id: str) -> list[WorkspaceFile]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM workspace_files
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_workspace_file(row) for row in rows]

    def update_status(
        self,
        file_id: str,
        *,
        status: str,
        failure_reason: str | None = None,
    ) -> WorkspaceFile | None:
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE workspace_files
                SET status = ?, failure_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    failure_reason,
                    datetime.now(timezone.utc).isoformat(),
                    file_id,
                ),
            )
        return self.get(file_id)

    def update_metadata(
        self,
        file_id: str,
        *,
        source_message_id: str | None,
        file_kind: str,
        display_name: str,
        relative_path: str,
        storage_path: str,
        mime_type: str | None,
        size_bytes: int,
        checksum: str,
        status: str = "ready",
        source_file_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
        failure_reason: str | None = None,
    ) -> WorkspaceFile | None:
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE workspace_files
                SET source_message_id = ?,
                    file_kind = ?,
                    display_name = ?,
                    relative_path = ?,
                    storage_path = ?,
                    mime_type = ?,
                    size_bytes = ?,
                    checksum = ?,
                    status = ?,
                    source_file_ids_json = ?,
                    source_document_ids_json = ?,
                    updated_at = ?,
                    failure_reason = ?
                WHERE id = ?
                """,
                (
                    source_message_id,
                    file_kind,
                    display_name,
                    relative_path,
                    storage_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    status,
                    json.dumps(source_file_ids or [], ensure_ascii=False),
                    json.dumps(source_document_ids or [], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    failure_reason,
                    file_id,
                ),
            )
        return self.get(file_id)

    def delete_record_only(self, file_id: str) -> WorkspaceFile | None:
        workspace_file = self.get(file_id)
        if workspace_file is None:
            return None
        with self.database.connection() as conn:
            conn.execute("DELETE FROM workspace_files WHERE id = ?", (file_id,))
        return workspace_file

    @classmethod
    def _row_to_workspace_file(cls, row: sqlite3.Row) -> WorkspaceFile:
        return WorkspaceFile(
            id=row["id"],
            session_id=row["session_id"],
            source_message_id=row["source_message_id"],
            created_by=row["created_by"],
            file_kind=row["file_kind"],
            display_name=row["display_name"],
            relative_path=row["relative_path"],
            storage_path=row["storage_path"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            checksum=row["checksum"],
            status=row["status"],
            source_file_ids=cls._load_string_list(row["source_file_ids_json"]),
            source_document_ids=cls._load_string_list(row["source_document_ids_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            failure_reason=row["failure_reason"],
        )

    @staticmethod
    def _load_string_list(payload: Any) -> list[str]:
        try:
            value = json.loads(payload or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item]
