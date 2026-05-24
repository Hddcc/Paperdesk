"""SQLite repository for session-scoped user file assets."""

from __future__ import annotations

from datetime import datetime
import sqlite3
from typing import Any

from app.models import FileAsset

from .base import BaseRepository


class FileAssetRepository(BaseRepository):
    """Store user workspace files outside the paper library."""

    def create(self, asset: FileAsset) -> FileAsset:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO file_assets (
                    id,
                    filename,
                    display_name,
                    mime_type,
                    extension,
                    size_bytes,
                    sha256,
                    storage_path,
                    source,
                    scope,
                    session_id,
                    kind,
                    status,
                    text_extract_status,
                    preview_text,
                    text_char_count,
                    failure_reason,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.id,
                    asset.filename,
                    asset.display_name,
                    asset.mime_type,
                    asset.extension,
                    asset.size_bytes,
                    asset.sha256,
                    asset.storage_path,
                    asset.source,
                    asset.scope,
                    asset.session_id,
                    asset.kind,
                    asset.status,
                    asset.text_extract_status,
                    asset.preview_text,
                    asset.text_char_count,
                    asset.failure_reason,
                    asset.created_at.isoformat(),
                ),
            )
        return asset

    def get(self, file_id: str) -> FileAsset | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM file_assets WHERE id = ?",
                (file_id,),
            ).fetchone()
        return self._row_to_asset(row) if row else None

    def list_by_session(self, session_id: str) -> list[FileAsset]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM file_assets
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_asset(row) for row in rows]

    def update_status(
        self,
        file_id: str,
        **changes: Any,
    ) -> FileAsset | None:
        if not changes:
            return self.get(file_id)

        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = [self._serialize_value(value) for value in changes.values()]
        values.append(file_id)

        with self.database.connection() as conn:
            conn.execute(
                f"UPDATE file_assets SET {assignments} WHERE id = ?",
                tuple(values),
            )
        return self.get(file_id)

    def delete(self, file_id: str) -> FileAsset | None:
        asset = self.get(file_id)
        if asset is None:
            return None
        with self.database.connection() as conn:
            conn.execute("DELETE FROM file_assets WHERE id = ?", (file_id,))
        return asset

    @staticmethod
    def _row_to_asset(row: sqlite3.Row) -> FileAsset:
        return FileAsset(
            id=row["id"],
            filename=row["filename"],
            display_name=row["display_name"],
            mime_type=row["mime_type"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            storage_path=row["storage_path"],
            source=row["source"],
            scope=row["scope"],
            session_id=row["session_id"],
            kind=row["kind"],
            status=row["status"],
            text_extract_status=row["text_extract_status"],
            preview_text=row["preview_text"],
            text_char_count=row["text_char_count"] or 0,
            failure_reason=row["failure_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value
