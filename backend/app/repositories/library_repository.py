"""SQLite repository for local library metadata."""

from __future__ import annotations

from datetime import datetime
import sqlite3
from typing import Any

from app.models import LibraryDocument

from .base import BaseRepository


class LibraryRepository(BaseRepository):
    """Store and query local PDF metadata."""

    def create_document(self, document: LibraryDocument) -> LibraryDocument:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO library_documents (
                    id,
                    filename,
                    display_name,
                    title,
                    file_path,
                    sha256,
                    page_count,
                    status,
                    parser_status,
                    indexed_at,
                    version,
                    created_at,
                    uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.filename,
                    document.display_name,
                    document.title,
                    document.file_path,
                    document.sha256,
                    document.page_count,
                    document.status,
                    document.parser_status,
                    document.indexed_at.isoformat() if document.indexed_at else None,
                    document.version,
                    document.created_at.isoformat(),
                    document.uploaded_at.isoformat(),
                ),
            )
        return document

    def list_documents(self) -> list[LibraryDocument]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM library_documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> LibraryDocument | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM library_documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_by_sha256(self, sha256: str) -> LibraryDocument | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM library_documents WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_by_display_name(self, display_name: str) -> LibraryDocument | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM library_documents
                WHERE display_name = ?
                ORDER BY uploaded_at DESC
                LIMIT 1
                """,
                (display_name,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def update_document(
        self,
        document_id: str,
        **changes: Any,
    ) -> LibraryDocument | None:
        if not changes:
            return self.get_document(document_id)

        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = [self._serialize_value(value) for value in changes.values()]
        values.append(document_id)

        with self.database.connection() as conn:
            conn.execute(
                f"UPDATE library_documents SET {assignments} WHERE id = ?",
                tuple(values),
            )

        return self.get_document(document_id)

    def delete_document(self, document_id: str) -> LibraryDocument | None:
        document = self.get_document(document_id)
        if document is None:
            return None
        with self.database.connection() as conn:
            conn.execute("DELETE FROM library_documents WHERE id = ?", (document_id,))
        return document

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> LibraryDocument:
        return LibraryDocument(
            id=row["id"],
            filename=row["filename"],
            display_name=row["display_name"],
            title=row["title"],
            file_path=row["file_path"],
            sha256=row["sha256"],
            page_count=row["page_count"],
            status=row["status"],
            parser_status=row["parser_status"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
            version=row["version"] or 1,
            created_at=datetime.fromisoformat(row["created_at"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        )

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value
