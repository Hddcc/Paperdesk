"""SQLite repository for local library metadata."""

from __future__ import annotations

from datetime import datetime
import sqlite3

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
                    created_at,
                    uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            created_at=datetime.fromisoformat(row["created_at"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        )

