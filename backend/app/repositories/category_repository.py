"""SQLite repository for document categories."""

from __future__ import annotations

from datetime import datetime
import sqlite3
from uuid import uuid4

from app.models import DocumentCategory

from .base import BaseRepository, utc_now


class CategoryRepository(BaseRepository):
    """Store category metadata and document/category links."""

    def list_categories(self) -> list[DocumentCategory]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_categories ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_category(row) for row in rows]

    def create_category(self, name: str, color: str | None = None) -> DocumentCategory:
        now = utc_now()
        category = DocumentCategory(
            id=uuid4().hex,
            name=self._normalize_name(name),
            color=self._normalize_color(color),
            created_at=now,
            updated_at=now,
        )
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO document_categories (id, name, color, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    category.id,
                    category.name,
                    category.color,
                    category.created_at.isoformat(),
                    category.updated_at.isoformat(),
                ),
            )
        return category

    def update_category(
        self,
        category_id: str,
        *,
        name: str | None = None,
        color: str | None = None,
    ) -> DocumentCategory | None:
        category = self.get_category(category_id)
        if category is None:
            return None

        next_name = self._normalize_name(name) if name is not None else category.name
        next_color = self._normalize_color(color) if color is not None else category.color
        updated_at = utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE document_categories
                SET name = ?, color = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_name, next_color, updated_at.isoformat(), category_id),
            )
        return self.get_category(category_id)

    def delete_category(self, category_id: str) -> DocumentCategory | None:
        category = self.get_category(category_id)
        if category is None:
            return None
        with self.database.connection() as conn:
            conn.execute("DELETE FROM document_categories WHERE id = ?", (category_id,))
        return category

    def get_category(self, category_id: str) -> DocumentCategory | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_categories WHERE id = ?",
                (category_id,),
            ).fetchone()
        return self._row_to_category(row) if row else None

    def list_document_categories(self, document_id: str) -> list[DocumentCategory]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT category.*
                FROM document_categories category
                JOIN document_category_links link ON link.category_id = category.id
                WHERE link.document_id = ?
                ORDER BY category.created_at ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_category(row) for row in rows]

    def list_categories_by_document_ids(self, document_ids: list[str]) -> dict[str, list[DocumentCategory]]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        with self.database.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT link.document_id, category.*
                FROM document_category_links link
                JOIN document_categories category ON category.id = link.category_id
                WHERE link.document_id IN ({placeholders})
                ORDER BY category.created_at ASC
                """,
                tuple(document_ids),
            ).fetchall()

        grouped: dict[str, list[DocumentCategory]] = {document_id: [] for document_id in document_ids}
        for row in rows:
            grouped[str(row["document_id"])].append(self._row_to_category(row))
        return grouped

    def replace_document_categories(
        self,
        document_id: str,
        category_ids: list[str],
    ) -> list[DocumentCategory] | None:
        unique_category_ids = list(dict.fromkeys(category_ids))
        with self.database.connection() as conn:
            document_exists = conn.execute(
                "SELECT 1 FROM library_documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if document_exists is None:
                return None

            if unique_category_ids:
                placeholders = ",".join("?" for _ in unique_category_ids)
                rows = conn.execute(
                    f"SELECT id FROM document_categories WHERE id IN ({placeholders})",
                    tuple(unique_category_ids),
                ).fetchall()
                existing_ids = {str(row["id"]) for row in rows}
                unique_category_ids = [
                    category_id for category_id in unique_category_ids if category_id in existing_ids
                ]

            conn.execute(
                "DELETE FROM document_category_links WHERE document_id = ?",
                (document_id,),
            )
            now = utc_now().isoformat()
            for category_id in unique_category_ids:
                conn.execute(
                    """
                    INSERT INTO document_category_links (category_id, document_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (category_id, document_id, now),
                )
        return self.list_document_categories(document_id)

    @staticmethod
    def _row_to_category(row: sqlite3.Row) -> DocumentCategory:
        return DocumentCategory(
            id=row["id"],
            name=row["name"],
            color=row["color"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Category name cannot be empty")
        return normalized[:40]

    @staticmethod
    def _normalize_color(color: str | None) -> str | None:
        if color is None:
            return None
        normalized = color.strip()
        return normalized[:32] or None
