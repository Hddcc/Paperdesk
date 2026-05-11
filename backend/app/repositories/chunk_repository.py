"""SQLite repository for persisted library chunk metadata."""

from __future__ import annotations

import json
from typing import Any

from app.models import ChunkRecord

from .base import BaseRepository


class ChunkRepository(BaseRepository):
    """Store and query chunk-level metadata for local library documents."""

    def replace_document_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        with self.database.connection() as conn:
            conn.execute("DELETE FROM library_chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO library_chunks (
                        id,
                        document_id,
                        source,
                        page_number,
                        chunk_index,
                        section,
                        title,
                        sha256,
                        version,
                        text,
                        token_estimate,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id or chunk.id,
                        chunk.document_id,
                        chunk.source or "",
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.section,
                        chunk.title,
                        chunk.sha256,
                        chunk.version,
                        chunk.content or chunk.text,
                        chunk.token_estimate,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    ),
                )

    def list_chunks(
        self,
        *,
        document_ids: list[str] | None = None,
    ) -> list[ChunkRecord]:
        query = "SELECT * FROM library_chunks"
        values: list[Any] = []
        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            query += f" WHERE document_id IN ({placeholders})"
            values.extend(document_ids)
        query += " ORDER BY document_id ASC, page_number ASC, chunk_index ASC"

        with self.database.connection() as conn:
            rows = conn.execute(query, tuple(values)).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def delete_document_chunks(self, document_id: str) -> None:
        with self.database.connection() as conn:
            conn.execute("DELETE FROM library_chunks WHERE document_id = ?", (document_id,))

    @staticmethod
    def _row_to_chunk(row) -> ChunkRecord:
        metadata_json = row["metadata_json"] or "{}"
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}

        return ChunkRecord(
            id=row["id"],
            chunk_id=row["id"],
            document_id=row["document_id"],
            source=row["source"] or None,
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            section=row["section"] or None,
            title=row["title"] or None,
            sha256=row["sha256"] or None,
            version=row["version"] or 1,
            text=row["text"],
            content=row["text"],
            token_estimate=row["token_estimate"] or 0,
            metadata=metadata,
        )
