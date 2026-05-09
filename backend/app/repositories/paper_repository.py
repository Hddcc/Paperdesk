"""SQLite repository for normalized online paper search results."""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3

from app.models import PaperRecord

from .base import BaseRepository


class PaperRepository(BaseRepository):
    """Store online paper results per task."""

    def save_task_papers(self, task_id: str, records: list[PaperRecord]) -> None:
        with self.database.connection() as conn:
            conn.execute("DELETE FROM paper_records WHERE task_id = ?", (task_id,))
            for record in records:
                conn.execute(
                    """
                    INSERT INTO paper_records (
                        task_id,
                        paper_id,
                        title,
                        authors_json,
                        abstract,
                        year,
                        venue,
                        doi,
                        url,
                        source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        record.paper_id,
                        record.title,
                        json.dumps(record.authors, ensure_ascii=False),
                        record.abstract,
                        record.year,
                        record.venue,
                        record.doi,
                        record.url,
                        record.source,
                    ),
                )

    def list_task_papers(self, task_id: str) -> list[PaperRecord]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM paper_records
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [self._row_to_paper(row) for row in rows]

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            abstract=row["abstract"],
            year=row["year"],
            venue=row["venue"],
            doi=row["doi"],
            url=row["url"],
            source=row["source"],
        )

