"""SQLite repository implementation for the runnable skeleton."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
import json

from app.models import (
    EvidenceItem,
    LibraryDocument,
    PaperRecord,
    ReportListItem,
    ResearchReport,
    ResearchRun,
    TaskSummary,
    TodoTask,
)
from app.models.enums import ResearchRunStatus


def utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class SQLiteRepository:
    """Store research runs, tasks, documents, and reports in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        """Create tables if they do not already exist."""
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS todo_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    task_summaries_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );
                """
            )

    def create_document(self, document: LibraryDocument) -> LibraryDocument:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, display_name, file_path, status, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.filename,
                    document.display_name,
                    document.file_path,
                    document.status,
                    document.uploaded_at.isoformat(),
                ),
            )
        return document

    def list_documents(self) -> list[LibraryDocument]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> LibraryDocument | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def delete_document(self, document_id: str) -> LibraryDocument | None:
        document = self.get_document(document_id)
        if document is None:
            return None
        with self.connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return document

    def create_run(self, run_id: str, topic: str) -> ResearchRun:
        timestamp = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO research_runs (id, topic, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    topic,
                    ResearchRunStatus.CREATED.value,
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
        return ResearchRun(
            id=run_id,
            topic=topic,
            status=ResearchRunStatus.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def update_run_status(self, run_id: str, status: ResearchRunStatus) -> None:
        timestamp = utc_now().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE research_runs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, timestamp, run_id),
            )

    def save_todo_tasks(self, run_id: str, tasks: list[TodoTask]) -> None:
        with self.connection() as conn:
            for index, task in enumerate(tasks):
                conn.execute(
                    """
                    INSERT INTO todo_tasks (id, run_id, task_index, title, intent, query_text, status, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.id,
                        run_id,
                        index,
                        task.title,
                        task.intent,
                        task.query,
                        task.status.value,
                        task.summary,
                    ),
                )

    def update_task(self, run_id: str, task: TodoTask) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE todo_tasks
                SET status = ?, summary = ?
                WHERE run_id = ? AND id = ?
                """,
                (
                    task.status.value,
                    task.summary,
                    run_id,
                    task.id,
                ),
            )

    def list_tasks(self, run_id: str) -> list[TodoTask]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM todo_tasks
                WHERE run_id = ?
                ORDER BY task_index ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def create_report(self, report: ResearchReport, run_id: str) -> ResearchReport:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO reports (id, run_id, topic, markdown, citations, task_summaries_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    run_id,
                    report.topic,
                    report.markdown,
                    "\n".join(report.citations),
                    json.dumps([summary.model_dump(mode="json") for summary in report.task_summaries], ensure_ascii=False),
                    report.created_at.isoformat(),
                ),
            )
        return report

    def list_reports(self) -> list[ReportListItem]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, topic, created_at FROM reports ORDER BY created_at DESC"
            ).fetchall()
        return [
            ReportListItem(
                id=row["id"],
                topic=row["topic"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_report(self, report_id: str) -> ResearchReport | None:
        with self.connection() as conn:
            report_row = conn.execute(
                "SELECT * FROM reports WHERE id = ?",
                (report_id,),
            ).fetchone()
            if report_row is None:
                return None
        return ResearchReport(
            id=report_row["id"],
            topic=report_row["topic"],
            markdown=report_row["markdown"],
            task_summaries=[
                TaskSummary(
                    task_id=item["task_id"],
                    title=item["title"],
                    intent=item["intent"],
                    summary=item["summary"],
                    evidence_items=[
                        EvidenceItem(**evidence)
                        for evidence in item.get("evidence_items", [])
                    ],
                    paper_records=[
                        PaperRecord(**paper)
                        for paper in item.get("paper_records", [])
                    ],
                )
                for item in json.loads(report_row["task_summaries_json"])
            ],
            citations=[line for line in report_row["citations"].splitlines() if line],
            created_at=datetime.fromisoformat(report_row["created_at"]),
        )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> LibraryDocument:
        return LibraryDocument(
            id=row["id"],
            filename=row["filename"],
            display_name=row["display_name"],
            file_path=row["file_path"],
            status=row["status"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TodoTask:
        return TodoTask(
            id=row["id"],
            title=row["title"],
            intent=row["intent"],
            query=row["query_text"],
            status=ResearchRunStatus(row["status"]),
            summary=row["summary"],
        )
