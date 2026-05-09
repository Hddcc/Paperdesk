"""SQLite repository for final reports and citations."""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3

from app.models import CitationRecord, EvidenceItem, PaperRecord, ReportListItem, ResearchReport, TaskSummary

from .base import BaseRepository


class ReportRepository(BaseRepository):
    """Store final reports and structured citation entries."""

    def create_report(self, report: ResearchReport, run_id: str) -> ResearchReport:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO report_records (
                    id,
                    run_id,
                    topic,
                    markdown,
                    citations_text,
                    task_summaries_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    run_id,
                    report.topic,
                    report.markdown,
                    "\n".join(report.citations),
                    json.dumps(
                        [summary.model_dump(mode="json") for summary in report.task_summaries],
                        ensure_ascii=False,
                    ),
                    report.created_at.isoformat(),
                ),
            )
            for index, citation in enumerate(report.citation_items):
                conn.execute(
                    """
                    INSERT INTO citation_records (
                        report_id,
                        citation_label,
                        source_type,
                        title,
                        url,
                        doi,
                        document_id,
                        page_number,
                        sort_order
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report.id,
                        citation.citation_label,
                        citation.source_type,
                        citation.title,
                        citation.url,
                        citation.doi,
                        citation.document_id,
                        citation.page_number,
                        index,
                    ),
                )
        return report

    def list_reports(self) -> list[ReportListItem]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT id, topic, created_at FROM report_records ORDER BY created_at DESC"
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
        with self.database.connection() as conn:
            report_row = conn.execute(
                "SELECT * FROM report_records WHERE id = ?",
                (report_id,),
            ).fetchone()
            if report_row is None:
                return None
            citation_rows = conn.execute(
                """
                SELECT citation_label, source_type, title, url, doi, document_id, page_number
                FROM citation_records
                WHERE report_id = ?
                ORDER BY sort_order ASC, id ASC
                """,
                (report_id,),
            ).fetchall()

        return ResearchReport(
            id=report_row["id"],
            topic=report_row["topic"],
            markdown=report_row["markdown"],
            task_summaries=self._load_task_summaries(report_row["task_summaries_json"]),
            citations=[line for line in report_row["citations_text"].splitlines() if line],
            citation_items=[CitationRecord(**dict(row)) for row in citation_rows],
            created_at=datetime.fromisoformat(report_row["created_at"]),
        )

    @staticmethod
    def _load_task_summaries(task_summaries_json: str) -> list[TaskSummary]:
        payload = json.loads(task_summaries_json)
        return [
            TaskSummary(
                task_id=item["task_id"],
                title=item["title"],
                intent=item["intent"],
                summary=item.get("summary") or item.get("summary_markdown") or "",
                summary_markdown=item.get("summary_markdown") or item.get("summary"),
                evidence_items=[
                    EvidenceItem(**evidence)
                    for evidence in item.get("evidence_items", [])
                ],
                paper_records=[
                    PaperRecord(**paper)
                    for paper in item.get("paper_records", [])
                ],
            )
            for item in payload
        ]

