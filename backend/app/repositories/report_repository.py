"""SQLite repository for final reports and citations."""

from __future__ import annotations

from datetime import datetime
import json
import re
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
                    created_at,
                    lifecycle_status,
                    source,
                    source_message_id,
                    paper_ids_json,
                    category_ids_json,
                    evidence_ids_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    report.lifecycle_status,
                    report.source,
                    report.source_message_id,
                    json.dumps(report.paper_ids, ensure_ascii=False),
                    json.dumps(report.category_ids, ensure_ascii=False),
                    json.dumps(report.evidence_ids, ensure_ascii=False),
                    (report.updated_at or report.created_at).isoformat(),
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

        return self._build_report(report_row, citation_rows)

    def get_report_by_run_id(self, run_id: str) -> ResearchReport | None:
        with self.database.connection() as conn:
            report_row = conn.execute(
                """
                SELECT * FROM report_records
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
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
                (report_row["id"],),
            ).fetchall()

        return self._build_report(report_row, citation_rows)

    def delete_report(self, report_id: str) -> ResearchReport | None:
        report = self.get_report(report_id)
        if report is None:
            return None

        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deleted_report_records (report_id, deleted_at)
                VALUES (?, ?)
                """,
                (report_id, datetime.now().astimezone().isoformat()),
            )
            conn.execute("DELETE FROM citation_records WHERE report_id = ?", (report_id,))
            conn.execute("DELETE FROM report_records WHERE id = ?", (report_id,))
            legacy_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reports'"
            ).fetchone()
            if legacy_table is not None:
                conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))

        return report

    def _build_report(
        self,
        report_row: sqlite3.Row,
        citation_rows: list[sqlite3.Row],
    ) -> ResearchReport:
        sanitized_markdown = self._sanitize_user_facing_markdown(report_row["markdown"])
        normalized_markdown = self._normalize_reference_section(sanitized_markdown)
        return ResearchReport(
            id=report_row["id"],
            topic=report_row["topic"],
            markdown=normalized_markdown,
            task_summaries=self._load_task_summaries(report_row["task_summaries_json"]),
            citations=[line for line in report_row["citations_text"].splitlines() if line],
            citation_items=[CitationRecord(**dict(row)) for row in citation_rows],
            created_at=datetime.fromisoformat(report_row["created_at"]),
            lifecycle_status=report_row["lifecycle_status"] if "lifecycle_status" in report_row.keys() else "saved_report",
            source=report_row["source"] if "source" in report_row.keys() else "research_task",
            source_message_id=(
                report_row["source_message_id"] if "source_message_id" in report_row.keys() else None
            ),
            paper_ids=self._load_string_list(
                report_row["paper_ids_json"] if "paper_ids_json" in report_row.keys() else "[]"
            ),
            category_ids=self._load_string_list(
                report_row["category_ids_json"] if "category_ids_json" in report_row.keys() else "[]"
            ),
            evidence_ids=self._load_string_list(
                report_row["evidence_ids_json"] if "evidence_ids_json" in report_row.keys() else "[]"
            ),
            updated_at=datetime.fromisoformat(
                report_row["updated_at"] if "updated_at" in report_row.keys() and report_row["updated_at"] else report_row["created_at"]
            ),
        )

    @staticmethod
    def _load_task_summaries(task_summaries_json: str) -> list[TaskSummary]:
        payload = json.loads(task_summaries_json)
        return [
            TaskSummary(
                task_id=item["task_id"],
                title=item["title"],
                intent=item["intent"],
                summary=ReportRepository._sanitize_user_facing_markdown(
                    item.get("summary") or item.get("summary_markdown") or ""
                ),
                summary_markdown=ReportRepository._sanitize_user_facing_markdown(
                    item.get("summary_markdown") or item.get("summary") or ""
                ),
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

    @staticmethod
    def _load_string_list(payload: str) -> list[str]:
        try:
            value = json.loads(payload or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item]

    @staticmethod
    def _sanitize_user_facing_markdown(markdown: str) -> str:
        cleaned = re.sub(
            r"(?m)^\s*[-*]\s*说明：.*(?:当前实现已接入在线论文检索|规则模板|SQLite|SSE).*$\n?",
            "",
            markdown,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _normalize_reference_section(markdown: str) -> str:
        parts = re.split(r"(?m)^##\s*参考来源\s*$", markdown, maxsplit=1)
        if len(parts) != 2:
            return markdown

        body, references = parts
        normalized_lines: list[str] = []
        for raw_line in references.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("- "):
                normalized_lines.append(line)
            elif re.match(r"^\[\d+\]\s+", line):
                normalized_lines.append(f"- {line}")
            else:
                normalized_lines.append(line)

        if not normalized_lines:
            normalized_lines.append("- 暂无可导出的参考来源。")

        return "\n".join(
            [
                body.rstrip(),
                "",
                "## 参考来源",
                "",
                *normalized_lines,
            ]
        ).strip()
