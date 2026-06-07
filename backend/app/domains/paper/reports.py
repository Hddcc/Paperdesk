"""Report lifecycle operations that require explicit user intent."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.models import CitationRecord, ResearchReport, ResearchRunStatus, TaskSummary
from app.repositories import ChatRepository, ReportRepository, ResearchRepository


class ReportLifecycleService:
    """Persist saved reports with source metadata and post-read verification."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        research_repository: ResearchRepository,
        report_repository: ReportRepository,
    ) -> None:
        self.chat_repository = chat_repository
        self.research_repository = research_repository
        self.report_repository = report_repository

    def save_from_message(
        self,
        *,
        session_id: str,
        message_id: str,
        optional_title: str | None = None,
    ) -> ResearchReport:
        session = self.chat_repository.get_session(session_id)
        if session is None:
            raise ValueError("Chat session not found")

        message = self.chat_repository.get_message(message_id)
        if message is None or message.session_id != session_id:
            raise ValueError("Chat message not found")
        if message.role != "assistant":
            raise ValueError("Only assistant messages can be saved as reports")

        if message.saved_report_id:
            existing = self.report_repository.get_report(message.saved_report_id)
            if existing is not None:
                return existing

        topic = self._report_topic(optional_title or session.title, message.content)
        now = datetime.now(timezone.utc)
        run_id = f"chat-report-{uuid4().hex}"
        self.research_repository.create_run(run_id, topic)
        self.research_repository.update_run_status(run_id, ResearchRunStatus.COMPLETED)

        citations = list(message.citations)
        evidence_ids = self._evidence_ids_from_citations(citations)
        report = ResearchReport(
            id=str(uuid4()),
            topic=topic,
            markdown=message.content,
            lifecycle_status="saved_report",
            source="knowledge_answer",
            source_message_id=message.id,
            paper_ids=list(message.used_document_ids),
            evidence_ids=evidence_ids,
            task_summaries=[
                TaskSummary(
                    task_id=message.id,
                    title=topic,
                    intent="Saved from a PaperDesk assistant chat response.",
                    summary=message.content,
                    summary_markdown=message.content,
                    evidence_items=[],
                    paper_records=[],
                )
            ],
            citations=citations,
            citation_items=[
                CitationRecord(
                    citation_label=citation,
                    source_type="knowledge_answer",
                    title=citation,
                )
                for citation in citations
            ],
            created_at=now,
            updated_at=now,
        )
        saved = self.report_repository.create_report(report, run_id)
        verified = self.verify_saved_report(
            saved.id,
            expected_title=topic,
            expected_source_message_id=message.id,
            expected_paper_ids=message.used_document_ids,
        )
        self.chat_repository.update_message_report(message.id, verified.id)
        return verified

    def verify_saved_report(
        self,
        report_id: str,
        *,
        expected_title: str,
        expected_source_message_id: str | None = None,
        expected_paper_ids: list[str] | None = None,
    ) -> ResearchReport:
        report = self.report_repository.get_report(report_id)
        if report is None:
            raise RuntimeError("Saved report could not be read back")
        if report.topic != expected_title:
            raise RuntimeError("Saved report title verification failed")
        if not report.markdown.strip():
            raise RuntimeError("Saved report content is empty")
        if expected_source_message_id and report.source_message_id != expected_source_message_id:
            raise RuntimeError("Saved report source message verification failed")
        if expected_paper_ids is not None and set(report.paper_ids) != set(expected_paper_ids):
            raise RuntimeError("Saved report paper association verification failed")
        return report

    @staticmethod
    def _report_topic(title: str, content: str) -> str:
        if title and title != "新对话":
            return title[:80]
        for line in content.splitlines():
            cleaned = line.strip(" #")
            if cleaned:
                return cleaned[:80]
        return "知识库聊天报告"

    @staticmethod
    def _evidence_ids_from_citations(citations: list[str]) -> list[str]:
        return [citation for citation in citations if citation]
