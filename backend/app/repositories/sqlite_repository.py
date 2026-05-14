"""Compatibility facade over the split phase-03 repositories."""

from __future__ import annotations

from pathlib import Path

from app.models import DocumentCategory, LibraryDocument, PaperRecord, ReportListItem, ResearchReport, ResearchRun, TodoTask
from app.models.enums import ResearchRunStatus

from .base import SQLiteDatabase
from .category_repository import CategoryRepository
from .chat_repository import ChatRepository
from .chunk_repository import ChunkRepository
from .library_repository import LibraryRepository
from .paper_repository import PaperRepository
from .report_repository import ReportRepository
from .research_repository import ResearchRepository
from .runtime_repository import RuntimeRepository


class SQLiteRepository:
    """Expose focused repositories while preserving legacy call sites."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database = SQLiteDatabase(database_path)
        self.category = CategoryRepository(self.database)
        self.chunk = ChunkRepository(self.database)
        self.library = LibraryRepository(self.database)
        self.research = ResearchRepository(self.database)
        self.paper = PaperRepository(self.database)
        self.report = ReportRepository(self.database)
        self.runtime = RuntimeRepository(self.database)
        self.chat = ChatRepository(self.database)

    def create_document(self, document: LibraryDocument) -> LibraryDocument:
        return self.library.create_document(document)

    def list_documents(self) -> list[LibraryDocument]:
        documents = self.library.list_documents()
        categories_by_document_id = self.category.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        return [
            document.model_copy(update={"categories": categories_by_document_id.get(document.id, [])})
            for document in documents
        ]

    def list_chunks(self, document_ids: list[str] | None = None):
        return self.chunk.list_chunks(document_ids=document_ids)

    def get_document(self, document_id: str) -> LibraryDocument | None:
        document = self.library.get_document(document_id)
        if document is None:
            return None
        return document.model_copy(
            update={"categories": self.category.list_document_categories(document_id)}
        )

    def get_document_by_sha256(self, sha256: str) -> LibraryDocument | None:
        return self.library.get_by_sha256(sha256)

    def update_document(self, document_id: str, **changes) -> LibraryDocument | None:
        return self.library.update_document(document_id, **changes)

    def delete_document(self, document_id: str) -> LibraryDocument | None:
        return self.library.delete_document(document_id)

    def list_categories(self) -> list[DocumentCategory]:
        return self.category.list_categories()

    def create_category(self, name: str, color: str | None = None) -> DocumentCategory:
        return self.category.create_category(name=name, color=color)

    def update_category(
        self,
        category_id: str,
        *,
        name: str | None = None,
        color: str | None = None,
    ) -> DocumentCategory | None:
        return self.category.update_category(category_id, name=name, color=color)

    def delete_category(self, category_id: str) -> DocumentCategory | None:
        return self.category.delete_category(category_id)

    def replace_document_categories(
        self,
        document_id: str,
        category_ids: list[str],
    ) -> LibraryDocument | None:
        categories = self.category.replace_document_categories(document_id, category_ids)
        if categories is None:
            return None
        document = self.library.get_document(document_id)
        if document is None:
            return None
        return document.model_copy(update={"categories": categories})

    def create_run(self, run_id: str, topic: str) -> ResearchRun:
        return self.research.create_run(run_id, topic)

    def update_run_status(self, run_id: str, status: ResearchRunStatus) -> None:
        self.research.update_run_status(run_id, status)

    def get_run(self, run_id: str) -> ResearchRun | None:
        return self.research.get_run(run_id)

    def save_todo_tasks(self, run_id: str, tasks: list[TodoTask]) -> None:
        self.research.save_todo_tasks(run_id, tasks)

    def update_task(self, run_id: str, task: TodoTask) -> None:
        self.research.update_task(run_id, task)

    def list_tasks(self, run_id: str) -> list[TodoTask]:
        return self.research.list_tasks(run_id)

    def save_task_papers(self, task_id: str, records: list[PaperRecord]) -> None:
        self.paper.save_task_papers(task_id, records)

    def list_task_papers(self, task_id: str) -> list[PaperRecord]:
        return self.paper.list_task_papers(task_id)

    def create_report(self, report: ResearchReport, run_id: str) -> ResearchReport:
        return self.report.create_report(report, run_id)

    def list_reports(self) -> list[ReportListItem]:
        return self.report.list_reports()

    def get_report(self, report_id: str) -> ResearchReport | None:
        return self.report.get_report(report_id)

    def get_report_by_run_id(self, run_id: str) -> ResearchReport | None:
        return self.report.get_report_by_run_id(run_id)

    def delete_report(self, report_id: str) -> ResearchReport | None:
        return self.report.delete_report(report_id)
