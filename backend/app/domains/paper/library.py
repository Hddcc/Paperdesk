"""Manage uploaded PDF metadata and storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from datetime import datetime, timezone
import hashlib
from threading import Lock
from pathlib import Path
import time
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.models import LibraryDocument
from app.repositories import CategoryRepository, LibraryRepository
from app.vectorstores import AbstractVectorStore

from .ingestion import KnowledgeIngestionService


class DocumentLibraryService:
    """Service for local PDF library operations."""

    def __init__(
        self,
        repository: LibraryRepository,
        vectorstore: AbstractVectorStore,
        upload_dir: Path,
        ingestion_service: KnowledgeIngestionService,
        category_repository: CategoryRepository | None = None,
    ) -> None:
        self.repository = repository
        self.category_repository = category_repository
        self.vectorstore = vectorstore
        self.upload_dir = upload_dir
        self.ingestion_service = ingestion_service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paperdesk-import")
        self._scheduled_document_ids: set[str] = set()
        self._schedule_lock = Lock()
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def upload_document(self, upload: UploadFile) -> LibraryDocument:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

        content = await upload.read()
        await upload.close()
        sha256 = hashlib.sha256(content).hexdigest()
        original_name = Path(upload.filename).name

        existing = self.repository.get_by_sha256(sha256)
        if existing is not None:
            if existing.status != "ready":
                self.repository.update_document(
                    existing.id,
                    status="processing",
                    parser_status="pending",
                    failure_reason=None,
                    indexed_at=None,
                )
                self._schedule_processing(existing.id)
                refreshed = self.repository.get_document(existing.id)
                if refreshed is not None:
                    return refreshed
            return existing

        existing_named_document = self.repository.get_by_display_name(original_name)
        if existing_named_document is not None and existing_named_document.sha256 != sha256:
            destination = self.upload_dir / f"{existing_named_document.id}_{original_name}"
            destination.write_bytes(content)
            now = datetime.now(timezone.utc)
            updated = self.repository.update_document(
                existing_named_document.id,
                filename=destination.name,
                display_name=original_name,
                title=Path(original_name).stem,
                file_path=str(destination),
                sha256=sha256,
                page_count=0,
                status="processing",
                parser_status="pending",
                failure_reason=None,
                indexed_at=None,
                version=max(existing_named_document.version, 1) + 1,
                uploaded_at=now,
            )
            if updated is not None:
                self._schedule_processing(updated.id)
                return updated

        document_id = str(uuid4())
        safe_filename = f"{document_id}_{original_name}"
        destination = self.upload_dir / safe_filename
        destination.write_bytes(content)
        now = datetime.now(timezone.utc)

        document = LibraryDocument(
            id=document_id,
            filename=safe_filename,
            display_name=original_name,
            title=Path(original_name).stem,
            file_path=str(destination),
            status="processing",
            parser_status="pending",
            failure_reason=None,
            sha256=sha256,
            page_count=0,
            indexed_at=None,
            version=1,
            created_at=now,
            uploaded_at=now,
        )
        self.repository.create_document(document)
        self._schedule_processing(document.id)
        return document

    def _schedule_processing(self, document_id: str) -> None:
        with self._schedule_lock:
            if document_id in self._scheduled_document_ids:
                return
            self._scheduled_document_ids.add(document_id)
        future = self._executor.submit(self._process_document, document_id)
        future.add_done_callback(lambda _: self._mark_processing_complete(document_id))

    def _mark_processing_complete(self, document_id: str) -> None:
        with self._schedule_lock:
            self._scheduled_document_ids.discard(document_id)

    def _process_document(self, document_id: str) -> None:
        try:
            self.ingestion_service.ingest_document(document_id)
        except Exception:
            return

    def list_documents(self) -> list[LibraryDocument]:
        self._recover_processing_documents()
        documents = self.repository.list_documents()
        if self.category_repository is None:
            return documents
        categories_by_document_id = self.category_repository.list_categories_by_document_ids(
            [document.id for document in documents]
        )
        return [
            document.model_copy(update={"categories": categories_by_document_id.get(document.id, [])})
            for document in documents
        ]

    def delete_document(self, document_id: str) -> LibraryDocument:
        document = self.repository.delete_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = Path(document.file_path)
        if file_path.exists():
            self._unlink_with_retry(file_path)
        try:
            self.vectorstore.delete_document(document.id)
        except Exception:
            pass
        return document

    def _recover_processing_documents(self) -> None:
        for document in self.repository.list_documents():
            if document.status != "processing":
                continue
            if not Path(document.file_path).exists():
                self.repository.update_document(
                    document.id,
                    status="failed",
                    parser_status="failed",
                    failure_reason="文档文件不存在，已无法继续处理。请删除后重新上传。",
                    indexed_at=None,
                )
                continue
            self._schedule_processing(document.id)

    @staticmethod
    def _unlink_with_retry(file_path: Path) -> bool:
        for _ in range(3):
            try:
                file_path.unlink()
                return True
            except PermissionError:
                gc.collect()
                time.sleep(0.05)
        return False
