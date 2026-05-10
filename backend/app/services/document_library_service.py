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
from app.repositories import LibraryRepository
from app.vectorstores import AbstractVectorStore

from .pdf_parser import PdfParser
from .text_chunker import TextChunker


class DocumentLibraryService:
    """Service for local PDF library operations."""

    def __init__(
        self,
        repository: LibraryRepository,
        vectorstore: AbstractVectorStore,
        upload_dir: Path,
        pdf_parser: PdfParser | None = None,
        text_chunker: TextChunker | None = None,
    ) -> None:
        self.repository = repository
        self.vectorstore = vectorstore
        self.upload_dir = upload_dir
        self.pdf_parser = pdf_parser or PdfParser()
        self.text_chunker = text_chunker or TextChunker()
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

        existing = self.repository.get_by_sha256(sha256)
        if existing is not None:
            if existing.status != "ready":
                self.repository.update_document(existing.id, status="processing")
                self._schedule_processing(existing.id)
                refreshed = self.repository.get_document(existing.id)
                if refreshed is not None:
                    return refreshed
            return existing

        document_id = str(uuid4())
        original_name = Path(upload.filename).name
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
            sha256=sha256,
            page_count=0,
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
        document = self.repository.get_document(document_id)
        if document is None:
            return

        destination = Path(document.file_path)
        original_name = document.display_name or document.filename
        try:
            parsed = self.pdf_parser.parse(destination)
            resolved_title = parsed.title or document.title or original_name
            chunks = self.text_chunker.chunk_document(
                document_id=document.id,
                filename=document.display_name or document.filename,
                title=resolved_title,
                file_path=str(destination),
                pages=parsed.pages,
            )
            indexed_document = self.repository.update_document(
                document.id,
                title=resolved_title,
                page_count=parsed.page_count,
                status="processing",
                file_path=str(destination),
            )
            if indexed_document is None:  # pragma: no cover - defensive guardrail
                raise RuntimeError("Document disappeared during import")
            self.vectorstore.upsert_document(indexed_document)
            self.vectorstore.add_chunks(chunks)
            ready_document = self.repository.update_document(
                document.id,
                title=resolved_title,
                page_count=parsed.page_count,
                status="ready",
            )
            if ready_document is None:  # pragma: no cover - defensive guardrail
                raise RuntimeError("Failed to finalize imported document")
            self.vectorstore.upsert_document(ready_document)
        except Exception as exc:
            try:
                self.vectorstore.delete_document(document.id)
            except Exception:
                pass
            failed_document = self.repository.update_document(
                document.id,
                status="failed",
                page_count=0,
            )
            if failed_document is not None:
                self.vectorstore.upsert_document(failed_document)

    def list_documents(self) -> list[LibraryDocument]:
        return self.repository.list_documents()

    def delete_document(self, document_id: str) -> LibraryDocument:
        document = self.repository.delete_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = Path(document.file_path)
        if file_path.exists():
            self._unlink_with_retry(file_path)
        self.vectorstore.delete_document(document.id)
        return document

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
