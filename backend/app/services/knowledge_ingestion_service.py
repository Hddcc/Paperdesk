"""Offline ingestion pipeline for local PDF knowledge indexing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.models import LibraryDocument
from app.repositories import ChunkRepository, LibraryRepository
from app.vectorstores import AbstractVectorStore

from .pdf_parser import PdfParser
from .text_chunker import TextChunker


class KnowledgeIngestionService:
    """Parse, chunk, persist, and index uploaded library documents."""

    def __init__(
        self,
        *,
        document_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        vectorstore: AbstractVectorStore,
        pdf_parser: PdfParser | None = None,
        text_chunker: TextChunker | None = None,
    ) -> None:
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.vectorstore = vectorstore
        self.pdf_parser = pdf_parser or PdfParser()
        self.text_chunker = text_chunker or TextChunker()

    def ingest_document(self, document_id: str) -> None:
        document = self.document_repository.get_document(document_id)
        if document is None:
            return

        destination = Path(document.file_path)
        try:
            self.document_repository.update_document(
                document.id,
                status="processing",
                parser_status="processing",
                indexed_at=None,
            )
            parsed = self.pdf_parser.parse(destination)
            resolved_title = parsed.title or document.title or document.display_name or document.filename
            working_document = self.document_repository.update_document(
                document.id,
                title=resolved_title,
                page_count=parsed.page_count,
                parser_status="parsed",
                file_path=str(destination),
            )
            if working_document is None:  # pragma: no cover - defensive guardrail
                raise RuntimeError("Document disappeared during parsing")

            chunks = self.text_chunker.chunk_document(
                document=working_document,
                pages=parsed.pages,
            )
            if not chunks:
                raise RuntimeError("PDF import produced no usable text chunks")

            self.vectorstore.delete_document(document.id)
            self.chunk_repository.delete_document_chunks(document.id)
            self.vectorstore.upsert_document(working_document)
            self.vectorstore.add_chunks(chunks)
            self.chunk_repository.replace_document_chunks(document.id, chunks)

            ready_document = self.document_repository.update_document(
                document.id,
                title=resolved_title,
                page_count=parsed.page_count,
                status="ready",
                parser_status="indexed",
                indexed_at=datetime.now(timezone.utc),
            )
            if ready_document is None:  # pragma: no cover - defensive guardrail
                raise RuntimeError("Failed to finalize imported document")
            self.vectorstore.upsert_document(ready_document)
        except Exception:
            self.chunk_repository.delete_document_chunks(document.id)
            try:
                self.vectorstore.delete_document(document.id)
            except Exception:
                pass
            failed_document = self.document_repository.update_document(
                document.id,
                status="failed",
                parser_status="failed",
                page_count=0,
                indexed_at=None,
            )
            if failed_document is not None:
                self.vectorstore.upsert_document(failed_document)
            raise
