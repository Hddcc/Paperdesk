"""Manage uploaded PDF metadata and storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.models import LibraryDocument
from app.repositories import SQLiteRepository
from app.vectorstores import AbstractVectorStore


class DocumentLibraryService:
    """Service for local PDF library operations."""

    def __init__(
        self,
        repository: SQLiteRepository,
        vectorstore: AbstractVectorStore,
        upload_dir: Path,
    ) -> None:
        self.repository = repository
        self.vectorstore = vectorstore
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def upload_document(self, upload: UploadFile) -> LibraryDocument:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

        document_id = str(uuid4())
        safe_filename = f"{document_id}.pdf"
        destination = self.upload_dir / safe_filename
        content = await upload.read()
        destination.write_bytes(content)

        document = LibraryDocument(
            id=document_id,
            filename=safe_filename,
            display_name=upload.filename,
            file_path=str(destination),
            status="uploaded",
            uploaded_at=datetime.now(timezone.utc),
        )
        self.repository.create_document(document)
        self.vectorstore.upsert_document(document)
        return document

    def list_documents(self) -> list[LibraryDocument]:
        return self.repository.list_documents()

    def delete_document(self, document_id: str) -> LibraryDocument:
        document = self.repository.delete_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
        marker_base_path = getattr(self.vectorstore, "base_path", None)
        if marker_base_path is not None:
            marker_path = Path(marker_base_path) / f"{document.id}.txt"
            if marker_path.exists():
                marker_path.unlink()
        return document
