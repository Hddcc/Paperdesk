"""Paper document application boundary."""

from __future__ import annotations

from fastapi import UploadFile

from app.domains.paper import DocumentLibraryService


class PaperUploadUseCase:
    """Use case for paper library upload and listing flows."""

    def __init__(self, document_library_service: DocumentLibraryService) -> None:
        self.document_library_service = document_library_service

    def list_documents(self):
        return self.document_library_service.list_documents()

    async def upload_document(self, file: UploadFile):
        return await self.document_library_service.upload_document(file)

    def delete_document(self, document_id: str):
        return self.document_library_service.delete_document(document_id)
