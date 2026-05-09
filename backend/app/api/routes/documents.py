"""Document library routes."""

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.main import get_document_library_service
from app.services import DocumentLibraryService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(
    service: DocumentLibraryService = Depends(get_document_library_service),
) -> list[dict]:
    return [document.model_dump(mode="json") for document in service.list_documents()]


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentLibraryService = Depends(get_document_library_service),
) -> dict:
    document = await service.upload_document(file)
    return document.model_dump(mode="json")


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    service: DocumentLibraryService = Depends(get_document_library_service),
) -> dict:
    document = service.delete_document(document_id)
    return document.model_dump(mode="json")

