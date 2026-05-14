"""Document library routes."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.main import get_category_repository, get_document_library_service, get_library_repository
from app.models import (
    DocumentCategoryAssignmentRequest,
    DocumentCategoryCreateRequest,
    DocumentCategoryUpdateRequest,
)
from app.repositories import CategoryRepository, LibraryRepository
from app.services import DocumentLibraryService

router = APIRouter(prefix="/documents", tags=["documents"])
category_router = APIRouter(prefix="/document-categories", tags=["document-categories"])


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


@router.put("/{document_id}/categories")
def assign_document_categories(
    document_id: str,
    payload: DocumentCategoryAssignmentRequest,
    library_repository: LibraryRepository = Depends(get_library_repository),
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> dict:
    categories = category_repository.replace_document_categories(document_id, payload.category_ids)
    if categories is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document = library_repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document.model_copy(update={"categories": categories}).model_dump(mode="json")


@category_router.get("")
def list_categories(
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> list[dict]:
    return [category.model_dump(mode="json") for category in category_repository.list_categories()]


@category_router.post("")
def create_category(
    payload: DocumentCategoryCreateRequest,
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> dict:
    try:
        category = category_repository.create_category(payload.name, payload.color)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return category.model_dump(mode="json")


@category_router.patch("/{category_id}")
def update_category(
    category_id: str,
    payload: DocumentCategoryUpdateRequest,
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> dict:
    try:
        category = category_repository.update_category(
            category_id,
            name=payload.name,
            color=payload.color,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return category.model_dump(mode="json")


@category_router.delete("/{category_id}")
def delete_category(
    category_id: str,
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> dict:
    category = category_repository.delete_category(category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return category.model_dump(mode="json")
