"""Document library routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.main import get_category_repository, get_library_repository, get_paper_upload_use_case
from app.application import PaperUploadUseCase
from app.models import (
    DocumentCategoryAssignmentRequest,
    DocumentCategoryCreateRequest,
    DocumentCategoryUpdateRequest,
)
from app.repositories import CategoryRepository, LibraryRepository

router = APIRouter(prefix="/documents", tags=["documents"])
category_router = APIRouter(prefix="/document-categories", tags=["document-categories"])


@router.get("")
def list_documents(
    use_case: PaperUploadUseCase = Depends(get_paper_upload_use_case),
) -> list[dict]:
    return [document.model_dump(mode="json") for document in use_case.list_documents()]


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_case: PaperUploadUseCase = Depends(get_paper_upload_use_case),
) -> dict:
    document = await use_case.upload_document(file)
    return document.model_dump(mode="json")


@router.get("/{document_id}/file")
def get_document_file(
    document_id: str,
    library_repository: LibraryRepository = Depends(get_library_repository),
) -> FileResponse:
    document = library_repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(document.file_path)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Document file not found")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=document.display_name,
        content_disposition_type="inline",
    )


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    use_case: PaperUploadUseCase = Depends(get_paper_upload_use_case),
) -> dict:
    document = use_case.delete_document(document_id)
    return document.model_dump(mode="json")


@router.put("/{document_id}/categories")
def assign_document_categories(
    document_id: str,
    payload: DocumentCategoryAssignmentRequest,
    library_repository: LibraryRepository = Depends(get_library_repository),
    category_repository: CategoryRepository = Depends(get_category_repository),
) -> dict:
    current_categories = category_repository.list_document_categories(document_id)
    if current_categories and not payload.category_ids and not payload.confirm_clear:
        raise HTTPException(
            status_code=400,
            detail="Clearing all categories for this document requires confirm_clear=true.",
        )
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
