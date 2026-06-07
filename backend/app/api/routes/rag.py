"""Standalone RAG routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.main import get_rag_service
from app.domains.paper import RagService
from app.models import RagAskRequest, RagAskResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask")
def ask_rag(
    request: RagAskRequest,
    service: RagService = Depends(get_rag_service),
) -> dict:
    response = service.ask(
        question=request.question,
        document_ids=request.document_ids,
        top_k=request.top_k,
        notes=request.notes,
    )
    return RagAskResponse.model_validate(response).model_dump(mode="json")
