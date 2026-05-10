"""Standalone online paper search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.main import get_paper_search_service
from app.models import PaperSearchRequest, PaperSearchResponse
from app.services import PaperSearchService

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("/search")
def search_papers(
    request: PaperSearchRequest,
    service: PaperSearchService = Depends(get_paper_search_service),
) -> dict:
    items = service.search(
        request.topic,
        search_provider=request.search_provider,
        top_k=request.top_k_online,
    )
    return PaperSearchResponse(items=items).model_dump(mode="json")
