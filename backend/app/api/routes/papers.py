"""Standalone online paper search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agents import PaperAnalysisAgent, PaperSelectionAgent
from app.api.main import get_paper_analysis_agent, get_paper_search_service, get_paper_selection_agent
from app.models import (
    PaperAnalysisRequest,
    PaperAnalysisResponse,
    PaperCurationRequest,
    PaperCurationResponse,
    PaperSearchRequest,
    PaperSearchResponse,
)
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


@router.post("/analyze")
def analyze_papers(
    request: PaperAnalysisRequest,
    agent: PaperAnalysisAgent = Depends(get_paper_analysis_agent),
) -> dict:
    if request.mode == "single" and len(request.document_ids) != 1:
        raise HTTPException(status_code=400, detail="single mode requires exactly one document_id")
    if request.mode == "compare" and len(request.document_ids) < 2:
        raise HTTPException(status_code=400, detail="compare mode requires at least two document_ids")

    response = agent.analyze(
        document_ids=request.document_ids,
        mode=request.mode,
        question=request.question,
    )
    return PaperAnalysisResponse.model_validate(response).model_dump(mode="json")


@router.post("/curate")
def curate_papers(
    request: PaperCurationRequest,
    agent: PaperSelectionAgent = Depends(get_paper_selection_agent),
) -> dict:
    response = agent.curate(
        topic=request.topic,
        search_provider=request.search_provider,
        top_k_online=request.top_k_online,
    )
    return PaperCurationResponse.model_validate(response).model_dump(mode="json")
