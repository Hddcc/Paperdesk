"""Agent wrapper for local paper analysis."""

from __future__ import annotations

from app.models import PaperAnalysisResponse

from app.services.paper_analysis_service import PaperAnalysisService


class PaperAnalysisAgent:
    """Delegate local paper analysis to the structured analysis service."""

    def __init__(self, service: PaperAnalysisService) -> None:
        self.service = service

    def analyze(
        self,
        *,
        document_ids: list[str],
        mode: str,
        question: str | None = None,
    ) -> PaperAnalysisResponse:
        return self.service.analyze(
            document_ids=document_ids,
            mode=mode,
            question=question,
        )
