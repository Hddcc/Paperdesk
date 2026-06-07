"""Heuristic paper curation service for online search candidates."""

from __future__ import annotations

from app.models import PaperCurationItem, PaperCurationResponse, PaperRecord

from .search import PaperSearchService


class PaperSelectionService:
    """Suggest whether online papers are worth adding to the local library."""

    def __init__(self, paper_search_service: PaperSearchService) -> None:
        self.paper_search_service = paper_search_service

    def curate(
        self,
        *,
        topic: str,
        search_provider: str | None,
        top_k_online: int,
    ) -> PaperCurationResponse:
        papers = self.paper_search_service.search(
            topic,
            search_provider=search_provider,
            top_k=top_k_online,
        )
        items = [
            PaperCurationItem(
                paper=paper,
                decision=self._decision_for(paper),
                reason=self._reason_for(paper),
            )
            for paper in papers
        ]
        return PaperCurationResponse(items=items)

    @staticmethod
    def _decision_for(paper: PaperRecord) -> str:
        score = 0
        if paper.abstract:
            score += 1
        if paper.url or paper.doi:
            score += 1
        if paper.year is not None and paper.year >= 2021:
            score += 1
        if score >= 3:
            return "recommended"
        if score == 2:
            return "consider"
        return "skip"

    @staticmethod
    def _reason_for(paper: PaperRecord) -> str:
        reasons: list[str] = []
        if paper.abstract:
            reasons.append("摘要完整，可快速判断相关性")
        else:
            reasons.append("摘要缺失，主题判断成本较高")
        if paper.year is not None:
            reasons.append(f"年份为 {paper.year}")
        if paper.doi:
            reasons.append("带 DOI，便于后续溯源")
        elif paper.url:
            reasons.append("可直接访问原始条目")
        return "；".join(reasons)
