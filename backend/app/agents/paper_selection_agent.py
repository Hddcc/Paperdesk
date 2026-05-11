"""Agent wrapper for paper curation suggestions."""

from __future__ import annotations

from app.models import PaperCurationResponse

from app.services.paper_selection_service import PaperSelectionService


class PaperSelectionAgent:
    """Delegate online paper curation to the selection service."""

    def __init__(self, service: PaperSelectionService) -> None:
        self.service = service

    def curate(
        self,
        *,
        topic: str,
        search_provider: str | None,
        top_k_online: int,
    ) -> PaperCurationResponse:
        return self.service.curate(
            topic=topic,
            search_provider=search_provider,
            top_k_online=top_k_online,
        )
