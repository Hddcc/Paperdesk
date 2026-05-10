"""Online paper search agent with deterministic query rewriting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import PaperRecord, TodoTask
from app.utils.query_utils import extract_primary_topic

if TYPE_CHECKING:
    from app.services.paper_search_service import PaperSearchService


class PaperSearchAgent:
    """Rewrite task queries and delegate actual retrieval to the service layer."""

    def __init__(self, search_service: PaperSearchService) -> None:
        self.search_service = search_service

    def search(
        self,
        task: TodoTask,
        *,
        top_k: int = 3,
        search_provider: str | None = None,
    ) -> list[PaperRecord]:
        query = self._rewrite_query(task)
        return self.search_service.search(
            query,
            search_provider=search_provider,
            top_k=top_k,
        )

    @staticmethod
    def _rewrite_query(task: TodoTask) -> str:
        query = extract_primary_topic(task.title or task.query or "").strip()
        intent_text = f"{task.title} {task.intent}".casefold()

        keywords: list[str] = []
        if any(token in intent_text for token in ("背景", "综述", "趋势", "survey", "review")):
            keywords.extend(["survey", "review", "recent"])
        if any(token in intent_text for token in ("方法", "模型", "benchmark", "framework", "method")):
            keywords.extend(["method", "framework", "benchmark"])
        if any(token in intent_text for token in ("应用", "证据", "实验", "evaluation", "experiment")):
            keywords.extend(["application", "evaluation", "experiment"])
        if any(token in intent_text for token in ("挑战", "局限", "方向", "future", "limitation")):
            keywords.extend(["challenge", "limitation", "future work"])

        extras = [keyword for keyword in keywords if keyword.casefold() not in query.casefold()]
        if extras:
            return f"{query} {' '.join(extras)}"
        return query
