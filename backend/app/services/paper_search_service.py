"""Online paper search orchestration across external providers."""

from __future__ import annotations

import re

from app.models import PaperRecord
from app.utils.query_utils import contains_cjk, extract_ascii_keywords, extract_primary_topic

from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient
from .paper_mapper import map_arxiv_entry, map_openalex_work, normalize_title
from .query_translation_service import QueryTranslationService


class PaperSearchService:
    """Aggregate, normalize, deduplicate, and rank online paper results."""

    def __init__(
        self,
        *,
        openalex_client: OpenAlexClient,
        arxiv_client: ArxivClient,
        translation_service: QueryTranslationService | None = None,
    ) -> None:
        self.openalex_client = openalex_client
        self.arxiv_client = arxiv_client
        self.translation_service = translation_service

    def search(
        self,
        query: str,
        *,
        search_provider: str | None = None,
        top_k: int = 5,
    ) -> list[PaperRecord]:
        provider = (search_provider or "all").lower()
        provider_limit = max(top_k * 2, top_k)
        collected: list[PaperRecord] = []
        for candidate_query in self._build_candidate_queries(query):
            collected.extend(self._search_once(candidate_query, provider=provider, provider_limit=provider_limit))
            deduplicated = self._deduplicate(collected)
            if len(deduplicated) >= top_k:
                deduplicated.sort(key=lambda item: self._sort_key(item, candidate_query))
                return deduplicated[:top_k]

        deduplicated = self._deduplicate(collected)
        deduplicated.sort(key=lambda item: self._sort_key(item, query))
        return deduplicated[:top_k]

    def _build_candidate_queries(self, query: str) -> list[str]:
        original = query.strip()
        if not original:
            return []

        core_topic = extract_primary_topic(original)
        english_hints = extract_ascii_keywords(original)
        candidates: list[str] = []

        if contains_cjk(core_topic) and self.translation_service is not None:
            translated = self.translation_service.translate_to_english(core_topic)
            if translated:
                translated = translated.strip()
                if english_hints:
                    candidates.append(f"{translated} {' '.join(english_hints)}")
                candidates.append(translated)

        candidates.append(original)
        if core_topic != original:
            candidates.append(core_topic)

        deduplicated_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if normalized and normalized.casefold() not in seen:
                seen.add(normalized.casefold())
                deduplicated_candidates.append(normalized)
        return deduplicated_candidates

    def _search_once(
        self,
        query: str,
        *,
        provider: str,
        provider_limit: int,
    ) -> list[PaperRecord]:
        collected: list[PaperRecord] = []

        if provider in {"all", "auto", "openalex"}:
            try:
                collected.extend(
                    map_openalex_work(work)
                    for work in self.openalex_client.search(query, limit=provider_limit)
                    if (work.get("display_name") or "").strip()
                )
            except Exception:
                pass

        if provider in {"all", "auto", "arxiv"}:
            try:
                collected.extend(
                    map_arxiv_entry(entry)
                    for entry in self.arxiv_client.search(query, limit=provider_limit)
                    if (entry.get("title") or "").strip()
                )
            except Exception:
                pass

        return collected

    @staticmethod
    def _deduplicate(records: list[PaperRecord]) -> list[PaperRecord]:
        unique_records: list[PaperRecord] = []
        seen_doi: set[str] = set()
        seen_title: set[str] = set()

        for record in records:
            if record.doi:
                if record.doi in seen_doi:
                    continue
                seen_doi.add(record.doi)
                seen_title.add(normalize_title(record.title))
                unique_records.append(record)
                continue

            normalized_title = normalize_title(record.title)
            if normalized_title in seen_title:
                continue
            seen_title.add(normalized_title)
            unique_records.append(record)

        return unique_records

    @staticmethod
    def _sort_key(record: PaperRecord, query: str) -> tuple[int, int, int, str]:
        return (
            -(record.year or 0),
            -_title_overlap_score(record.title, query),
            0 if record.abstract else 1,
            record.title.casefold(),
        )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token}


def _title_overlap_score(title: str, query: str) -> int:
    title_tokens = _tokenize(title)
    query_tokens = _tokenize(query)
    if not title_tokens or not query_tokens:
        return 0
    return len(title_tokens & query_tokens)
