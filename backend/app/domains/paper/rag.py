"""Standalone RAG service for local knowledge-base Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
import re
from typing import Any

from openai import OpenAI

from app.models import ChunkRecord, EvidenceItem, EvidenceQuality, LibraryDocument, RagAskResponse
from app.models.enums import EvidenceSourceType
from app.repositories import ChunkRepository, LibraryRepository
from app.vectorstores import AbstractVectorStore

from .query_translation import QueryTranslationService


@dataclass(slots=True)
class RetrievalResult:
    """Evidence retrieval output with quality and cache diagnostics."""

    evidence_items: list[EvidenceItem]
    evidence_quality: EvidenceQuality
    cache_hit: bool = False
    retrieval_strategy: str = "hybrid"


@dataclass(slots=True)
class _CacheEntry:
    result: RetrievalResult
    expires_at: datetime


class RagService:
    """Retrieve local evidence and generate grounded answers."""

    def __init__(
        self,
        *,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        vectorstore: AbstractVectorStore,
        translation_service: QueryTranslationService | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
        cache_ttl_seconds: int = 1800,
    ) -> None:
        self.library_repository = library_repository
        self.chunk_repository = chunk_repository
        self.vectorstore = vectorstore
        self.translation_service = translation_service
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._retrieval_cache: dict[str, _CacheEntry] = {}
        self.last_retrieval_cache_hit = False
        self.last_retrieval_strategy = "hybrid"
        self.last_evidence_quality = EvidenceQuality()

    def ask(
        self,
        *,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int = 4,
        notes: str | None = None,
    ) -> RagAskResponse:
        documents = self._select_documents(document_ids)
        evidence_items = self.retrieve_evidence(
            question=question,
            documents=documents,
            top_k=top_k,
        )
        answer = self._generate_answer(question=question, notes=notes, evidence_items=evidence_items)
        return RagAskResponse(
            answer=answer,
            citations=self._collect_citations(evidence_items),
            sources=self._collect_sources(evidence_items),
            pages=self._collect_pages(evidence_items),
            retrieval_count=len(evidence_items),
            confidence=self._estimate_confidence(evidence_items),
            evidence_items=evidence_items,
            evidence_quality=self.last_evidence_quality,
        )

    def retrieve_evidence(
        self,
        *,
        question: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> list[EvidenceItem]:
        result = self.retrieve_evidence_with_quality(
            question=question,
            documents=documents,
            top_k=top_k,
        )
        return result.evidence_items

    def retrieve_evidence_with_quality(
        self,
        *,
        question: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> RetrievalResult:
        if not documents:
            quality = self.assess_evidence_quality([], documents)
            result = RetrievalResult(
                evidence_items=[],
                evidence_quality=quality,
                cache_hit=False,
            )
            self._remember_retrieval(result)
            return result

        cache_key = self._cache_key(question=question, documents=documents, top_k=top_k)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            self._remember_retrieval(cached)
            return cached

        synthesis_query = self._is_synthesis_query(question)
        candidate_limit = max(top_k * (5 if synthesis_query else 3), top_k)
        query_candidates = self._expanded_queries(question)
        translated = self._translate_query(question)
        if translated and translated not in query_candidates:
            query_candidates.append(translated)

        candidates: list[EvidenceItem] = []
        vector_failed = False
        vector_available = True
        for query in query_candidates:
            if vector_available:
                try:
                    vector_items = self.vectorstore.query_evidence(query, documents, candidate_limit)
                except Exception:
                    vector_failed = True
                    vector_available = False
                    vector_items = []
                for item in vector_items:
                    candidates.append(self._with_strategy(item, "vector"))
            candidates.extend(
                self._keyword_search(
                    query=query,
                    documents=documents,
                    limit=candidate_limit,
                )
            )

        merged = self._deduplicate(candidates)
        filtered = self._similarity_filter(merged)
        reranked = self._rerank(filtered, documents=documents, question=question)
        evidence_items = self._select_top_evidence(reranked, documents=documents, top_k=top_k, ensure_coverage=synthesis_query)
        quality = self.assess_evidence_quality(evidence_items, documents)
        retrieval_strategy = "hybrid"
        if vector_failed:
            warnings = list(quality.warnings)
            if "vector_unavailable" not in warnings:
                warnings.append("vector_unavailable")
            quality = quality.model_copy(update={"warnings": warnings})
            retrieval_strategy = "keyword_only_vector_unavailable"
        result = RetrievalResult(
            evidence_items=evidence_items,
            evidence_quality=quality,
            cache_hit=False,
            retrieval_strategy=retrieval_strategy,
        )
        if not vector_failed:
            self._set_cached_result(cache_key, result)
        self._remember_retrieval(result)
        return result

    def assess_evidence_quality(
        self,
        evidence_items: list[EvidenceItem],
        documents: list[LibraryDocument],
    ) -> EvidenceQuality:
        document_coverage: dict[str, int] = {}
        for item in evidence_items:
            document_id = item.document_id or item.source_id
            if document_id:
                document_coverage[document_id] = document_coverage.get(document_id, 0) + 1

        target_count = max(1, len(documents))
        covered_count = len(document_coverage)
        coverage_score = min(1.0, covered_count / target_count)

        if not evidence_items:
            diversity_score = 0.0
        else:
            largest_bucket = max(document_coverage.values(), default=0)
            diversity_score = 1.0 - max(0.0, (largest_bucket / len(evidence_items)) - (1 / max(1, covered_count)))
            diversity_score = max(0.0, min(1.0, diversity_score))

        citation_ready = [
            item
            for item in evidence_items
            if item.citation_label and item.page_number is not None
        ]
        citation_score = len(citation_ready) / len(evidence_items) if evidence_items else 0.0

        scored = [
            item.rerank_score if item.rerank_score is not None else item.score
            for item in evidence_items
            if (item.rerank_score if item.rerank_score is not None else item.score) is not None
        ]
        relevance_score = min(1.0, max(0.0, sum(scored) / len(scored))) if scored else 0.0

        warnings: list[str] = []
        if not evidence_items:
            warnings.append("insufficient_evidence")
        if coverage_score < 0.35 and documents:
            warnings.append("low_document_coverage")
        if diversity_score < 0.45 and len(evidence_items) > 1:
            warnings.append("low_evidence_diversity")
        if citation_score < 0.8 and evidence_items:
            warnings.append("weak_citation_metadata")
        if relevance_score < 0.35 and evidence_items:
            warnings.append("low_relevance")

        return EvidenceQuality(
            coverage_score=round(coverage_score, 4),
            diversity_score=round(diversity_score, 4),
            citation_score=round(citation_score, 4),
            relevance_score=round(relevance_score, 4),
            document_coverage=document_coverage,
            warnings=warnings,
        )

    def _keyword_search(
        self,
        *,
        query: str,
        documents: list[LibraryDocument],
        limit: int,
    ) -> list[EvidenceItem]:
        document_ids = [document.id for document in documents]
        chunks = self.chunk_repository.list_chunks(document_ids=document_ids)
        terms = self._query_terms(query)
        if not terms:
            return []
        document_by_id = {document.id: document for document in documents}
        scored: list[tuple[float, int, ChunkRecord]] = []
        for chunk in chunks:
            text = chunk.content or chunk.text
            score, hit_count = self._keyword_score(text, terms)
            if score <= 0.0:
                continue
            scored.append((score, hit_count, chunk))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        results: list[EvidenceItem] = []
        for score, hit_count, chunk in scored[:limit]:
            document = document_by_id.get(chunk.document_id)
            title = chunk.title or (document.title if document else None) or (document.filename if document else "")
            filename = (document.display_name or document.filename) if document else (chunk.source or title)
            snippet = self._keyword_snippet(chunk.content or chunk.text, terms)
            item = EvidenceItem(
                id=chunk.chunk_id or chunk.id,
                evidence_id=chunk.chunk_id or chunk.id,
                source_type=EvidenceSourceType.LOCAL_DOCUMENT,
                source_id=chunk.document_id,
                title=title or filename,
                snippet=snippet,
                quote=snippet,
                citation_label=f"{filename} p.{chunk.page_number}",
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                score=round(score, 4),
                strategy="keyword",
                strategies=["keyword"],
                raw_scores={"keyword": round(score, 4), "keyword_hits": float(hit_count)},
                metadata={
                    **chunk.metadata,
                    "filename": filename,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id or chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "version": chunk.version,
                    "strategy": "keyword",
                },
            )
            results.append(item)
        return results

    @staticmethod
    def _with_strategy(item: EvidenceItem, strategy: str) -> EvidenceItem:
        next_item = item.model_copy(deep=True)
        score = next_item.score if next_item.score is not None else 0.0
        next_item.strategy = strategy
        next_item.strategies = list(dict.fromkeys([*next_item.strategies, strategy]))
        next_item.raw_scores = {**next_item.raw_scores, strategy: round(score, 4)}
        next_item.metadata = {**next_item.metadata, "strategy": strategy}
        next_item.multi_route_hit = len(next_item.strategies) > 1
        return next_item

    def _deduplicate(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}
        for item in items:
            key = self._dedupe_key(item)
            existing = merged.get(key)
            if existing is None:
                merged[key] = item.model_copy(deep=True)
                continue
            merged[key] = self._merge_evidence(existing, item)
        return list(merged.values())

    @staticmethod
    def _merge_evidence(left: EvidenceItem, right: EvidenceItem) -> EvidenceItem:
        merged = left.model_copy(deep=True)
        strategies = list(dict.fromkeys([*merged.strategies, *right.strategies]))
        if right.strategy and right.strategy not in strategies:
            strategies.append(right.strategy)
        merged.strategies = strategies
        merged.strategy = "hybrid" if len(strategies) > 1 else strategies[0] if strategies else merged.strategy
        merged.multi_route_hit = len(strategies) > 1
        merged.raw_scores = {**merged.raw_scores, **right.raw_scores}
        if (right.score or 0.0) > (merged.score or 0.0):
            merged.score = right.score
        if len(right.snippet or right.quote) > len(merged.snippet or merged.quote):
            merged.snippet = right.snippet
            merged.quote = right.quote
        merged.metadata = {
            **merged.metadata,
            **right.metadata,
            "strategies": strategies,
            "multi_route_hit": merged.multi_route_hit,
        }
        return merged

    @staticmethod
    def _similarity_filter(items: list[EvidenceItem]) -> list[EvidenceItem]:
        filtered: list[EvidenceItem] = []
        for item in items:
            strategies = set(item.strategies or ([item.strategy] if item.strategy else []))
            vector_score = item.raw_scores.get("vector")
            keyword_score = item.raw_scores.get("keyword")
            if "vector" in strategies and len(strategies) == 1 and vector_score is not None and vector_score < 0.12:
                continue
            if "keyword" in strategies and len(strategies) == 1:
                text = item.quote or item.snippet
                if keyword_score is not None and keyword_score < 0.18:
                    continue
                if len(text.strip()) < 24:
                    continue
            filtered.append(item)
        return filtered

    @classmethod
    def _rerank(cls, items: list[EvidenceItem], *, documents: list[LibraryDocument], question: str = "") -> list[EvidenceItem]:
        selected_ids = {document.id for document in documents}
        document_counts: dict[str, int] = {}
        synthesis_query = cls._is_synthesis_query(question)
        for item in items:
            document_id = item.document_id or item.source_id
            document_counts[document_id] = document_counts.get(document_id, 0) + 1

        reranked: list[EvidenceItem] = []
        for item in items:
            vector_score = item.raw_scores.get("vector", item.score or 0.0)
            keyword_score = item.raw_scores.get("keyword", 0.0)
            base_score = max(vector_score, keyword_score)
            score = base_score
            if item.multi_route_hit:
                score += 0.24
            if item.document_id in selected_ids:
                score += 0.08
            if item.page_number is not None and item.citation_label:
                score += 0.05
            if item.title and item.title in (item.quote or item.snippet):
                score += 0.03
            section_score = cls._section_quality_score(item)
            if synthesis_query:
                score += section_score
            document_id = item.document_id or item.source_id
            if document_counts.get(document_id, 0) > 2:
                score -= min(0.08, (document_counts[document_id] - 2) * 0.02)
            next_item = item.model_copy(deep=True)
            next_item.rerank_score = round(max(0.0, min(1.0, score)), 4)
            next_item.score = next_item.rerank_score
            next_item.metadata = {
                **next_item.metadata,
                "rerank_score": next_item.rerank_score,
                "strategies": next_item.strategies,
                "multi_route_hit": next_item.multi_route_hit,
            }
            reranked.append(next_item)
        return sorted(reranked, key=lambda item: item.rerank_score or item.score or 0.0, reverse=True)

    @classmethod
    def _expanded_queries(cls, question: str) -> list[str]:
        queries = [question]
        if not cls._is_synthesis_query(question):
            return queries
        expansions = [
            f"{question} abstract introduction method contribution experiment result conclusion",
            f"{question} summary overview abstract introduction approach contribution experiment result conclusion",
            f"{question} 摘要 引言 方法 贡献 实验 结果 结论 主要内容",
        ]
        for query in expansions:
            if query not in queries:
                queries.append(query)
        return queries

    @staticmethod
    def _is_synthesis_query(question: str) -> bool:
        normalized = question.casefold()
        markers = (
            "总结",
            "综述",
            "报告",
            "概述",
            "对比",
            "比较",
            "分析",
            "创新点",
            "贡献",
            "summary",
            "summarize",
            "review",
            "survey",
            "report",
            "compare",
            "comparison",
            "analysis",
            "explain",
            "overview",
            "what is this about",
            "讲了什么",
            "说了什么",
            "主要讲",
            "主要内容",
            "一段话",
            "一句话",
            "解释",
            "概括",
            "介绍",
            "这篇论文",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _section_quality_score(item: EvidenceItem) -> float:
        metadata = item.metadata or {}
        section_text = " ".join(
            str(value)
            for value in (
                item.title,
                metadata.get("section"),
                metadata.get("heading"),
                metadata.get("title"),
                item.quote or item.snippet,
            )
            if value
        ).casefold()
        positive_markers = (
            "abstract",
            "introduction",
            "method",
            "approach",
            "contribution",
            "experiment",
            "evaluation",
            "result",
            "conclusion",
            "overview",
            "summary",
            "摘要",
            "引言",
            "方法",
            "贡献",
            "实验",
            "结果",
            "结论",
            "主要内容",
        )
        negative_markers = (
            "references",
            "reference",
            "bibliography",
            "acknowledgement",
            "acknowledgment",
            "appendix",
            "supplementary",
            "致谢",
            "参考文献",
            "附录",
        )
        score = 0.0
        if any(marker in section_text for marker in positive_markers):
            score += 0.24
        if any(marker in section_text for marker in negative_markers):
            score -= 0.55
        return score

    @staticmethod
    def _select_top_evidence(
        items: list[EvidenceItem],
        *,
        documents: list[LibraryDocument],
        top_k: int,
        ensure_coverage: bool,
    ) -> list[EvidenceItem]:
        if not ensure_coverage or len(documents) <= 1:
            return items[:top_k]
        selected: list[EvidenceItem] = []
        selected_ids: set[str] = set()
        for document in documents:
            match = next(
                (
                    item
                    for item in items
                    if (item.document_id or item.source_id) == document.id and item.id not in selected_ids
                ),
                None,
            )
            if match is not None:
                selected.append(match)
                selected_ids.add(match.id)
            if len(selected) >= top_k:
                return selected
        for item in items:
            if item.id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.id)
            if len(selected) >= top_k:
                break
        return selected

    def _cache_key(
        self,
        *,
        question: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> str:
        payload = "|".join(
            [
                self._hash_text(question),
                self._hash_text(",".join(sorted(document.id for document in documents))),
                "hybrid",
                str(top_k),
                self._library_version(documents),
            ]
        )
        return self._hash_text(payload)

    def _get_cached_result(self, cache_key: str) -> RetrievalResult | None:
        now = datetime.now(timezone.utc)
        entry = self._retrieval_cache.get(cache_key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._retrieval_cache.pop(cache_key, None)
            return None
        return RetrievalResult(
            evidence_items=[item.model_copy(deep=True) for item in entry.result.evidence_items],
            evidence_quality=entry.result.evidence_quality.model_copy(deep=True),
            cache_hit=True,
            retrieval_strategy=entry.result.retrieval_strategy,
        )

    def _set_cached_result(self, cache_key: str, result: RetrievalResult) -> None:
        if self.cache_ttl.total_seconds() <= 0:
            return
        self._retrieval_cache[cache_key] = _CacheEntry(
            result=RetrievalResult(
                evidence_items=[item.model_copy(deep=True) for item in result.evidence_items],
                evidence_quality=result.evidence_quality.model_copy(deep=True),
                cache_hit=False,
                retrieval_strategy=result.retrieval_strategy,
            ),
            expires_at=datetime.now(timezone.utc) + self.cache_ttl,
        )

    def _remember_retrieval(self, result: RetrievalResult) -> None:
        self.last_retrieval_cache_hit = result.cache_hit
        self.last_retrieval_strategy = result.retrieval_strategy
        self.last_evidence_quality = result.evidence_quality

    @staticmethod
    def _library_version(documents: list[LibraryDocument]) -> str:
        parts = []
        for document in sorted(documents, key=lambda item: item.id):
            parts.append(
                ":".join(
                    [
                        document.id,
                        str(document.version),
                        document.sha256 or "",
                        document.indexed_at.isoformat() if document.indexed_at else "",
                        document.status,
                    ]
                )
            )
        return RagService._hash_text("|".join(parts))

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _dedupe_key(item: EvidenceItem) -> str:
        text = RagService._normalize_text(item.quote or item.snippet)
        if len(text) > 180:
            text = text[:180]
        return "|".join([item.document_id or item.source_id, str(item.page_number or 0), text])

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", value.casefold())).strip()

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", query.casefold())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "paper",
            "please",
            "how",
            "what",
        }
        deduped: list[str] = []
        for term in terms:
            if term in stopwords or term in deduped:
                continue
            deduped.append(term)
        return deduped[:12]

    @staticmethod
    def _keyword_score(text: str, terms: list[str]) -> tuple[float, int]:
        normalized = text.casefold()
        hits = 0
        weighted = 0.0
        for term in terms:
            count = normalized.count(term)
            if count <= 0:
                continue
            hits += 1
            weighted += min(3, count) * (1.5 if len(term) >= 6 else 1.0)
        if hits == 0:
            return 0.0, 0
        length_penalty = 1 / math.sqrt(max(1.0, len(text) / 500))
        score = min(1.0, (weighted / max(1, len(terms))) * 0.45 * length_penalty)
        return score, hits

    @staticmethod
    def _keyword_snippet(text: str, terms: list[str], *, max_chars: int = 360) -> str:
        lowered = text.casefold()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        if not positions:
            return text[:max_chars].strip()
        start = max(0, min(positions) - 120)
        snippet = text[start : start + max_chars].strip()
        return re.sub(r"\s+", " ", snippet)

    def _select_documents(self, document_ids: list[str] | None) -> list[LibraryDocument]:
        documents = self.library_repository.list_documents()
        ready_documents = [document for document in documents if document.status == "ready"]
        if not document_ids:
            return ready_documents
        selected_ids = set(document_ids)
        return [document for document in ready_documents if document.id in selected_ids]

    def _translate_query(self, query: str) -> str | None:
        if self.translation_service is None:
            return None
        try:
            return self.translation_service.translate_to_english(query)
        except Exception:
            return None

    def _generate_answer(
        self,
        *,
        question: str,
        notes: str | None,
        evidence_items: list[EvidenceItem],
    ) -> str:
        if not evidence_items:
            return "当前知识库中没有检索到足够相关的本地证据，暂不足以回答这个问题。"

        prompt = self._build_prompt(question=question, notes=notes, evidence_items=evidence_items)
        polished = self._call_llm(prompt)
        if polished:
            return polished
        return self._build_template_answer(question=question, evidence_items=evidence_items)

    def _build_prompt(
        self,
        *,
        question: str,
        notes: str | None,
        evidence_items: list[EvidenceItem],
    ) -> str:
        evidence_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"来源：{item.citation_label}",
                        f"标题：{item.title}",
                        f"页码：{item.page_number if item.page_number is not None else '未知'}",
                        f"证据：{item.quote or item.snippet}",
                    ]
                )
                for item in evidence_items
            ]
        )
        parts = [
            f"用户问题：{question}",
            "请仅基于给定证据，用中文回答，并在关键结论处引用来源标签。",
        ]
        if notes:
            parts.append(f"补充说明：{notes}")
        parts.extend(["证据清单：", evidence_block])
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> str | None:
        if not self.api_key:
            return None
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer in Chinese using only provided evidence. Cite sources inline "
                            "with the original citation labels and explicitly say when evidence is insufficient."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return None
        return self._extract_message_text(response)

    def _build_template_answer(self, *, question: str, evidence_items: list[EvidenceItem]) -> str:
        top = evidence_items[: min(3, len(evidence_items))]
        bullets = [
            f"- 证据 {index}: {item.quote or item.snippet}（{item.citation_label}）"
            for index, item in enumerate(top, start=1)
        ]
        return "\n".join(
            [
                f"围绕“{question}”，当前检索到的本地证据主要集中在以下内容：",
                *bullets,
                "基于这些证据，可以先从原文所涉及的方法、结论和限制展开核验；若需要更完整判断，建议继续补充相关 PDF。",  # noqa: E501
            ]
        )

    @staticmethod
    def _collect_citations(evidence_items: list[EvidenceItem]) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []
        for item in evidence_items:
            if item.citation_label in seen:
                continue
            seen.add(item.citation_label)
            citations.append(item.citation_label)
        return citations

    @staticmethod
    def _collect_sources(evidence_items: list[EvidenceItem]) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for item in evidence_items:
            source = str(item.metadata.get("filename") or item.title or item.source_id)
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
        return sources

    @staticmethod
    def _collect_pages(evidence_items: list[EvidenceItem]) -> list[int]:
        pages = {item.page_number for item in evidence_items if item.page_number is not None}
        return sorted(pages)

    @staticmethod
    def _estimate_confidence(evidence_items: list[EvidenceItem]) -> float | None:
        scores = [item.score for item in evidence_items if item.score is not None]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def _extract_message_text(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip() if parts else None
        return None
