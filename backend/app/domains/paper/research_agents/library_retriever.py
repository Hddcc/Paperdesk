"""Local evidence retrieval agent stub."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.models import EvidenceItem, LibraryDocument, TodoTask
from app.vectorstores import AbstractVectorStore

if TYPE_CHECKING:
    from app.domains.paper import QueryTranslationService, RagService

_ENGLISH_STOPWORDS = {
    "about",
    "across",
    "analysis",
    "approach",
    "approaches",
    "background",
    "challenge",
    "challenges",
    "definition",
    "definitions",
    "direction",
    "directions",
    "evidence",
    "for",
    "from",
    "future",
    "methods",
    "models",
    "overview",
    "paper",
    "papers",
    "problem",
    "problems",
    "research",
    "scenarios",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "the",
    "their",
    "theme",
    "topic",
    "trends",
    "with",
    "year",
}

_CHINESE_STOPWORDS = {
    "研究",
    "背景",
    "问题",
    "定义",
    "代表性",
    "方法",
    "论文",
    "脉络",
    "应用",
    "场景",
    "证据",
    "线索",
    "挑战",
    "局限",
    "后续",
    "方向",
    "了解",
    "关注",
    "总结",
    "主题",
}


class LibraryRetrieverAgent:
    """Delegate local retrieval to the configured vectorstore."""

    def __init__(
        self,
        vectorstore: AbstractVectorStore,
        translation_service: "QueryTranslationService | None" = None,
        rag_service: "RagService | None" = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.translation_service = translation_service
        self.rag_service = rag_service

    def retrieve(
        self,
        task: TodoTask,
        documents: list[LibraryDocument],
        *,
        top_k: int = 3,
    ) -> list[EvidenceItem]:
        if self.rag_service is not None:
            translated_query = self._translate_query(task.query)
            retrieved = self.rag_service.retrieve_evidence(
                question=task.query,
                documents=documents,
                top_k=top_k,
            )
            filtered = [
                item
                for item in retrieved
                if self._is_relevant_local_evidence(
                    task=task,
                    evidence=item,
                    translated_query=translated_query,
                )
            ]
            return filtered[:top_k]

        candidate_limit = max(top_k * 3, top_k)
        query_candidates = [task.query]
        translated_query = self._translate_query(task.query)
        if translated_query and translated_query not in query_candidates:
            query_candidates.append(translated_query)

        merged_candidates: dict[str, EvidenceItem] = {}
        for query_text in query_candidates:
            for item in self.vectorstore.query_evidence(query_text, documents, candidate_limit):
                existing = merged_candidates.get(item.id)
                if existing is None or (item.score or 0.0) > (existing.score or 0.0):
                    merged_candidates[item.id] = item

        filtered = [
            item
            for item in merged_candidates.values()
            if self._is_relevant_local_evidence(
                task=task,
                evidence=item,
                translated_query=translated_query,
            )
        ]
        filtered.sort(key=lambda item: item.score or 0.0, reverse=True)
        return filtered[:top_k]

    def _translate_query(self, query: str) -> str | None:
        if self.translation_service is None:
            return None
        try:
            return self.translation_service.translate_to_english(query)
        except Exception:
            return None

    def _is_relevant_local_evidence(
        self,
        *,
        task: TodoTask,
        evidence: EvidenceItem,
        translated_query: str | None,
    ) -> bool:
        evidence_text = " ".join(
            filter(
                None,
                [
                    evidence.title,
                    evidence.snippet,
                    evidence.quote,
                    str(evidence.metadata.get("filename") or ""),
                ],
            )
        )
        evidence_keywords = self._extract_keywords(evidence_text)
        if not evidence_keywords:
            return False

        query_text = "\n".join(
            filter(
                None,
                [
                    task.title,
                    task.intent,
                    task.query,
                    translated_query,
                ],
            )
        )
        query_keywords = self._extract_keywords(query_text)
        if not query_keywords:
            return False

        overlap = query_keywords & evidence_keywords
        if overlap:
            evidence.metadata["matched_keywords"] = sorted(overlap)
            return True

        return False

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        english_tokens = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
            if token not in _ENGLISH_STOPWORDS
        }

        chinese_tokens: set[str] = set()
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if chunk in _CHINESE_STOPWORDS:
                continue
            if len(chunk) <= 4:
                chinese_tokens.add(chunk)
                continue
            for start in range(len(chunk) - 1):
                token = chunk[start : start + 4]
                if len(token) >= 2 and token not in _CHINESE_STOPWORDS:
                    chinese_tokens.add(token)
            chinese_tokens.add(chunk)

        return english_tokens | chinese_tokens
