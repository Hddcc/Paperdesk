"""Standalone RAG service for local knowledge-base Q&A."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.models import EvidenceItem, LibraryDocument, RagAskResponse
from app.repositories import LibraryRepository
from app.vectorstores import AbstractVectorStore

from .query_translation_service import QueryTranslationService


class RagService:
    """Retrieve local evidence and generate grounded answers."""

    def __init__(
        self,
        *,
        library_repository: LibraryRepository,
        vectorstore: AbstractVectorStore,
        translation_service: QueryTranslationService | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.library_repository = library_repository
        self.vectorstore = vectorstore
        self.translation_service = translation_service
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

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
        )

    def retrieve_evidence(
        self,
        *,
        question: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> list[EvidenceItem]:
        if not documents:
            return []

        candidate_limit = max(top_k * 3, top_k)
        query_candidates = [question]
        translated = self._translate_query(question)
        if translated and translated not in query_candidates:
            query_candidates.append(translated)

        merged: dict[str, EvidenceItem] = {}
        for query in query_candidates:
            for item in self.vectorstore.query_evidence(query, documents, candidate_limit):
                existing = merged.get(item.id)
                if existing is None or (item.score or 0.0) > (existing.score or 0.0):
                    merged[item.id] = item

        ranked = sorted(merged.values(), key=lambda item: item.score or 0.0, reverse=True)
        return ranked[:top_k]

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
