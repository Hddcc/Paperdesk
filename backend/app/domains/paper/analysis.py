"""Structured local paper analysis built on top of the RAG service."""

from __future__ import annotations

from app.models import PaperAnalysisResponse, PaperAnalysisSection

from .rag import RagService


class PaperAnalysisService:
    """Analyze one or more local papers using retrieved evidence."""

    def __init__(self, rag_service: RagService) -> None:
        self.rag_service = rag_service

    def analyze(
        self,
        *,
        document_ids: list[str],
        mode: str,
        question: str | None = None,
    ) -> PaperAnalysisResponse:
        prompt = question or self._default_question(mode=mode)
        rag_answer = self.rag_service.ask(
            question=prompt,
            document_ids=document_ids,
            top_k=max(4, min(8, len(document_ids) * 3)),
            notes="请优先提炼研究问题、方法、实验设置、结论与局限。",
        )
        sections = self._build_sections(mode=mode, question=prompt, answer=rag_answer.answer)
        return PaperAnalysisResponse(
            mode=mode,
            answer=rag_answer.answer,
            sections=sections,
            citations=rag_answer.citations,
            evidence_items=rag_answer.evidence_items,
            retrieval_count=rag_answer.retrieval_count,
        )

    @staticmethod
    def _default_question(*, mode: str) -> str:
        if mode == "compare":
            return "请比较这些论文的研究问题、方法设计、实验设置、关键结果与局限。"
        return "请分析这篇论文的研究问题、方法设计、实验设置、关键结果与局限。"

    @staticmethod
    def _build_sections(
        *,
        mode: str,
        question: str,
        answer: str,
    ) -> list[PaperAnalysisSection]:
        prefix = "多篇比较" if mode == "compare" else "论文分析"
        return [
            PaperAnalysisSection(title=f"{prefix}任务", content=question),
            PaperAnalysisSection(title="基于证据的结论", content=answer),
        ]
