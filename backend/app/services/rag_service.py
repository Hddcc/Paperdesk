"""Compatibility wrapper for paper domain RAG service."""

from app.domains.paper.rag import RagService, RetrievalResult

__all__ = ["RagService", "RetrievalResult"]
