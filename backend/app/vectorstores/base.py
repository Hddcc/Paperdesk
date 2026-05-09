"""Abstract vectorstore contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import EvidenceItem, LibraryDocument


class AbstractVectorStore(ABC):
    """Minimal interface for future pluggable vectorstores."""

    @abstractmethod
    def upsert_document(self, document: LibraryDocument) -> None:
        """Register a document with the local knowledge store."""

    @abstractmethod
    def query_evidence(
        self,
        query: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> list[EvidenceItem]:
        """Return evidence items relevant to the given query."""

