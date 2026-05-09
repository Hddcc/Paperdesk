"""Local evidence retrieval agent stub."""

from __future__ import annotations

from app.models import EvidenceItem, LibraryDocument, TodoTask
from app.vectorstores import AbstractVectorStore


class LibraryRetrieverAgent:
    """Delegate local retrieval to the configured vectorstore."""

    def __init__(self, vectorstore: AbstractVectorStore) -> None:
        self.vectorstore = vectorstore

    def retrieve(
        self,
        task: TodoTask,
        documents: list[LibraryDocument],
        *,
        top_k: int = 3,
    ) -> list[EvidenceItem]:
        return self.vectorstore.query_evidence(task.query, documents, top_k)

