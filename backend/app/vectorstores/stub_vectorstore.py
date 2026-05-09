"""Stub vectorstore used for the 00/01 runnable skeleton."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.models import EvidenceItem, LibraryDocument
from app.models.enums import EvidenceSourceType

from .base import AbstractVectorStore


class StubVectorStore(AbstractVectorStore):
    """Deterministic local evidence provider based on document metadata."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upsert_document(self, document: LibraryDocument) -> None:
        marker_file = self.base_path / f"{document.id}.txt"
        marker_file.write_text(
            f"{document.display_name}\n{document.file_path}\nstatus={document.status}\n",
            encoding="utf-8",
        )

    def query_evidence(
        self,
        query: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        for index, document in enumerate(documents[:top_k], start=1):
            evidence_items.append(
                EvidenceItem(
                    id=str(uuid4()),
                    source_type=EvidenceSourceType.LOCAL_DOCUMENT,
                    source_id=document.id,
                    quote=(
                        f"Mock local evidence for '{query}' derived from uploaded PDF "
                        f"'{document.display_name}'. This is a placeholder retrieval result."
                    ),
                    citation_label=f"[L{index}] {document.display_name}",
                    metadata={
                        "document_id": document.id,
                        "filename": document.filename,
                        "file_path": document.file_path,
                    },
                )
            )
        return evidence_items
