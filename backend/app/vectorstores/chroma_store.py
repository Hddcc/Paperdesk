"""Chroma-backed vectorstore for local PDF evidence retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import chromadb

from app.models import ChunkRecord, EvidenceItem, LibraryDocument
from app.models.enums import EvidenceSourceType

from .base import AbstractVectorStore

if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService


class ChromaVectorStore(AbstractVectorStore):
    """Persist local PDF chunks in Chroma and retrieve semantic evidence."""

    def __init__(
        self,
        base_path: Path,
        embedding_service: EmbeddingService,
        *,
        collection_name: str = "paperdesk_local_library",
    ) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    def upsert_document(self, document: LibraryDocument) -> None:
        _ = document

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            raise RuntimeError("PDF import produced no usable text chunks")

        collection = self._get_collection()
        texts = [chunk.content or chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        collection.upsert(
            ids=[chunk.chunk_id or chunk.id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[self._to_metadata(chunk) for chunk in chunks],
        )

    def query_evidence(
        self,
        query: str,
        documents: list[LibraryDocument],
        top_k: int,
    ) -> list[EvidenceItem]:
        ready_documents = {
            document.id: document for document in documents if document.status == "ready"
        }
        if not ready_documents:
            return []

        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_query(query)
        result_limit = max(top_k * 4, top_k)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_limit,
        )

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents_payload = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        evidence_items: list[EvidenceItem] = []
        seen_chunk_ids: set[str] = set()

        for chunk_id, metadata, text, distance in zip(ids, metadatas, documents_payload, distances):
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            record = metadata or {}
            document_id = str(record.get("document_id") or "")
            if document_id not in ready_documents:
                continue

            document = ready_documents[document_id]
            page_number = self._to_int(record.get("page_number"))
            chunk_index = self._to_int(record.get("chunk_index"))
            title = str(record.get("title") or document.title or document.display_name or document.filename)
            display_name = document.display_name or document.filename
            snippet = str(text or "").strip()
            citation_label = f"{display_name} p.{page_number or 0}"

            evidence_items.append(
                EvidenceItem(
                    id=chunk_id,
                    evidence_id=chunk_id,
                    source_type=EvidenceSourceType.LOCAL_DOCUMENT,
                    source_id=document_id,
                    title=title,
                    snippet=snippet,
                    quote=snippet,
                    citation_label=citation_label,
                    document_id=document_id,
                    page_number=page_number,
                    score=self._distance_to_score(distance),
                    metadata={
                        "document_id": document_id,
                        "filename": str(record.get("filename") or document.filename),
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "title": title,
                        "file_path": str(record.get("file_path") or document.file_path),
                        "source_type": "local_document",
                    },
                )
            )

            if len(evidence_items) >= top_k:
                break

        return evidence_items

    def delete_document(self, document_id: str) -> None:
        collection = self._get_collection()
        collection.delete(where={"document_id": document_id})

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            if self._client is None:
                self._client = chromadb.PersistentClient(path=str(self.base_path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # pragma: no cover - depends on local chroma runtime
            raise RuntimeError(f"Failed to initialize Chroma vector store: {exc}") from exc

        return self._collection

    @staticmethod
    def _to_metadata(chunk: ChunkRecord) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("page_number", chunk.page_number)
        metadata.setdefault("chunk_index", chunk.chunk_index)
        metadata.setdefault("source_type", "local_document")
        return metadata

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance_to_score(distance: Any) -> float | None:
        if distance is None:
            return None
        try:
            return round(1 / (1 + float(distance)), 4)
        except (TypeError, ValueError):
            return None
