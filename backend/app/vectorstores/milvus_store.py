"""Milvus-backed vectorstore for local PDF evidence retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models import ChunkRecord, EvidenceItem, LibraryDocument
from app.models.enums import EvidenceSourceType

from .base import AbstractVectorStore


class MilvusVectorStore(AbstractVectorStore):
    """Persist local PDF chunks in Milvus and retrieve semantic evidence."""

    STRING_PRIMARY_KEY_MAX_LENGTH = 512

    def __init__(
        self,
        *,
        uri: str,
        token: str | None,
        database: str,
        collection_name: str,
        embedding_service,
    ) -> None:
        self.uri = uri
        self.token = token
        self.database = database
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self._client = None
        self._dimension: int | None = None

    def ensure_available(self) -> None:
        """Eagerly initialize the Milvus client so startup failures surface early."""
        self._get_client()

    def upsert_document(self, document: LibraryDocument) -> None:
        _ = document

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            raise RuntimeError("PDF import produced no usable text chunks")

        texts = [chunk.content or chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        self._ensure_collection(len(embeddings[0]))
        client = self._get_client()
        payload = [self._to_row(chunk, embedding) for chunk, embedding in zip(chunks, embeddings)]
        chunk_ids = [chunk.chunk_id or chunk.id for chunk in chunks]
        client.delete(
            collection_name=self.collection_name,
            filter=self._in_filter("chunk_id", chunk_ids),
        )
        client.insert(
            collection_name=self.collection_name,
            data=payload,
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

        client = self._get_client()
        if not client.has_collection(collection_name=self.collection_name):
            return []

        query_embedding = self.embedding_service.embed_query(query)
        result_limit = max(top_k * 4, top_k)
        filter_expr = self._in_filter("document_id", list(ready_documents))
        results = client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=result_limit,
            filter=filter_expr,
            output_fields=[
                "chunk_id",
                "document_id",
                "filename",
                "page_number",
                "chunk_index",
                "title",
                "file_path",
                "sha256",
                "version",
                "text",
            ],
        )

        hits = results[0] if results and isinstance(results[0], list) else results
        evidence_items: list[EvidenceItem] = []
        seen_chunk_ids: set[str] = set()

        for hit in hits:
            payload = self._normalize_hit(hit)
            chunk_id = str(payload.get("chunk_id") or payload.get("id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            document_id = str(payload.get("document_id") or "")
            if document_id not in ready_documents:
                continue

            document = ready_documents[document_id]
            page_number = self._to_int(payload.get("page_number"))
            chunk_index = self._to_int(payload.get("chunk_index"))
            title = str(payload.get("title") or document.title or document.display_name or document.filename)
            display_name = document.display_name or document.filename
            snippet = str(payload.get("text") or "").strip()

            evidence_items.append(
                EvidenceItem(
                    id=chunk_id,
                    evidence_id=chunk_id,
                    source_type=EvidenceSourceType.LOCAL_DOCUMENT,
                    source_id=document_id,
                    title=title,
                    snippet=snippet,
                    quote=snippet,
                    citation_label=f"{display_name} p.{page_number or 0}",
                    document_id=document_id,
                    page_number=page_number,
                    score=self._distance_to_score(payload.get("distance") or payload.get("score")),
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": str(payload.get("filename") or document.filename),
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "title": title,
                        "file_path": str(payload.get("file_path") or document.file_path),
                        "source": str(payload.get("file_path") or document.file_path),
                        "source_type": "local_document",
                        "sha256": str(payload.get("sha256") or document.sha256),
                        "version": self._to_int(payload.get("version")) or document.version,
                    },
                )
            )
            if len(evidence_items) >= top_k:
                break

        return evidence_items

    def delete_document(self, document_id: str) -> None:
        client = self._get_client()
        if not client.has_collection(collection_name=self.collection_name):
            return
        client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{self._escape_value(document_id)}"',
        )

    def _ensure_collection(self, dimension: int) -> None:
        client = self._get_client()
        if client.has_collection(collection_name=self.collection_name):
            self._dimension = self._dimension or dimension
            return

        client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type="COSINE",
            primary_field_name="chunk_id",
            id_type="string",
            max_length=self.STRING_PRIMARY_KEY_MAX_LENGTH,
            vector_field_name="embedding",
            auto_id=False,
            enable_dynamic_field=True,
        )
        try:
            client.load_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._dimension = dimension

    def _get_client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        try:
            from pymilvus import MilvusClient
        except Exception as exc:  # pragma: no cover - depends on installed packages
            raise RuntimeError(
                "pymilvus is required for PaperDesk 09.5 vector retrieval. "
                "Install the updated backend dependencies before using the local library."
            ) from exc

        kwargs: dict[str, Any] = {"uri": self.uri, "db_name": self.database}
        if self.token:
            kwargs["token"] = self.token
        return MilvusClient(**kwargs)

    @staticmethod
    def _to_row(chunk: ChunkRecord, embedding: list[float]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id or chunk.id,
            "document_id": chunk.document_id,
            "filename": str(chunk.metadata.get("filename") or ""),
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title or "",
            "file_path": chunk.source or "",
            "sha256": chunk.sha256 or "",
            "version": chunk.version,
            "text": chunk.content or chunk.text,
            "embedding": embedding,
        }

    @staticmethod
    def _normalize_hit(hit: Any) -> dict[str, Any]:
        if isinstance(hit, dict):
            entity = hit.get("entity")
            if isinstance(entity, dict):
                payload = dict(entity)
                payload.setdefault("id", hit.get("id"))
                payload.setdefault("distance", hit.get("distance"))
                payload.setdefault("score", hit.get("score"))
                return payload
            return dict(hit)
        return {}

    @staticmethod
    def _distance_to_score(distance: Any) -> float | None:
        if distance is None:
            return None
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return None
        return round(1 / (1 + value), 4)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _in_filter(cls, field_name: str, values: list[str]) -> str:
        escaped = ", ".join(f'"{cls._escape_value(value)}"' for value in values)
        return f"{field_name} in [{escaped}]"

    @staticmethod
    def _escape_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
