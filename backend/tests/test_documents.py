from datetime import datetime, timezone
from io import BytesIO
import os
import sqlite3
import time

import fitz

from app.api.main import get_vectorstore
from app.models import ChunkRecord, LibraryDocument
from app.repositories import SQLiteRepository
from app.services.document_library_service import DocumentLibraryService
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.vectorstores.milvus_store import MilvusVectorStore


def _build_pdf_bytes(text: str, *, title: str = "Sample Paper") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 760),
        text,
        fontsize=11,
    )
    document.set_metadata({"title": title})
    return document.tobytes()


def _wait_for_document_status(client, document_id: str, expected_status: str, *, timeout: float = 5.0):
    deadline = time.time() + timeout
    last_payload = None
    while time.time() < deadline:
        response = client.get("/api/documents")
        assert response.status_code == 200
        documents = response.json()
        for item in documents:
            if item["id"] == document_id:
                last_payload = item
                if item["status"] == expected_status:
                    return item
        time.sleep(0.05)
    raise AssertionError(f"Document {document_id} did not reach status {expected_status}: {last_payload}")


def test_document_upload_list_delete_flow(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 1.0, 0.5],
    )

    pdf_bytes = _build_pdf_bytes(
        "This sample paper discusses retrieval augmented generation evaluation strategies. " * 12,
        title="Sample Paper",
    )
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["display_name"] == "sample.pdf"
    assert len(payload["sha256"]) == 64
    assert payload["status"] in {"processing", "ready"}
    assert payload["filename"].startswith(f"{payload['id']}_sample.pdf")
    assert payload["created_at"]

    ready_payload = _wait_for_document_status(client, payload["id"], "ready")
    assert ready_payload["title"] == "Sample Paper"
    assert ready_payload["page_count"] == 1
    assert ready_payload["parser_status"] == "indexed"
    assert ready_payload["indexed_at"]
    assert ready_payload["version"] == 1

    duplicate_response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", BytesIO(pdf_bytes), "application/pdf")},
    )
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["id"] == payload["id"]

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    documents = list_response.json()
    assert len(documents) == 1
    assert documents[0]["id"] == payload["id"]
    assert documents[0]["sha256"] == payload["sha256"]
    assert documents[0]["status"] == "ready"

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        library_count = conn.execute("SELECT COUNT(*) FROM library_documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM library_chunks").fetchone()[0]
    finally:
        conn.close()
    assert library_count == 1
    assert chunk_count > 0
    assert get_vectorstore()._get_client().has_collection(collection_name=get_vectorstore().collection_name)

    delete_response = client.delete(f"/api/documents/{payload['id']}")
    assert delete_response.status_code == 200

    list_again_response = client.get("/api/documents")
    assert list_again_response.status_code == 200
    assert list_again_response.json() == []
    assert get_vectorstore().query_evidence("sample", [], top_k=1) == []


def test_document_upload_accepts_second_different_pdf_without_connection_break(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 1.0, 0.5],
    )

    first = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "first.pdf",
                BytesIO(_build_pdf_bytes("first paper text " * 20, title="First Paper")),
                "application/pdf",
            )
        },
    )
    second = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "second.pdf",
                BytesIO(_build_pdf_bytes("second paper text " * 20, title="Second Paper")),
                "application/pdf",
            )
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = _wait_for_document_status(client, first.json()["id"], "ready")
    second_payload = _wait_for_document_status(client, second.json()["id"], "ready")
    assert first_payload["display_name"] == "first.pdf"
    assert second_payload["display_name"] == "second.pdf"

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    documents = list_response.json()
    assert len(documents) == 2
    assert {item["status"] for item in documents} == {"ready"}


def test_document_upload_marks_failed_when_embedding_generation_breaks(client, monkeypatch):
    def fake_embed_failure(self, texts):
        _ = texts
        raise RuntimeError("embedding model unavailable")

    monkeypatch.setattr(EmbeddingService, "embed_texts", fake_embed_failure)

    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "failed.pdf",
                BytesIO(
                    _build_pdf_bytes(
                        "This PDF still parses, but embedding generation will fail.",
                        title="Broken Embed",
                    )
                ),
                "application/pdf",
            )
        },
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["status"] in {"processing", "failed"}
    failed_payload = _wait_for_document_status(client, payload["id"], "failed")
    assert failed_payload["page_count"] == 1
    assert failed_payload["title"] == "Broken Embed"
    assert "嵌入生成失败" in failed_payload["failure_reason"]

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        rows = conn.execute("SELECT id, status FROM library_documents").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][1] == "failed"


def test_document_upload_surfaces_milvus_connection_failure_reason(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)],
    )

    monkeypatch.setattr(
        MilvusVectorStore,
        "_create_client",
        lambda self: (_ for _ in ()).throw(
            RuntimeError("Fail connecting to server on 127.0.0.1:19530, server unavailable")
        ),
    )

    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "milvus-down.pdf",
                BytesIO(
                    _build_pdf_bytes(
                        "This document is valid, but the Milvus connection will fail.",
                        title="Milvus Down",
                    )
                ),
                "application/pdf",
            )
        },
    )
    assert upload_response.status_code == 200
    failed_payload = _wait_for_document_status(client, upload_response.json()["id"], "failed")
    assert failed_payload["page_count"] == 1
    assert failed_payload["title"] == "Milvus Down"
    assert "Milvus 服务不可用" in failed_payload["failure_reason"]


def test_document_reupload_with_same_name_and_new_content_reindexes_in_place(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 0.9, 0.1] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 0.9, 0.1],
    )

    first_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "iterative.pdf",
                BytesIO(_build_pdf_bytes("first revision " * 24, title="Iterative Study")),
                "application/pdf",
            )
        },
    )
    assert first_upload.status_code == 200
    first_payload = _wait_for_document_status(client, first_upload.json()["id"], "ready")

    second_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "iterative.pdf",
                BytesIO(_build_pdf_bytes("second revision with more evidence " * 24, title="Iterative Study")),
                "application/pdf",
            )
        },
    )
    assert second_upload.status_code == 200
    second_payload = _wait_for_document_status(client, second_upload.json()["id"], "ready")

    assert second_payload["id"] == first_payload["id"]
    assert second_payload["version"] == first_payload["version"] + 1
    assert second_payload["sha256"] != first_payload["sha256"]

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        row = conn.execute(
            "SELECT COUNT(*), MAX(version) FROM library_chunks WHERE document_id = ?",
            (second_payload["id"],),
        ).fetchone()
    finally:
        conn.close()

    assert row[0] > 0
    assert row[1] == second_payload["version"]


def test_milvus_store_adds_and_queries_chunks(sandbox_dir, monkeypatch):
    class FakeEmbeddingService:
        def embed_texts(self, texts):
            return [[float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)]

        def embed_query(self, query):
            _ = query
            return [1.0, 0.5, 0.25]

    store = MilvusVectorStore(
        uri="http://fake-milvus:19530",
        token=None,
        database="paperdesk_test",
        collection_name="paperdesk_collection",
        embedding_service=FakeEmbeddingService(),
    )
    document = LibraryDocument(
        id="doc-1",
        filename="doc-1.pdf",
        display_name="doc-1.pdf",
        title="Doc 1",
        file_path=str(sandbox_dir / "doc-1.pdf"),
        sha256="x" * 64,
        page_count=1,
        status="ready",
        parser_status="indexed",
        version=1,
        created_at=datetime.now(timezone.utc),
        uploaded_at=datetime.now(timezone.utc),
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id=document.id,
            page_number=1,
            chunk_index=0,
            text="sample text",
            title="Doc 1",
            sha256=document.sha256,
            version=1,
            metadata={
                "document_id": document.id,
                "filename": document.display_name,
                "page_number": 1,
                "chunk_index": 0,
                "title": document.title,
                "file_path": document.file_path,
                "sha256": document.sha256,
                "version": 1,
            },
        )
    ]

    store.add_chunks(chunks)
    results = store.query_evidence("sample", [document], top_k=1)
    assert len(results) == 1
    assert results[0].document_id == document.id


def test_milvus_store_creates_string_primary_key_with_max_length():
    class FakeEmbeddingService:
        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_query(self, query):
            _ = query
            return [0.1, 0.2, 0.3]

    class InspectingClient:
        def __init__(self) -> None:
            self.created_kwargs = None

        def has_collection(self, *, collection_name: str) -> bool:
            _ = collection_name
            return False

        def create_collection(self, **kwargs) -> None:
            self.created_kwargs = kwargs

        def load_collection(self, *, collection_name: str) -> None:
            _ = collection_name

        def delete(self, *, collection_name: str, filter: str) -> None:
            _ = (collection_name, filter)

        def insert(self, *, collection_name: str, data: list[dict]) -> None:
            _ = (collection_name, data)

    client = InspectingClient()
    store = MilvusVectorStore(
        uri="http://fake-milvus:19530",
        token=None,
        database="paperdesk_test",
        collection_name="paperdesk_collection",
        embedding_service=FakeEmbeddingService(),
    )
    store._client = client

    store.add_chunks(
        [
            ChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                page_number=1,
                chunk_index=0,
                text="sample text",
                title="Doc 1",
                sha256="x" * 64,
                version=1,
                metadata={},
            )
        ]
    )

    assert client.created_kwargs is not None
    assert client.created_kwargs["id_type"] == "string"
    assert client.created_kwargs["max_length"] == MilvusVectorStore.STRING_PRIMARY_KEY_MAX_LENGTH


def test_list_documents_reschedules_orphaned_processing_documents(sandbox_dir, monkeypatch):
    data_dir = sandbox_dir / "data"
    repo = SQLiteRepository(data_dir / "paperdesk.db")
    upload_dir = sandbox_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / "orphan.pdf"
    file_path.write_bytes(_build_pdf_bytes("orphan processing doc", title="Orphan"))

    document = LibraryDocument(
        id="doc-orphan",
        filename="doc-orphan_orphan.pdf",
        display_name="orphan.pdf",
        title="Orphan",
        file_path=str(file_path),
        sha256="a" * 64,
        page_count=1,
        status="processing",
        parser_status="pending",
        version=1,
        created_at=datetime.now(timezone.utc),
        uploaded_at=datetime.now(timezone.utc),
    )
    repo.library.create_document(document)

    service = DocumentLibraryService(
        repository=repo.library,
        vectorstore=get_vectorstore(),
        upload_dir=upload_dir,
        ingestion_service=KnowledgeIngestionService(
            document_repository=repo.library,
            chunk_repository=repo.chunk,
            vectorstore=get_vectorstore(),
        ),
    )

    scheduled: list[str] = []
    monkeypatch.setattr(service, "_schedule_processing", lambda document_id: scheduled.append(document_id))

    listed = service.list_documents()

    assert listed[0].id == document.id
    assert scheduled == [document.id]


def test_delete_document_ignores_vectorstore_cleanup_failures(sandbox_dir):
    data_dir = sandbox_dir / "data"
    repo = SQLiteRepository(data_dir / "paperdesk.db")
    upload_dir = sandbox_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / "delete-me.pdf"
    file_path.write_bytes(_build_pdf_bytes("delete me", title="Delete Me"))

    document = LibraryDocument(
        id="doc-delete",
        filename="doc-delete_delete-me.pdf",
        display_name="delete-me.pdf",
        title="Delete Me",
        file_path=str(file_path),
        sha256="b" * 64,
        page_count=1,
        status="processing",
        parser_status="pending",
        version=1,
        created_at=datetime.now(timezone.utc),
        uploaded_at=datetime.now(timezone.utc),
    )
    repo.library.create_document(document)

    class FailingVectorStore:
        def delete_document(self, document_id: str) -> None:
            _ = document_id
            raise RuntimeError("vector cleanup failed")

    service = DocumentLibraryService(
        repository=repo.library,
        vectorstore=FailingVectorStore(),
        upload_dir=upload_dir,
        ingestion_service=KnowledgeIngestionService(
            document_repository=repo.library,
            chunk_repository=repo.chunk,
            vectorstore=get_vectorstore(),
        ),
    )

    deleted = service.delete_document(document.id)

    assert deleted.id == document.id
    assert repo.library.get_document(document.id) is None
