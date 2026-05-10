from datetime import datetime, timezone
from io import BytesIO
import os
import sqlite3
import time

import chromadb
import fitz

from app.api.main import get_vectorstore
from app.models import ChunkRecord, LibraryDocument
from app.services.embedding_service import EmbeddingService
from app.vectorstores.chroma_store import ChromaVectorStore


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
    finally:
        conn.close()
    assert library_count == 1
    assert get_vectorstore()._get_collection().count() > 0

    delete_response = client.delete(f"/api/documents/{payload['id']}")
    assert delete_response.status_code == 200

    list_again_response = client.get("/api/documents")
    assert list_again_response.status_code == 200
    assert list_again_response.json() == []
    assert get_vectorstore()._get_collection().count() == 0


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
    assert failed_payload["page_count"] == 0

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        rows = conn.execute("SELECT id, status FROM library_documents").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][1] == "failed"


def test_chroma_store_falls_back_to_ephemeral_when_persistent_client_breaks(sandbox_dir, monkeypatch):
    class FakeEmbeddingService:
        def embed_texts(self, texts):
            return [[float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)]

        def embed_query(self, query):
            _ = query
            return [1.0, 0.5, 0.25]

    monkeypatch.setattr(
        "app.vectorstores.chroma_store.chromadb.PersistentClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk I/O error")),
    )
    monkeypatch.setattr(
        "app.vectorstores.chroma_store.chromadb.EphemeralClient",
        chromadb.EphemeralClient,
    )

    store = ChromaVectorStore(sandbox_dir / "chroma", FakeEmbeddingService())
    document = LibraryDocument(
        id="doc-1",
        filename="doc-1.pdf",
        display_name="doc-1.pdf",
        title="Doc 1",
        file_path=str(sandbox_dir / "doc-1.pdf"),
        sha256="x" * 64,
        page_count=1,
        status="ready",
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
            metadata={
                "document_id": document.id,
                "filename": document.display_name,
                "page_number": 1,
                "chunk_index": 0,
                "title": document.title,
                "file_path": document.file_path,
            },
        )
    ]

    store.add_chunks(chunks)
    assert store._client_mode == "ephemeral"
    results = store.query_evidence("sample", [document], top_k=1)
    assert len(results) == 1
