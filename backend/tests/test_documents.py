from io import BytesIO
import os
import sqlite3

import fitz

from app.api.main import get_vectorstore
from app.services.embedding_service import EmbeddingService


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
    assert payload["title"] == "Sample Paper"
    assert len(payload["sha256"]) == 64
    assert payload["page_count"] == 1
    assert payload["status"] == "ready"
    assert payload["filename"].startswith(f"{payload['id']}_sample.pdf")
    assert payload["created_at"]

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
    assert upload_response.status_code == 500
    assert "PDF import failed" in upload_response.text
    assert "embedding model unavailable" in upload_response.text

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        rows = conn.execute("SELECT id, status FROM library_documents").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][1] == "failed"
