from io import BytesIO
import time

import fitz

from app.services.arxiv_client import ArxivClient
from app.services.embedding_service import EmbeddingService
from app.services.openalex_client import OpenAlexClient
from app.services.query_translation_service import QueryTranslationService


def _build_pdf_bytes(text: str, *, title: str) -> bytes:
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
    while time.time() < deadline:
        response = client.get("/api/documents")
        assert response.status_code == 200
        documents = response.json()
        for item in documents:
            if item["id"] == document_id and item["status"] == expected_status:
                return item
        time.sleep(0.05)
    raise AssertionError(f"Document {document_id} did not reach status {expected_status}")


def _upload_document(client, name: str, title: str, text: str) -> dict:
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                name,
                BytesIO(_build_pdf_bytes(text, title=title)),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200
    return _wait_for_document_status(client, response.json()["id"], "ready")


def test_rag_ask_returns_grounded_answer(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 0.8, 0.2] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 0.8, 0.2] if "评估" in query else [1.0, 0.2, 0.8],
    )
    monkeypatch.setattr(
        QueryTranslationService,
        "translate_to_english",
        lambda self, query: "retrieval evaluation methods" if "评估" in query else query,
    )

    document = _upload_document(
        client,
        "rag-eval.pdf",
        "RAG Evaluation",
        "This paper studies retrieval evaluation methods, grounding quality, attribution, and faithfulness. "
        * 10,
    )

    response = client.post(
        "/api/rag/ask",
        json={
            "question": "这篇论文如何讨论评估方法？",
            "document_ids": [document["id"]],
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["sources"] == ["rag-eval.pdf"]
    assert payload["pages"] == [1]
    assert payload["retrieval_count"] >= 1
    assert payload["evidence_items"][0]["document_id"] == document["id"]


def test_paper_analysis_supports_single_and_compare_modes(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 0.7, 0.3] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 0.7, 0.3],
    )

    first = _upload_document(
        client,
        "first.pdf",
        "First Paper",
        "This paper focuses on retrieval grounding, task decomposition, and evaluation metrics. " * 8,
    )
    second = _upload_document(
        client,
        "second.pdf",
        "Second Paper",
        "This paper focuses on citation quality, robustness, and benchmark comparison. " * 8,
    )

    single_response = client.post(
        "/api/papers/analyze",
        json={"document_ids": [first["id"]], "mode": "single"},
    )
    assert single_response.status_code == 200
    single_payload = single_response.json()
    assert single_payload["mode"] == "single"
    assert single_payload["answer"]
    assert single_payload["sections"]
    assert single_payload["retrieval_count"] >= 1

    compare_response = client.post(
        "/api/papers/analyze",
        json={"document_ids": [first["id"], second["id"]], "mode": "compare"},
    )
    assert compare_response.status_code == 200
    compare_payload = compare_response.json()
    assert compare_payload["mode"] == "compare"
    assert compare_payload["answer"]
    assert compare_payload["sections"]
    assert compare_payload["retrieval_count"] >= 1


def test_paper_curation_returns_recommendations(client, monkeypatch):
    monkeypatch.setattr(
        QueryTranslationService,
        "translate_to_english",
        lambda self, query: "retrieval augmented generation evaluation",
    )
    monkeypatch.setattr(
        OpenAlexClient,
        "search",
        lambda self, query, *, limit: [
            {
                "id": "https://openalex.org/W1",
                "display_name": "Grounded RAG Evaluation",
                "authorships": [{"author": {"display_name": "Author One"}}],
                "abstract_inverted_index": {"grounded": [0], "rag": [1], "evaluation": [2]},
                "publication_year": 2025,
                "doi": "https://doi.org/10.1000/rag-eval",
                "primary_location": {
                    "landing_page_url": "https://openalex.example/rag-eval",
                    "source": {"display_name": "OpenAlex Venue"},
                },
            }
        ],
    )
    monkeypatch.setattr(
        ArxivClient,
        "search",
        lambda self, query, *, limit: [],
    )

    response = client.post(
        "/api/papers/curate",
        json={"topic": "RAG 评估", "search_provider": "openalex", "top_k_online": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"]
    assert payload["items"][0]["paper"]["title"] == "Grounded RAG Evaluation"
    assert payload["items"][0]["decision"] in {"recommended", "consider"}
    assert payload["items"][0]["reason"]
