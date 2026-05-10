from __future__ import annotations

from app.services.arxiv_client import ArxivClient
from app.services.openalex_client import OpenAlexClient
from app.services.query_translation_service import QueryTranslationService


def test_paper_search_route_aggregates_and_deduplicates(client, monkeypatch):
    def fake_openalex_search(self, query: str, *, limit: int):
        return [
            {
                "id": "https://openalex.org/W100",
                "display_name": "RAG Systems Evaluation",
                "authorships": [
                    {"author": {"display_name": "Alice"}},
                    {"author": {"display_name": "Bob"}},
                ],
                "abstract_inverted_index": {"RAG": [0], "systems": [1], "evaluation": [2]},
                "publication_year": 2025,
                "doi": "https://doi.org/10.1000/rag-eval",
                "primary_location": {
                    "landing_page_url": "https://openalex.example/paper-1",
                    "source": {"display_name": "ACL"},
                },
            },
            {
                "id": "https://openalex.org/W101",
                "display_name": "Benchmarking RAG Pipelines",
                "authorships": [{"author": {"display_name": "Carol"}}],
                "abstract_inverted_index": {"benchmarking": [0], "rag": [1], "pipelines": [2]},
                "publication_year": 2024,
                "doi": None,
                "primary_location": {
                    "landing_page_url": "https://openalex.example/paper-2",
                    "source": {"display_name": "EMNLP"},
                },
            },
        ]

    def fake_arxiv_search(self, query: str, *, limit: int):
        return [
            {
                "id": "http://arxiv.org/abs/2401.00001v1",
                "title": "RAG Systems Evaluation",
                "authors": ["Dan"],
                "summary": "Duplicate by DOI",
                "published": "2024-01-01T00:00:00Z",
                "doi": "10.1000/rag-eval",
                "url": "https://arxiv.org/abs/2401.00001",
            },
            {
                "id": "http://arxiv.org/abs/2402.00002v1",
                "title": "RAG Pipeline Diagnostics",
                "authors": ["Eve"],
                "summary": "Fresh arXiv-only evidence",
                "published": "2024-02-01T00:00:00Z",
                "doi": None,
                "url": "https://arxiv.org/abs/2402.00002",
            },
        ]

    monkeypatch.setattr(OpenAlexClient, "search", fake_openalex_search)
    monkeypatch.setattr(ArxivClient, "search", fake_arxiv_search)

    response = client.post(
        "/api/papers/search",
        json={"topic": "RAG systems evaluation", "top_k_online": 5},
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert len(items) == 3
    assert items[0]["title"] == "RAG Systems Evaluation"
    assert items[0]["doi"] == "10.1000/rag-eval"
    assert items[0]["source"] == "openalex"
    assert items[0]["abstract"] == "RAG systems evaluation"
    assert {item["source"] for item in items} == {"openalex", "arxiv"}


def test_paper_search_route_supports_provider_filter_and_partial_failure(client, monkeypatch):
    def fake_openalex_search(self, query: str, *, limit: int):
        return [
            {
                "id": "https://openalex.org/W200",
                "display_name": "Filtered OpenAlex Result",
                "authorships": [{"author": {"display_name": "Filtered Author"}}],
                "abstract_inverted_index": {"filtered": [0], "result": [1]},
                "publication_year": 2026,
                "doi": None,
                "primary_location": {
                    "landing_page_url": "https://openalex.example/paper-3",
                    "source": {"display_name": "NeurIPS"},
                },
            }
        ]

    def failing_arxiv_search(self, query: str, *, limit: int):
        raise RuntimeError("simulated arXiv outage")

    monkeypatch.setattr(OpenAlexClient, "search", fake_openalex_search)
    monkeypatch.setattr(ArxivClient, "search", failing_arxiv_search)

    filtered_response = client.post(
        "/api/papers/search",
        json={
            "topic": "agentic retrieval",
            "search_provider": "openalex",
            "top_k_online": 5,
        },
    )
    assert filtered_response.status_code == 200
    filtered_items = filtered_response.json()["items"]
    assert len(filtered_items) == 1
    assert filtered_items[0]["source"] == "openalex"

    partial_response = client.post(
        "/api/papers/search",
        json={"topic": "agentic retrieval", "top_k_online": 5},
    )
    assert partial_response.status_code == 200
    partial_items = partial_response.json()["items"]
    assert len(partial_items) == 1
    assert partial_items[0]["title"] == "Filtered OpenAlex Result"


def test_paper_search_route_rejects_invalid_provider(client):
    response = client.post(
        "/api/papers/search",
        json={"topic": "agentic retrieval", "search_provider": "semantic-scholar"},
    )
    assert response.status_code == 422


def test_query_translation_service_uses_glossary_without_llm_credentials():
    service = QueryTranslationService(model="test-model", api_key=None, base_url=None)
    assert service.translate_to_english("人脸超分") == "face super-resolution"


def test_paper_search_route_falls_back_to_translated_query_for_chinese_topic(client, monkeypatch):
    observed_queries: list[str] = []

    def fake_translate(self, query: str) -> str | None:
        assert query == "人脸超分"
        return "face super-resolution"

    def fake_openalex_search(self, query: str, *, limit: int):
        observed_queries.append(query)
        if query.startswith("face super-resolution"):
            return [
                {
                    "id": "https://openalex.org/W300",
                    "display_name": "Deep Learning-based Face Super-resolution: A Survey",
                    "authorships": [{"author": {"display_name": "Survey Author"}}],
                    "abstract_inverted_index": {"face": [0], "super-resolution": [1]},
                    "publication_year": 2021,
                    "doi": "10.1145/3485132",
                    "primary_location": {
                        "landing_page_url": "https://openalex.example/fsr-survey",
                        "source": {"display_name": "ACM Computing Surveys"},
                    },
                }
            ]
        return []

    def fake_arxiv_search(self, query: str, *, limit: int):
        observed_queries.append(f"arxiv:{query}")
        return []

    monkeypatch.setattr(QueryTranslationService, "translate_to_english", fake_translate)
    monkeypatch.setattr(OpenAlexClient, "search", fake_openalex_search)
    monkeypatch.setattr(ArxivClient, "search", fake_arxiv_search)

    response = client.post(
        "/api/papers/search",
        json={"topic": "人脸超分", "top_k_online": 5},
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["title"] == "Deep Learning-based Face Super-resolution: A Survey"
    assert observed_queries[0] == "face super-resolution"
