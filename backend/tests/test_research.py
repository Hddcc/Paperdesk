from io import BytesIO
import os
import sqlite3

from app.services.arxiv_client import ArxivClient
from app.services.openalex_client import OpenAlexClient
from app.services.query_translation_service import QueryTranslationService


def _build_openalex_work(query: str, index: int) -> dict:
    return {
        "id": f"https://openalex.org/W{index}{abs(hash(query)) % 10000}",
        "display_name": f"{query} OpenAlex Paper {index}",
        "authorships": [
            {"author": {"display_name": f"OpenAlex Author {index}"}},
        ],
        "abstract_inverted_index": {
            query: [0],
            "study": [1],
            str(index): [2],
        },
        "publication_year": 2026 - index,
        "doi": f"https://doi.org/10.1000/{abs(hash(query)) % 1000}.{index}",
        "primary_location": {
            "landing_page_url": f"https://openalex.example/{abs(hash(query)) % 10000}/{index}",
            "source": {"display_name": "OpenAlex Venue"},
        },
    }


def _build_arxiv_entry(query: str) -> dict:
    seed = abs(hash(query)) % 10000
    return {
        "id": f"http://arxiv.org/abs/2605.{seed:04d}v1",
        "title": f"{query} arXiv Evidence",
        "authors": ["arXiv Author"],
        "summary": f"{query} arXiv summary",
        "published": "2026-05-10T00:00:00Z",
        "doi": None,
        "url": f"https://arxiv.org/abs/2605.{seed:04d}",
    }


def test_research_stream_and_report_persistence(client, monkeypatch):
    observed_queries: list[str] = []

    def fake_translate(self, query: str) -> str | None:
        if "RAG 系统中的评估方法" in query:
            return "RAG systems evaluation"
        return query

    def fake_openalex_search(self, query: str, *, limit: int):
        observed_queries.append(query)
        return [_build_openalex_work(query, 1), _build_openalex_work(query, 2)]

    def fake_arxiv_search(self, query: str, *, limit: int):
        return [_build_arxiv_entry(query)]

    monkeypatch.setattr(QueryTranslationService, "translate_to_english", fake_translate)
    monkeypatch.setattr(OpenAlexClient, "search", fake_openalex_search)
    monkeypatch.setattr(ArxivClient, "search", fake_arxiv_search)

    client.post(
        "/api/documents/upload",
        files={"file": ("library.pdf", BytesIO(b"%PDF-1.4 library"), "application/pdf")},
    )

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    assert '"type": "run_created"' in body
    assert '"type": "todo_list"' in body
    assert '"type": "final_report"' in body
    assert '"type": "done"' in body
    assert "任务总结仍采用教程阶段的规则模板" not in body
    assert "在线论文参考：" in body
    assert "[1]" in body
    assert "OpenAlex Paper 1." in body

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    reports = reports_response.json()
    assert len(reports) == 1

    report_response = client.get(f"/api/reports/{reports[0]['id']}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["topic"] == "RAG 系统中的评估方法"
    assert "当前阶段已接入真实在线论文检索" in report["markdown"]
    assert "导出路径：" in report["markdown"]
    assert report["citation_items"]
    assert "任务总结仍采用教程阶段的规则模板" not in report["markdown"]
    assert "在线论文参考：" in report["markdown"]

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        run_count = conn.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]
        task_count = conn.execute("SELECT COUNT(*) FROM todo_tasks").fetchone()[0]
        paper_count = conn.execute("SELECT COUNT(*) FROM paper_records").fetchone()[0]
        library_count = conn.execute("SELECT COUNT(*) FROM library_documents").fetchone()[0]
        report_count = conn.execute("SELECT COUNT(*) FROM report_records").fetchone()[0]
        citation_count = conn.execute("SELECT COUNT(*) FROM citation_records").fetchone()[0]
    finally:
        conn.close()

    assert run_count == 1
    assert task_count == 4
    assert paper_count == 12
    assert library_count == 1
    assert report_count == 1
    assert citation_count > 0
    assert observed_queries
    assert observed_queries[0].startswith("RAG systems evaluation")
