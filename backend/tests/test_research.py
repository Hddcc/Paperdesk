from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3

import fitz

from app.agents.reading_summarizer import ReadingSummarizerAgent
from app.services.arxiv_client import ArxivClient
from app.services.embedding_service import EmbeddingService
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


def _build_pdf_bytes(text: str, *, title: str = "Library Paper") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 760),
        text,
        fontsize=11,
    )
    document.set_metadata({"title": title})
    return document.tobytes()


def _patch_research_dependencies(monkeypatch, observed_queries: list[str] | None = None) -> None:
    def fake_translate(self, query: str) -> str | None:
        if "RAG 系统中的评估方法" in query:
            return "RAG systems evaluation"
        return query

    def fake_openalex_search(self, query: str, *, limit: int):
        if observed_queries is not None:
            observed_queries.append(query)
        return [_build_openalex_work(query, 1), _build_openalex_work(query, 2)]

    def fake_arxiv_search(self, query: str, *, limit: int):
        return [_build_arxiv_entry(query)]

    monkeypatch.setattr(QueryTranslationService, "translate_to_english", fake_translate)
    monkeypatch.setattr(OpenAlexClient, "search", fake_openalex_search)
    monkeypatch.setattr(ArxivClient, "search", fake_arxiv_search)
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 0.75, 0.25] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 0.75, 0.25],
    )


def _upload_library_pdf(client) -> None:
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "library.pdf",
                BytesIO(
                    _build_pdf_bytes(
                        "RAG systems evaluation methods compare faithfulness, attribution, robustness, "
                        "and grounded answer quality across retrieval settings. " * 10,
                        title="RAG Evaluation Library",
                    )
                ),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200
    document_id = response.json()["id"]
    _wait_for_document_status(client, document_id, "ready")


def _wait_for_document_status(client, document_id: str, expected_status: str, *, timeout: float = 5.0):
    import time

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


def test_research_stream_and_report_persistence(client, monkeypatch):
    observed_queries: list[str] = []
    _patch_research_dependencies(monkeypatch, observed_queries)
    _upload_library_pdf(client)

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
    assert "Mock local evidence" not in body
    assert "library.pdf p.1" in body

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    reports = reports_response.json()
    assert len(reports) == 1

    report_response = client.get(f"/api/reports/{reports[0]['id']}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["topic"] == "RAG 系统中的评估方法"
    assert "固定多 Agent 研究工作流自动生成" in report["markdown"]
    assert "导出路径：" in report["markdown"]
    assert report["citation_items"]
    assert "任务总结仍采用教程阶段的规则模板" not in report["markdown"]
    assert "在线论文参考：" in report["markdown"]
    assert "library.pdf p.1" in report["markdown"]

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        run_row = conn.execute("SELECT id, status FROM research_runs").fetchone()
        task_rows = conn.execute("SELECT id, status FROM todo_tasks ORDER BY task_order ASC").fetchall()
        run_count = conn.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]
        task_count = conn.execute("SELECT COUNT(*) FROM todo_tasks").fetchone()[0]
        paper_count = conn.execute("SELECT COUNT(*) FROM paper_records").fetchone()[0]
        library_count = conn.execute("SELECT COUNT(*) FROM library_documents").fetchone()[0]
        report_count = conn.execute("SELECT COUNT(*) FROM report_records").fetchone()[0]
        citation_count = conn.execute("SELECT COUNT(*) FROM citation_records").fetchone()[0]
    finally:
        conn.close()

    assert run_row is not None
    assert run_count == 1
    assert task_count == 4
    assert paper_count == 12
    assert library_count == 1
    assert report_count == 1
    assert citation_count > 0
    assert run_row[1] == "completed"
    assert all(task_row[1] == "completed" for task_row in task_rows)
    assert observed_queries
    assert observed_queries[0].startswith("RAG systems evaluation")

    run_dir = Path(os.environ["WORKSPACE_DIR"]) / "runs" / run_row[0]
    assert (run_dir / "todo_tasks.json").exists()
    assert (run_dir / "task_1_summary.md").exists()
    assert (run_dir / "task_2_summary.md").exists()
    assert (run_dir / "task_3_summary.md").exists()
    assert (run_dir / "task_4_summary.md").exists()
    assert (run_dir / "final_report.md").exists()

    todo_payload = json.loads((run_dir / "todo_tasks.json").read_text(encoding="utf-8"))
    assert len(todo_payload) == 4
    assert all(item["status"] == "pending" for item in todo_payload)
    assert "在线论文参考：" in (run_dir / "task_1_summary.md").read_text(encoding="utf-8")
    assert "导出路径：" in (run_dir / "final_report.md").read_text(encoding="utf-8")


def test_research_stream_marks_run_and_task_failed_when_task_stage_breaks(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    def broken_summarize(self, task, paper_records, evidence_items):
        raise RuntimeError("summary stage exploded")

    monkeypatch.setattr(ReadingSummarizerAgent, "summarize", broken_summarize)

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    assert '"type": "error"' in body
    assert "summary stage exploded" in body
    assert '"type": "done"' not in body
    assert '"type": "final_report"' not in body

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        run_row = conn.execute("SELECT id, status FROM research_runs").fetchone()
        task_rows = conn.execute("SELECT status FROM todo_tasks ORDER BY task_order ASC").fetchall()
        report_count = conn.execute("SELECT COUNT(*) FROM report_records").fetchone()[0]
    finally:
        conn.close()

    assert run_row is not None
    assert run_row[1] == "failed"
    assert task_rows[0][0] == "failed"
    assert task_rows[1][0] == "pending"
    assert report_count == 0

    run_dir = Path(os.environ["WORKSPACE_DIR"]) / "runs" / run_row[0]
    assert (run_dir / "todo_tasks.json").exists()
    assert not (run_dir / "task_1_summary.md").exists()
    assert not (run_dir / "final_report.md").exists()
