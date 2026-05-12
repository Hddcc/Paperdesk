from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3

import fitz

from app.agents.report_writer import ReportWriterAgent
from app.agents.reading_summarizer import ReadingSummarizerAgent
from app.models import EvidenceItem, PaperRecord, TaskSummary
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


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        raw_event = block.strip()
        if not raw_event.startswith("data:"):
            continue
        payload_text = raw_event[5:].strip()
        if payload_text:
            events.append(json.loads(payload_text))
    return events


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
    events = _parse_sse_events(body)
    event_types = [event["type"] for event in events]
    assert "status" in event_types
    assert "coordinator_status" in event_types
    assert "todo_list" in event_types
    assert "task_status" in event_types
    assert "subagent_spawned" in event_types
    assert "subagent_started" in event_types
    assert "subagent_completed" in event_types
    assert "task_merge_started" in event_types
    assert "task_merge_completed" in event_types
    assert "task_summary" in event_types
    assert "report_completed" in event_types
    assert "report" in event_types
    assert "done" in event_types
    assert "任务总结仍采用教程阶段的规则模板" not in body
    assert "在线论文参考：" in body
    assert "[1]" in body
    assert "OpenAlex Paper 1." in body
    assert "Mock local evidence" not in body
    assert "library.pdf p.1" in body

    first_status = next(event for event in events if event["type"] == "status")
    run_id = first_status["run_id"]
    report_event = next(event for event in events if event["type"] == "report")
    report_id = report_event["report_id"]

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    reports = reports_response.json()
    assert len(reports) == 1

    report_response = client.get(f"/api/reports/{reports[0]['id']}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["topic"] == "RAG 系统中的评估方法"
    assert "## 1. 研究概览" in report["markdown"]
    assert "## 2. 分任务总结整合" in report["markdown"]
    assert "## 3. 关键观点归纳" in report["markdown"]
    assert "## 4. 局限与待补充问题" in report["markdown"]
    assert "## 参考来源" in report["markdown"]
    assert "### 2.1 " in report["markdown"]
    assert "### 2.4 " in report["markdown"]
    assert "固定多 Agent 研究工作流自动生成" not in report["markdown"]
    assert "导出路径：" not in report["markdown"]
    assert report["citation_items"]
    assert "任务总结仍采用教程阶段的规则模板" not in report["markdown"]
    assert "在线论文参考：" not in report["markdown"]
    assert "Local PDF: library.pdf, p.1." in report["markdown"]
    assert report["citations"]
    assert report["citations"][0].startswith("[1] ")
    assert any(line.startswith("[") for line in report["citations"])

    run_response = client.get(f"/api/research/{run_id}")
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["run"]["id"] == run_id
    assert run_payload["run"]["status"] == "completed"
    assert len(run_payload["tasks"]) == 4
    assert len(run_payload["task_summaries"]) == 4
    assert len(run_payload["subagent_tasks"]) == 8
    assert len(run_payload["task_notifications"]) == 8
    assert len(run_payload["task_traces"]) >= 8
    assert len(run_payload["task_artifacts"]) >= 8
    assert run_payload["report"]["id"] == report_id

    export_response = client.get(f"/api/export/{report_id}")
    assert export_response.status_code == 200
    assert export_response.headers["content-disposition"] == (
        f'attachment; filename="{report_id}.md"'
    )
    assert "text/markdown" in export_response.headers["content-type"]
    assert export_response.text == report["markdown"]

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
        subagent_task_count = conn.execute("SELECT COUNT(*) FROM subagent_tasks").fetchone()[0]
        notification_count = conn.execute("SELECT COUNT(*) FROM task_notifications").fetchone()[0]
        artifact_count = conn.execute("SELECT COUNT(*) FROM task_artifacts").fetchone()[0]
        trace_count = conn.execute("SELECT COUNT(*) FROM task_execution_traces").fetchone()[0]
    finally:
        conn.close()

    assert run_row is not None
    assert run_count == 1
    assert task_count == 4
    assert paper_count == 12
    assert library_count == 1
    assert report_count == 1
    assert citation_count > 0
    assert subagent_task_count == 8
    assert notification_count == 8
    assert artifact_count >= 16
    assert trace_count >= 16
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
    final_report_markdown = (run_dir / "final_report.md").read_text(encoding="utf-8").strip()
    assert len(todo_payload) == 4
    assert all(item["status"] == "pending" for item in todo_payload)
    assert "在线论文参考：" in (run_dir / "task_1_summary.md").read_text(encoding="utf-8")
    assert final_report_markdown == export_response.text.strip()

def test_irrelevant_local_document_is_not_used_as_evidence(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)

    unrelated_upload = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "unrelated.pdf",
                BytesIO(
                    _build_pdf_bytes(
                        "This document discusses portrait and video synthesis using identity-aware multimodal generation. "
                        * 8,
                        title="Identity-aware Generation",
                    )
                ),
                "application/pdf",
            )
        },
    )
    assert unrelated_upload.status_code == 200
    _wait_for_document_status(client, unrelated_upload.json()["id"], "ready")

    response = client.post(
        "/api/research/stream",
        json={"topic": "生物医学中的大语言模型应用"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    report_event = next(event for event in events if event["type"] == "report")
    report_payload = report_event["report"]
    report_markdown = report_payload["markdown"]

    assert "## 4. 局限与待补充问题" in report_markdown
    assert "本轮未检索到可直接采用的本地 PDF 证据" in report_markdown
    assert "Local PDF:" not in report_markdown
    assert "unrelated.pdf" not in report_markdown

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    report_id = reports_response.json()[0]["id"]
    report_response = client.get(f"/api/reports/{report_id}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert "本轮未检索到可直接采用的本地 PDF 证据" in report["markdown"]
    assert "Local PDF:" not in report["markdown"]
    assert "unrelated.pdf" not in report["markdown"]


def test_report_can_be_deleted_from_history(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    report_event = next(event for event in events if event["type"] == "report")
    report_id = report_event["report_id"]

    report_path = Path(os.environ["REPORT_DIR"]) / f"{report_id}.md"
    assert report_path.exists()

    delete_response = client.delete(f"/api/reports/{report_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == report_id

    list_response = client.get("/api/reports")
    assert list_response.status_code == 200
    assert list_response.json() == []

    get_response = client.get(f"/api/reports/{report_id}")
    assert get_response.status_code == 404

    export_response = client.get(f"/api/export/{report_id}")
    assert export_response.status_code == 404

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        report_count = conn.execute("SELECT COUNT(*) FROM report_records").fetchone()[0]
        citation_count = conn.execute("SELECT COUNT(*) FROM citation_records").fetchone()[0]
    finally:
        conn.close()

    assert report_count == 0
    assert citation_count == 0
    assert not report_path.exists()


def test_legacy_technical_note_is_hidden_in_report_preview(client):
    from app.api.main import get_repository

    get_repository()
    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        conn.execute(
            """
            INSERT INTO research_runs (id, topic, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-legacy",
                "Legacy Topic",
                "completed",
                "2026-05-10T00:00:00+00:00",
                "2026-05-10T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO report_records (id, run_id, topic, markdown, citations_text, task_summaries_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "report-legacy",
                "run-legacy",
                "Legacy Topic",
                "# Legacy Topic\n\n- 说明：当前实现已接入在线论文检索；任务总结仍采用教程阶段的规则模板，用于验证多 Agent 职责边界、状态流、SQLite 持久化与 SSE 展示。\n\n正文保留。",
                "",
                json.dumps(
                    [
                        {
                            "task_id": "task-1",
                            "title": "任务一",
                            "intent": "示例",
                            "summary": "- 说明：当前实现已接入在线论文检索；任务总结仍采用教程阶段的规则模板，用于验证多 Agent 职责边界、状态流、SQLite 持久化与 SSE 展示。\n\n任务正文。",
                            "summary_markdown": "- 说明：当前实现已接入在线论文检索；任务总结仍采用教程阶段的规则模板，用于验证多 Agent 职责边界、状态流、SQLite 持久化与 SSE 展示。\n\n任务正文。",
                            "evidence_items": [],
                            "paper_records": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                "2026-05-10T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    response = client.get("/api/reports/report-legacy")
    assert response.status_code == 200
    payload = response.json()
    assert "当前实现已接入在线论文检索" not in payload["markdown"]
    assert "SQLite 持久化" not in payload["markdown"]
    assert "正文保留。" in payload["markdown"]
    assert "当前实现已接入在线论文检索" not in payload["task_summaries"][0]["summary"]
    assert "任务正文。" in payload["task_summaries"][0]["summary"]


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
    events = _parse_sse_events(body)
    event_types = [event["type"] for event in events]
    assert "error" in event_types
    assert "summary stage exploded" in body
    assert "done" not in event_types
    assert "report" not in event_types

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


def test_report_writer_uses_fixed_structure_and_deduplicated_citations():
    shared_paper = PaperRecord(
        title="Shared Paper",
        authors=["Author One", "Author Two"],
        year=2025,
        doi="10.1000/shared",
        url="https://example.com/shared",
        source_type="online_paper",
    )
    shared_evidence = EvidenceItem(
        id="evidence-1",
        source_type="local_document",
        source_id="doc-1",
        title="shared.pdf",
        quote="important quote",
        citation_label="shared.pdf p.3",
        document_id="doc-1",
        page_number=3,
        metadata={"filename": "shared.pdf", "document_id": "doc-1"},
    )
    task_summaries = [
        TaskSummary(
            task_id="task-1",
            title="代表方法",
            intent="比较代表性方法",
            summary="任务聚焦：比较代表性方法\n\n- 在线论文结果：1 条\n- 本地证据结果：1 条\n\n在线论文参考：\n[1] Shared Paper\n本地证据引用：shared.pdf p.3",
            summary_markdown="任务聚焦：比较代表性方法\n\n- 在线论文结果：1 条\n- 本地证据结果：1 条\n\n在线论文参考：\n[1] Shared Paper\n本地证据引用：shared.pdf p.3",
            paper_records=[shared_paper],
            evidence_items=[shared_evidence],
        ),
        TaskSummary(
            task_id="task-2",
            title="评估基准",
            intent="梳理评估设置",
            summary="任务聚焦：梳理评估设置\n\n- 在线论文结果：1 条\n- 本地证据结果：1 条",
            summary_markdown="任务聚焦：梳理评估设置\n\n- 在线论文结果：1 条\n- 本地证据结果：1 条",
            paper_records=[shared_paper.model_copy(deep=True)],
            evidence_items=[shared_evidence.model_copy(deep=True)],
        ),
    ]

    report = ReportWriterAgent().write("RAG 评估", task_summaries)

    assert "## 1. 研究概览" in report.markdown
    assert "## 2. 分任务总结整合" in report.markdown
    assert "## 3. 关键观点归纳" in report.markdown
    assert "## 4. 局限与待补充问题" in report.markdown
    assert "## 参考来源" in report.markdown
    assert "### 2.1 代表方法" in report.markdown
    assert "### 2.2 评估基准" in report.markdown
    assert len(report.citation_items) == 2
    assert [item.citation_label for item in report.citation_items] == ["[1]", "[2]"]
    assert report.markdown.count("Shared Paper.") == 1
    assert report.markdown.count("Local PDF: shared.pdf, p.3.") == 1
    assert "关联引用：[1] [2]" in report.markdown


def test_report_writer_uses_llm_polish_when_response_is_valid(monkeypatch):
    task_summaries = [
        TaskSummary(
            task_id="task-1",
            title="任务一",
            intent="整合主题背景",
            summary="任务聚焦：整合主题背景",
            summary_markdown="任务聚焦：整合主题背景",
            paper_records=[
                PaperRecord(
                    title="Paper One",
                    authors=["Author One"],
                    year=2026,
                    url="https://example.com/paper-1",
                    source_type="online_paper",
                )
            ],
        )
    ]
    agent = ReportWriterAgent(api_key="test-key")

    def fake_call_llm(*, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return "\n".join(
            [
                "# 临时标题",
                "",
                "## 1. 研究概览",
                "",
                "润色后的概览 [1]。",
                "",
                "## 2. 分任务总结整合",
                "",
                "### 2.1 任务一",
                "",
                "润色后的任务整合 [1]。",
                "",
                "## 3. 关键观点归纳",
                "",
                "- 润色后的关键观点 [1]。",
                "",
                "## 4. 局限与待补充问题",
                "",
                "- 润色后的局限 [1]。",
                "",
                "## 参考来源",
                "",
                "[1] fake",
            ]
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    report = agent.write("测试主题", task_summaries)

    assert report.markdown.startswith("# 测试主题")
    assert "润色后的概览 [1]。" in report.markdown
    assert "润色后的任务整合 [1]。" in report.markdown
    assert "润色后的关键观点 [1]。" in report.markdown
    assert "## 参考来源" in report.markdown
    assert "[1] fake" not in report.markdown
    assert report.citations[0] in report.markdown


def test_report_writer_falls_back_to_template_when_llm_response_is_invalid(monkeypatch):
    task_summaries = [
        TaskSummary(
            task_id="task-1",
            title="任务一",
            intent="整合主题背景",
            summary="任务聚焦：整合主题背景",
            summary_markdown="任务聚焦：整合主题背景",
            paper_records=[
                PaperRecord(
                    title="Paper One",
                    authors=["Author One"],
                    year=2026,
                    url="https://example.com/paper-1",
                    source_type="online_paper",
                )
            ],
        )
    ]
    agent = ReportWriterAgent(api_key="test-key")
    monkeypatch.setattr(agent, "_call_llm", lambda **kwargs: "只有一段没有结构的文本")

    report = agent.write("测试主题", task_summaries)

    assert "本次研究围绕“测试主题”拆解为 1 个子任务" in report.markdown
    assert "只有一段没有结构的文本" not in report.markdown
    assert "## 参考来源" in report.markdown
