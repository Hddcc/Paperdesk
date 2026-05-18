from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3

import fitz

from app.agents.report_writer import ReportWriterAgent
from app.agents.reading_summarizer import ReadingSummarizerAgent
from app.models import (
    EvidenceItem,
    PaperRecord,
    ResearchEvidenceAssessment,
    ResearchEvidenceBufferItem,
    ResearchPlanOperation,
    ResearchPlanOperationType,
    ResearchPlanItem,
    ResearchRuntimePhase,
    ResearchRuntimeState,
    TaskSummary,
)
from app.runtime.main_agent_runtime import MainAgentRuntime
from app.runtime.planner_candidate_provider import RuleBasedPlannerCandidateProvider
from app.services.research_orchestrator import ResearchOrchestrator
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
        json={"topic": "RAG 系统中的评估方法综述"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    events = _parse_sse_events(body)
    event_types = [event["type"] for event in events]
    assert "run_created" in event_types
    assert "checkpoint_saved" in event_types
    assert "status" in event_types
    assert "coordinator_status" in event_types
    assert "todo_list" in event_types
    assert "task_status" in event_types
    assert "agent_step_started" in event_types
    assert "agent_step_completed" in event_types
    assert "task_summary" in event_types
    assert "report_completed" in event_types
    assert "report" in event_types
    assert "done" in event_types
    checkpoint_event = next(event for event in events if event["type"] == "checkpoint_saved")
    assert checkpoint_event["context_state"]["budget_tokens"] > 0
    assert checkpoint_event["context_state"]["sources"]
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
    assert report["topic"] == "RAG 系统中的评估方法综述"
    assert "## 综述脉络与主题分组" in report["markdown"]
    assert "## 关键研究方向" in report["markdown"]
    assert "## 趋势归纳" in report["markdown"]
    assert "## 证据来源与引用映射" in report["markdown"]
    assert "## 局限与待补充问题" in report["markdown"]
    assert "## 参考来源" in report["markdown"]
    assert "固定多 Agent 研究工作流自动生成" not in report["markdown"]
    assert "导出路径：" not in report["markdown"]
    assert report["citation_items"]
    assert "任务总结仍采用教程阶段的规则模板" not in report["markdown"]
    assert "在线论文参考：" not in report["markdown"]
    assert "Local PDF: library.pdf, p.1." in report["markdown"]
    assert "本地知识库证据" in report["markdown"]
    assert "在线检索证据" in report["markdown"]
    assert "混合结论" in report["markdown"]
    assert "强结论只应来自有明确引用编号或本地页码支撑的内容" in report["markdown"]
    assert "产物协议" in report["markdown"]
    assert "多篇论文综述" in report["markdown"]
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
    assert run_payload["runtime_state"]["current_phase"] == "completed"
    assert run_payload["runtime_state"]["context_state"]["budget_tokens"] > 0
    assert run_payload["runtime_state"]["context_state"]["stage"] in {
        "normal",
        "evidence_compacted",
        "history_compacted",
        "truncated",
    }
    assert "working_summary" in run_payload["runtime_state"]
    assert "证据进展" in run_payload["runtime_state"]["working_summary"]
    assert len(run_payload["runtime_state"]["completed_items"]) == 4
    assert run_payload["runtime_state"]["tool_history"]
    assert run_payload["runtime_state"]["plan_items"][0]["objective"]
    assert run_payload["runtime_state"]["plan_items"][0]["done_criteria"]
    assert run_payload["runtime_state"]["plan_items"][0]["suggested_tools"]
    assert run_payload["task_route"]["active_skill_id"] == "multi_paper_review"
    planned_tools = {
        tool
        for item in run_payload["runtime_state"]["plan_items"]
        for tool in item["suggested_tools"]
    }
    assert planned_tools >= {
        "plan/rule_based_initial",
        "search_local/vector_recall_default",
        "search_online/mixed_broad_recall",
        "mcp/academic_search",
        "summarize_evidence/task_level_merge",
        "finalize_report/report_writer_default",
    }
    assert "no_progress_count" in run_payload["runtime_state"]
    assert "same_tool_streak" in run_payload["runtime_state"]
    assert run_payload["runtime_state"]["planner_provider"] == "rule_based"
    assert "plan_revision_history" in run_payload["runtime_state"]
    assert run_payload["task_route"]["task_type"] == "multi_paper_review"
    assert run_payload["task_route"]["evidence_policy"] == "local_first"
    assert run_payload["task_route"]["artifact_protocol"]["title"] == "多篇论文综述"
    assert "产物协议" in run_payload["report"]["markdown"]
    assert "多篇论文综述" in run_payload["report"]["markdown"]
    assert "## 综述脉络与主题分组" in run_payload["report"]["markdown"]
    assert "## 证据来源与引用映射" in run_payload["report"]["markdown"]
    first_tool_record = run_payload["runtime_state"]["tool_history"][0]
    assert first_tool_record["selected_tool"] == "plan/rule_based_initial"
    assert first_tool_record["tool_strategy"]["strategy_id"] == first_tool_record["selected_tool"]
    assert first_tool_record["decision_reason"]
    assert first_tool_record["result_classification"]
    online_tool_record = next(
        record for record in run_payload["runtime_state"]["tool_history"]
        if record["action"] == "search_online"
    )
    assert online_tool_record["selected_tool"] == "mcp/academic_search"
    assert online_tool_record["tool_strategy"]["action_type"] == "search_online"
    assert online_tool_record["tool_strategy"]["label"] == "External academic search"
    first_evidence_action = next(
        record["action"]
        for record in run_payload["runtime_state"]["tool_history"]
        if record["action"] in {"search_online", "search_local"}
    )
    assert first_evidence_action == "search_local"
    first_buffer = run_payload["runtime_state"]["evidence_buffer"][0]
    assert first_buffer["compacted_evidence"]
    assert first_buffer["evidence_assessment"]["total_item_count"] > 0
    assert run_payload["subagent_tasks"] == []
    assert run_payload["task_notifications"] == []
    assert len(run_payload["task_traces"]) >= 8
    assert run_payload["task_artifacts"] == []
    assert run_payload["report"]["id"] == report_id

    report_path = Path(os.environ["REPORT_DIR"]) / f"{report_id}.md"
    assert not report_path.exists()

    export_response = client.get(f"/api/reports/{report_id}/export.md")
    assert export_response.status_code == 200
    assert export_response.headers["content-disposition"] == (
        f'attachment; filename="{report_id}.md"'
    )
    assert "text/markdown" in export_response.headers["content-type"]
    assert export_response.text == report["markdown"]
    assert report_path.exists()

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
    assert subagent_task_count == 0
    assert notification_count == 0
    assert artifact_count == 0
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
    assert (run_dir / "runtime_state.json").exists()

    todo_payload = json.loads((run_dir / "todo_tasks.json").read_text(encoding="utf-8"))
    final_report_markdown = (run_dir / "final_report.md").read_text(encoding="utf-8").strip()
    assert len(todo_payload) == 4
    assert all(item["status"] == "pending" for item in todo_payload)
    assert "在线论文参考：" in (run_dir / "task_1_summary.md").read_text(encoding="utf-8")
    assert report["markdown"] == export_response.text.strip()
    assert "产物协议" in final_report_markdown

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

    assert "## 结论边界" in report_markdown
    assert "本轮存在证据不足或无直接证据的任务" in report_markdown
    assert "Local PDF:" not in report_markdown
    assert "unrelated.pdf" not in report_markdown

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    report_id = reports_response.json()[0]["id"]
    report_response = client.get(f"/api/reports/{report_id}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert "本轮存在证据不足或无直接证据的任务" in report["markdown"]
    assert "Local PDF:" not in report["markdown"]
    assert "unrelated.pdf" not in report["markdown"]


def test_report_can_be_deleted_from_history(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法综述"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    report_event = next(event for event in events if event["type"] == "report")
    report_id = report_event["report_id"]

    report_path = Path(os.environ["REPORT_DIR"]) / f"{report_id}.md"
    assert not report_path.exists()

    export_response = client.get(f"/api/reports/{report_id}/export.md")
    assert export_response.status_code == 200
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


def test_legacy_technical_note_is_hidden_in_report_preview(client):
    from app.api.main import get_repository

    repository = get_repository()
    with repository.database.connection() as conn:
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

    response = client.get("/api/reports/report-legacy")
    assert response.status_code == 200
    payload = response.json()
    assert "当前实现已接入在线论文检索" not in payload["markdown"]
    assert "SQLite 持久化" not in payload["markdown"]
    assert "正文保留。" in payload["markdown"]
    assert "当前实现已接入在线论文检索" not in payload["task_summaries"][0]["summary"]
    assert "任务正文。" in payload["task_summaries"][0]["summary"]


def test_deleted_legacy_report_is_not_migrated_back(client):
    from app.api.main import get_repository
    from app.repositories import SQLiteRepository

    repository = get_repository()
    with repository.database.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                markdown TEXT NOT NULL,
                citations TEXT NOT NULL,
                task_summaries_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO research_runs (id, topic, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-legacy-delete",
                "Legacy Delete Topic",
                "completed",
                "2026-05-10T00:00:00+00:00",
                "2026-05-10T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO reports (id, run_id, topic, markdown, citations, task_summaries_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "report-legacy-delete",
                "run-legacy-delete",
                "Legacy Delete Topic",
                "# Legacy Delete",
                "",
                "[]",
                "2026-05-10T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO report_records (id, run_id, topic, markdown, citations_text, task_summaries_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "report-legacy-delete",
                "run-legacy-delete",
                "Legacy Delete Topic",
                "# Legacy Delete",
                "",
                "[]",
                "2026-05-10T00:00:00+00:00",
            ),
        )

    delete_response = client.delete("/api/reports/report-legacy-delete")
    assert delete_response.status_code == 200

    SQLiteRepository(Path(os.environ["SQLITE_PATH"]))
    list_response = client.get("/api/reports")
    assert list_response.status_code == 200
    assert all(item["id"] != "report-legacy-delete" for item in list_response.json())

    conn = sqlite3.connect(os.environ["SQLITE_PATH"])
    try:
        report_count = conn.execute(
            "SELECT COUNT(*) FROM report_records WHERE id = 'report-legacy-delete'"
        ).fetchone()[0]
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE id = 'report-legacy-delete'"
        ).fetchone()[0]
        tombstone_count = conn.execute(
            "SELECT COUNT(*) FROM deleted_report_records WHERE report_id = 'report-legacy-delete'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert report_count == 0
    assert legacy_count == 0
    assert tombstone_count == 1


def test_research_stream_marks_run_and_task_failed_when_task_stage_breaks(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    def broken_summarize(self, task, paper_records, evidence_items):
        raise RuntimeError("summary stage exploded")

    monkeypatch.setattr(ReadingSummarizerAgent, "summarize", broken_summarize)

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法综述", "search_provider": "all"},
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
    assert (run_dir / "runtime_state.json").exists()
    assert not (run_dir / "task_1_summary.md").exists()
    assert not (run_dir / "final_report.md").exists()


def test_research_task_route_respects_selected_knowledge_documents(client, monkeypatch):
    observed_queries: list[str] = []
    _patch_research_dependencies(monkeypatch, observed_queries)
    _upload_library_pdf(client)
    documents_response = client.get("/api/documents")
    assert documents_response.status_code == 200
    document_id = documents_response.json()[0]["id"]

    response = client.post(
        "/api/research/stream",
        json={
            "topic": "总结这篇论文的方法和贡献",
            "input_modes": ["prompt", "knowledge_base"],
            "selected_document_ids": [document_id],
            "top_k_local": 3,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    route_event = next(event for event in events if event["type"] == "task_route")
    assert route_event["task_route"]["task_type"] == "paper_summary"
    assert route_event["task_route"]["evidence_policy"] == "local_only"
    assert route_event["task_route"]["needs_online_search"] is False
    assert route_event["task_route"]["selected_document_ids"] == [document_id]

    run_id = next(event["run_id"] for event in events if event["type"] == "run_created")
    run_response = client.get(f"/api/research/{run_id}")
    assert run_response.status_code == 200
    payload = run_response.json()
    assert payload["task_route"]["task_type"] == "paper_summary"
    assert payload["task_route"]["use_main_agent_loop"] is False
    assert payload["task_route"]["allow_single_pass"] is True
    actions = [record["action"] for record in payload["runtime_state"]["tool_history"]]
    assert "search_local" in actions
    assert "search_online" not in actions
    assert "revise_plan" not in actions
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["title"].startswith("单篇论文总结")
    assert "单篇论文总结" in payload["report"]["markdown"]
    assert "## 论文主题" in payload["report"]["markdown"]
    assert "## 核心方法" in payload["report"]["markdown"]
    assert "## 证据来源与引用映射" in payload["report"]["markdown"]
    assert "## 结论边界" in payload["report"]["markdown"]
    assert "本地知识库证据" in payload["report"]["markdown"]
    assert "## 1. 研究概览" not in payload["report"]["markdown"]
    assert not observed_queries


def test_lightweight_qa_route_uses_single_pass_result_builder(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)
    documents_response = client.get("/api/documents")
    assert documents_response.status_code == 200
    document_id = documents_response.json()[0]["id"]

    response = client.post(
        "/api/research/stream",
        json={
            "topic": "这篇材料如何评估 RAG 系统",
            "input_modes": ["prompt", "knowledge_base"],
            "selected_document_ids": [document_id],
            "top_k_local": 3,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    run_id = next(event["run_id"] for event in events if event["type"] == "run_created")
    run_response = client.get(f"/api/research/{run_id}")
    assert run_response.status_code == 200
    payload = run_response.json()

    assert payload["task_route"]["task_type"] == "qa"
    assert payload["task_route"]["allow_single_pass"] is True
    assert len(payload["tasks"]) == 1
    actions = [record["action"] for record in payload["runtime_state"]["tool_history"]]
    assert actions == ["plan", "search_local", "summarize_evidence", "finalize_report", "finish"]
    assert "## 直接答案" in payload["report"]["markdown"]
    assert "## 关键证据" in payload["report"]["markdown"]
    assert "## 证据来源与引用映射" in payload["report"]["markdown"]
    assert "## 结论边界" in payload["report"]["markdown"]
    assert "## 1. 研究概览" not in payload["report"]["markdown"]


def test_uploaded_file_input_mode_participates_in_task_route(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)
    documents_response = client.get("/api/documents")
    assert documents_response.status_code == 200
    document_id = documents_response.json()[0]["id"]

    response = client.post(
        "/api/research/stream",
        json={
            "topic": "总结刚上传论文的核心贡献",
            "input_modes": ["prompt", "uploaded_file"],
            "selected_document_ids": [document_id],
            "top_k_local": 3,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    route_event = next(event for event in events if event["type"] == "task_route")

    assert route_event["task_route"]["task_type"] == "paper_summary"
    assert "uploaded_file" in route_event["task_route"]["input_modes"]
    assert "knowledge_base" in route_event["task_route"]["input_modes"]
    assert route_event["task_route"]["selected_document_ids"] == [document_id]
    assert route_event["task_route"]["needs_local_knowledge"] is True


def test_comparison_route_stays_on_main_agent_path(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    response = client.post(
        "/api/research/stream",
        json={"topic": "对比 RAG 评估中的忠实性和鲁棒性方法"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    route_event = next(event for event in events if event["type"] == "task_route")
    assert route_event["task_route"]["task_type"] == "comparison"
    assert route_event["task_route"]["use_main_agent_loop"] is True
    assert route_event["task_route"]["allow_single_pass"] is False


def test_research_resume_stream_recovers_failed_run_from_checkpoint(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    original_summarize = ReadingSummarizerAgent.summarize
    call_state = {"failed_once": False}

    def flaky_summarize(self, task, paper_records, evidence_items):
        if not call_state["failed_once"]:
            call_state["failed_once"] = True
            raise RuntimeError("summary stage exploded once")
        return original_summarize(self, task, paper_records, evidence_items)

    monkeypatch.setattr(ReadingSummarizerAgent, "summarize", flaky_summarize)

    first_response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法综述", "search_provider": "all"},
        headers={"Accept": "text/event-stream"},
    )
    assert first_response.status_code == 200
    first_events = _parse_sse_events(first_response.text)
    assert "error" in [event["type"] for event in first_events]
    run_id = next(event["run_id"] for event in first_events if event["type"] == "run_created")

    resume_response = client.post(
        f"/api/research/{run_id}/resume/stream",
        headers={"Accept": "text/event-stream"},
    )
    assert resume_response.status_code == 200
    resume_events = _parse_sse_events(resume_response.text)
    resume_types = [event["type"] for event in resume_events]
    assert "research_resumed" in resume_types
    assert "report" in resume_types
    assert "done" in resume_types

    run_response = client.get(f"/api/research/{run_id}")
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["run"]["status"] == "completed"
    assert run_payload["runtime_state"]["current_phase"] == "completed"
    assert run_payload["runtime_state"]["report_id"]


def test_retryable_tool_error_records_decision_classification(client, monkeypatch):
    _patch_research_dependencies(monkeypatch)
    _upload_library_pdf(client)

    call_state = {"failed_once": False}

    def flaky_search(self, task, *, top_k, search_provider):
        _ = top_k
        _ = search_provider
        if not call_state["failed_once"]:
            call_state["failed_once"] = True
            raise RuntimeError("temporary provider outage")
        return []

    monkeypatch.setattr("app.agents.paper_search_agent.PaperSearchAgent.search", flaky_search)

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法综述", "search_provider": "all"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert "done" in [event["type"] for event in events]
    failed_event = next(event for event in events if event["type"] == "agent_step_failed")
    assert failed_event["result_classification"] == "retryable_error"
    assert failed_event["selected_tool"] == "mcp/academic_search"
    assert failed_event["tool_strategy"]["strategy_id"] == "mcp/academic_search"
    assert "temporary provider outage" in failed_event["error"]


def _runtime_for_decision(assessment: ResearchEvidenceAssessment, *, revise_count: int = 0):
    plan_item = ResearchPlanItem(
        task_id="task-1",
        title="RAG 评估方法",
        intent="梳理 RAG 评估指标",
        query="RAG evaluation metrics",
        revise_count=revise_count,
        query_history=["RAG evaluation metrics"],
    )
    evidence = ResearchEvidenceBufferItem(
        task_id="task-1",
        online_completed=True,
        local_completed=True,
        evidence_assessment=assessment,
    )
    return ResearchRuntimeState(
        run_id="run-decision",
        goal="RAG 评估",
        current_phase=ResearchRuntimePhase.EXECUTING,
        plan_items=[plan_item],
        evidence_buffer=[evidence],
    )


def test_main_agent_runtime_revises_when_evidence_quality_is_weak():
    state = _runtime_for_decision(
        ResearchEvidenceAssessment(
            total_item_count=2,
            relevant_item_count=0,
            visible_item_count=2,
            sufficiency_score=0.2,
            relevance_score=0.0,
            has_relevant_evidence=False,
        )
    )

    decision = MainAgentRuntime().next_action(state)

    assert decision.action_type.value == "revise_plan"
    assert decision.target_task_id == "task-1"
    assert decision.selected_tool == "revise_plan/rewrite_query"
    assert decision.tool_strategy.strategy_id == "revise_plan/rewrite_query"
    assert "证据不足" in decision.reason


def test_main_agent_runtime_degrades_after_revise_when_evidence_stays_weak():
    state = _runtime_for_decision(
        ResearchEvidenceAssessment(
            total_item_count=2,
            relevant_item_count=0,
            visible_item_count=2,
            sufficiency_score=0.2,
            relevance_score=0.0,
            has_relevant_evidence=False,
        ),
        revise_count=1,
    )

    decision = MainAgentRuntime().next_action(state)

    assert decision.action_type.value == "summarize_evidence"
    assert decision.target_task_id == "task-1"
    assert decision.selected_tool == "summarize_evidence/degraded_closeout"
    assert MainAgentRuntime.should_degrade(state.plan_items[0], state.evidence_buffer[0])


def test_main_agent_runtime_summarizes_when_evidence_quality_is_sufficient():
    state = _runtime_for_decision(
        ResearchEvidenceAssessment(
            total_item_count=3,
            relevant_item_count=2,
            visible_item_count=3,
            sufficiency_score=0.72,
            relevance_score=0.67,
            diversity_score=1.0,
            coverage=["method", "evaluation"],
            has_relevant_evidence=True,
        )
    )

    decision = MainAgentRuntime().next_action(state)

    assert decision.action_type.value == "summarize_evidence"
    assert decision.target_task_id == "task-1"
    assert decision.selected_tool == "summarize_evidence/task_level_merge"


def test_main_agent_runtime_replans_on_stale_repeated_tool():
    state = _runtime_for_decision(
        ResearchEvidenceAssessment(
            total_item_count=0,
            sufficiency_score=0.0,
            relevance_score=0.0,
            has_relevant_evidence=False,
        )
    )
    state.same_tool_streak = 2
    state.no_progress_count = 1
    state.last_tool_signature = "search_local:task-1:abc"

    decision = MainAgentRuntime().next_action(state)

    assert decision.action_type.value == "revise_plan"
    assert decision.target_task_id == "task-1"
    assert decision.selected_tool == "revise_plan/reorder_priority"
    assert "没有新增信息" in decision.reason


def test_main_agent_runtime_stops_after_no_progress_limit():
    state = _runtime_for_decision(
        ResearchEvidenceAssessment(
            total_item_count=0,
            sufficiency_score=0.0,
            relevance_score=0.0,
            has_relevant_evidence=False,
        )
    )
    state.no_progress_count = MainAgentRuntime.max_no_progress_count

    decision = MainAgentRuntime().next_action(state)

    assert decision.action_type.value == "fail"
    assert "无增量" in decision.reason


def test_rule_planner_candidate_splits_weak_broad_task():
    plan_item = ResearchPlanItem(
        task_id="task-1",
        title="RAG 方法、应用与挑战",
        intent="同时梳理方法、应用与挑战",
        query="RAG methods applications limitations",
        query_history=["RAG methods applications limitations"],
    )
    state = ResearchRuntimeState(
        run_id="run-decision",
        goal="RAG 评估",
        current_phase=ResearchRuntimePhase.EXECUTING,
        plan_items=[plan_item],
        evidence_buffer=[
            ResearchEvidenceBufferItem(
                task_id="task-1",
                online_completed=True,
                local_completed=True,
                evidence_assessment=ResearchEvidenceAssessment(
                    total_item_count=2,
                    relevant_item_count=0,
                    visible_item_count=2,
                    sufficiency_score=0.2,
                    relevance_score=0.0,
                    has_relevant_evidence=False,
                ),
            )
        ],
    )
    decision = MainAgentRuntime().next_action(state)

    candidate = RuleBasedPlannerCandidateProvider().propose(state, decision, state.plan_items[0])

    assert candidate.provider.value == "rule_based"
    assert candidate.candidate_tool == "revise_plan/split_task"
    assert candidate.candidate_plan_ops[0].operation_type == ResearchPlanOperationType.SPLIT_ITEM
    assert candidate.candidate_plan_ops[0].new_task_id.startswith("task-1-split-")


def test_plan_operations_insert_split_task_and_reorder_pending_items():
    first = ResearchPlanItem(
        task_id="task-1",
        title="宽泛任务",
        intent="同时梳理方法与应用",
        query="topic methods applications",
        priority=1,
    )
    second = ResearchPlanItem(
        task_id="task-2",
        title="后续任务",
        intent="梳理局限",
        query="topic limitations",
        priority=2,
    )
    state = ResearchRuntimeState(
        run_id="run-plan-ops",
        goal="topic",
        current_phase=ResearchRuntimePhase.EXECUTING,
        plan_items=[first, second],
        evidence_buffer=[ResearchEvidenceBufferItem(task_id="task-1")],
    )
    split_op = ResearchPlanOperation(
        operation_type=ResearchPlanOperationType.SPLIT_ITEM,
        target_task_id="task-1",
        new_task_id="task-1-split-test",
        title="宽泛任务：补充证据线索",
        intent="补充更窄证据",
        query="topic focused evidence",
        priority=2,
        reason="测试拆分",
    )

    applied = ResearchOrchestrator._apply_plan_operations(
        state,
        first,
        [split_op],
        revised_query="topic methods applications evidence detail",
    )

    assert [operation.operation_type for operation in applied] == [ResearchPlanOperationType.SPLIT_ITEM]
    assert [item.task_id for item in state.plan_items] == ["task-1", "task-1-split-test", "task-2"]
    assert state.plan_revision_history[0].operation_type == ResearchPlanOperationType.SPLIT_ITEM
    assert state.last_plan_operation.operation_type == ResearchPlanOperationType.SPLIT_ITEM
    assert any(buffer.task_id == "task-1-split-test" for buffer in state.evidence_buffer)

    reorder_op = ResearchPlanOperation(
        operation_type=ResearchPlanOperationType.REORDER_ITEMS,
        target_task_id="task-1",
        ordered_task_ids=["task-2", "task-1-split-test", "task-1"],
        reason="测试重排",
    )
    applied = ResearchOrchestrator._apply_plan_operations(
        state,
        first,
        [reorder_op],
        revised_query=first.query,
    )

    assert [operation.operation_type for operation in applied] == [ResearchPlanOperationType.REORDER_ITEMS]
    assert [item.task_id for item in state.plan_items] == ["task-2", "task-1-split-test", "task-1"]
    assert [item.priority for item in state.plan_items] == [1, 2, 3]


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
