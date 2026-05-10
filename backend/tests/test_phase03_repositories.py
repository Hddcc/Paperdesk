from __future__ import annotations

import sqlite3

from app.models import TodoTask
from app.models.enums import ResearchRunStatus
from app.repositories import SQLiteRepository


def test_phase03_schema_and_legacy_backfill_are_idempotent(sandbox_dir):
    database_path = sandbox_dir / "paperdesk.db"
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                display_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE research_runs (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE todo_tasks (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                intent TEXT NOT NULL,
                query_text TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT
            );

            CREATE TABLE reports (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                markdown TEXT NOT NULL,
                citations TEXT NOT NULL,
                task_summaries_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, filename, display_name, file_path, status, uploaded_at)
            VALUES ('doc-1', 'doc-1.pdf', 'Sample Paper.pdf', '/tmp/doc-1.pdf', 'uploaded', '2026-05-09T12:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO research_runs (id, topic, status, created_at, updated_at)
            VALUES ('run-1', 'Legacy Topic', 'completed', '2026-05-09T12:00:00+00:00', '2026-05-09T12:30:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO todo_tasks (id, run_id, task_index, title, intent, query_text, status, summary)
            VALUES ('task-1', 'run-1', 0, 'Legacy Task', 'Legacy Intent', 'legacy query', 'completed', 'legacy summary')
            """
        )
        conn.execute(
            """
            INSERT INTO reports (id, run_id, topic, markdown, citations, task_summaries_json, created_at)
            VALUES (
                'report-1',
                'run-1',
                'Legacy Topic',
                '# Legacy',
                'Legacy citation line',
                '[{"task_id":"task-1","title":"Legacy Task","intent":"Legacy Intent","summary":"legacy summary","evidence_items":[{"id":"evi-1","source_type":"local_document","source_id":"doc-1","quote":"legacy quote","citation_label":"[L1] Sample Paper.pdf","metadata":{"document_id":"doc-1","filename":"Sample Paper.pdf"}}],"paper_records":[{"title":"Legacy Paper","authors":["Legacy Author"],"abstract":"legacy abstract","url":"https://example.com/legacy","doi":"10.0000/legacy","source_type":"online_paper"}]}]',
                '2026-05-09T12:30:00+00:00'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    first_repository = SQLiteRepository(database_path)
    second_repository = SQLiteRepository(database_path)

    documents = first_repository.list_documents()
    assert len(documents) == 1
    assert documents[0].display_name == "Sample Paper.pdf"
    assert documents[0].title == "Sample Paper.pdf"

    tasks = first_repository.list_tasks("run-1")
    assert len(tasks) == 1
    assert tasks[0].query == "legacy query"
    assert tasks[0].summary_markdown == "legacy summary"

    report = first_repository.get_report("report-1")
    assert report is not None
    assert report.citation_items
    assert report.task_summaries[0].paper_records[0].title == "Legacy Paper"

    conn = sqlite3.connect(database_path)
    try:
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = NORMAL")
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        library_count = conn.execute("SELECT COUNT(*) FROM library_documents").fetchone()[0]
        report_count = conn.execute("SELECT COUNT(*) FROM report_records").fetchone()[0]
        citation_count = conn.execute("SELECT COUNT(*) FROM citation_records").fetchone()[0]
        todo_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(todo_tasks)").fetchall()
        }
    finally:
        conn.close()

    assert {"paper_records", "library_documents", "report_records", "citation_records"}.issubset(
        table_names
    )
    assert {"task_order", "query", "summary_markdown"}.issubset(todo_columns)
    assert library_count == 1
    assert report_count == 1
    assert citation_count >= 1
    assert second_repository.get_report("report-1") is not None

    new_run = first_repository.create_run("run-2", "New Topic")
    first_repository.save_todo_tasks(
        new_run.id,
        [
            TodoTask(
                id="task-2",
                title="New Task",
                intent="Check compatibility insert",
                query="new query",
                status=ResearchRunStatus.CREATED,
            )
        ],
    )
    inserted_tasks = first_repository.list_tasks("run-2")
    assert len(inserted_tasks) == 1
    assert inserted_tasks[0].query == "new query"
