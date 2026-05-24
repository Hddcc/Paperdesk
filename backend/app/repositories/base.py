"""Shared SQLite helpers for repository implementations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.models import CitationRecord


def utc_now() -> datetime:
    """Return timezone-aware current UTC time."""

    return datetime.now(timezone.utc)


class SQLiteDatabase:
    """Own the shared SQLite connection lifecycle and schema management."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # Some Windows-backed workspaces reject SQLite's default rollback journal writes.
        # Keeping the journal in memory preserves local test/runtime stability here.
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        """Create phase-03 tables and backfill compatible legacy data."""

        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS todo_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_order INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_markdown TEXT,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );

                CREATE TABLE IF NOT EXISTS paper_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    paper_id TEXT,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    abstract TEXT,
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    url TEXT,
                    source TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS library_documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    title TEXT,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    parser_status TEXT NOT NULL DEFAULT 'pending',
                    failure_reason TEXT,
                    indexed_at TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS library_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section TEXT,
                    title TEXT,
                    sha256 TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    text TEXT NOT NULL,
                    token_estimate INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (document_id) REFERENCES library_documents (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_category_links (
                    category_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (category_id, document_id),
                    FOREIGN KEY (category_id) REFERENCES document_categories (id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES library_documents (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS report_records (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    citations_text TEXT NOT NULL,
                    task_summaries_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL DEFAULT 'saved_report',
                    source TEXT NOT NULL DEFAULT 'research_task',
                    source_message_id TEXT,
                    paper_ids_json TEXT NOT NULL DEFAULT '[]',
                    category_ids_json TEXT NOT NULL DEFAULT '[]',
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );

                CREATE TABLE IF NOT EXISTS citation_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    citation_label TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT,
                    doi TEXT,
                    document_id TEXT,
                    page_number INTEGER,
                    sort_order INTEGER NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES report_records (id)
                );

                CREATE TABLE IF NOT EXISTS deleted_report_records (
                    report_id TEXT PRIMARY KEY,
                    deleted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS subagent_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_task_id TEXT,
                    profile TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    context_bundle_json TEXT NOT NULL,
                    done_criteria TEXT NOT NULL,
                    tool_policy_json TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );

                CREATE TABLE IF NOT EXISTS task_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    result_payload_json TEXT NOT NULL,
                    token_usage_json TEXT NOT NULL,
                    artifact_refs_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id),
                    FOREIGN KEY (task_id) REFERENCES subagent_tasks (id)
                );

                CREATE TABLE IF NOT EXISTS task_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    description TEXT,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id),
                    FOREIGN KEY (task_id) REFERENCES subagent_tasks (id)
                );

                CREATE TABLE IF NOT EXISTS task_execution_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    trace_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs (id)
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retrieval_status TEXT,
                    warning TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    used_document_ids_json TEXT NOT NULL DEFAULT '[]',
                    used_file_ids_json TEXT NOT NULL DEFAULT '[]',
                    memory_hits_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_attachments (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    mime_type TEXT,
                    document_id TEXT,
                    file_asset_id TEXT,
                    data_url TEXT,
                    file_path TEXT,
                    status TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES chat_messages (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS file_assets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    mime_type TEXT,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text_extract_status TEXT NOT NULL,
                    preview_text TEXT,
                    text_char_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    detail TEXT,
                    source_kind TEXT,
                    source_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_verified_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory_records (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_refresh_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory_records (id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_todo_task_columns(conn)
            self._ensure_library_document_columns(conn)
            self._ensure_chat_message_columns(conn)
            self._ensure_chat_attachment_columns(conn)
            self._ensure_report_columns(conn)
            self._ensure_research_run_columns(conn)
            self._migrate_legacy_documents(conn)
            self._migrate_legacy_reports(conn)

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        columns = SQLiteDatabase._column_names(conn, table_name)
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _ensure_todo_task_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "todo_tasks"):
            return

        self._ensure_column(conn, "todo_tasks", "task_order", "INTEGER")
        self._ensure_column(conn, "todo_tasks", "query", "TEXT")
        self._ensure_column(conn, "todo_tasks", "summary_markdown", "TEXT")

        columns = self._column_names(conn, "todo_tasks")
        if "task_index" in columns:
            conn.execute(
                """
                UPDATE todo_tasks
                SET task_order = COALESCE(task_order, task_index)
                """
            )
        if "query_text" in columns:
            conn.execute(
                """
                UPDATE todo_tasks
                SET query = COALESCE(query, query_text)
                """
            )
        if "summary" in columns:
            conn.execute(
                """
                UPDATE todo_tasks
                SET summary_markdown = COALESCE(summary_markdown, summary)
                """
            )

    def _ensure_library_document_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "library_documents"):
            return

        self._ensure_column(conn, "library_documents", "parser_status", "TEXT NOT NULL DEFAULT 'pending'")
        self._ensure_column(conn, "library_documents", "failure_reason", "TEXT")
        self._ensure_column(conn, "library_documents", "indexed_at", "TEXT")
        self._ensure_column(conn, "library_documents", "version", "INTEGER NOT NULL DEFAULT 1")

        conn.execute(
            """
            UPDATE library_documents
            SET parser_status = CASE
                WHEN status = 'ready' THEN 'indexed'
                WHEN status = 'failed' THEN 'failed'
                WHEN status = 'processing' THEN 'processing'
                ELSE COALESCE(parser_status, 'pending')
            END
            WHERE parser_status IS NULL
               OR parser_status = ''
               OR parser_status = 'pending'
            """
        )
        conn.execute(
            """
            UPDATE library_documents
            SET indexed_at = COALESCE(indexed_at, uploaded_at)
            WHERE status = 'ready' AND indexed_at IS NULL
            """
        )
        conn.execute(
            """
            UPDATE library_documents
            SET version = COALESCE(version, 1)
            WHERE version IS NULL OR version <= 0
            """
        )

    def _ensure_chat_message_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "chat_messages"):
            return

        self._ensure_column(conn, "chat_messages", "saved_report_id", "TEXT")
        self._ensure_column(conn, "chat_messages", "agent_trace_id", "TEXT")
        self._ensure_column(conn, "chat_messages", "action_status", "TEXT")
        self._ensure_column(conn, "chat_messages", "used_file_ids_json", "TEXT NOT NULL DEFAULT '[]'")

    def _ensure_chat_attachment_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "chat_attachments"):
            return

        self._ensure_column(conn, "chat_attachments", "file_asset_id", "TEXT")

    def _ensure_report_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "report_records"):
            return

        self._ensure_column(conn, "report_records", "lifecycle_status", "TEXT NOT NULL DEFAULT 'saved_report'")
        self._ensure_column(conn, "report_records", "source", "TEXT NOT NULL DEFAULT 'research_task'")
        self._ensure_column(conn, "report_records", "source_message_id", "TEXT")
        self._ensure_column(conn, "report_records", "paper_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column(conn, "report_records", "category_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column(conn, "report_records", "evidence_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column(conn, "report_records", "updated_at", "TEXT")

        conn.execute(
            """
            UPDATE report_records
            SET
                lifecycle_status = COALESCE(NULLIF(lifecycle_status, ''), 'saved_report'),
                source = COALESCE(NULLIF(source, ''), 'research_task'),
                paper_ids_json = COALESCE(NULLIF(paper_ids_json, ''), '[]'),
                category_ids_json = COALESCE(NULLIF(category_ids_json, ''), '[]'),
                evidence_ids_json = COALESCE(NULLIF(evidence_ids_json, ''), '[]'),
                updated_at = COALESCE(updated_at, created_at)
            """
        )

    def _ensure_research_run_columns(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "research_runs"):
            return

        self._ensure_column(conn, "research_runs", "runtime_state_json", "TEXT")
        self._ensure_column(conn, "research_runs", "request_payload_json", "TEXT")
        self._ensure_column(conn, "research_runs", "stop_reason", "TEXT")
        self._ensure_column(conn, "research_runs", "last_checkpoint_at", "TEXT")

    def _migrate_legacy_documents(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "documents"):
            return

        conn.execute(
            """
            INSERT INTO library_documents (
                id,
                filename,
                display_name,
                title,
                file_path,
                sha256,
                page_count,
                status,
                parser_status,
                indexed_at,
                version,
                created_at,
                uploaded_at
            )
            SELECT
                documents.id,
                documents.filename,
                documents.display_name,
                documents.display_name,
                documents.file_path,
                '',
                0,
                documents.status,
                CASE
                    WHEN documents.status = 'ready' THEN 'indexed'
                    WHEN documents.status = 'failed' THEN 'failed'
                    WHEN documents.status = 'processing' THEN 'processing'
                    ELSE 'pending'
                END,
                NULL,
                1,
                documents.uploaded_at,
                documents.uploaded_at
            FROM documents
            WHERE NOT EXISTS (
                SELECT 1 FROM library_documents
                WHERE library_documents.id = documents.id
            )
            """
        )

    def _migrate_legacy_reports(self, conn: sqlite3.Connection) -> None:
        if not self._table_exists(conn, "reports"):
            return

        conn.execute(
            """
            INSERT INTO report_records (
                id,
                run_id,
                topic,
                markdown,
                citations_text,
                task_summaries_json,
                created_at,
                lifecycle_status,
                source,
                updated_at
            )
            SELECT
                reports.id,
                reports.run_id,
                reports.topic,
                reports.markdown,
                reports.citations,
                reports.task_summaries_json,
                reports.created_at,
                'saved_report',
                'research_task',
                reports.created_at
            FROM reports
            WHERE NOT EXISTS (
                SELECT 1 FROM report_records
                WHERE report_records.id = reports.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM deleted_report_records
                WHERE deleted_report_records.report_id = reports.id
            )
            """
        )

        rows = conn.execute(
            """
            SELECT id, citations, task_summaries_json
            FROM reports
            ORDER BY created_at ASC
            """
        ).fetchall()
        for row in rows:
            existing = conn.execute(
                "SELECT 1 FROM citation_records WHERE report_id = ? LIMIT 1",
                (row["id"],),
            ).fetchone()
            if existing is not None:
                continue
            deleted = conn.execute(
                "SELECT 1 FROM deleted_report_records WHERE report_id = ? LIMIT 1",
                (row["id"],),
            ).fetchone()
            if deleted is not None:
                continue
            citation_items = self._extract_legacy_citation_items(
                task_summaries_json=row["task_summaries_json"],
                citations_text=row["citations"],
            )
            for index, citation in enumerate(citation_items):
                conn.execute(
                    """
                    INSERT INTO citation_records (
                        report_id,
                        citation_label,
                        source_type,
                        title,
                        url,
                        doi,
                        document_id,
                        page_number,
                        sort_order
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        citation.citation_label,
                        citation.source_type,
                        citation.title,
                        citation.url,
                        citation.doi,
                        citation.document_id,
                        citation.page_number,
                        index,
                    ),
                )

    @staticmethod
    def _extract_legacy_citation_items(
        *,
        task_summaries_json: str,
        citations_text: str,
    ) -> list[CitationRecord]:
        items: list[CitationRecord] = []
        seen: set[tuple[str, str, str | None, str | None, str | None, int | None]] = set()

        try:
            task_summaries = json.loads(task_summaries_json)
        except json.JSONDecodeError:
            task_summaries = []

        for task_summary in task_summaries:
            for evidence in task_summary.get("evidence_items", []):
                metadata = evidence.get("metadata") or {}
                citation = CitationRecord(
                    citation_label=evidence.get("citation_label") or "Legacy local evidence",
                    source_type=str(evidence.get("source_type") or "local_document"),
                    title=str(
                        evidence.get("title")
                        or metadata.get("filename")
                        or evidence.get("citation_label")
                        or "Legacy local evidence"
                    ),
                    url=evidence.get("url"),
                    document_id=evidence.get("document_id")
                    or metadata.get("document_id")
                    or evidence.get("source_id"),
                    page_number=evidence.get("page_number"),
                )
                key = (
                    citation.citation_label,
                    citation.title,
                    citation.url,
                    citation.doi,
                    citation.document_id,
                    citation.page_number,
                )
                if key not in seen:
                    items.append(citation)
                    seen.add(key)

            for paper in task_summary.get("paper_records", []):
                citation = CitationRecord(
                    citation_label=paper.get("title") or "Legacy online paper",
                    source_type=str(paper.get("source_type") or "online_paper"),
                    title=paper.get("title") or "Legacy online paper",
                    url=paper.get("url"),
                    doi=paper.get("doi"),
                )
                key = (
                    citation.citation_label,
                    citation.title,
                    citation.url,
                    citation.doi,
                    citation.document_id,
                    citation.page_number,
                )
                if key not in seen:
                    items.append(citation)
                    seen.add(key)

        if items:
            return items

        for line in [entry.strip() for entry in citations_text.splitlines() if entry.strip()]:
            items.append(
                CitationRecord(
                    citation_label=line,
                    source_type="legacy",
                    title=line,
                )
            )
        return items


class BaseRepository:
    """Base class for repositories backed by a shared SQLite database."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
