"""SQLite repository for research runs and todo tasks."""

from __future__ import annotations

import sqlite3

from app.models import ResearchRun, TodoTask
from app.models.enums import ResearchRunStatus

from .base import BaseRepository, utc_now


class ResearchRepository(BaseRepository):
    """Store research execution state."""

    @staticmethod
    def _todo_task_columns(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(todo_tasks)").fetchall()
        return {str(row["name"]) for row in rows}

    def create_run(self, run_id: str, topic: str) -> ResearchRun:
        timestamp = utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO research_runs (id, topic, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    topic,
                    ResearchRunStatus.CREATED.value,
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
        return ResearchRun(
            id=run_id,
            topic=topic,
            status=ResearchRunStatus.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get_run(self, run_id: str) -> ResearchRun | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT id, topic, status, created_at, updated_at
                FROM research_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchRun(**dict(row))

    def update_run_status(self, run_id: str, status: ResearchRunStatus) -> None:
        timestamp = utc_now().isoformat()
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE research_runs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, timestamp, run_id),
            )

    def save_todo_tasks(self, run_id: str, tasks: list[TodoTask]) -> None:
        with self.database.connection() as conn:
            columns = self._todo_task_columns(conn)
            for index, task in enumerate(tasks):
                payload: dict[str, object | None] = {
                    "id": task.id,
                    "run_id": run_id,
                    "title": task.title,
                    "intent": task.intent,
                    "status": task.status.value,
                }
                if "task_order" in columns:
                    payload["task_order"] = index
                if "task_index" in columns:
                    payload["task_index"] = index
                if "query" in columns:
                    payload["query"] = task.query
                if "query_text" in columns:
                    payload["query_text"] = task.query
                if "summary_markdown" in columns:
                    payload["summary_markdown"] = task.summary_markdown
                if "summary" in columns:
                    payload["summary"] = task.summary_markdown

                column_sql = ", ".join(payload.keys())
                placeholder_sql = ", ".join("?" for _ in payload)
                conn.execute(
                    f"INSERT INTO todo_tasks ({column_sql}) VALUES ({placeholder_sql})",
                    tuple(payload.values()),
                )

    def update_task(self, run_id: str, task: TodoTask) -> None:
        with self.database.connection() as conn:
            columns = self._todo_task_columns(conn)
            assignments: list[str] = ["status = ?"]
            values: list[object | None] = [task.status.value]
            if "summary_markdown" in columns:
                assignments.append("summary_markdown = ?")
                values.append(task.summary_markdown)
            if "summary" in columns:
                assignments.append("summary = ?")
                values.append(task.summary_markdown)
            values.extend([run_id, task.id])
            conn.execute(
                f"""
                UPDATE todo_tasks
                SET {", ".join(assignments)}
                WHERE run_id = ? AND id = ?
                """,
                tuple(values),
            )

    def list_tasks(self, run_id: str) -> list[TodoTask]:
        with self.database.connection() as conn:
            columns = self._todo_task_columns(conn)
            if "task_order" in columns and "task_index" in columns:
                order_sql = "COALESCE(task_order, task_index, 0)"
            elif "task_order" in columns:
                order_sql = "task_order"
            elif "task_index" in columns:
                order_sql = "task_index"
            else:
                order_sql = "rowid"
            rows = conn.execute(
                f"""
                SELECT * FROM todo_tasks
                WHERE run_id = ?
                ORDER BY {order_sql} ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TodoTask:
        columns = set(row.keys())
        query = row["query"] if "query" in columns else row["query_text"]
        summary_markdown = (
            row["summary_markdown"] if "summary_markdown" in columns else row["summary"]
        )
        return TodoTask(
            id=row["id"],
            title=row["title"],
            intent=row["intent"],
            query=query,
            status=row["status"],
            summary=summary_markdown,
            summary_markdown=summary_markdown,
        )
