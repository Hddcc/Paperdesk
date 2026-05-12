"""SQLite repository for research runs and todo tasks."""

from __future__ import annotations

import sqlite3
import json

from app.models import ResearchRequest, ResearchRun, ResearchRuntimeState, TodoTask
from app.models.enums import ResearchRunStatus

from .base import BaseRepository, utc_now


class ResearchRepository(BaseRepository):
    """Store research execution state."""

    @staticmethod
    def _todo_task_columns(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(todo_tasks)").fetchall()
        return {str(row["name"]) for row in rows}

    @staticmethod
    def _research_run_columns(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(research_runs)").fetchall()
        return {str(row["name"]) for row in rows}

    def create_run(
        self,
        run_id: str,
        topic: str,
        *,
        request_payload: ResearchRequest | dict | None = None,
    ) -> ResearchRun:
        timestamp = utc_now()
        serialized_request = None
        if request_payload is not None:
            if hasattr(request_payload, "model_dump"):
                request_payload = request_payload.model_dump(mode="json")
            serialized_request = json.dumps(request_payload, ensure_ascii=False)
        with self.database.connection() as conn:
            columns = self._research_run_columns(conn)
            payload: dict[str, object | None] = {
                "id": run_id,
                "topic": topic,
                "status": ResearchRunStatus.CREATED.value,
                "created_at": timestamp.isoformat(),
                "updated_at": timestamp.isoformat(),
            }
            if "request_payload_json" in columns:
                payload["request_payload_json"] = serialized_request
            conn.execute(
                f"INSERT INTO research_runs ({', '.join(payload.keys())}) VALUES ({', '.join('?' for _ in payload)})",
                tuple(payload.values()),
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
                SELECT id, topic, status, created_at, updated_at, stop_reason, last_checkpoint_at
                FROM research_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchRun(**dict(row))

    def update_run_status(
        self,
        run_id: str,
        status: ResearchRunStatus,
        *,
        stop_reason: str | None = None,
    ) -> None:
        timestamp = utc_now().isoformat()
        with self.database.connection() as conn:
            columns = self._research_run_columns(conn)
            assignments = ["status = ?", "updated_at = ?"]
            values: list[object | None] = [status.value, timestamp]
            if "stop_reason" in columns:
                assignments.append("stop_reason = ?")
                values.append(stop_reason)
            conn.execute(
                f"""
                UPDATE research_runs
                SET {", ".join(assignments)}
                WHERE id = ?
                """,
                (*values, run_id),
            )

    def save_runtime_state(
        self,
        run_id: str,
        runtime_state: ResearchRuntimeState,
        *,
        request_payload: ResearchRequest | dict | None = None,
        status: ResearchRunStatus | None = None,
        stop_reason: str | None = None,
    ) -> None:
        timestamp = utc_now().isoformat()
        runtime_payload = json.dumps(runtime_state.model_dump(mode="json"), ensure_ascii=False)
        serialized_request = None
        if request_payload is not None:
            if hasattr(request_payload, "model_dump"):
                request_payload = request_payload.model_dump(mode="json")
            serialized_request = json.dumps(request_payload, ensure_ascii=False)

        with self.database.connection() as conn:
            columns = self._research_run_columns(conn)
            assignments = ["updated_at = ?"]
            values: list[object | None] = [timestamp]
            if "runtime_state_json" in columns:
                assignments.append("runtime_state_json = ?")
                values.append(runtime_payload)
            if "last_checkpoint_at" in columns:
                assignments.append("last_checkpoint_at = ?")
                values.append(timestamp)
            if "request_payload_json" in columns and serialized_request is not None:
                assignments.append("request_payload_json = ?")
                values.append(serialized_request)
            if "status" in columns and status is not None:
                assignments.append("status = ?")
                values.append(status.value)
            if "stop_reason" in columns:
                assignments.append("stop_reason = ?")
                values.append(stop_reason)
            conn.execute(
                f"""
                UPDATE research_runs
                SET {", ".join(assignments)}
                WHERE id = ?
                """,
                (*values, run_id),
            )

    def get_runtime_state(self, run_id: str) -> ResearchRuntimeState | None:
        with self.database.connection() as conn:
            columns = self._research_run_columns(conn)
            if "runtime_state_json" not in columns:
                return None
            row = conn.execute(
                "SELECT runtime_state_json FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None or not row["runtime_state_json"]:
            return None
        return ResearchRuntimeState(**json.loads(row["runtime_state_json"]))

    def get_request_payload(self, run_id: str) -> ResearchRequest | None:
        with self.database.connection() as conn:
            columns = self._research_run_columns(conn)
            if "request_payload_json" not in columns:
                return None
            row = conn.execute(
                "SELECT request_payload_json FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None or not row["request_payload_json"]:
            return None
        return ResearchRequest(**json.loads(row["request_payload_json"]))

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
            if "query" in columns:
                assignments.append("query = ?")
                values.append(task.query)
            if "query_text" in columns:
                assignments.append("query_text = ?")
                values.append(task.query)
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
