"""SQLite repository for subagent runtime traces and notifications."""

from __future__ import annotations

import json
import sqlite3

from app.models import (
    AgentTask,
    AgentTaskStatus,
    StoredAgentTask,
    TaskArtifactRef,
    TaskExecutionTrace,
    TaskNotification,
)
from app.models.runtime import ToolPolicy

from .base import BaseRepository, utc_now


class RuntimeRepository(BaseRepository):
    """Persist runtime-managed subagent state and communication records."""

    def create_task(self, task: AgentTask, status: AgentTaskStatus = AgentTaskStatus.CREATED) -> StoredAgentTask:
        timestamp = utc_now()
        payload = task.model_dump(mode="json")
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO subagent_tasks (
                    id,
                    run_id,
                    parent_task_id,
                    profile,
                    goal,
                    context_bundle_json,
                    done_criteria,
                    tool_policy_json,
                    artifact_dir,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.run_id,
                    task.parent_task_id,
                    task.profile.value,
                    task.goal,
                    json.dumps(payload["context_bundle"], ensure_ascii=False),
                    task.done_criteria,
                    json.dumps(payload["tool_policy"], ensure_ascii=False),
                    task.artifact_dir,
                    status.value,
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
        return StoredAgentTask(
            **payload,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get_task(self, task_id: str) -> StoredAgentTask | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM subagent_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self, run_id: str) -> list[StoredAgentTask]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM subagent_tasks
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_task_status(self, task_id: str, status: AgentTaskStatus) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE subagent_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, utc_now().isoformat(), task_id),
            )

    def record_notification(self, run_id: str, notification: TaskNotification) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO task_notifications (
                    run_id,
                    task_id,
                    agent_profile,
                    status,
                    summary,
                    result_payload_json,
                    token_usage_json,
                    artifact_refs_json,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    notification.task_id,
                    notification.agent_profile.value,
                    notification.status.value,
                    notification.summary,
                    json.dumps(notification.result_payload, ensure_ascii=False),
                    json.dumps(notification.token_usage, ensure_ascii=False),
                    json.dumps(
                        [artifact.model_dump(mode="json") for artifact in notification.artifact_refs],
                        ensure_ascii=False,
                    ),
                    notification.error,
                    notification.created_at.isoformat(),
                ),
            )

    def list_notifications(self, run_id: str, *, task_id: str | None = None) -> list[TaskNotification]:
        sql = """
            SELECT *
            FROM task_notifications
            WHERE run_id = ?
        """
        params: list[object] = [run_id]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, id ASC"

        with self.database.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_notification(row) for row in rows]

    def save_artifacts(self, run_id: str, task_id: str, artifacts: list[TaskArtifactRef]) -> None:
        with self.database.connection() as conn:
            conn.execute("DELETE FROM task_artifacts WHERE task_id = ?", (task_id,))
            for index, artifact in enumerate(artifacts):
                conn.execute(
                    """
                    INSERT INTO task_artifacts (
                        run_id,
                        task_id,
                        name,
                        kind,
                        path,
                        description,
                        sort_order,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        artifact.name,
                        artifact.kind,
                        artifact.path,
                        artifact.description,
                        index,
                        utc_now().isoformat(),
                    ),
                )

    def list_artifacts(self, run_id: str, *, task_id: str | None = None) -> list[TaskArtifactRef]:
        sql = """
            SELECT name, kind, path, description
            FROM task_artifacts
            WHERE run_id = ?
        """
        params: list[object] = [run_id]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, sort_order ASC, id ASC"

        with self.database.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            TaskArtifactRef(
                name=row["name"],
                kind=row["kind"],
                path=row["path"],
                description=row["description"],
            )
            for row in rows
        ]

    def append_trace(self, trace: TaskExecutionTrace) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO task_execution_traces (
                    run_id,
                    task_id,
                    trace_type,
                    status,
                    message,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.run_id,
                    trace.task_id,
                    trace.trace_type.value,
                    trace.status,
                    trace.message,
                    json.dumps(trace.payload, ensure_ascii=False),
                    trace.created_at.isoformat(),
                ),
            )

    def list_traces(self, run_id: str, *, task_id: str | None = None) -> list[TaskExecutionTrace]:
        sql = """
            SELECT *
            FROM task_execution_traces
            WHERE run_id = ?
        """
        params: list[object] = [run_id]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, id ASC"

        with self.database.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            TaskExecutionTrace(
                run_id=row["run_id"],
                task_id=row["task_id"],
                trace_type=row["trace_type"],
                status=row["status"],
                message=row["message"],
                payload=json.loads(row["payload_json"] or "{}"),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> StoredAgentTask:
        return StoredAgentTask(
            id=row["id"],
            run_id=row["run_id"],
            parent_task_id=row["parent_task_id"],
            profile=row["profile"],
            goal=row["goal"],
            context_bundle=json.loads(row["context_bundle_json"] or "{}"),
            done_criteria=row["done_criteria"],
            tool_policy=ToolPolicy(**json.loads(row["tool_policy_json"] or "{}")),
            artifact_dir=row["artifact_dir"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_notification(row: sqlite3.Row) -> TaskNotification:
        return TaskNotification(
            task_id=row["task_id"],
            agent_profile=row["agent_profile"],
            status=row["status"],
            summary=row["summary"],
            result_payload=json.loads(row["result_payload_json"] or "{}"),
            token_usage=json.loads(row["token_usage_json"] or "{}"),
            artifact_refs=[
                TaskArtifactRef(**item)
                for item in json.loads(row["artifact_refs_json"] or "[]")
            ],
            error=row["error"],
            created_at=row["created_at"],
        )
