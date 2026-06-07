"""Message-bus style helpers for coordinator and subagent communication."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from app.models import TaskExecutionTrace, TaskNotification, TraceEventType
from app.repositories import RuntimeRepository

EventSink = Callable[[dict], None]


class MessageBus:
    """Persist traces and forward UI-facing events through a single helper."""

    def __init__(self, runtime_repository: RuntimeRepository) -> None:
        self.runtime_repository = runtime_repository

    @staticmethod
    def emit(event_sink: EventSink | None, event: dict[str, Any]) -> None:
        if event_sink is not None:
            event_sink(event)

    def append_trace(
        self,
        *,
        run_id: str,
        task_id: str | None,
        trace_type: TraceEventType,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> TaskExecutionTrace:
        trace = TaskExecutionTrace(
            run_id=run_id,
            task_id=task_id,
            trace_type=trace_type,
            status=status,
            message=message,
            payload=payload or {},
            created_at=datetime.now(timezone.utc),
        )
        self.runtime_repository.append_trace(trace)
        return trace

    def publish_control(
        self,
        *,
        run_id: str,
        task_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        safe_payload = payload or {}
        self.append_trace(
            run_id=run_id,
            task_id=task_id,
            trace_type=TraceEventType.CONTROL,
            status=event_type,
            message=message,
            payload=safe_payload,
        )
        self.emit(
            event_sink,
            {
                "type": event_type,
                "run_id": run_id,
                "task_id": task_id,
                "message": message,
                **safe_payload,
            },
        )

    def publish_notification(
        self,
        *,
        run_id: str,
        notification: TaskNotification,
        event_type: str,
        event_sink: EventSink | None = None,
    ) -> None:
        self.runtime_repository.record_notification(run_id, notification)
        payload = notification.model_dump(mode="json")
        self.append_trace(
            run_id=run_id,
            task_id=notification.task_id,
            trace_type=TraceEventType.NOTIFICATION,
            status=notification.status.value,
            message=notification.summary,
            payload=payload,
        )
        self.emit(
            event_sink,
            {
                "type": event_type,
                "run_id": run_id,
                "task_id": notification.task_id,
                "agent_profile": notification.agent_profile.value,
                "status": notification.status.value,
                "summary": notification.summary,
                "result_payload": notification.result_payload,
                "artifact_refs": [artifact.model_dump(mode="json") for artifact in notification.artifact_refs],
                "token_usage": notification.token_usage,
                "error": notification.error,
                "notification": payload,
            },
        )

    def publish_merge(
        self,
        *,
        run_id: str,
        task_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        safe_payload = payload or {}
        self.append_trace(
            run_id=run_id,
            task_id=task_id,
            trace_type=TraceEventType.MERGE,
            status=event_type,
            message=message,
            payload=safe_payload,
        )
        self.emit(
            event_sink,
            {
                "type": event_type,
                "run_id": run_id,
                "task_id": task_id,
                "message": message,
                **safe_payload,
            },
        )
