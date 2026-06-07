"""Subagent execution helpers with task registration and communication hooks."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models import AgentTask, AgentTaskStatus, TaskNotification
from app.repositories import RuntimeRepository

from .message_bus import EventSink, MessageBus
from .scratchpad_store import ScratchpadStore
from .task_registry import TaskRegistry


ProgressCallback = Callable[[str, dict[str, Any] | None], None]
WorkerCallable = Callable[[AgentTask, ProgressCallback], "WorkerResult"]


@dataclass(slots=True)
class WorkerResult:
    """Normalized output returned by a subagent worker."""

    summary: str
    result_payload: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    artifact_refs: list = field(default_factory=list)


class SubagentRunner:
    """Run subagent workers and convert their outputs into notifications."""

    def __init__(
        self,
        runtime_repository: RuntimeRepository,
        task_registry: TaskRegistry,
        message_bus: MessageBus,
        scratchpad_store: ScratchpadStore,
    ) -> None:
        self.runtime_repository = runtime_repository
        self.task_registry = task_registry
        self.message_bus = message_bus
        self.scratchpad_store = scratchpad_store

    def spawn(
        self,
        task: AgentTask,
        worker: WorkerCallable,
        *,
        event_sink: EventSink | None = None,
        continue_existing: bool = False,
    ) -> TaskNotification:
        if continue_existing:
            existing = self.task_registry.continue_task(task.id)
            if existing is None:
                self.task_registry.register(task)
            self.message_bus.publish_control(
                run_id=task.run_id,
                task_id=task.id,
                event_type="subagent_spawned",
                message="Continuing subagent task with existing context.",
                payload={"continued": True, "agent_profile": task.profile.value},
                event_sink=event_sink,
            )
        else:
            self.task_registry.register(task)
            self.message_bus.publish_control(
                run_id=task.run_id,
                task_id=task.id,
                event_type="subagent_spawned",
                message="Spawned subagent task.",
                payload={"agent_profile": task.profile.value, "parent_task_id": task.parent_task_id},
                event_sink=event_sink,
            )

        self.task_registry.mark_status(task.id, AgentTaskStatus.RUNNING)
        self.message_bus.publish_control(
            run_id=task.run_id,
            task_id=task.id,
            event_type="subagent_started",
            message="Subagent execution started.",
            payload={"agent_profile": task.profile.value},
            event_sink=event_sink,
        )

        def progress(message: str, payload: dict[str, Any] | None = None) -> None:
            self.message_bus.publish_control(
                run_id=task.run_id,
                task_id=task.id,
                event_type="subagent_progress",
                message=message,
                payload={"agent_profile": task.profile.value, **(payload or {})},
                event_sink=event_sink,
            )

        try:
            result = worker(task, progress)
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=task.profile,
                status=AgentTaskStatus.COMPLETED,
                summary=result.summary,
                result_payload=result.result_payload,
                token_usage=result.token_usage,
                artifact_refs=result.artifact_refs,
                created_at=datetime.now(timezone.utc),
            )
            self.task_registry.mark_status(task.id, notification.status)
            self.runtime_repository.save_artifacts(task.run_id, task.id, notification.artifact_refs)
            self.message_bus.publish_notification(
                run_id=task.run_id,
                notification=notification,
                event_type="subagent_completed",
                event_sink=event_sink,
            )
            return notification
        except Exception as exc:
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=task.profile,
                status=AgentTaskStatus.FAILED,
                summary=f"Subagent failed while executing {task.goal}",
                result_payload={},
                token_usage={},
                artifact_refs=[],
                error=str(exc),
                created_at=datetime.now(timezone.utc),
            )
            self.task_registry.mark_status(task.id, notification.status)
            self.message_bus.publish_notification(
                run_id=task.run_id,
                notification=notification,
                event_type="subagent_failed",
                event_sink=event_sink,
            )
            return notification

    def run_parallel(
        self,
        tasks_with_workers: list[tuple[AgentTask, WorkerCallable]],
        *,
        event_sink: EventSink | None = None,
    ) -> list[TaskNotification]:
        if not tasks_with_workers:
            return []

        notifications: dict[str, TaskNotification] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(tasks_with_workers))) as executor:
            futures = {
                executor.submit(self.spawn, task, worker, event_sink=event_sink): task.id
                for task, worker in tasks_with_workers
            }
            for future in as_completed(futures):
                notification = future.result()
                notifications[notification.task_id] = notification

        ordered: list[TaskNotification] = []
        for task, _worker in tasks_with_workers:
            ordered.append(notifications[task.id])
        return ordered
