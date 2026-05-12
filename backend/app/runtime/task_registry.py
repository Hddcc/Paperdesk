"""In-memory registry for active runtime-managed agent tasks."""

from __future__ import annotations

from threading import Lock

from app.models import AgentTask, AgentTaskStatus, StoredAgentTask
from app.repositories import RuntimeRepository


class TaskRegistry:
    """Keep a small in-memory view over persisted subagent task state."""

    def __init__(self, runtime_repository: RuntimeRepository) -> None:
        self.runtime_repository = runtime_repository
        self._lock = Lock()
        self._tasks: dict[str, StoredAgentTask] = {}

    def register(self, task: AgentTask) -> StoredAgentTask:
        with self._lock:
            stored = self._tasks.get(task.id)
            if stored is not None:
                return stored
            persisted = self.runtime_repository.get_task(task.id)
            if persisted is not None:
                self._tasks[task.id] = persisted
                return persisted
            created = self.runtime_repository.create_task(task)
            self._tasks[task.id] = created
            return created

    def get(self, task_id: str) -> StoredAgentTask | None:
        with self._lock:
            stored = self._tasks.get(task_id)
            if stored is not None:
                return stored
        persisted = self.runtime_repository.get_task(task_id)
        if persisted is not None:
            with self._lock:
                self._tasks[task_id] = persisted
        return persisted

    def list_for_run(self, run_id: str) -> list[StoredAgentTask]:
        tasks = self.runtime_repository.list_tasks(run_id)
        with self._lock:
            for task in tasks:
                self._tasks[task.id] = task
        return tasks

    def mark_status(self, task_id: str, status: AgentTaskStatus) -> StoredAgentTask | None:
        self.runtime_repository.update_task_status(task_id, status)
        stored = self.runtime_repository.get_task(task_id)
        if stored is not None:
            with self._lock:
                self._tasks[task_id] = stored
        return stored

    def continue_task(self, task_id: str) -> StoredAgentTask | None:
        return self.get(task_id)
