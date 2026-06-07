"""Scratchpad persistence for subagent outputs."""

from __future__ import annotations

from app.models import AgentTask, TaskArtifactRef
from app.services.research_workspace_service import ResearchWorkspaceService


class ScratchpadStore:
    """Own task-scoped scratchpad writes under workspace/runs/<run_id>/scratch."""

    def __init__(self, workspace_service: ResearchWorkspaceService) -> None:
        self.workspace_service = workspace_service

    def write_json(
        self,
        task: AgentTask,
        filename: str,
        payload: object,
        *,
        description: str | None = None,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_json(task.run_id, task.id, filename, payload)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="json",
            description=description,
        )

    def write_markdown(
        self,
        task: AgentTask,
        filename: str,
        content: str,
        *,
        description: str | None = None,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_markdown(task.run_id, task.id, filename, content)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="markdown",
            description=description,
        )
