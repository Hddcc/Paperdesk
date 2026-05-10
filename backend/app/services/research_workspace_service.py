"""Workspace persistence helpers for phase-06 research runs."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import ResearchReport, TaskSummary, TodoTask


class ResearchWorkspaceService:
    """Persist run-scoped intermediate research artifacts under workspace/runs."""

    def __init__(self, workspace_dir: Path) -> None:
        self.runs_dir = workspace_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_todo_tasks(self, run_id: str, tasks: list[TodoTask]) -> Path:
        destination = self.get_run_dir(run_id) / "todo_tasks.json"
        payload = [task.model_dump(mode="json") for task in tasks]
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def write_task_summary(self, run_id: str, task_index: int, task_summary: TaskSummary) -> Path:
        destination = self.get_run_dir(run_id) / f"task_{task_index}_summary.md"
        content = (task_summary.summary_markdown or task_summary.summary).rstrip() + "\n"
        destination.write_text(content, encoding="utf-8")
        return destination

    def write_final_report(self, run_id: str, report: ResearchReport) -> Path:
        destination = self.get_run_dir(run_id) / "final_report.md"
        destination.write_text(report.markdown.rstrip() + "\n", encoding="utf-8")
        return destination
