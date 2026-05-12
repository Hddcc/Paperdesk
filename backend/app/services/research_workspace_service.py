"""Workspace persistence helpers for phase-06 research runs."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import ResearchReport, ResearchRuntimeState, TaskSummary, TodoTask


class ResearchWorkspaceService:
    """Persist run-scoped intermediate research artifacts under workspace/runs."""

    def __init__(self, workspace_dir: Path) -> None:
        self.runs_dir = workspace_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def get_scratch_dir(self, run_id: str, task_id: str) -> Path:
        compact_task_id = task_id.replace("-", "")[:12]
        scratch_dir = self.get_run_dir(run_id) / "s" / compact_task_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        return scratch_dir

    def write_todo_tasks(self, run_id: str, tasks: list[TodoTask]) -> Path:
        destination = self.get_run_dir(run_id) / "todo_tasks.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [task.model_dump(mode="json") for task in tasks]
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def write_task_summary(self, run_id: str, task_index: int, task_summary: TaskSummary) -> Path:
        destination = self.get_run_dir(run_id) / f"task_{task_index}_summary.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = (task_summary.summary_markdown or task_summary.summary).rstrip() + "\n"
        destination.write_text(content, encoding="utf-8")
        return destination

    def write_final_report(self, run_id: str, report: ResearchReport) -> Path:
        destination = self.get_run_dir(run_id) / "final_report.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.markdown.rstrip() + "\n", encoding="utf-8")
        return destination

    def write_runtime_state(self, run_id: str, runtime_state: ResearchRuntimeState) -> Path:
        destination = self.get_run_dir(run_id) / "runtime_state.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(runtime_state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    def read_runtime_state(self, run_id: str) -> ResearchRuntimeState | None:
        destination = self.get_run_dir(run_id) / "runtime_state.json"
        if not destination.exists():
            return None
        return ResearchRuntimeState(**json.loads(destination.read_text(encoding="utf-8")))

    def write_scratch_json(self, run_id: str, task_id: str, filename: str, payload: object) -> Path:
        destination = self.get_scratch_dir(run_id, task_id) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def write_scratch_markdown(self, run_id: str, task_id: str, filename: str, content: str) -> Path:
        destination = self.get_scratch_dir(run_id, task_id) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content.rstrip() + "\n", encoding="utf-8")
        return destination
