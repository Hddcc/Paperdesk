"""Persist final Markdown reports to the workspace."""

from __future__ import annotations

from pathlib import Path

from app.models import ResearchReport


class ExportService:
    """Handle report export paths for the skeleton."""

    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def export_markdown(self, report: ResearchReport) -> Path:
        destination = self.get_export_path(report.id)
        destination.write_text(report.markdown, encoding="utf-8")
        return destination

    def get_export_path(self, report_id: str) -> Path:
        return self.report_dir / f"{report_id}.md"
