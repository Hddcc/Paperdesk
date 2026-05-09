"""Final report writer agent stub."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.models import ResearchReport, TaskSummary


class ReportWriterAgent:
    """Create a Markdown report from task summaries."""

    def write(self, topic: str, task_summaries: list[TaskSummary]) -> ResearchReport:
        citations: list[str] = []
        sections: list[str] = []
        for index, task_summary in enumerate(task_summaries, start=1):
            sections.append(
                "\n".join(
                    [
                        f"## {index}. {task_summary.title}",
                        "",
                        task_summary.summary,
                        "",
                    ]
                )
            )
            for evidence in task_summary.evidence_items:
                citations.append(
                    f"{evidence.citation_label} - {evidence.metadata.get('file_path', evidence.source_id)}"
                )
            for paper in task_summary.paper_records:
                citations.append(f"{paper.title} - {paper.url or paper.doi or 'mock source'}")

        citation_lines = [f"- {citation}" for citation in citations] if citations else ["- 暂无引用"]
        markdown = "\n".join(
            [
                f"# {topic}",
                "",
                "## 概述",
                "",
                "本报告由 PaperDesk 00/01 可运行骨架自动生成，用于验证固定工作流、"
                "多 Agent 边界、SQLite 持久化与前端研究工作台展示。",
                "",
                *sections,
                "## 参考线索",
                "",
                *citation_lines,
            ]
        )
        return ResearchReport(
            id=str(uuid4()),
            topic=topic,
            markdown=markdown,
            task_summaries=task_summaries,
            citations=citations,
            created_at=datetime.now(timezone.utc),
        )
