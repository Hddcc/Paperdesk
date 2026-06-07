"""Reading summarizer agent stub."""

from __future__ import annotations

from app.models import EvidenceItem, PaperRecord, TaskSummary, TodoTask


class ReadingSummarizerAgent:
    """Merge online and local evidence into a task summary."""

    def summarize(
        self,
        task: TodoTask,
        paper_records: list[PaperRecord],
        evidence_items: list[EvidenceItem],
    ) -> TaskSummary:
        citation_lines = [item.citation_label for item in evidence_items]
        paper_lines = [
            self._format_paper_reference(index, record)
            for index, record in enumerate(paper_records, start=1)
        ]
        summary = (
            f"任务聚焦：{task.intent}\n\n"
            f"- 在线论文结果：{len(paper_records)} 条\n"
            f"- 本地证据结果：{len(evidence_items)} 条\n"
            "\n"
            f"在线论文参考：\n{chr(10).join(paper_lines) if paper_lines else '暂无'}\n"
            f"本地证据引用：{'; '.join(citation_lines) if citation_lines else '暂无'}"
        )
        return TaskSummary(
            task_id=task.id,
            title=task.title,
            intent=task.intent,
            summary=summary,
            summary_markdown=summary,
            evidence_items=evidence_items,
            paper_records=paper_records,
        )

    @staticmethod
    def _format_paper_reference(index: int, record: PaperRecord) -> str:
        authors = ReadingSummarizerAgent._format_authors(record.authors)
        year = str(record.year) if record.year is not None else "年份未知"
        venue = f" {record.venue}." if record.venue else ""
        locator = ""
        if record.doi:
            locator = f" DOI: {record.doi}"
        elif record.url:
            locator = f" URL: {record.url}"
        return f"[{index}] {authors}. {record.title}. {year}.{venue}{locator}"

    @staticmethod
    def _format_authors(authors: list[str]) -> str:
        if not authors:
            return "作者未知"
        if len(authors) <= 3:
            return ", ".join(authors)
        return f"{', '.join(authors[:3])} 等"
