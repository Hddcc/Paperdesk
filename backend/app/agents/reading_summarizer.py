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
        paper_lines = [record.title for record in paper_records]
        summary = (
            f"这是针对“{task.title}”的骨架版任务总结。\n\n"
            f"- 在线论文结果：{len(paper_records)} 条\n"
            f"- 本地证据结果：{len(evidence_items)} 条\n"
            f"- 说明：当前实现为 00/01 章节可运行骨架，摘要内容为 mock/stub，"
            f"用于验证多 Agent 职责边界、状态流和 SSE 展示。\n\n"
            f"在线论文样例：{'; '.join(paper_lines) if paper_lines else '暂无'}\n"
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
