"""Research orchestrator coordinating the fixed PaperDesk workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.agents import (
    LibraryRetrieverAgent,
    PaperSearchAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)
from app.models import ResearchRequest, ResearchReport, TaskSummary
from app.models.enums import ResearchRunStatus
from app.repositories import LibraryRepository, PaperRepository, ReportRepository, ResearchRepository

from .export_service import ExportService


class ResearchOrchestrator:
    """Run the fixed sequential research workflow and emit SSE events."""

    def __init__(
        self,
        research_repository: ResearchRepository,
        paper_repository: PaperRepository,
        library_repository: LibraryRepository,
        report_repository: ReportRepository,
        topic_planner: TopicPlannerAgent,
        paper_search_agent: PaperSearchAgent,
        library_retriever: LibraryRetrieverAgent,
        reading_summarizer: ReadingSummarizerAgent,
        report_writer: ReportWriterAgent,
        export_service: ExportService,
    ) -> None:
        self.research_repository = research_repository
        self.paper_repository = paper_repository
        self.library_repository = library_repository
        self.report_repository = report_repository
        self.topic_planner = topic_planner
        self.paper_search_agent = paper_search_agent
        self.library_retriever = library_retriever
        self.reading_summarizer = reading_summarizer
        self.report_writer = report_writer
        self.export_service = export_service

    def run_stream(self, request: ResearchRequest) -> Iterator[dict]:
        run = self.research_repository.create_run(str(uuid4()), request.topic)
        yield {"type": "run_created", "run": run.model_dump(mode="json")}

        self.research_repository.update_run_status(run.id, ResearchRunStatus.PLANNING)
        yield {
            "type": "status",
            "status": ResearchRunStatus.PLANNING.value,
            "message": "正在规划研究任务",
        }

        tasks = self.topic_planner.plan(request.topic)
        self.research_repository.save_todo_tasks(run.id, tasks)
        yield {
            "type": "todo_list",
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }

        task_summaries: list[TaskSummary] = []
        documents = self.library_repository.list_documents()

        for task in tasks:
            task.status = ResearchRunStatus.SEARCHING_ONLINE
            self.research_repository.update_task(run.id, task)
            self.research_repository.update_run_status(run.id, ResearchRunStatus.SEARCHING_ONLINE)
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": task.status.value,
                "title": task.title,
                "message": "正在执行在线论文检索",
            }
            paper_records = self.paper_search_agent.search(
                task,
                top_k=request.top_k_online,
                search_provider=request.search_provider,
            )
            self.paper_repository.save_task_papers(task.id, paper_records)

            task.status = ResearchRunStatus.RETRIEVING_LOCAL
            self.research_repository.update_task(run.id, task)
            self.research_repository.update_run_status(run.id, ResearchRunStatus.RETRIEVING_LOCAL)
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": task.status.value,
                "title": task.title,
                "message": "正在检索本地论文库",
            }
            evidence_items = self.library_retriever.retrieve(
                task,
                documents,
                top_k=request.top_k_local,
            )

            task.status = ResearchRunStatus.SUMMARIZING_TASK
            self.research_repository.update_task(run.id, task)
            self.research_repository.update_run_status(run.id, ResearchRunStatus.SUMMARIZING_TASK)
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": task.status.value,
                "title": task.title,
                "message": "正在合并证据并生成任务总结",
            }
            task_summary = self.reading_summarizer.summarize(task, paper_records, evidence_items)
            task.summary = task_summary.summary
            task.summary_markdown = task_summary.summary_markdown
            task.status = ResearchRunStatus.COMPLETED
            self.research_repository.update_task(run.id, task)
            task_summaries.append(task_summary)
            yield {
                "type": "task_result",
                "task_id": task.id,
                "task": task.model_dump(mode="json"),
                "papers": [paper.model_dump(mode="json") for paper in paper_records],
                "evidence_items": [item.model_dump(mode="json") for item in evidence_items],
                "summary": task_summary.model_dump(mode="json"),
            }

        self.research_repository.update_run_status(run.id, ResearchRunStatus.WRITING_REPORT)
        yield {
            "type": "status",
            "status": ResearchRunStatus.WRITING_REPORT.value,
            "message": "正在生成最终综述",
        }

        report = self.report_writer.write(request.topic, task_summaries)
        report = self._attach_export_path(report, self.export_service.get_export_path(report.id))
        self.export_service.export_markdown(report)
        self.report_repository.create_report(report, run.id)
        self.research_repository.update_run_status(run.id, ResearchRunStatus.COMPLETED)

        yield {
            "type": "final_report",
            "report": report.model_dump(mode="json"),
        }
        yield {"type": "done", "run_id": run.id}

    @staticmethod
    def _attach_export_path(report: ResearchReport, export_path: Path) -> ResearchReport:
        markdown = report.markdown + f"\n\n> 导出路径：{export_path}"
        return report.model_copy(update={"markdown": markdown})
