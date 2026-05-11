"""Research orchestrator coordinating the fixed PaperDesk workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from uuid import uuid4

from app.agents import (
    LibraryRetrieverAgent,
    PaperSearchAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)
from app.models import ResearchRequest, ResearchReport, ResearchState, TodoTask
from app.models.enums import ResearchRunStatus, TodoTaskStatus
from app.repositories import LibraryRepository, PaperRepository, ReportRepository, ResearchRepository

from .export_service import ExportService
from .research_workspace_service import ResearchWorkspaceService


EventSink = Callable[[dict], None]


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
        workspace_service: ResearchWorkspaceService,
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
        self.workspace_service = workspace_service

    def run(
        self,
        request: ResearchRequest,
        event_sink: EventSink | None = None,
    ) -> ResearchState:
        run = self.research_repository.create_run(str(uuid4()), request.topic)
        state = ResearchState(
            run_id=run.id,
            topic=request.topic,
            status=ResearchRunStatus.CREATED,
        )

        current_task: TodoTask | None = None
        try:
            state.status = ResearchRunStatus.PLANNING
            self.research_repository.update_run_status(run.id, state.status)
            self._emit_status(
                event_sink,
                state.status,
                "正在规划研究任务",
                run=run.model_dump(mode="json"),
                run_id=run.id,
            )

            tasks = self.topic_planner.plan(request.topic)
            state.todo_tasks = tasks
            self.research_repository.save_todo_tasks(run.id, tasks)
            self.workspace_service.write_todo_tasks(run.id, tasks)
            self._emit(
                event_sink,
                {
                    "type": "todo_list",
                    "run_id": run.id,
                    "tasks": [task.model_dump(mode="json") for task in tasks],
                },
            )

            documents = self.library_repository.list_documents()

            for task_index, task in enumerate(tasks, start=1):
                current_task = task

                self._mark_task_in_progress(
                    state,
                    task,
                    event_sink,
                    "正在执行在线论文检索",
                )
                paper_records = self.paper_search_agent.search(
                    task,
                    top_k=request.top_k_online,
                    search_provider=request.search_provider,
                )
                self.paper_repository.save_task_papers(task.id, paper_records)

                self._mark_task_in_progress(
                    state,
                    task,
                    event_sink,
                    "正在检索本地论文库",
                )
                evidence_items = self.library_retriever.retrieve(
                    task,
                    documents,
                    top_k=request.top_k_local,
                )

                self._mark_task_in_progress(
                    state,
                    task,
                    event_sink,
                    "正在合并证据并生成任务总结",
                )
                task_summary = self.reading_summarizer.summarize(task, paper_records, evidence_items)
                task.summary = task_summary.summary
                task.summary_markdown = task_summary.summary_markdown
                task.status = TodoTaskStatus.COMPLETED
                self.research_repository.update_task(run.id, task)
                state.task_summaries.append(task_summary)
                self.workspace_service.write_task_summary(run.id, task_index, task_summary)
                self._emit(
                    event_sink,
                    {
                        "type": "task_status",
                        "run_id": run.id,
                        "task_id": task.id,
                        "status": task.status.value,
                        "title": task.title,
                        "message": "任务总结已完成",
                    },
                )
                self._emit(
                    event_sink,
                    {
                        "type": "task_summary",
                        "run_id": run.id,
                        "task_id": task.id,
                        "title": task.title,
                        "summary_markdown": task_summary.summary_markdown,
                        "summary": task_summary.model_dump(mode="json"),
                    },
                )
                current_task = None

            state.status = ResearchRunStatus.WRITING_REPORT
            self.research_repository.update_run_status(run.id, state.status)
            self._emit_status(event_sink, state.status, "正在生成最终综述", run_id=run.id)

            report = self.report_writer.write(request.topic, state.task_summaries)
            report = self._attach_export_path(report, self.export_service.get_export_path(report.id))
            self.workspace_service.write_final_report(run.id, report)
            self.export_service.export_markdown(report)
            self.report_repository.create_report(report, run.id)

            state.status = ResearchRunStatus.COMPLETED
            state.report = report
            self.research_repository.update_run_status(run.id, state.status)
            self._emit(
                event_sink,
                {
                    "type": "report",
                    "run_id": run.id,
                    "report_id": report.id,
                    "markdown": report.markdown,
                    "report": report.model_dump(mode="json"),
                },
            )
            self._emit(event_sink, {"type": "done", "run_id": run.id})
            return state
        except Exception as exc:
            self._handle_failure(state, current_task, event_sink, exc)
            raise

    def run_stream(self, request: ResearchRequest) -> Iterator[dict]:
        event_queue: Queue[dict | object] = Queue()
        sentinel = object()

        def emit(event: dict) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                self.run(request, event_sink=emit)
            except Exception:
                pass
            finally:
                event_queue.put(sentinel)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            event = event_queue.get()
            if event is sentinel:
                break
            yield event
        thread.join()

    def _mark_task_in_progress(
        self,
        state: ResearchState,
        task: TodoTask,
        event_sink: EventSink | None,
        message: str,
    ) -> None:
        task.status = TodoTaskStatus.IN_PROGRESS
        state.status = ResearchRunStatus.RUNNING_TASK
        self.research_repository.update_task(state.run_id, task)
        self.research_repository.update_run_status(state.run_id, state.status)
        self._emit_status(event_sink, state.status, message)
        self._emit(
            event_sink,
            {
                "type": "task_status",
                "run_id": state.run_id,
                "task_id": task.id,
                "status": task.status.value,
                "title": task.title,
                "message": message,
            },
        )

    def _handle_failure(
        self,
        state: ResearchState,
        current_task: TodoTask | None,
        event_sink: EventSink | None,
        exc: Exception,
    ) -> None:
        if current_task is not None:
            current_task.status = TodoTaskStatus.FAILED
            self.research_repository.update_task(state.run_id, current_task)
            self._emit(
                event_sink,
                {
                    "type": "task_status",
                    "run_id": state.run_id,
                    "task_id": current_task.id,
                    "status": current_task.status.value,
                    "title": current_task.title,
                    "message": "任务执行失败",
                },
            )

        state.status = ResearchRunStatus.FAILED
        self.research_repository.update_run_status(state.run_id, state.status)
        self._emit_status(event_sink, state.status, "研究流程执行失败")
        self._emit(event_sink, {"type": "error", "detail": str(exc)})

    @staticmethod
    def _emit(event_sink: EventSink | None, event: dict) -> None:
        if event_sink is not None:
            event_sink(event)

    def _emit_status(
        self,
        event_sink: EventSink | None,
        status: ResearchRunStatus,
        message: str,
        **extra: object,
    ) -> None:
        event = {
            "type": "status",
            "status": status.value,
            "message": message,
        }
        event.update(extra)
        self._emit(
            event_sink,
            event,
        )

    @staticmethod
    def _attach_export_path(report: ResearchReport, export_path: Path) -> ResearchReport:
        _ = export_path
        return report
