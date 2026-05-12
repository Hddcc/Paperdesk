"""Research orchestrator coordinating a Claude Code-style subagent workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
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
from app.models import (
    AgentTask,
    AgentTaskStatus,
    ResearchRequest,
    ResearchReport,
    ResearchState,
    ResearchRunStatus,
    TaskNotification,
    TodoTask,
    TodoTaskStatus,
    TraceEventType,
)
from app.repositories import (
    LibraryRepository,
    PaperRepository,
    ReportRepository,
    ResearchRepository,
    RuntimeRepository,
)
from app.runtime import MainAgentRuntime, MessageBus, ScratchpadStore, SubagentRunner, TaskRegistry, WorkerResult

from .export_service import ExportService
from .research_workspace_service import ResearchWorkspaceService


EventSink = Callable[[dict], None]


class ResearchOrchestrator:
    """Run the research workflow through a main-agent + subagent runtime."""

    def __init__(
        self,
        research_repository: ResearchRepository,
        paper_repository: PaperRepository,
        library_repository: LibraryRepository,
        report_repository: ReportRepository,
        runtime_repository: RuntimeRepository,
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
        self.runtime_repository = runtime_repository
        self.topic_planner = topic_planner
        self.paper_search_agent = paper_search_agent
        self.library_retriever = library_retriever
        self.reading_summarizer = reading_summarizer
        self.report_writer = report_writer
        self.export_service = export_service
        self.workspace_service = workspace_service

        self.main_runtime = MainAgentRuntime()
        self.scratchpad_store = ScratchpadStore(workspace_service)
        self.message_bus = MessageBus(runtime_repository)
        self.task_registry = TaskRegistry(runtime_repository)
        self.subagent_runner = SubagentRunner(
            runtime_repository=runtime_repository,
            task_registry=self.task_registry,
            message_bus=self.message_bus,
            scratchpad_store=self.scratchpad_store,
        )

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
            decision = self.main_runtime.decide(request)
            self._emit_coordinator_status(
                run_id=run.id,
                status=state.status,
                message="Research coordinator is planning the execution strategy.",
                payload={"decision": decision.model_dump(mode="json"), "run": run.model_dump(mode="json")},
                event_sink=event_sink,
            )

            if decision.spawn_subagents:
                tasks = self.topic_planner.plan(request.topic)
            else:
                tasks = [self._build_direct_task(request.topic)]
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
                    "Coordinator is dispatching research work for this task.",
                )

                if decision.spawn_subagents:
                    notifications = self._run_explore_subagents(
                        run_id=run.id,
                        task=task,
                        request=request,
                        documents=documents,
                        event_sink=event_sink,
                    )
                else:
                    notifications = self._run_direct_path(
                        run_id=run.id,
                        task=task,
                        request=request,
                        documents=documents,
                    )

                failed_notifications = [item for item in notifications if item.status != AgentTaskStatus.COMPLETED]
                if failed_notifications:
                    failure = failed_notifications[0]
                    raise RuntimeError(failure.error or failure.summary)

                if decision.spawn_subagents and self.main_runtime.should_verify(notifications):
                    verify_notification = self.subagent_runner.spawn(
                        self.main_runtime.build_verify_task(
                            run_id=run.id,
                            parent_task=task,
                            notifications=notifications,
                        ),
                        self._verify_worker,
                        event_sink=event_sink,
                    )
                    notifications.append(verify_notification)
                    if verify_notification.status != AgentTaskStatus.COMPLETED:
                        raise RuntimeError(verify_notification.error or verify_notification.summary)

                self.message_bus.publish_merge(
                    run_id=run.id,
                    task_id=task.id,
                    event_type="task_merge_started",
                    message="Main agent is merging subagent evidence.",
                    payload={"notification_count": len(notifications)},
                    event_sink=event_sink,
                )

                paper_records, evidence_items = self._collect_notification_payloads(notifications)
                self.paper_repository.save_task_papers(task.id, paper_records)

                task_summary = self.reading_summarizer.summarize(task, paper_records, evidence_items)
                task.summary = task_summary.summary
                task.summary_markdown = task_summary.summary_markdown
                task.status = TodoTaskStatus.COMPLETED
                self.research_repository.update_task(run.id, task)
                state.task_summaries.append(task_summary)
                self.workspace_service.write_task_summary(run.id, task_index, task_summary)

                self.message_bus.publish_merge(
                    run_id=run.id,
                    task_id=task.id,
                    event_type="task_merge_completed",
                    message="Main agent finished merging evidence for this task.",
                    payload={
                        "paper_count": len(paper_records),
                        "evidence_count": len(evidence_items),
                        "notification_xml": self.main_runtime.format_notifications_xml(notifications),
                    },
                    event_sink=event_sink,
                )
                self._emit(
                    event_sink,
                    {
                        "type": "task_status",
                        "run_id": run.id,
                        "task_id": task.id,
                        "status": task.status.value,
                        "title": task.title,
                        "message": "Task summary completed.",
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
            self._emit_coordinator_status(
                run_id=run.id,
                status=state.status,
                message="Coordinator is writing the final report.",
                payload={"task_count": len(state.task_summaries)},
                event_sink=event_sink,
            )

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
                    "type": "report_completed",
                    "run_id": run.id,
                    "report_id": report.id,
                    "message": "Final report generated.",
                },
            )
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

    def _run_explore_subagents(
        self,
        *,
        run_id: str,
        task: TodoTask,
        request: ResearchRequest,
        documents,
        event_sink: EventSink | None,
    ) -> list[TaskNotification]:
        online_task = self.main_runtime.build_explore_task(
            run_id=run_id,
            parent_task=task,
            channel="online",
            context_bundle={
                "search_provider": request.search_provider,
                "top_k_online": request.top_k_online,
            },
            done_criteria="Collect a concise set of online paper candidates and summarize them without mutating state.",
        )
        local_task = self.main_runtime.build_explore_task(
            run_id=run_id,
            parent_task=task,
            channel="local",
            context_bundle={
                "top_k_local": request.top_k_local,
                "document_count": len(documents),
            },
            done_criteria="Collect local evidence candidates from the library and summarize them without mutating state.",
        )
        return self.subagent_runner.run_parallel(
            [
                (online_task, lambda agent_task, progress: self._online_explore_worker(agent_task, request, progress)),
                (
                    local_task,
                    lambda agent_task, progress: self._local_explore_worker(
                        agent_task,
                        documents,
                        request.top_k_local,
                        progress,
                    ),
                ),
            ],
            event_sink=event_sink,
        )

    def _run_direct_path(
        self,
        *,
        run_id: str,
        task: TodoTask,
        request: ResearchRequest,
        documents,
    ) -> list[TaskNotification]:
        online_task = self.main_runtime.build_explore_task(
            run_id=run_id,
            parent_task=task,
            channel="online-direct",
            context_bundle={
                "search_provider": request.search_provider,
                "top_k_online": request.top_k_online,
            },
            done_criteria="Directly gather online evidence without spawning a separate subagent runtime.",
        )
        local_task = self.main_runtime.build_explore_task(
            run_id=run_id,
            parent_task=task,
            channel="local-direct",
            context_bundle={"top_k_local": request.top_k_local},
            done_criteria="Directly gather local evidence without spawning a separate subagent runtime.",
        )
        return [
            self._complete_direct_notification(self._online_explore_worker(online_task, request, lambda *_args, **_kwargs: None), online_task),
            self._complete_direct_notification(self._local_explore_worker(local_task, documents, request.top_k_local, lambda *_args, **_kwargs: None), local_task),
        ]

    @staticmethod
    def _complete_direct_notification(result: WorkerResult, task: AgentTask) -> TaskNotification:
        return TaskNotification(
            task_id=task.id,
            agent_profile=task.profile,
            status=AgentTaskStatus.COMPLETED,
            summary=result.summary,
            result_payload=result.result_payload,
            token_usage=result.token_usage,
            artifact_refs=result.artifact_refs,
            created_at=datetime.now(timezone.utc),
        )

    def _online_explore_worker(
        self,
        task: AgentTask,
        request: ResearchRequest,
        progress,
    ) -> WorkerResult:
        progress("Searching online papers.", {"channel": "online"})
        todo_task = TodoTask(**task.context_bundle["todo_task"])
        paper_records = self.paper_search_agent.search(
            todo_task,
            top_k=request.top_k_online,
            search_provider=request.search_provider,
        )
        payload = [record.model_dump(mode="json") for record in paper_records]
        artifacts = [
            self.scratchpad_store.write_json(
                task,
                "papers.json",
                payload,
                description="Normalized online paper candidates",
            ),
            self.scratchpad_store.write_markdown(
                task,
                "analysis.md",
                "\n".join(
                    [
                        f"# Online Explore: {todo_task.title}",
                        "",
                        f"Collected {len(paper_records)} paper candidates.",
                        "",
                        *[f"- {record.title}" for record in paper_records],
                    ]
                ),
                description="Compact online paper summary",
            ),
        ]
        return WorkerResult(
            summary=f"Collected {len(paper_records)} online paper candidates.",
            result_payload={"channel": "online", "paper_records": payload},
            token_usage={"result_items": len(paper_records)},
            artifact_refs=artifacts,
        )

    def _local_explore_worker(
        self,
        task: AgentTask,
        documents,
        top_k_local: int,
        progress,
    ) -> WorkerResult:
        progress("Retrieving local evidence.", {"channel": "local"})
        todo_task = TodoTask(**task.context_bundle["todo_task"])
        evidence_items = self.library_retriever.retrieve(
            todo_task,
            documents,
            top_k=top_k_local,
        )
        payload = [item.model_dump(mode="json") for item in evidence_items]
        artifacts = [
            self.scratchpad_store.write_json(
                task,
                "evidence.json",
                payload,
                description="Retrieved local evidence items",
            ),
            self.scratchpad_store.write_markdown(
                task,
                "analysis.md",
                "\n".join(
                    [
                        f"# Local Explore: {todo_task.title}",
                        "",
                        f"Collected {len(evidence_items)} local evidence items.",
                        "",
                        *[f"- {item.citation_label}" for item in evidence_items],
                    ]
                ),
                description="Compact local evidence summary",
            ),
        ]
        return WorkerResult(
            summary=f"Collected {len(evidence_items)} local evidence items.",
            result_payload={"channel": "local", "evidence_items": payload},
            token_usage={"result_items": len(evidence_items)},
            artifact_refs=artifacts,
        )

    def _verify_worker(self, task: AgentTask, progress) -> WorkerResult:
        progress("Verifying evidence completeness.", {"channel": "verify"})
        notifications = [TaskNotification(**item) for item in task.context_bundle.get("notifications", [])]
        has_content = any(notification.result_payload for notification in notifications)
        needs_followup = not has_content
        markdown = "\n".join(
            [
                "# Verification",
                "",
                f"Notifications reviewed: {len(notifications)}",
                f"Needs follow-up: {'yes' if needs_followup else 'no'}",
            ]
        )
        artifacts = [
            self.scratchpad_store.write_markdown(
                task,
                "analysis.md",
                markdown,
                description="Verification outcome for merged evidence",
            )
        ]
        return WorkerResult(
            summary="Verification finished for the collected evidence.",
            result_payload={"channel": "verify", "needs_followup": needs_followup},
            token_usage={"result_items": len(notifications)},
            artifact_refs=artifacts,
        )

    @staticmethod
    def _collect_notification_payloads(notifications: list[TaskNotification]):
        from app.models import EvidenceItem, PaperRecord

        paper_records: list[PaperRecord] = []
        evidence_items: list[EvidenceItem] = []
        for notification in notifications:
            for item in notification.result_payload.get("paper_records", []):
                paper_records.append(PaperRecord(**item))
            for item in notification.result_payload.get("evidence_items", []):
                evidence_items.append(EvidenceItem(**item))
        return paper_records, evidence_items

    @staticmethod
    def _build_direct_task(topic: str) -> TodoTask:
        return TodoTask(
            id=str(uuid4()),
            title=topic,
            intent="Direct research pass for a compact topic",
            query=topic,
            status=TodoTaskStatus.PENDING,
        )

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
        self._emit_coordinator_status(
            run_id=state.run_id,
            status=state.status,
            message=message,
            payload={"task_id": task.id, "title": task.title},
            event_sink=event_sink,
        )
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
                    "message": "Task execution failed.",
                },
            )

        state.status = ResearchRunStatus.FAILED
        self.research_repository.update_run_status(state.run_id, state.status)
        self._emit_coordinator_status(
            run_id=state.run_id,
            status=state.status,
            message="Research workflow failed.",
            payload={"error": str(exc)},
            event_sink=event_sink,
        )
        self._emit(event_sink, {"type": "error", "detail": str(exc)})

    @staticmethod
    def _emit(event_sink: EventSink | None, event: dict) -> None:
        if event_sink is not None:
            event_sink(event)

    def _emit_coordinator_status(
        self,
        *,
        run_id: str,
        status: ResearchRunStatus,
        message: str,
        payload: dict | None,
        event_sink: EventSink | None,
    ) -> None:
        safe_payload = payload or {}
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=safe_payload.get("task_id"),
            trace_type=TraceEventType.STATUS,
            status=status.value,
            message=message,
            payload=safe_payload,
        )
        self._emit(
            event_sink,
            {
                "type": "status",
                "status": status.value,
                "message": message,
                "run_id": run_id,
                **safe_payload,
            },
        )
        self._emit(
            event_sink,
            {
                "type": "coordinator_status",
                "status": status.value,
                "message": message,
                "run_id": run_id,
                **safe_payload,
            },
        )

    @staticmethod
    def _attach_export_path(report: ResearchReport, export_path: Path) -> ResearchReport:
        _ = export_path
        return report
