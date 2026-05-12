"""Research orchestrator coordinating the phase-13 main-agent loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
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
    EvidenceItem,
    PaperRecord,
    ResearchActionType,
    ResearchPlanItem,
    ResearchRequest,
    ResearchReport,
    ResearchRuntimePhase,
    ResearchRuntimeState,
    ResearchRuntimeStep,
    ResearchRunStatus,
    ResearchState,
    ResearchStepStatus,
    ResearchToolCallRecord,
    ResearchToolResult,
    ResearchToolResultStatus,
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
from app.runtime import (
    MainAgentRuntime,
    MessageBus,
    ResearchToolExecutor,
    ScratchpadStore,
    SubagentRunner,
    TaskRegistry,
)

from .export_service import ExportService
from .research_workspace_service import ResearchWorkspaceService


EventSink = Callable[[dict], None]


class ResearchOrchestrator:
    """Run a research request through a single-main-agent step loop."""

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
        self.tool_executor = ResearchToolExecutor(
            topic_planner=topic_planner,
            paper_search_agent=paper_search_agent,
            library_retriever=library_retriever,
            reading_summarizer=reading_summarizer,
            report_writer=report_writer,
            workspace_service=workspace_service,
        )

    def run(
        self,
        request: ResearchRequest,
        event_sink: EventSink | None = None,
    ) -> ResearchState:
        run_id = str(uuid4())
        run = self.research_repository.create_run(run_id, request.topic, request_payload=request)
        runtime_state = ResearchRuntimeState(
            run_id=run.id,
            goal=request.topic,
            current_phase=self.main_runtime.initial_phase(),
        )
        state = ResearchState(
            run_id=run.id,
            topic=request.topic,
            status=ResearchRunStatus.CREATED,
            runtime_state=runtime_state,
        )

        self._emit(
            event_sink,
            {
                "type": "run_created",
                "run_id": run.id,
                "run": run.model_dump(mode="json"),
            },
        )
        self._checkpoint(
            runtime_state,
            request,
            status=self._status_from_phase(runtime_state.current_phase),
            stop_reason=None,
            event_sink=event_sink,
        )
        return self._run_loop(state, request, event_sink=event_sink, resumed=False)

    def resume(
        self,
        run_id: str,
        event_sink: EventSink | None = None,
    ) -> ResearchState:
        run = self.research_repository.get_run(run_id)
        if run is None:
            raise ValueError("Research run not found")

        runtime_state = self.research_repository.get_runtime_state(run_id)
        if runtime_state is None:
            runtime_state = self.workspace_service.read_runtime_state(run_id)
        if runtime_state is None:
            raise ValueError("Research runtime checkpoint not found")

        request = self.research_repository.get_request_payload(run_id)
        if request is None:
            raise ValueError("Research request payload not found")

        if run.status not in {
            ResearchRunStatus.PLANNING,
            ResearchRunStatus.RUNNING_TASK,
            ResearchRunStatus.WRITING_REPORT,
            ResearchRunStatus.FAILED,
        }:
            raise ValueError("Current research run cannot be resumed")

        runtime_state.stop_reason = None
        if runtime_state.current_phase == ResearchRuntimePhase.FAILED:
            runtime_state.current_phase = ResearchRuntimePhase.EXECUTING
        runtime_state.active_step = None

        state = self._load_state_snapshot(run_id, run.topic, runtime_state)
        self._emit(
            event_sink,
            {
                "type": "research_resumed",
                "run_id": run_id,
                "runtime_state": runtime_state.model_dump(mode="json"),
            },
        )
        self._emit(
            event_sink,
            {
                "type": "todo_list",
                "run_id": run_id,
                "tasks": [task.model_dump(mode="json") for task in state.todo_tasks],
            },
        )
        self._checkpoint(
            runtime_state,
            request,
            status=self._status_from_phase(runtime_state.current_phase),
            stop_reason=None,
            event_sink=event_sink,
        )
        return self._run_loop(state, request, event_sink=event_sink, resumed=True)

    def run_stream(self, request: ResearchRequest) -> Iterator[dict]:
        yield from self._stream_worker(lambda emit: self.run(request, event_sink=emit))

    def resume_stream(self, run_id: str) -> Iterator[dict]:
        yield from self._stream_worker(lambda emit: self.resume(run_id, event_sink=emit))

    def _stream_worker(self, work: Callable[[EventSink], object]) -> Iterator[dict]:
        event_queue: Queue[dict | object] = Queue()
        sentinel = object()

        def emit(event: dict) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                work(emit)
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

    def _run_loop(
        self,
        state: ResearchState,
        request: ResearchRequest,
        *,
        event_sink: EventSink | None,
        resumed: bool,
    ) -> ResearchState:
        runtime_state = state.runtime_state
        assert runtime_state is not None

        documents = self.library_repository.list_documents()
        current_task: TodoTask | None = None

        try:
            while True:
                action, task_id = self.main_runtime.next_action(runtime_state)
                plan_item = self._find_plan_item(runtime_state, task_id)
                current_task = self._to_todo_task(plan_item) if plan_item is not None else None
                self._start_step(runtime_state, action, plan_item, event_sink)

                result = self._execute_action(
                    runtime_state,
                    request,
                    documents,
                    action,
                    plan_item,
                )
                if result.status == ResearchToolResultStatus.FAILED:
                    self._handle_tool_failure(runtime_state, request, action, plan_item, result, event_sink)
                    if result.retryable and runtime_state.failure_count < 3:
                        continue
                    raise RuntimeError(result.error or result.summary)

                self._apply_tool_result(state, request, action, plan_item, result, event_sink)
                current_task = None

                if action == ResearchActionType.FINISH:
                    runtime_state.current_phase = ResearchRuntimePhase.COMPLETED
                    state.status = ResearchRunStatus.COMPLETED
                    self.research_repository.update_run_status(state.run_id, state.status, stop_reason=None)
                    self._emit(event_sink, {"type": "done", "run_id": state.run_id})
                    return state
                if action == ResearchActionType.FAIL:
                    raise RuntimeError(runtime_state.stop_reason or "Research runtime failed")
        except Exception as exc:
            self._handle_failure(state, current_task, request, event_sink, exc, resumed=resumed)
            raise

    def _execute_action(
        self,
        runtime_state: ResearchRuntimeState,
        request: ResearchRequest,
        documents,
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
    ) -> ResearchToolResult:
        if action == ResearchActionType.PLAN:
            return self.tool_executor.plan(
                runtime_state.run_id,
                request,
                direct_task=self.main_runtime.should_use_direct_task(request),
            )
        if action == ResearchActionType.SEARCH_ONLINE and plan_item is not None:
            return self.tool_executor.search_online(runtime_state.run_id, plan_item, request)
        if action == ResearchActionType.SEARCH_LOCAL and plan_item is not None:
            return self.tool_executor.search_local(
                runtime_state.run_id,
                plan_item,
                documents,
                top_k_local=request.top_k_local,
            )
        if action == ResearchActionType.REVISE_PLAN and plan_item is not None:
            revised_query = f"{request.topic} {plan_item.intent}".strip()
            return ResearchToolResult(
                status=ResearchToolResultStatus.COMPLETED,
                summary=f"Revised task query to focus on {plan_item.intent}.",
                payload={"query": revised_query},
                retryable=False,
            )
        if action == ResearchActionType.SUMMARIZE_EVIDENCE and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            return self.tool_executor.summarize_evidence(
                runtime_state.run_id,
                plan_item,
                evidence.paper_records,
                evidence.evidence_items,
                degraded=self.main_runtime.should_degrade(plan_item, evidence),
            )
        if action == ResearchActionType.FINALIZE_REPORT:
            return self.tool_executor.finalize_report(
                runtime_state.run_id,
                request.topic,
                state_task_summaries(runtime_state),
            )
        if action == ResearchActionType.FINISH:
            return ResearchToolResult(
                status=ResearchToolResultStatus.COMPLETED,
                summary="Research workflow finished.",
            )
        if action == ResearchActionType.FAIL:
            return ResearchToolResult(
                status=ResearchToolResultStatus.FAILED,
                summary=runtime_state.stop_reason or "Research workflow failed.",
                retryable=False,
                error=runtime_state.stop_reason or "Research workflow failed.",
            )
        return ResearchToolResult(
            status=ResearchToolResultStatus.FAILED,
            summary=f"Unsupported action: {action.value}",
            retryable=False,
            error=f"Unsupported action: {action.value}",
        )

    def _apply_tool_result(
        self,
        state: ResearchState,
        request: ResearchRequest,
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
        result: ResearchToolResult,
        event_sink: EventSink | None,
    ) -> None:
        runtime_state = state.runtime_state
        assert runtime_state is not None
        now = datetime.now(timezone.utc)
        active_step = runtime_state.active_step
        task_id = plan_item.task_id if plan_item is not None else None

        if action == ResearchActionType.PLAN:
            runtime_state.plan_items = [
                ResearchPlanItem(**item)
                for item in result.payload.get("plan_items", [])
            ]
            state.todo_tasks = [self._to_todo_task(item) for item in runtime_state.plan_items]
            self.research_repository.save_todo_tasks(state.run_id, state.todo_tasks)
            self.workspace_service.write_todo_tasks(state.run_id, state.todo_tasks)
            self._emit(
                event_sink,
                {
                    "type": "todo_list",
                    "run_id": state.run_id,
                    "tasks": [task.model_dump(mode="json") for task in state.todo_tasks],
                },
            )
        elif action == ResearchActionType.SEARCH_ONLINE and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            evidence.paper_records = [PaperRecord(**item) for item in result.payload.get("paper_records", [])]
            evidence.online_completed = True
            plan_item.status = TodoTaskStatus.IN_PROGRESS
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            self._emit_task_status(state.run_id, plan_item, "在线论文检索完成。", event_sink)
        elif action == ResearchActionType.SEARCH_LOCAL and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            evidence.evidence_items = [EvidenceItem(**item) for item in result.payload.get("evidence_items", [])]
            evidence.local_completed = True
            plan_item.status = TodoTaskStatus.IN_PROGRESS
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            self._emit_task_status(state.run_id, plan_item, "本地证据检索完成。", event_sink)
        elif action == ResearchActionType.REVISE_PLAN and plan_item is not None:
            revised_query = str(result.payload.get("query") or plan_item.query).strip()
            if revised_query and revised_query != plan_item.query:
                plan_item.query = revised_query
                plan_item.query_history.append(revised_query)
            plan_item.revise_count += 1
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            evidence.paper_records = []
            evidence.evidence_items = []
            evidence.online_completed = False
            evidence.local_completed = False
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            self._emit_task_status(state.run_id, plan_item, result.summary, event_sink)
        elif action == ResearchActionType.SUMMARIZE_EVIDENCE and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            if self.main_runtime.should_degrade(plan_item, evidence):
                evidence.degraded = True
                plan_item.degraded = True
            task_summary = plan_item.to_task_summary(evidence)
            if result.payload.get("task_summary"):
                task_summary = task_summary.model_validate(result.payload["task_summary"])
            plan_item.summary = task_summary.summary
            plan_item.summary_markdown = task_summary.summary_markdown
            plan_item.status = TodoTaskStatus.COMPLETED
            self.paper_repository.save_task_papers(plan_item.task_id, evidence.paper_records)
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            if plan_item.task_id not in runtime_state.completed_items:
                runtime_state.completed_items.append(plan_item.task_id)
            state.todo_tasks = [self._to_todo_task(item) for item in runtime_state.plan_items]
            state.task_summaries = state_task_summaries(runtime_state)
            task_index = runtime_state.plan_items.index(plan_item) + 1
            self.workspace_service.write_task_summary(state.run_id, task_index, task_summary)
            self._emit_task_status(state.run_id, plan_item, "Task summary completed.", event_sink)
            self._emit(
                event_sink,
                {
                    "type": "task_summary",
                    "run_id": state.run_id,
                    "task_id": plan_item.task_id,
                    "title": plan_item.title,
                    "summary_markdown": task_summary.summary_markdown,
                    "summary": task_summary.model_dump(mode="json"),
                },
            )
        elif action == ResearchActionType.FINALIZE_REPORT:
            report_payload = result.payload.get("report")
            if report_payload:
                report = ResearchReport.model_validate(report_payload)
            else:
                report = self.report_writer.write(request.topic, state_task_summaries(runtime_state))
            state.report = report
            self.export_service.export_markdown(report)
            self.report_repository.create_report(report, state.run_id)
            runtime_state.report_id = report.id
            self._emit(
                event_sink,
                {
                    "type": "report_completed",
                    "run_id": state.run_id,
                    "report_id": report.id,
                    "message": "Final report generated.",
                },
            )
            self._emit(
                event_sink,
                {
                    "type": "report",
                    "run_id": state.run_id,
                    "report_id": report.id,
                    "markdown": report.markdown,
                    "report": report.model_dump(mode="json"),
                },
            )
        elif action == ResearchActionType.FINISH:
            runtime_state.stop_reason = None

        runtime_state.step_count += 1
        runtime_state.failure_count = 0
        runtime_state.working_summary = self.main_runtime.summarize_working_notes(runtime_state)
        if active_step is not None:
            active_step.status = ResearchStepStatus.COMPLETED
            runtime_state.tool_history.append(
                ResearchToolCallRecord(
                    step_id=active_step.step_id,
                    action=action,
                    task_id=task_id,
                    status=ResearchToolResultStatus.COMPLETED,
                    summary=result.summary,
                    retryable=result.retryable,
                    paper_count=len(result.payload.get("paper_records", [])),
                    evidence_count=len(result.payload.get("evidence_items", [])),
                    artifact_refs=result.artifacts,
                    created_at=now,
                )
            )
        self._emit(
            event_sink,
            {
                "type": "agent_step_completed",
                "run_id": state.run_id,
                "task_id": task_id,
                "action": action.value,
                "summary": result.summary,
            },
        )
        runtime_state.active_step = None
        runtime_state.current_phase = self.main_runtime.step_phase(action)
        state.status = self._status_from_phase(runtime_state.current_phase)
        self._checkpoint(runtime_state, request, status=state.status, stop_reason=None, event_sink=event_sink)

    def _handle_tool_failure(
        self,
        runtime_state: ResearchRuntimeState,
        request: ResearchRequest,
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
        result: ResearchToolResult,
        event_sink: EventSink | None,
    ) -> None:
        runtime_state.failure_count += 1
        runtime_state.stop_reason = result.error or result.summary
        active_step = runtime_state.active_step
        if active_step is not None:
            active_step.status = ResearchStepStatus.FAILED
            runtime_state.tool_history.append(
                ResearchToolCallRecord(
                    step_id=active_step.step_id,
                    action=action,
                    task_id=plan_item.task_id if plan_item is not None else None,
                    status=ResearchToolResultStatus.FAILED,
                    summary=result.summary,
                    retryable=result.retryable,
                    error=result.error or result.summary,
                    artifact_refs=result.artifacts,
                    created_at=datetime.now(timezone.utc),
                )
            )
        self._emit(
            event_sink,
            {
                "type": "agent_step_failed",
                "run_id": runtime_state.run_id,
                "task_id": plan_item.task_id if plan_item is not None else None,
                "action": action.value,
                "error": result.error or result.summary,
                "retryable": result.retryable,
            },
        )
        runtime_state.active_step = None
        if result.retryable and runtime_state.failure_count < 3:
            self._checkpoint(
                runtime_state,
                request,
                status=self._status_from_phase(self.main_runtime.step_phase(action)),
                stop_reason=runtime_state.stop_reason,
                event_sink=event_sink,
            )
            return
        runtime_state.current_phase = ResearchRuntimePhase.FAILED
        self._checkpoint(
            runtime_state,
            request,
            status=ResearchRunStatus.FAILED,
            stop_reason=runtime_state.stop_reason,
            event_sink=event_sink,
        )

    def _start_step(
        self,
        runtime_state: ResearchRuntimeState,
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
        event_sink: EventSink | None,
    ) -> None:
        runtime_state.current_phase = self.main_runtime.step_phase(action)
        runtime_state.active_step = ResearchRuntimeStep(
            step_id=str(uuid4()),
            action=action,
            task_id=plan_item.task_id if plan_item is not None else None,
            attempt=runtime_state.failure_count + 1,
            started_at=datetime.now(timezone.utc),
        )
        self.message_bus.append_trace(
            run_id=runtime_state.run_id,
            task_id=plan_item.task_id if plan_item is not None else None,
            trace_type=TraceEventType.STATUS,
            status=f"step:{action.value}",
            message=f"Main agent started step {action.value}.",
            payload={
                "action": action.value,
                "task_id": plan_item.task_id if plan_item is not None else None,
                "title": plan_item.title if plan_item is not None else None,
                "attempt": runtime_state.active_step.attempt,
            },
        )
        self._emit(
            event_sink,
            {
                "type": "agent_step_started",
                "run_id": runtime_state.run_id,
                "task_id": plan_item.task_id if plan_item is not None else None,
                "action": action.value,
                "title": plan_item.title if plan_item is not None else None,
                "attempt": runtime_state.active_step.attempt,
            },
        )

    def _checkpoint(
        self,
        runtime_state: ResearchRuntimeState,
        request: ResearchRequest,
        *,
        status: ResearchRunStatus,
        stop_reason: str | None,
        event_sink: EventSink | None,
    ) -> None:
        runtime_state.last_checkpoint_at = datetime.now(timezone.utc)
        runtime_state.stop_reason = stop_reason
        self.research_repository.save_runtime_state(
            runtime_state.run_id,
            runtime_state,
            request_payload=request,
            status=status,
            stop_reason=stop_reason,
        )
        self.workspace_service.write_runtime_state(runtime_state.run_id, runtime_state)
        self.message_bus.append_trace(
            run_id=runtime_state.run_id,
            task_id=runtime_state.active_step.task_id if runtime_state.active_step is not None else None,
            trace_type=TraceEventType.STATUS,
            status="checkpoint_saved",
            message="Research runtime checkpoint saved.",
            payload={
                "current_phase": runtime_state.current_phase.value,
                "step_count": runtime_state.step_count,
                "stop_reason": runtime_state.stop_reason,
            },
        )
        self._emit(
            event_sink,
            {
                "type": "checkpoint_saved",
                "run_id": runtime_state.run_id,
                "current_phase": runtime_state.current_phase.value,
                "step_count": runtime_state.step_count,
                "stop_reason": runtime_state.stop_reason,
            },
        )
        self._emit_status(
            run_id=runtime_state.run_id,
            status=status,
            message=self._status_message(runtime_state),
            payload={"stop_reason": runtime_state.stop_reason} if runtime_state.stop_reason else None,
            event_sink=event_sink,
        )

    def _handle_failure(
        self,
        state: ResearchState,
        current_task: TodoTask | None,
        request: ResearchRequest,
        event_sink: EventSink | None,
        exc: Exception,
        *,
        resumed: bool,
    ) -> None:
        runtime_state = state.runtime_state
        assert runtime_state is not None
        _ = resumed
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

        runtime_state.current_phase = ResearchRuntimePhase.FAILED
        runtime_state.stop_reason = str(exc)
        state.status = ResearchRunStatus.FAILED
        self.research_repository.update_run_status(state.run_id, state.status, stop_reason=str(exc))
        self._checkpoint(runtime_state, request, status=state.status, stop_reason=str(exc), event_sink=event_sink)
        self._emit(event_sink, {"type": "error", "detail": str(exc), "run_id": state.run_id})

    def _load_state_snapshot(
        self,
        run_id: str,
        topic: str,
        runtime_state: ResearchRuntimeState,
    ) -> ResearchState:
        tasks = self.research_repository.list_tasks(run_id)
        task_summaries = state_task_summaries(runtime_state)
        report = self.report_repository.get_report_by_run_id(run_id)
        return ResearchState(
            run_id=run_id,
            topic=topic,
            status=self._status_from_phase(runtime_state.current_phase),
            runtime_state=runtime_state,
            todo_tasks=tasks,
            task_summaries=task_summaries,
            subagent_tasks=self.runtime_repository.list_tasks(run_id),
            task_notifications=self.runtime_repository.list_notifications(run_id),
            task_traces=self.runtime_repository.list_traces(run_id),
            task_artifacts=self.runtime_repository.list_artifacts(run_id),
            report=report,
        )

    @staticmethod
    def _find_plan_item(
        runtime_state: ResearchRuntimeState,
        task_id: str | None,
    ) -> ResearchPlanItem | None:
        if task_id is None:
            return None
        for item in runtime_state.plan_items:
            if item.task_id == task_id:
                return item
        return None

    @staticmethod
    def _get_or_create_buffer(runtime_state: ResearchRuntimeState, task_id: str):
        for item in runtime_state.evidence_buffer:
            if item.task_id == task_id:
                return item
        from app.models import ResearchEvidenceBufferItem

        evidence = ResearchEvidenceBufferItem(task_id=task_id)
        runtime_state.evidence_buffer.append(evidence)
        return evidence

    @staticmethod
    def _to_todo_task(plan_item: ResearchPlanItem | None) -> TodoTask:
        assert plan_item is not None
        return TodoTask(
            id=plan_item.task_id,
            title=plan_item.title,
            intent=plan_item.intent,
            query=plan_item.query,
            status=plan_item.status,
            summary=plan_item.summary,
            summary_markdown=plan_item.summary_markdown,
        )

    @staticmethod
    def _status_from_phase(phase: ResearchRuntimePhase) -> ResearchRunStatus:
        if phase == ResearchRuntimePhase.PLANNING:
            return ResearchRunStatus.PLANNING
        if phase in {ResearchRuntimePhase.EXECUTING, ResearchRuntimePhase.SUMMARIZING}:
            return ResearchRunStatus.RUNNING_TASK
        if phase == ResearchRuntimePhase.WRITING_REPORT:
            return ResearchRunStatus.WRITING_REPORT
        if phase == ResearchRuntimePhase.COMPLETED:
            return ResearchRunStatus.COMPLETED
        return ResearchRunStatus.FAILED

    @staticmethod
    def _status_message(runtime_state: ResearchRuntimeState) -> str:
        mapping = {
            ResearchRuntimePhase.PLANNING: "Research coordinator is planning the execution strategy.",
            ResearchRuntimePhase.EXECUTING: "Main agent is gathering evidence.",
            ResearchRuntimePhase.SUMMARIZING: "Main agent is revising or summarizing evidence.",
            ResearchRuntimePhase.WRITING_REPORT: "Coordinator is writing the final report.",
            ResearchRuntimePhase.COMPLETED: "Research workflow completed.",
            ResearchRuntimePhase.FAILED: "Research workflow failed.",
        }
        return mapping[runtime_state.current_phase]

    def _emit_task_status(
        self,
        run_id: str,
        plan_item: ResearchPlanItem,
        message: str,
        event_sink: EventSink | None,
    ) -> None:
        self._emit(
            event_sink,
            {
                "type": "task_status",
                "run_id": run_id,
                "task_id": plan_item.task_id,
                "status": plan_item.status.value,
                "title": plan_item.title,
                "message": message,
            },
        )

    def _emit_status(
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
    def _emit(event_sink: EventSink | None, event: dict) -> None:
        if event_sink is not None:
            event_sink(event)


def state_task_summaries(runtime_state: ResearchRuntimeState):
    summaries = []
    for item in runtime_state.plan_items:
        if item.summary is None and item.summary_markdown is None:
            continue
        evidence = next(
            (buffer for buffer in runtime_state.evidence_buffer if buffer.task_id == item.task_id),
            None,
        )
        if evidence is None:
            from app.models import ResearchEvidenceBufferItem

            evidence = ResearchEvidenceBufferItem(task_id=item.task_id)
        summaries.append(item.to_task_summary(evidence))
    return summaries
