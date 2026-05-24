"""Research orchestrator coordinating the phase-13 main-agent loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
    PlannerProviderType,
    ResearchActionDecision,
    ResearchActionType,
    ResearchEvidenceBufferItem,
    ResearchEvidencePolicy,
    ResearchPlanItem,
    ResearchPlanOperation,
    ResearchPlanOperationType,
    ResearchPlannerCandidate,
    ResearchRequest,
    ResearchReport,
    ResearchRuntimePhase,
    ResearchRuntimeState,
    ResearchRuntimeStep,
    ResearchRunStatus,
    ResearchState,
    ResearchTaskRoute,
    ResearchTaskType,
    ResearchStepStatus,
    ResearchToolStrategy,
    ResearchToolCallRecord,
    ResearchToolResult,
    ResearchToolResultClassification,
    ResearchToolResultStatus,
    SkillDefinition,
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
    default_read_only_academic_mcp_declarations,
    MainAgentRuntime,
    MessageBus,
    ReadOnlyMcpAdapter,
    RuleBasedPlannerCandidateProvider,
    ResearchToolExecutor,
    ScratchpadStore,
    SkillRegistry,
    SubagentRunner,
    TaskRegistry,
    ToolRegistry,
)

from .export_service import ExportService
from .research_context_assembler import ResearchContextAssembler
from .research_skill_consistency_checker import (
    ResearchSkillConsistencyChecker,
    ResearchSkillConsistencyReport,
)
from .research_task_router import ResearchTaskRouter
from .research_workspace_service import ResearchWorkspaceService
from .skill_selector import SkillSelector


EventSink = Callable[[dict], None]


@dataclass(frozen=True)
class _RouteRequestResult:
    task_route: ResearchTaskRoute
    consistency_report: ResearchSkillConsistencyReport


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
        context_assembler: ResearchContextAssembler,
        enable_experimental_mcp: bool = False,
        enable_subagent_execution: bool = False,
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
        self.context_assembler = context_assembler
        self.enable_experimental_mcp = enable_experimental_mcp
        self.enable_subagent_execution = enable_subagent_execution

        mcp_tools = (
            ReadOnlyMcpAdapter(
                allowed_tool_ids={
                    "mcp/academic_search",
                    "mcp/academic_metadata",
                    "mcp/read_only_web_fetch",
                }
            ).normalize(default_read_only_academic_mcp_declarations())
            if enable_experimental_mcp
            else []
        )
        self.tool_registry = ToolRegistry(mcp_tools, enable_experimental_mcp=enable_experimental_mcp)
        self.skill_registry = SkillRegistry()
        self.main_runtime = MainAgentRuntime(
            {tool.tool_id: tool for tool in self.tool_registry.list_enabled()}
        )
        self.scratchpad_store = ScratchpadStore(workspace_service)
        self.message_bus = MessageBus(runtime_repository)
        self.task_registry = TaskRegistry(runtime_repository)
        self.subagent_runner = SubagentRunner(
            runtime_repository=runtime_repository,
            task_registry=self.task_registry,
            message_bus=self.message_bus,
            scratchpad_store=self.scratchpad_store,
        )
        self.planner_candidate_provider = RuleBasedPlannerCandidateProvider()
        self.task_router = ResearchTaskRouter()
        self.skill_selector = SkillSelector()
        self.skill_consistency_checker = ResearchSkillConsistencyChecker()
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
        route_result = self._route_request(request)
        task_route = route_result.task_route
        run = self.research_repository.create_run(run_id, request.topic, request_payload=request)
        runtime_state = ResearchRuntimeState(
            run_id=run.id,
            goal=request.topic,
            current_phase=self.main_runtime.initial_phase(),
            task_route=task_route,
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
                "task_route": task_route.model_dump(mode="json"),
            },
        )
        self._emit_task_route(run.id, task_route, event_sink)
        self._emit_research_skill_selected(run.id, task_route, event_sink)
        self._append_research_skill_consistency_trace(run.id, route_result.consistency_report)
        self._checkpoint(
            runtime_state,
            request,
            status=self._status_from_phase(runtime_state.current_phase),
            stop_reason=None,
            event_sink=event_sink,
        )
        if task_route.allow_single_pass and not task_route.use_main_agent_loop:
            return self._run_single_pass(state, request, event_sink=event_sink)
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
        consistency_report = None
        if runtime_state.task_route is None:
            route_result = self._route_request(request)
            runtime_state.task_route = route_result.task_route
            consistency_report = route_result.consistency_report
        runtime_state.active_step = None

        state = self._load_state_snapshot(run_id, run.topic, runtime_state)
        self._emit(
            event_sink,
            {
                "type": "research_resumed",
                "run_id": run_id,
                "runtime_state": runtime_state.model_dump(mode="json"),
                "task_route": runtime_state.task_route.model_dump(mode="json") if runtime_state.task_route else None,
            },
        )
        if runtime_state.task_route is not None:
            self._emit_task_route(run_id, runtime_state.task_route, event_sink)
        if consistency_report is not None:
            self._append_research_skill_consistency_trace(run_id, consistency_report)
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
        if (
            runtime_state.task_route is not None
            and runtime_state.task_route.allow_single_pass
            and not runtime_state.task_route.use_main_agent_loop
            and not runtime_state.tool_history
        ):
            return self._run_single_pass(state, request, event_sink=event_sink)
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
                candidate_item = self.main_runtime.peek_next_pending_item(runtime_state)
                self.context_assembler.refresh(runtime_state, active_task=candidate_item)
                decision = self.main_runtime.next_action(runtime_state, request)
                runtime_state.last_decision = decision
                action = decision.action_type
                task_id = decision.target_task_id
                plan_item = self._find_plan_item(runtime_state, task_id)
                current_task = self._to_todo_task(plan_item) if plan_item is not None else None
                self._start_step(runtime_state, decision, plan_item, event_sink)

                try:
                    result = self._execute_action(
                        runtime_state,
                        request,
                        documents,
                        action,
                        plan_item,
                    )
                except Exception as exc:
                    result = ResearchToolResult(
                        status=ResearchToolResultStatus.FAILED,
                        summary=str(exc),
                        retryable=action in {ResearchActionType.SEARCH_ONLINE, ResearchActionType.SEARCH_LOCAL},
                        error=str(exc),
                    )
                if result.status == ResearchToolResultStatus.FAILED:
                    self._handle_tool_failure(runtime_state, request, decision, plan_item, result, event_sink)
                    if result.retryable and runtime_state.failure_count < 3:
                        continue
                    raise RuntimeError(result.error or result.summary)

                self._apply_tool_result(state, request, decision, plan_item, result, event_sink)
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

    def _run_single_pass(
        self,
        state: ResearchState,
        request: ResearchRequest,
        *,
        event_sink: EventSink | None,
    ) -> ResearchState:
        runtime_state = state.runtime_state
        assert runtime_state is not None
        documents = self.library_repository.list_documents()
        current_task: TodoTask | None = None

        try:
            plan_decision = self._single_pass_decision(
                ResearchActionType.PLAN,
                "轻量任务由路由允许单轮执行，先生成单项计划。",
            )
            runtime_state.last_decision = plan_decision
            self._start_step(runtime_state, plan_decision, None, event_sink)
            plan_result = self._execute_action(
                runtime_state,
                request,
                documents,
                ResearchActionType.PLAN,
                None,
            )
            self._apply_tool_result(state, request, plan_decision, None, plan_result, event_sink)

            plan_item = runtime_state.plan_items[0] if runtime_state.plan_items else None
            current_task = self._to_todo_task(plan_item) if plan_item is not None else None
            if plan_item is None:
                raise RuntimeError("Lightweight route did not create a plan item")

            if runtime_state.task_route is not None and runtime_state.task_route.needs_local_knowledge:
                local_decision = self._single_pass_decision(
                    ResearchActionType.SEARCH_LOCAL,
                    "轻量任务优先读取已指定的本地材料。",
                    task_id=plan_item.task_id,
                )
                runtime_state.last_decision = local_decision
                self._start_step(runtime_state, local_decision, plan_item, event_sink)
                local_result = self._execute_action(
                    runtime_state,
                    request,
                    documents,
                    ResearchActionType.SEARCH_LOCAL,
                    plan_item,
                )
                self._apply_tool_result(state, request, local_decision, plan_item, local_result, event_sink)
                if self._should_promote_single_pass_after_local_search(runtime_state, plan_item):
                    self._promote_route_to_main_loop(runtime_state, request, state)
                    return self._run_loop(state, request, event_sink=event_sink, resumed=True)

            if runtime_state.task_route is not None and runtime_state.task_route.needs_online_search:
                online_decision = self._single_pass_decision(
                    ResearchActionType.SEARCH_ONLINE,
                    "轻量任务按路由要求补充在线证据。",
                    task_id=plan_item.task_id,
                    request=request,
                )
                runtime_state.last_decision = online_decision
                self._start_step(runtime_state, online_decision, plan_item, event_sink)
                online_result = self._execute_action(
                    runtime_state,
                    request,
                    documents,
                    ResearchActionType.SEARCH_ONLINE,
                    plan_item,
                )
                self._apply_tool_result(state, request, online_decision, plan_item, online_result, event_sink)

            summary_decision = self._single_pass_decision(
                ResearchActionType.SUMMARIZE_EVIDENCE,
                "轻量任务按当前证据直接生成任务总结。",
                task_id=plan_item.task_id,
            )
            runtime_state.last_decision = summary_decision
            self._start_step(runtime_state, summary_decision, plan_item, event_sink)
            summary_result = self._execute_action(
                runtime_state,
                request,
                documents,
                ResearchActionType.SUMMARIZE_EVIDENCE,
                plan_item,
            )
            self._apply_tool_result(state, request, summary_decision, plan_item, summary_result, event_sink)

            finalize_decision = self._single_pass_decision(
                ResearchActionType.FINALIZE_REPORT,
                "轻量任务已完成总结，进入任务型结果生成。",
            )
            runtime_state.last_decision = finalize_decision
            self._start_step(runtime_state, finalize_decision, None, event_sink)
            finalize_result = self._execute_action(
                runtime_state,
                request,
                documents,
                ResearchActionType.FINALIZE_REPORT,
                None,
            )
            self._apply_tool_result(state, request, finalize_decision, None, finalize_result, event_sink)

            finish_decision = self._single_pass_decision(
                ResearchActionType.FINISH,
                "轻量结果已生成，研究流程可以结束。",
            )
            runtime_state.last_decision = finish_decision
            self._start_step(runtime_state, finish_decision, None, event_sink)
            finish_result = self._execute_action(
                runtime_state,
                request,
                documents,
                ResearchActionType.FINISH,
                None,
            )
            self._apply_tool_result(state, request, finish_decision, None, finish_result, event_sink)

            runtime_state.current_phase = ResearchRuntimePhase.COMPLETED
            state.status = ResearchRunStatus.COMPLETED
            self.research_repository.update_run_status(state.run_id, state.status, stop_reason=None)
            self._emit(event_sink, {"type": "done", "run_id": state.run_id})
            return state
        except Exception as exc:
            self._handle_failure(state, current_task, request, event_sink, exc, resumed=False)
            raise

    def _promote_route_to_main_loop(
        self,
        runtime_state: ResearchRuntimeState,
        request: ResearchRequest,
        state: ResearchState,
    ) -> None:
        if runtime_state.task_route is not None:
            runtime_state.task_route.use_main_agent_loop = True
            runtime_state.task_route.allow_single_pass = False
            runtime_state.task_route.needs_online_search = True
            runtime_state.task_route.evidence_policy = ResearchEvidencePolicy.ONLINE_SUPPLEMENT
            runtime_state.task_route.rationale = (
                f"{runtime_state.task_route.rationale} 本地证据不足，已升级到主 Agent 路径。"
            )
        for item in runtime_state.plan_items:
            item.status = TodoTaskStatus.PENDING
            if item.task_id in runtime_state.completed_items:
                runtime_state.completed_items.remove(item.task_id)
        state.todo_tasks = [self._to_todo_task(item) for item in runtime_state.plan_items]
        for task in state.todo_tasks:
            self.research_repository.update_task(state.run_id, task)

    def _should_promote_single_pass_after_local_search(
        self,
        runtime_state: ResearchRuntimeState,
        plan_item: ResearchPlanItem,
    ) -> bool:
        if runtime_state.task_route is None:
            return False
        if runtime_state.task_route.task_type == ResearchTaskType.PAPER_SUMMARY:
            return False
        if runtime_state.task_route.task_type not in {
            ResearchTaskType.QA,
            ResearchTaskType.METHOD_EXPLAINER,
        }:
            return False
        evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
        assessment = evidence.evidence_assessment
        if not evidence.evidence_items:
            return True
        return not assessment.has_relevant_evidence or assessment.sufficiency_score < 0.35

    def _single_pass_decision(
        self,
        action: ResearchActionType,
        reason: str,
        *,
        task_id: str | None = None,
        request: ResearchRequest | None = None,
    ) -> ResearchActionDecision:
        strategy_id = self._strategy_id_for_action(action, request)
        return ResearchActionDecision(
            action_type=action,
            selected_tool=strategy_id,
            tool_strategy=self.main_runtime.tool_strategy(
                action,
                strategy_id=strategy_id,
                request=request,
                rationale="由任务路由直接选择的轻量执行策略。",
            ),
            reason=reason,
            target_task_id=task_id,
        )

    def _strategy_id_for_action(self, action: ResearchActionType, request: ResearchRequest | None = None) -> str:
        if action == ResearchActionType.PLAN:
            return "plan/rule_based_initial"
        if action == ResearchActionType.SEARCH_ONLINE:
            provider = (request.search_provider if request is not None else None) or ""
            provider = provider.casefold()
            if provider in {"", "all", "auto"} and self.tool_registry.get("mcp/academic_search") is not None:
                return "mcp/academic_search"
            if provider == "openalex":
                return "search_online/openalex_primary"
            if provider == "arxiv":
                return "search_online/arxiv_primary"
            return "search_online/mixed_broad_recall"
        if action == ResearchActionType.SEARCH_LOCAL:
            return "search_local/vector_recall_default"
        if action == ResearchActionType.SUMMARIZE_EVIDENCE:
            return "summarize_evidence/task_level_merge"
        if action == ResearchActionType.FINALIZE_REPORT:
            return "finalize_report/task_artifact_writer"
        if action == ResearchActionType.FINISH:
            return "finish/runtime_complete"
        return "fail/runtime_stop"

    @staticmethod
    def _strategy_label(strategy_id: str) -> str:
        labels = {
            "plan/rule_based_initial": "规则初始规划",
            "search_online/openalex_primary": "OpenAlex 优先检索",
            "search_online/arxiv_primary": "arXiv 优先检索",
            "search_online/mixed_broad_recall": "混合宽召回检索",
            "search_local/vector_recall_default": "默认向量召回",
            "summarize_evidence/task_level_merge": "任务级证据合并",
            "finalize_report/task_artifact_writer": "任务型结果生成",
            "finish/runtime_complete": "运行完成",
            "fail/runtime_stop": "运行停止",
        }
        return labels.get(strategy_id, strategy_id)

    @staticmethod
    def _strategy_parameters(action: ResearchActionType, request: ResearchRequest | None) -> dict[str, object]:
        if action == ResearchActionType.SEARCH_ONLINE and request is not None:
            return {"search_provider": request.search_provider, "top_k_online": request.top_k_online}
        if action == ResearchActionType.SEARCH_LOCAL and request is not None:
            return {"top_k_local": request.top_k_local}
        return {}

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
                task_route=runtime_state.task_route,
                active_skill=self._active_skill(runtime_state),
            )
        if action == ResearchActionType.SEARCH_ONLINE and plan_item is not None:
            selected_tool = (
                runtime_state.active_step.selected_tool
                if runtime_state.active_step is not None
                else None
            )
            return self.tool_executor.search_online(
                runtime_state.run_id,
                plan_item,
                request,
                selected_tool=selected_tool,
            )
        if action == ResearchActionType.SEARCH_LOCAL and plan_item is not None:
            route_documents = self._documents_for_request(request, documents)
            return self.tool_executor.search_local(
                runtime_state.run_id,
                plan_item,
                route_documents,
                top_k_local=request.top_k_local,
            )
        if action == ResearchActionType.REVISE_PLAN and plan_item is not None:
            candidate = self.planner_candidate_provider.propose(
                runtime_state,
                runtime_state.last_decision or ResearchActionDecision(
                    action_type=action,
                    selected_tool="revise_plan/rewrite_query",
                    reason="Fallback planner decision.",
                    target_task_id=plan_item.task_id,
                ),
                plan_item,
            )
            operation = candidate.candidate_plan_ops[0] if candidate.candidate_plan_ops else None
            revised_query = (
                f"{request.topic} {plan_item.intent}".strip()
                if operation is None or not operation.query
                else operation.query
            )
            return ResearchToolResult(
                status=ResearchToolResultStatus.COMPLETED,
                summary=candidate.reason or f"Revised task query to focus on {plan_item.intent}.",
                payload={
                    "query": revised_query,
                    "planner_candidate": candidate.model_dump(mode="json"),
                },
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
                task_route=runtime_state.task_route,
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
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem | None,
        result: ResearchToolResult,
        event_sink: EventSink | None,
    ) -> None:
        runtime_state = state.runtime_state
        assert runtime_state is not None
        now = datetime.now(timezone.utc)
        active_step = runtime_state.active_step
        task_id = plan_item.task_id if plan_item is not None else None
        action = decision.action_type
        tool_signature = self._tool_signature(action, plan_item, request, selected_tool=decision.selected_tool)
        before_keys = self._evidence_keys(runtime_state, plan_item.task_id) if plan_item is not None else set()
        applied_plan_ops: list[ResearchPlanOperation] = []
        planner_provider: PlannerProviderType | None = None
        planner_fallback_used = False
        existing_task_ids = {task.id for task in state.todo_tasks}

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
            plan_item.attempt_count += 1
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            self._emit_task_status(state.run_id, plan_item, "在线论文检索完成。", event_sink)
        elif action == ResearchActionType.SEARCH_LOCAL and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            evidence.evidence_items = [EvidenceItem(**item) for item in result.payload.get("evidence_items", [])]
            evidence.local_completed = True
            plan_item.status = TodoTaskStatus.IN_PROGRESS
            plan_item.attempt_count += 1
            self.research_repository.update_task(state.run_id, self._to_todo_task(plan_item))
            self._emit_task_status(state.run_id, plan_item, "本地证据检索完成。", event_sink)
            if (
                runtime_state.task_route is not None
                and runtime_state.task_route.evidence_policy.value == "local_first"
                and not evidence.evidence_items
            ):
                runtime_state.task_route.needs_online_search = True
                runtime_state.task_route.evidence_policy = ResearchEvidencePolicy.ONLINE_SUPPLEMENT
        elif action == ResearchActionType.REVISE_PLAN and plan_item is not None:
            candidate = self._candidate_from_result(result)
            if candidate is not None:
                planner_provider = candidate.provider
                planner_fallback_used = candidate.fallback_used
            revised_query = str(result.payload.get("query") or plan_item.query).strip()
            plan_item.revise_count += 1
            plan_item.attempt_count += 1
            runtime_state.replan_count += 1
            if runtime_state.task_route is None or runtime_state.task_route.needs_online_search:
                plan_item.required_evidence = ["more_relevant_online_paper", "more_relevant_local_document"]
            else:
                plan_item.required_evidence = ["more_relevant_local_document"]
            plan_item.notes.append(result.summary)
            applied_plan_ops = self._apply_plan_operations(
                runtime_state,
                plan_item,
                candidate.candidate_plan_ops if candidate is not None else [],
                revised_query=revised_query,
            )
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            evidence.paper_records = []
            evidence.evidence_items = []
            evidence.compacted_evidence = []
            evidence.online_completed = False
            evidence.local_completed = False
            state.todo_tasks = [self._to_todo_task(item) for item in runtime_state.plan_items]
            new_tasks = [task for task in state.todo_tasks if task.id not in existing_task_ids]
            if new_tasks:
                self.research_repository.save_todo_tasks(state.run_id, new_tasks)
            for task in state.todo_tasks:
                if task.id in existing_task_ids:
                    self.research_repository.update_task(state.run_id, task)
            self.workspace_service.write_todo_tasks(state.run_id, state.todo_tasks)
            self._emit_task_status(state.run_id, plan_item, result.summary, event_sink)
            self._emit(
                event_sink,
                {
                    "type": "todo_list",
                    "run_id": state.run_id,
                    "tasks": [task.model_dump(mode="json") for task in state.todo_tasks],
                },
            )
        elif action == ResearchActionType.SUMMARIZE_EVIDENCE and plan_item is not None:
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            if self.main_runtime.should_degrade(plan_item, evidence):
                evidence.degraded = True
                plan_item.degraded = True
                plan_item.notes.append("证据不足，按当前材料降级收束。")
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
        self.context_assembler.refresh(runtime_state, active_task=plan_item)
        result.classification = self._classify_result(runtime_state, action, plan_item, result, before_keys)
        self._update_progress_controls(runtime_state, tool_signature, result.classification)
        if active_step is not None:
            active_step.status = ResearchStepStatus.COMPLETED
            runtime_state.tool_history.append(
                ResearchToolCallRecord(
                    step_id=active_step.step_id,
                    action=action,
                    task_id=task_id,
                    status=ResearchToolResultStatus.COMPLETED,
                    summary=result.summary,
                    selected_tool=decision.selected_tool,
                    tool_strategy=decision.tool_strategy,
                    decision_reason=decision.reason,
                    result_classification=result.classification,
                    planner_provider=planner_provider,
                    planner_fallback_used=planner_fallback_used,
                    plan_operations=applied_plan_ops,
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
                "selected_tool": decision.selected_tool,
                "tool_strategy": decision.tool_strategy.model_dump(mode="json") if decision.tool_strategy else None,
                "reason": decision.reason,
                "planner_provider": planner_provider.value if planner_provider else None,
                "planner_fallback_used": planner_fallback_used,
                "plan_operations": [operation.model_dump(mode="json") for operation in applied_plan_ops],
                "result_classification": result.classification.value if result.classification else None,
                "summary": result.summary,
            },
        )
        runtime_state.active_step = None
        runtime_state.current_phase = self.main_runtime.step_phase(action)
        self.context_assembler.refresh(
            runtime_state,
            active_task=self.main_runtime.peek_next_pending_item(runtime_state),
        )
        state.status = self._status_from_phase(runtime_state.current_phase)
        self._checkpoint(runtime_state, request, status=state.status, stop_reason=None, event_sink=event_sink)

    def _handle_tool_failure(
        self,
        runtime_state: ResearchRuntimeState,
        request: ResearchRequest,
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem | None,
        result: ResearchToolResult,
        event_sink: EventSink | None,
    ) -> None:
        action = decision.action_type
        runtime_state.failure_count += 1
        result.classification = (
            ResearchToolResultClassification.RETRYABLE_ERROR
            if result.retryable
            else ResearchToolResultClassification.NON_RETRYABLE_ERROR
        )
        self._update_progress_controls(
            runtime_state,
            self._tool_signature(action, plan_item, request, selected_tool=decision.selected_tool),
            result.classification,
        )
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
                    selected_tool=decision.selected_tool,
                    tool_strategy=decision.tool_strategy,
                    decision_reason=decision.reason,
                    result_classification=result.classification,
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
                "selected_tool": decision.selected_tool,
                "tool_strategy": decision.tool_strategy.model_dump(mode="json") if decision.tool_strategy else None,
                "reason": decision.reason,
                "result_classification": result.classification.value,
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
        decision: ResearchActionDecision,
        plan_item: ResearchPlanItem | None,
        event_sink: EventSink | None,
    ) -> None:
        action = decision.action_type
        runtime_state.current_phase = self.main_runtime.step_phase(action)
        runtime_state.active_step = ResearchRuntimeStep(
            step_id=str(uuid4()),
            action=action,
            task_id=plan_item.task_id if plan_item is not None else None,
            attempt=runtime_state.failure_count + 1,
            selected_tool=decision.selected_tool,
            tool_strategy=decision.tool_strategy,
            reason=decision.reason,
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
                "selected_tool": decision.selected_tool,
                "tool_strategy": decision.tool_strategy.model_dump(mode="json") if decision.tool_strategy else None,
                "reason": decision.reason,
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
                "selected_tool": decision.selected_tool,
                "tool_strategy": decision.tool_strategy.model_dump(mode="json") if decision.tool_strategy else None,
                "reason": decision.reason,
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
        self.context_assembler.refresh(
            runtime_state,
            active_task=self.main_runtime.peek_next_pending_item(runtime_state),
        )
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
                "context_state": runtime_state.context_state.model_dump(mode="json"),
                "planner_provider": runtime_state.planner_provider.value,
                "planner_fallback_used": runtime_state.planner_fallback_used,
                "task_route": runtime_state.task_route.model_dump(mode="json") if runtime_state.task_route else None,
                "active_skill_id": (
                    runtime_state.task_route.active_skill_id
                    if runtime_state.task_route is not None
                    else None
                ),
                "primary_skill_id": self._primary_skill_id(runtime_state.task_route),
                "used_skill_ids": self._used_skill_ids(runtime_state.task_route),
                "used_skills": self._used_skills_payload(runtime_state.task_route),
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
                "context_state": runtime_state.context_state.model_dump(mode="json"),
                "planner_provider": runtime_state.planner_provider.value,
                "planner_fallback_used": runtime_state.planner_fallback_used,
                "task_route": runtime_state.task_route.model_dump(mode="json") if runtime_state.task_route else None,
                "active_skill_id": (
                    runtime_state.task_route.active_skill_id
                    if runtime_state.task_route is not None
                    else None
                ),
                "primary_skill_id": self._primary_skill_id(runtime_state.task_route),
                "used_skill_ids": self._used_skill_ids(runtime_state.task_route),
                "used_skills": self._used_skills_payload(runtime_state.task_route),
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

    def _route_request(self, request: ResearchRequest) -> _RouteRequestResult:
        documents = self._documents_for_request(request, self.library_repository.list_documents())
        available_skills = self.skill_registry.list_enabled()
        task_route = self.task_router.route(request, ready_documents=documents)
        default_skill = self.skill_registry.default_for(task_route.task_type)
        skill_selection = self.skill_selector.select(
            prompt=request.topic,
            command=None,
            intent_hint=None,
            selected_document_count=max(len(documents), len(request.selected_document_ids)),
            attachments=[],
            available_skills=available_skills,
            task_type=task_route.task_type.value,
            route=task_route.execution_route.value,
        )
        task_route.used_skills = skill_selection.used_skills
        primary_skill_id = (
            skill_selection.primary_skill.skill_id
            if skill_selection.primary_skill is not None
            else None
        )
        active_skill_id = primary_skill_id or (default_skill.skill_id if default_skill is not None else None)
        active_skill = (
            self.skill_registry.load_definition(active_skill_id)
            if active_skill_id is not None
            else None
        ) or default_skill
        if active_skill is not None:
            task_route.active_skill_id = active_skill.skill_id
            task_route.artifact_protocol = active_skill.artifact_protocol
            if active_skill.default_execution_mode.value == "lightweight" and task_route.allow_single_pass:
                task_route.use_main_agent_loop = False
        consistency_report = self.skill_consistency_checker.check_route_selection(
            task_route=task_route,
            skill_selection_result=skill_selection,
            available_skills=available_skills,
            warning_only=True,
        )
        return _RouteRequestResult(task_route=task_route, consistency_report=consistency_report)

    def _active_skill(self, runtime_state: ResearchRuntimeState) -> SkillDefinition | None:
        if runtime_state.task_route is None or runtime_state.task_route.active_skill_id is None:
            return None
        definition = self.skill_registry.load_definition(runtime_state.task_route.active_skill_id)
        if definition is not None:
            return definition
        manifest = self.skill_registry.default_for(runtime_state.task_route.task_type)
        if manifest is None:
            return None
        return self.skill_registry.load_definition(manifest.skill_id)

    @staticmethod
    def _documents_for_request(request: ResearchRequest, documents) -> list:
        ready_documents = [document for document in documents if document.status == "ready"]
        if not request.selected_document_ids:
            return ready_documents
        selected = set(request.selected_document_ids)
        return [document for document in ready_documents if document.id in selected]

    def _emit_task_route(
        self,
        run_id: str,
        task_route: ResearchTaskRoute,
        event_sink: EventSink | None,
    ) -> None:
        self._emit(
            event_sink,
            {
                "type": "task_route",
                "run_id": run_id,
                "task_route": task_route.model_dump(mode="json"),
            },
        )

    def _emit_research_skill_selected(
        self,
        run_id: str,
        task_route: ResearchTaskRoute,
        event_sink: EventSink | None,
    ) -> None:
        payload = {
            "active_skill_id": task_route.active_skill_id,
            "primary_skill_id": self._primary_skill_id(task_route),
            "used_skill_ids": self._used_skill_ids(task_route),
            "used_skills": self._used_skills_payload(task_route),
            "task_type": task_route.task_type.value,
            "execution_route": task_route.execution_route.value,
        }
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="research_skill_selected",
            message="Research skill selected for route trace.",
            payload=payload,
        )
        self._emit(
            event_sink,
            {
                "type": "research_skill_selected",
                "run_id": run_id,
                **payload,
            },
        )

    def _append_research_skill_consistency_trace(
        self,
        run_id: str,
        report: ResearchSkillConsistencyReport,
    ) -> None:
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="research_skill_consistency_checked",
            message="Research skill consistency checked in debug trace mode.",
            payload=self._research_skill_consistency_payload(report),
        )

    @staticmethod
    def _research_skill_consistency_payload(report: ResearchSkillConsistencyReport) -> dict:
        checked_fields = [
            "primary_skill_id",
            "active_skill_id",
            "execution_route",
            "artifact_protocol.protocol_type",
            "manifest.supported_task_types",
            "manifest.trigger.task_types",
            "manifest.trigger.routes",
            "manifest.artifact_protocol.protocol_type",
        ]
        mismatches = [
            {
                "field": mismatch.field,
                "expected": mismatch.expected,
                "actual": mismatch.actual,
                "message": mismatch.message,
            }
            for mismatch in report.mismatches
        ]
        return {
            "ok": report.ok,
            "severity": report.severity,
            "checked_fields": checked_fields,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "task_type": report.task_type,
            "active_skill_id": report.active_skill_id,
            "primary_skill_id": report.primary_skill_id,
            "execution_route": report.execution_route,
        }

    @staticmethod
    def _primary_skill_id(task_route: ResearchTaskRoute | None) -> str | None:
        if task_route is None:
            return None
        primary_skill = next((skill for skill in task_route.used_skills if skill.is_primary), None)
        return primary_skill.skill_id if primary_skill is not None else task_route.active_skill_id

    @staticmethod
    def _used_skill_ids(task_route: ResearchTaskRoute | None) -> list[str]:
        if task_route is None:
            return []
        return [skill.skill_id for skill in task_route.used_skills]

    @staticmethod
    def _used_skills_payload(task_route: ResearchTaskRoute | None) -> list[dict]:
        if task_route is None:
            return []
        return [skill.model_dump(mode="json") for skill in task_route.used_skills]

    @staticmethod
    def _candidate_from_result(result: ResearchToolResult) -> ResearchPlannerCandidate | None:
        candidate_payload = result.payload.get("planner_candidate")
        if not isinstance(candidate_payload, dict):
            return None
        return ResearchPlannerCandidate.model_validate(candidate_payload)

    @staticmethod
    def _apply_plan_operations(
        runtime_state: ResearchRuntimeState,
        plan_item: ResearchPlanItem,
        operations: list[ResearchPlanOperation],
        *,
        revised_query: str,
    ) -> list[ResearchPlanOperation]:
        if not operations:
            operations = [
                ResearchPlanOperation(
                    operation_type=ResearchPlanOperationType.REWRITE_QUERY,
                    target_task_id=plan_item.task_id,
                    query=revised_query,
                    reason="默认改写 query，不调整计划结构。",
                )
            ]

        applied: list[ResearchPlanOperation] = []
        now = datetime.now(timezone.utc)
        for operation in operations:
            op = operation.model_copy(deep=True)
            op.applied_at = now
            if op.operation_type == ResearchPlanOperationType.REWRITE_QUERY:
                if revised_query and revised_query != plan_item.query:
                    plan_item.query = revised_query
                    plan_item.query_history.append(revised_query)
                applied.append(op)
            elif op.operation_type == ResearchPlanOperationType.SPLIT_ITEM:
                if ResearchOrchestrator._insert_split_item(runtime_state, plan_item, op):
                    applied.append(op)
            elif op.operation_type == ResearchPlanOperationType.REORDER_ITEMS:
                if ResearchOrchestrator._reorder_pending_items(runtime_state, op):
                    applied.append(op)
            elif op.operation_type in {
                ResearchPlanOperationType.INSERT_ITEM,
                ResearchPlanOperationType.MERGE_ITEMS,
                ResearchPlanOperationType.CLOSE_ITEM,
            }:
                continue

        for op in applied:
            runtime_state.plan_revision_history.append(op)
            runtime_state.last_plan_operation = op
            plan_item.notes.append(f"计划调整：{op.operation_type.value} - {op.reason}")
        runtime_state.planner_provider = PlannerProviderType.RULE_BASED
        runtime_state.planner_fallback_used = False
        return applied

    @staticmethod
    def _insert_split_item(
        runtime_state: ResearchRuntimeState,
        plan_item: ResearchPlanItem,
        operation: ResearchPlanOperation,
    ) -> bool:
        if not operation.new_task_id:
            return False
        if any(item.task_id == operation.new_task_id for item in runtime_state.plan_items):
            return False
        try:
            index = runtime_state.plan_items.index(plan_item)
        except ValueError:
            return False
        new_item = ResearchPlanItem(
            task_id=operation.new_task_id,
            title=operation.title or f"{plan_item.title}：补充证据线索",
            intent=operation.intent or plan_item.intent,
            query=operation.query or plan_item.query,
            objective=operation.intent or plan_item.objective,
            done_criteria=plan_item.done_criteria,
            priority=operation.priority or plan_item.priority + 1,
            suggested_tools=plan_item.suggested_tools,
            required_evidence=plan_item.required_evidence,
            query_history=[operation.query or plan_item.query],
            status=TodoTaskStatus.PENDING,
            notes=[f"由任务 {plan_item.task_id} 拆分生成。"],
        )
        runtime_state.plan_items.insert(index + 1, new_item)
        runtime_state.evidence_buffer.append(ResearchEvidenceBufferItem(task_id=new_item.task_id))
        return True

    @staticmethod
    def _reorder_pending_items(runtime_state: ResearchRuntimeState, operation: ResearchPlanOperation) -> bool:
        if not operation.ordered_task_ids:
            return False
        completed = [item for item in runtime_state.plan_items if item.task_id in runtime_state.completed_items]
        pending = [item for item in runtime_state.plan_items if item.task_id not in runtime_state.completed_items]
        pending_by_id = {item.task_id: item for item in pending}
        if any(task_id not in pending_by_id for task_id in operation.ordered_task_ids):
            return False
        ordered_pending = [pending_by_id[task_id] for task_id in operation.ordered_task_ids]
        ordered_pending.extend(item for item in pending if item.task_id not in operation.ordered_task_ids)
        for index, item in enumerate(ordered_pending, start=1):
            item.priority = index
        runtime_state.plan_items = completed + ordered_pending
        return True

    @staticmethod
    def _tool_signature(
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
        request: ResearchRequest,
        *,
        selected_tool: str | None = None,
    ) -> str:
        parts = [
            action.value,
            selected_tool or action.value,
            plan_item.task_id if plan_item is not None else "",
            plan_item.query if plan_item is not None else request.topic,
            request.search_provider or "",
            str(request.top_k_online if action == ResearchActionType.SEARCH_ONLINE else request.top_k_local),
        ]
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
        task_part = plan_item.task_id if plan_item is not None else "run"
        return f"{action.value}:{task_part}:{digest}"

    def _classify_result(
        self,
        runtime_state: ResearchRuntimeState,
        action: ResearchActionType,
        plan_item: ResearchPlanItem | None,
        result: ResearchToolResult,
        before_keys: set[str],
    ) -> ResearchToolResultClassification:
        if result.status == ResearchToolResultStatus.FAILED:
            return (
                ResearchToolResultClassification.RETRYABLE_ERROR
                if result.retryable
                else ResearchToolResultClassification.NON_RETRYABLE_ERROR
            )
        if action in {
            ResearchActionType.PLAN,
            ResearchActionType.SUMMARIZE_EVIDENCE,
            ResearchActionType.FINALIZE_REPORT,
            ResearchActionType.FINISH,
        }:
            return ResearchToolResultClassification.SUCCESS_SUFFICIENT
        if action == ResearchActionType.REVISE_PLAN:
            return ResearchToolResultClassification.SUCCESS_INSUFFICIENT
        if action in {ResearchActionType.SEARCH_ONLINE, ResearchActionType.SEARCH_LOCAL} and plan_item is not None:
            after_keys = self._evidence_keys(runtime_state, plan_item.task_id)
            if after_keys <= before_keys:
                return ResearchToolResultClassification.NO_INCREMENT
            evidence = self._get_or_create_buffer(runtime_state, plan_item.task_id)
            if self.main_runtime.should_degrade(plan_item, evidence):
                return ResearchToolResultClassification.SUCCESS_INSUFFICIENT
            return (
                ResearchToolResultClassification.SUCCESS_SUFFICIENT
                if self.main_runtime._has_sufficient_evidence(evidence)
                else ResearchToolResultClassification.SUCCESS_INSUFFICIENT
            )
        return result.classification or ResearchToolResultClassification.SUCCESS_SUFFICIENT

    @staticmethod
    def _update_progress_controls(
        runtime_state: ResearchRuntimeState,
        tool_signature: str,
        classification: ResearchToolResultClassification,
    ) -> None:
        if runtime_state.last_tool_signature == tool_signature:
            runtime_state.same_tool_streak += 1
        else:
            runtime_state.same_tool_streak = 1
            runtime_state.last_tool_signature = tool_signature

        if classification in {
            ResearchToolResultClassification.NO_INCREMENT,
            ResearchToolResultClassification.RETRYABLE_ERROR,
            ResearchToolResultClassification.NON_RETRYABLE_ERROR,
        }:
            runtime_state.no_progress_count += 1
        else:
            runtime_state.no_progress_count = 0

    @staticmethod
    def _evidence_keys(runtime_state: ResearchRuntimeState, task_id: str) -> set[str]:
        evidence = next(
            (item for item in runtime_state.evidence_buffer if item.task_id == task_id),
            None,
        )
        if evidence is None:
            return set()
        keys: set[str] = set()
        for paper in evidence.paper_records:
            if paper.doi:
                keys.add(f"paper:doi:{paper.doi.casefold()}")
            elif paper.url:
                keys.add(f"paper:url:{paper.url.casefold()}")
            else:
                keys.add(f"paper:title:{' '.join(paper.title.casefold().split())}")
        for item in evidence.evidence_items:
            quote = " ".join((item.quote or item.snippet).casefold().split())[:120]
            keys.add(
                ":".join(
                    [
                        "local",
                        item.document_id or item.source_id,
                        str(item.page_number or ""),
                        quote,
                    ]
                )
            )
        return keys

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
