"""Unified research tool wrappers for the phase-13 main-agent loop."""

from __future__ import annotations

from datetime import datetime, timezone

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
    ResearchPlanItem,
    ResearchRequest,
    ResearchRuntimeState,
    ResearchToolResult,
    ResearchToolResultClassification,
    ResearchToolResultStatus,
    TaskArtifactRef,
    TodoTask,
    TodoTaskStatus,
)
from app.services.research_workspace_service import ResearchWorkspaceService


class ResearchToolExecutor:
    """Wrap existing research capabilities behind a unified tool contract."""

    def __init__(
        self,
        *,
        topic_planner: TopicPlannerAgent,
        paper_search_agent: PaperSearchAgent,
        library_retriever: LibraryRetrieverAgent,
        reading_summarizer: ReadingSummarizerAgent,
        report_writer: ReportWriterAgent,
        workspace_service: ResearchWorkspaceService,
    ) -> None:
        self.topic_planner = topic_planner
        self.paper_search_agent = paper_search_agent
        self.library_retriever = library_retriever
        self.reading_summarizer = reading_summarizer
        self.report_writer = report_writer
        self.workspace_service = workspace_service

    def plan(self, run_id: str, request: ResearchRequest, *, direct_task: bool) -> ResearchToolResult:
        if direct_task:
            tasks = [
                TodoTask(
                    id=run_id,
                    title=request.topic,
                    intent="Direct research pass for a compact topic",
                    query=request.topic,
                    status=TodoTaskStatus.PENDING,
                )
            ]
        else:
            tasks = self.topic_planner.plan(request.topic)
        plan_items = [
            ResearchPlanItem(
                task_id=task.id,
                title=task.title,
                intent=task.intent,
                query=task.query,
                objective=task.intent or task.title,
                done_criteria="形成可引用的任务级研究总结；若证据不足，说明降级边界。",
                priority=index + 1,
                suggested_tools=["search_online", "search_local", "summarize_evidence"],
                required_evidence=["online_paper", "local_document"],
                query_history=[task.query],
                status=task.status,
            )
            for index, task in enumerate(tasks)
        ]
        artifact = self._write_json_artifact(
            run_id,
            "__plan__",
            "plan_items.json",
            [item.model_dump(mode="json") for item in plan_items],
            description="Main-agent plan items",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary=f"Planned {len(plan_items)} research tasks.",
            payload={"plan_items": [item.model_dump(mode="json") for item in plan_items]},
            artifacts=[artifact],
        )

    def search_online(
        self,
        run_id: str,
        task: ResearchPlanItem,
        request: ResearchRequest,
    ) -> ResearchToolResult:
        todo_task = self._to_todo_task(task)
        paper_records = self.paper_search_agent.search(
            todo_task,
            top_k=request.top_k_online,
            search_provider=request.search_provider,
        )
        payload = [record.model_dump(mode="json") for record in paper_records]
        artifacts = [
            self._write_json_artifact(
                run_id,
                task.task_id,
                "papers.json",
                payload,
                description="Normalized online paper candidates",
            ),
            self._write_markdown_artifact(
                run_id,
                task.task_id,
                "online-analysis.md",
                "\n".join(
                    [
                        f"# Online Search: {task.title}",
                        "",
                        f"Collected {len(paper_records)} paper candidates.",
                        "",
                        *[f"- {record.title}" for record in paper_records],
                    ]
                ),
                description="Compact online paper summary",
            ),
        ]
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_INSUFFICIENT,
            summary=f"Collected {len(paper_records)} online paper candidates.",
            payload={"paper_records": payload},
            artifacts=artifacts,
            retryable=True,
        )

    def search_local(
        self,
        run_id: str,
        task: ResearchPlanItem,
        documents,
        *,
        top_k_local: int,
    ) -> ResearchToolResult:
        todo_task = self._to_todo_task(task)
        evidence_items = self.library_retriever.retrieve(
            todo_task,
            documents,
            top_k=top_k_local,
        )
        payload = [item.model_dump(mode="json") for item in evidence_items]
        artifacts = [
            self._write_json_artifact(
                run_id,
                task.task_id,
                "evidence.json",
                payload,
                description="Retrieved local evidence items",
            ),
            self._write_markdown_artifact(
                run_id,
                task.task_id,
                "local-analysis.md",
                "\n".join(
                    [
                        f"# Local Search: {task.title}",
                        "",
                        f"Collected {len(evidence_items)} local evidence items.",
                        "",
                        *[f"- {item.citation_label}" for item in evidence_items],
                    ]
                ),
                description="Compact local evidence summary",
            ),
        ]
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_INSUFFICIENT,
            summary=f"Collected {len(evidence_items)} local evidence items.",
            payload={"evidence_items": payload},
            artifacts=artifacts,
            retryable=True,
        )

    def summarize_evidence(
        self,
        run_id: str,
        task: ResearchPlanItem,
        paper_records: list[PaperRecord],
        evidence_items: list[EvidenceItem],
        *,
        degraded: bool = False,
    ) -> ResearchToolResult:
        summary = self.reading_summarizer.summarize(
            self._to_todo_task(task),
            paper_records,
            evidence_items,
        )
        if degraded:
            degraded_note = "证据不足：本任务未检索到足够可用材料，以下总结按当前材料降级收口。"
            summary.summary = f"{degraded_note}\n\n{summary.summary}".strip()
            summary.summary_markdown = f"{degraded_note}\n\n{summary.summary_markdown}".strip()

        artifact = self._write_markdown_artifact(
            run_id,
            task.task_id,
            "summary.md",
            summary.summary_markdown or summary.summary,
            description="Task-level merged summary",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary="Task summary completed.",
            payload={"task_summary": summary.model_dump(mode="json")},
            artifacts=[artifact],
        )

    def finalize_report(
        self,
        run_id: str,
        topic: str,
        task_summaries,
    ) -> ResearchToolResult:
        report = self.report_writer.write(topic, task_summaries)
        artifact = TaskArtifactRef(
            name="final_report.md",
            path=str(self.workspace_service.write_final_report(run_id, report)),
            kind="markdown",
            description="Final report markdown",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary="Final report generated.",
            payload={"report": report.model_dump(mode="json")},
            artifacts=[artifact],
            retryable=False,
        )

    @staticmethod
    def _to_todo_task(task: ResearchPlanItem) -> TodoTask:
        return TodoTask(
            id=task.task_id,
            title=task.title,
            intent=task.intent,
            query=task.query,
            status=task.status,
            summary=task.summary,
            summary_markdown=task.summary_markdown,
        )

    def _write_json_artifact(
        self,
        run_id: str,
        task_id: str,
        filename: str,
        payload: object,
        *,
        description: str,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_json(run_id, task_id, filename, payload)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="json",
            description=description,
        )

    def _write_markdown_artifact(
        self,
        run_id: str,
        task_id: str,
        filename: str,
        content: str,
        *,
        description: str,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_markdown(run_id, task_id, filename, content)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="markdown",
            description=description,
        )
