"""Research streaming route."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.main import (
    get_report_repository,
    get_research_orchestrator,
    get_research_repository,
    get_runtime_repository,
)
from app.config import Settings, get_settings
from app.models import ResearchRequest, ResearchRunDetail, TaskSummary
from app.repositories import ReportRepository, ResearchRepository, RuntimeRepository
from app.services.research_orchestrator import ResearchOrchestrator

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/stream")
def stream_research(
    request: ResearchRequest,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if not settings.enable_research_task_agent:
        raise HTTPException(
            status_code=403,
            detail="Research Task Agent is experimental and disabled. Set ENABLE_RESEARCH_TASK_AGENT=true to use this endpoint.",
        )

    def event_iterator():
        try:
            for event in orchestrator.run_stream(request):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive guardrail
            error_payload = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/{run_id}/resume/stream")
def resume_research_stream(
    run_id: str,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if not settings.enable_research_task_agent:
        raise HTTPException(
            status_code=403,
            detail="Research Task Agent is experimental and disabled. Set ENABLE_RESEARCH_TASK_AGENT=true to use this endpoint.",
        )

    def event_iterator():
        try:
            for event in orchestrator.resume_stream(run_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive guardrail
            error_payload = {"type": "error", "detail": str(exc), "run_id": run_id}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{task_id}")
def get_research_run(
    task_id: str,
    research_repository: ResearchRepository = Depends(get_research_repository),
    report_repository: ReportRepository = Depends(get_report_repository),
    runtime_repository: RuntimeRepository = Depends(get_runtime_repository),
) -> ResearchRunDetail:
    run = research_repository.get_run(task_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Research run not found")

    tasks = research_repository.list_tasks(task_id)
    report = report_repository.get_report_by_run_id(task_id)
    task_summaries = report.task_summaries if report is not None else _build_task_summaries(tasks)
    runtime_state = research_repository.get_runtime_state(task_id)
    return ResearchRunDetail(
        run=run,
        tasks=tasks,
        task_summaries=task_summaries,
        runtime_state=runtime_state,
        subagent_tasks=runtime_repository.list_tasks(task_id),
        task_notifications=runtime_repository.list_notifications(task_id),
        task_traces=runtime_repository.list_traces(task_id),
        task_artifacts=runtime_repository.list_artifacts(task_id),
        report=report,
        task_route=runtime_state.task_route if runtime_state is not None else None,
    )


def _build_task_summaries(tasks) -> list[TaskSummary]:
    return [
        TaskSummary(
            task_id=task.id,
            title=task.title,
            intent=task.intent,
            summary=task.summary_markdown or task.summary or "",
            summary_markdown=task.summary_markdown or task.summary or "",
        )
        for task in tasks
        if task.summary_markdown or task.summary
    ]
