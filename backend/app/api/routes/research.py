"""Research streaming route."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.main import get_report_repository, get_research_orchestrator, get_research_repository
from app.models import ResearchRequest, ResearchRunDetail, TaskSummary
from app.repositories import ReportRepository, ResearchRepository
from app.services.research_orchestrator import ResearchOrchestrator

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/stream")
def stream_research(
    request: ResearchRequest,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> StreamingResponse:
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


@router.get("/{task_id}")
def get_research_run(
    task_id: str,
    research_repository: ResearchRepository = Depends(get_research_repository),
    report_repository: ReportRepository = Depends(get_report_repository),
) -> ResearchRunDetail:
    run = research_repository.get_run(task_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Research run not found")

    tasks = research_repository.list_tasks(task_id)
    report = report_repository.get_report_by_run_id(task_id)
    task_summaries = report.task_summaries if report is not None else _build_task_summaries(tasks)
    return ResearchRunDetail(
        run=run,
        tasks=tasks,
        task_summaries=task_summaries,
        report=report,
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
