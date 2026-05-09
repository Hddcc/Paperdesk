"""Research streaming route."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.main import get_research_orchestrator
from app.models import ResearchRequest
from app.services import ResearchOrchestrator

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

