"""Chat-style knowledge routes."""

from __future__ import annotations

import json
from queue import Queue
from threading import Thread
import time
from datetime import datetime, timezone
from contextlib import suppress
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.main import get_chat_repository, get_chat_service, get_report_repository, get_research_repository
from app.models import (
    ChatContextState,
    ChatMessageRequest,
    ChatSendResponse,
    ChatSessionCreateRequest,
    ChatSessionDetail,
    CitationRecord,
    MemorySnapshot,
    ResearchReport,
    ResearchRunStatus,
    TaskSummary,
)
from app.repositories import ChatRepository, ReportRepository, ResearchRepository
from app.services import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/sessions")
def list_sessions(service: ChatService = Depends(get_chat_service)) -> list[dict]:
    return [item.model_dump(mode="json") for item in service.list_sessions()]


@router.post("/sessions")
def create_session(
    request: ChatSessionCreateRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    session = service.create_session(request.title)
    return session.model_dump(mode="json")


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        session = service.delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.model_dump(mode="json")


@router.get("/sessions/{session_id}")
def get_session_detail(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        detail = service.get_session_detail(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatSessionDetail.model_validate(detail).model_dump(mode="json")


@router.get("/sessions/{session_id}/memory")
def get_memory_snapshot(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        snapshot = service.get_memory_snapshot(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MemorySnapshot.model_validate(snapshot).model_dump(mode="json")


@router.get("/sessions/{session_id}/context")
def get_context_state(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        state = service.get_context_state(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatContextState.model_validate(state).model_dump(mode="json")


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    request: ChatMessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        response = service.send_message(session_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatSendResponse.model_validate(response).model_dump(mode="json")


@router.post("/sessions/{session_id}/messages/stream")
def send_message_stream(
    session_id: str,
    request: ChatMessageRequest,
    service: ChatService = Depends(get_chat_service),
):
    try:
        service.get_context_state(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def event_generator():
        events: Queue[tuple[str, dict]] = Queue()
        streamed_any_delta = False
        cancelled = False

        def push_delta(delta: str) -> None:
            if delta and not cancelled:
                events.put(("assistant_delta", {"type": "assistant_delta", "delta": delta}))

        def worker() -> None:
            try:
                response = service.send_message(session_id, request, delta_sink=push_delta)
            except Exception as exc:
                events.put(("error", {"type": "error", "message": str(exc) or "Chat message failed"}))
                return

            payload = ChatSendResponse.model_validate(response).model_dump(mode="json")
            events.put(("done", {"type": "done", "response": payload}))

        try:
            yield _sse_event("status", {"type": "status", "status": "processing"})
            Thread(target=worker, daemon=True).start()

            while True:
                event, payload = events.get()
                if event == "assistant_delta":
                    streamed_any_delta = True
                    yield _sse_event(event, payload)
                    continue
                if event == "error":
                    yield _sse_event(event, payload)
                    return
                if event == "done":
                    response_payload = payload["response"]
                    if not streamed_any_delta:
                        for chunk in _chunk_text(response_payload["assistant_message"]["content"], size=1):
                            yield _sse_event("assistant_delta", {"type": "assistant_delta", "delta": chunk})
                            time.sleep(0.004)
                    yield _sse_event(event, payload)
                    return
        except GeneratorExit:
            cancelled = True
            _record_stream_cancelled(service, session_id)
            raise
        except Exception:
            cancelled = True
            _record_stream_cancelled(service, session_id)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/messages/{message_id}/report")
def save_message_as_report(
    session_id: str,
    message_id: str,
    chat_repository: ChatRepository = Depends(get_chat_repository),
    research_repository: ResearchRepository = Depends(get_research_repository),
    report_repository: ReportRepository = Depends(get_report_repository),
) -> dict:
    session = chat_repository.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    message = chat_repository.get_message(message_id)
    if message is None or message.session_id != session_id:
        raise HTTPException(status_code=404, detail="Chat message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Only assistant messages can be saved as reports")

    if message.saved_report_id:
        existing = report_repository.get_report(message.saved_report_id)
        if existing is not None:
            return existing.model_dump(mode="json")

    topic = _report_topic(session.title, message.content)
    run_id = f"chat-report-{uuid4().hex}"
    research_repository.create_run(run_id, topic)
    research_repository.update_run_status(run_id, ResearchRunStatus.COMPLETED)

    citations = list(message.citations)
    report = ResearchReport(
        id=str(uuid4()),
        topic=topic,
        markdown=message.content,
        task_summaries=[
            TaskSummary(
                task_id=message.id,
                title=topic,
                intent="Saved from a PaperDesk assistant chat response.",
                summary=message.content,
                summary_markdown=message.content,
            )
        ],
        citations=citations,
        citation_items=[
            CitationRecord(
                citation_label=citation,
                source_type="chat",
                title=citation,
            )
            for citation in citations
        ],
        created_at=datetime.now(timezone.utc),
    )
    saved = report_repository.create_report(report, run_id)
    chat_repository.update_message_report(message.id, saved.id)
    return saved.model_dump(mode="json")


def _sse_event(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _chunk_text(text: str, size: int = 18):
    if not text:
        return
    index = 0
    while index < len(text):
        next_index = min(len(text), index + size)
        newline_index = text.find("\n", index + 1, next_index + 1)
        if newline_index != -1:
            next_index = newline_index + 1
        yield text[index:next_index]
        index = next_index


def _record_stream_cancelled(service: ChatService, session_id: str) -> None:
    orchestrator = getattr(service, "agent_orchestrator", None)
    if orchestrator is None:
        return
    trace_id = f"chat-stream-cancel-{uuid4().hex}"
    repository = getattr(orchestrator, "research_repository", None)
    if repository is None:
        return
    with suppress(Exception):
        repository.create_run(trace_id, f"Chat Stream Cancelled: {session_id}")
        orchestrator.append_trace(
            trace_id,
            status="generation_cancelled",
            message="User stopped the streaming chat response.",
            payload={"session_id": session_id, "user_stopped": True},
        )
        repository.update_run_status(trace_id, ResearchRunStatus.COMPLETED)


def _report_topic(session_title: str, content: str) -> str:
    if session_title and session_title != "新对话":
        return session_title[:80]
    for line in content.splitlines():
        cleaned = line.strip(" #")
        if cleaned:
            return cleaned[:80]
    return "知识库聊天报告"
