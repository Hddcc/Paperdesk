"""Chat-style knowledge routes."""

from __future__ import annotations

from queue import Queue
from threading import Thread
import time
from contextlib import suppress
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.main import get_chat_service, get_chat_use_case, get_report_lifecycle_service
from app.application import ChatUseCase
from app.domains.paper import ReportLifecycleService
from app.models import (
    ChatContextState,
    ChatMessageRequest,
    ChatSendResponse,
    ChatSessionCreateRequest,
    ChatSessionDetail,
    MemorySnapshot,
    ResearchRunStatus,
)
from app.services import ChatService
from .sse import chunk_text, sse_event

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/sessions")
def list_sessions(use_case: ChatUseCase = Depends(get_chat_use_case)) -> list[dict]:
    return [item.model_dump(mode="json") for item in use_case.list_sessions()]


@router.post("/sessions")
def create_session(
    request: ChatSessionCreateRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    session = use_case.create_session(request)
    return session.model_dump(mode="json")


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    try:
        session = use_case.delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.model_dump(mode="json")


@router.get("/sessions/{session_id}")
def get_session_detail(
    session_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    try:
        detail = use_case.get_session_detail(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatSessionDetail.model_validate(detail).model_dump(mode="json")


@router.get("/sessions/{session_id}/memory")
def get_memory_snapshot(
    session_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    try:
        snapshot = use_case.get_memory_snapshot(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MemorySnapshot.model_validate(snapshot).model_dump(mode="json")


@router.get("/sessions/{session_id}/context")
def get_context_state(
    session_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    try:
        state = use_case.get_context_state(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatContextState.model_validate(state).model_dump(mode="json")


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    request: ChatMessageRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> dict:
    try:
        response = use_case.send_message(session_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatSendResponse.model_validate(response).model_dump(mode="json")


@router.post("/sessions/{session_id}/messages/stream")
def send_message_stream(
    session_id: str,
    request: ChatMessageRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    try:
        use_case.get_context_state(session_id)
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
                response = use_case.send_message(session_id, request, delta_sink=push_delta)
            except Exception as exc:
                events.put(("error", {"type": "error", "message": str(exc) or "Chat message failed"}))
                return

            payload = ChatSendResponse.model_validate(response).model_dump(mode="json")
            events.put(("done", {"type": "done", "response": payload}))

        try:
            yield sse_event("status", {"type": "status", "status": "processing"})
            Thread(target=worker, daemon=True).start()

            while True:
                event, payload = events.get()
                if event == "assistant_delta":
                    streamed_any_delta = True
                    yield sse_event(event, payload)
                    continue
                if event == "error":
                    yield sse_event(event, payload)
                    return
                if event == "done":
                    response_payload = payload["response"]
                    if not streamed_any_delta:
                        for chunk in chunk_text(response_payload["assistant_message"]["content"], size=1):
                            yield sse_event("assistant_delta", {"type": "assistant_delta", "delta": chunk})
                            time.sleep(0.004)
                    yield sse_event(event, payload)
                    return
        except GeneratorExit:
            cancelled = True
            _record_stream_cancelled(use_case.chat_service, session_id)
            raise
        except Exception:
            cancelled = True
            _record_stream_cancelled(use_case.chat_service, session_id)
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
    service: ReportLifecycleService = Depends(get_report_lifecycle_service),
) -> dict:
    try:
        report = service.save_from_message(session_id=session_id, message_id=message_id)
    except ValueError as exc:
        status_code = 400 if str(exc).startswith("Only assistant") else 404
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return report.model_dump(mode="json")

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
