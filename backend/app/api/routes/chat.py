"""Chat-style knowledge routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.main import get_chat_service
from app.models import (
    ChatContextState,
    ChatMessageRequest,
    ChatSendResponse,
    ChatSessionCreateRequest,
    ChatSessionDetail,
    MemorySnapshot,
)
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
