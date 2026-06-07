"""Chat application boundary for API routes."""

from __future__ import annotations

from collections.abc import Callable

from app.models import ChatMessageRequest, ChatSessionCreateRequest
from app.services import ChatService


class ChatUseCase:
    """Thin use case around chat session and Agent lifecycle behavior."""

    def __init__(self, chat_service: ChatService) -> None:
        self.chat_service = chat_service

    def list_sessions(self):
        return self.chat_service.list_sessions()

    def create_session(self, request: ChatSessionCreateRequest):
        return self.chat_service.create_session(request.title)

    def delete_session(self, session_id: str):
        return self.chat_service.delete_session(session_id)

    def get_session_detail(self, session_id: str):
        return self.chat_service.get_session_detail(session_id)

    def get_memory_snapshot(self, session_id: str):
        return self.chat_service.get_memory_snapshot(session_id)

    def get_context_state(self, session_id: str):
        return self.chat_service.get_context_state(session_id)

    def send_message(
        self,
        session_id: str,
        request: ChatMessageRequest,
        *,
        delta_sink: Callable[[str], None] | None = None,
    ):
        return self.chat_service.send_message(session_id, request, delta_sink=delta_sink)
