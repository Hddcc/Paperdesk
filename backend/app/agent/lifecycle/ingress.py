"""Chat ingress normalization for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from uuid import uuid4

from app.models import (
    AgentLifecycleStage,
    ChatMessageRequest,
    ContextPacket,
    PaperDeskRoute,
    RouteDecisionPacket,
    RuntimeRequest,
)


class AgentIngressService:
    """Build lifecycle runtime requests from chat API input."""

    def build_request(
        self,
        *,
        session_id: str,
        request: ChatMessageRequest,
        message_id: str | None = None,
        pending_action: dict | None = None,
        workspace_scope: dict | None = None,
        preferences: dict | None = None,
    ) -> RuntimeRequest:
        runtime_request = RuntimeRequest(
            session_id=session_id,
            message_id=message_id or f"lifecycle-{uuid4().hex}",
            user_prompt=request.content.strip(),
            route=RouteDecisionPacket(
                route=PaperDeskRoute.DIRECT_CHAT,
                reason="ingress default before route decision",
            ),
            context=ContextPacket(
                selected_document_ids=list(request.selected_document_ids),
                selected_file_ids=list(request.selected_file_ids),
                pending_action=pending_action,
                workspace_scope=workspace_scope or {},
                preferences=preferences or {},
            ),
            attachments=list(request.attachments),
        )
        runtime_request.add_trace(
            AgentLifecycleStage.INGRESS,
            "chat request normalized",
            {
                "selected_document_count": len(request.selected_document_ids),
                "selected_file_count": len(request.selected_file_ids),
                "attachment_count": len(request.attachments),
                "command": request.command,
                "intent_hint": request.intent_hint,
                "has_pending_action": pending_action is not None,
            },
        )
        return runtime_request
