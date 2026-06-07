"""Lifecycle context assembly for PaperDesk Agent requests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.models import AgentLifecycleStage, ContextPacket, RuntimeRequest


class AgentContextLifecycleService:
    """Build route-aware context packets for lifecycle runtimes."""

    def __init__(self, *, default_token_budget: int = 6000) -> None:
        self.default_token_budget = default_token_budget

    def build_context(
        self,
        *,
        recent_messages: Iterable[dict[str, Any]] | None = None,
        selected_document_ids: Iterable[str] | None = None,
        selected_file_ids: Iterable[str] | None = None,
        evidence: Iterable[dict[str, Any]] | None = None,
        pending_action: dict[str, Any] | None = None,
        workspace_scope: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
        token_budget: int | None = None,
    ) -> ContextPacket:
        return ContextPacket(
            recent_messages=list(recent_messages or []),
            selected_document_ids=self._dedupe(selected_document_ids or []),
            selected_file_ids=self._dedupe(selected_file_ids or []),
            evidence=list(evidence or []),
            pending_action=pending_action,
            workspace_scope=workspace_scope or {},
            preferences=preferences or {},
            token_budget=token_budget or self.default_token_budget,
        )

    def attach_context(self, request: RuntimeRequest, context: ContextPacket) -> RuntimeRequest:
        request.context = context
        request.add_trace(
            AgentLifecycleStage.CONTEXT,
            "lifecycle context assembled",
            {
                "recent_message_count": len(context.recent_messages),
                "selected_document_count": len(context.selected_document_ids),
                "selected_file_count": len(context.selected_file_ids),
                "evidence_count": len(context.evidence),
                "has_pending_action": context.pending_action is not None,
                "has_workspace_scope": bool(context.workspace_scope),
                "has_preferences": bool(context.preferences),
                "token_budget": context.token_budget,
            },
        )
        return request

    def direct_chat_context(self, request: RuntimeRequest) -> ContextPacket:
        """Return the minimal context needed for a direct chat route."""

        return self.build_context(
            recent_messages=request.context.recent_messages,
            selected_file_ids=request.context.selected_file_ids,
            pending_action=request.context.pending_action,
            workspace_scope=request.context.workspace_scope,
            preferences=request.context.preferences,
            token_budget=request.context.token_budget,
        )

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip() if value is not None else ""
            if not item or item in seen:
                continue
            seen.add(item)
            results.append(item)
        return results
