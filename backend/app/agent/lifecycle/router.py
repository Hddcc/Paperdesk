"""Route decisions for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from app.models import (
    AgentOrchestrationPattern,
    ChatMessageRequest,
    PaperDeskRoute,
    PaperDeskRuntimeKind,
    RouteDecisionPacket,
    WriteOperationLevel,
)


class AgentRouteDecisionService:
    """Classify chat requests into explicit PaperDesk product routes."""

    _LIBRARY_TERMS = ("library", "paper library", "papers", "categories", "tags", "metadata")
    _RAG_TERMS = ("summarize", "summary", "compare", "review", "explain", "method", "evidence")
    _WRITE_TERMS = (
        "delete",
        "remove",
        "rename",
        "assign",
        "clear",
        "save",
        "overwrite",
        "tag",
        "category",
        "report",
    )
    _REPORT_TERMS = ("save report", "export report", "report")
    _WORKSPACE_TERMS = ("workspace", "file", "folder", "path")

    def decide(
        self,
        request: ChatMessageRequest,
        *,
        has_pending_action: bool = False,
        confirmation_received: bool = False,
    ) -> RouteDecisionPacket:
        prompt = request.content.strip()
        normalized = prompt.casefold()
        command = (request.command or "").casefold()
        selected_document_ids = list(request.selected_document_ids)

        if has_pending_action and confirmation_received:
            return RouteDecisionPacket(
                route=PaperDeskRoute.WRITE_CONFIRMED,
                reason="user confirmed an existing pending write action",
                capability_id="paper",
                confidence=0.95,
                requires_tools=True,
                requires_confirmation=True,
                write_operation_level=WriteOperationLevel.CONTENT,
                target_runtime=PaperDeskRuntimeKind.CONFIRMED_WRITE,
                orchestration_pattern=AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY,
                selected_document_ids=selected_document_ids,
            )

        if self._mentions_workspace(normalized) and self._mentions_write(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.WORKSPACE_WRITE,
                reason="workspace write intent detected",
                capability_id="workspace",
                confidence=0.78,
                requires_tools=True,
                requires_confirmation=True,
                write_operation_level=WriteOperationLevel.CONTENT,
                target_runtime=PaperDeskRuntimeKind.WORKSPACE_ACTION,
                orchestration_pattern=AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY,
                selected_document_ids=selected_document_ids,
                target_scope={
                    "scope_type": "workspace",
                    "scope_status": "needs_explicit_path",
                    "requires_confirmation": True,
                },
            )

        if self._mentions_workspace(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.WORKSPACE_READ,
                reason="workspace read intent detected",
                capability_id="workspace",
                confidence=0.72,
                requires_tools=True,
                target_runtime=PaperDeskRuntimeKind.WORKSPACE_ACTION,
                orchestration_pattern=AgentOrchestrationPattern.SERVICE_WORKFLOW,
                selected_document_ids=selected_document_ids,
                target_scope={"scope_type": "workspace", "scope_status": "read_only"},
            )

        if self._mentions_report(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.REPORT_ACTION,
                reason="report action intent detected",
                capability_id="paper",
                confidence=0.72,
                requires_tools=True,
                write_operation_level=WriteOperationLevel.CONTENT if "save" in normalized else WriteOperationLevel.NONE,
                target_runtime=PaperDeskRuntimeKind.REPORT_ACTION,
                orchestration_pattern=AgentOrchestrationPattern.SERVICE_WORKFLOW,
                selected_document_ids=selected_document_ids,
                target_scope={
                    "scope_type": "report",
                    "scope_status": "content_write" if "save" in normalized else "read_or_export",
                },
            )

        if self._mentions_write(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.WRITE_PENDING,
                reason="library, report, tag, category, or content write intent requires preview",
                capability_id="paper",
                confidence=0.82,
                requires_tools=True,
                requires_confirmation=True,
                write_operation_level=self._write_level(normalized),
                target_runtime=PaperDeskRuntimeKind.TOOL_ACTION,
                orchestration_pattern=AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY,
                selected_document_ids=selected_document_ids,
                target_scope={
                    "scope_type": "paper_library",
                    "scope_status": "selected_documents" if selected_document_ids else "needs_explicit_scope",
                    "selected_document_ids": selected_document_ids,
                    "requires_confirmation": True,
                },
            )

        if command in {"summary", "compare"} or selected_document_ids or self._mentions_rag(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.PAPER_RAG,
                reason="paper-grounded request detected",
                capability_id="paper",
                confidence=0.84 if selected_document_ids else 0.68,
                requires_rag=True,
                target_runtime=PaperDeskRuntimeKind.PAPER_RAG,
                orchestration_pattern=AgentOrchestrationPattern.RETRIEVE_THEN_SYNTHESIZE,
                selected_document_ids=selected_document_ids,
                target_scope={
                    "scope_type": "paper_corpus",
                    "scope_status": "selected_documents" if selected_document_ids else "query_scoped_retrieval",
                    "selected_document_ids": selected_document_ids,
                },
            )

        if command == "library" or self._mentions_library(normalized):
            return RouteDecisionPacket(
                route=PaperDeskRoute.LIBRARY_READ,
                reason="paper library read intent detected",
                capability_id="paper",
                confidence=0.78,
                requires_tools=True,
                target_runtime=PaperDeskRuntimeKind.TOOL_ACTION,
                orchestration_pattern=AgentOrchestrationPattern.BOUNDED_REACT,
                selected_document_ids=selected_document_ids,
                target_scope={"scope_type": "paper_library", "scope_status": "read_only"},
            )

        return RouteDecisionPacket(
            route=PaperDeskRoute.DIRECT_CHAT,
            reason="no paper, tool, or write signal detected",
            capability_id="chat",
            confidence=0.64,
            target_runtime=PaperDeskRuntimeKind.DIRECT_CHAT,
            orchestration_pattern=AgentOrchestrationPattern.SINGLE_TURN,
            selected_document_ids=selected_document_ids,
            target_scope={"scope_type": "chat", "scope_status": "conversation_only"},
        )

    @classmethod
    def _mentions_library(cls, normalized: str) -> bool:
        return any(term in normalized for term in cls._LIBRARY_TERMS)

    @classmethod
    def _mentions_rag(cls, normalized: str) -> bool:
        return any(term in normalized for term in cls._RAG_TERMS)

    @classmethod
    def _mentions_report(cls, normalized: str) -> bool:
        return any(term in normalized for term in cls._REPORT_TERMS)

    @classmethod
    def _mentions_workspace(cls, normalized: str) -> bool:
        return any(term in normalized for term in cls._WORKSPACE_TERMS)

    @classmethod
    def _mentions_write(cls, normalized: str) -> bool:
        return any(term in normalized for term in cls._WRITE_TERMS)

    @classmethod
    def _write_level(cls, normalized: str) -> WriteOperationLevel:
        if any(term in normalized for term in ("assign", "tag", "category")):
            return WriteOperationLevel.RELATION
        if any(term in normalized for term in ("delete", "rename", "clear")):
            return WriteOperationLevel.ENTITY
        if any(term in normalized for term in ("report", "save", "overwrite")):
            return WriteOperationLevel.CONTENT
        return WriteOperationLevel.QUERY
