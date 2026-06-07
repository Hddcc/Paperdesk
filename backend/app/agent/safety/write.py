"""Scoped write safety helpers for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.models import PaperDeskRoute, PendingWriteAction, ToolDeclaration, ToolVerification, WriteActionPlan, WriteOperationLevel


class AgentWriteSafetyService:
    """Create write previews and validate pending write confirmations."""

    def create_plan(
        self,
        *,
        route: PaperDeskRoute,
        operation_level: WriteOperationLevel,
        description: str,
        target_scope: dict,
        affected_objects: list[dict] | None = None,
        confirmation_text: str | None = None,
    ) -> WriteActionPlan:
        executable = self.has_explicit_scope(target_scope)
        action_id = f"write-{uuid4().hex}"
        return WriteActionPlan(
            action_id=action_id,
            route=route,
            operation_level=operation_level,
            description=description,
            target_scope=target_scope,
            affected_objects=affected_objects or [],
            confirmation_text=confirmation_text or f"confirm {action_id}",
            executable=executable,
            requires_confirmation=True,
            reason="explicit scope resolved" if executable else "write target scope is missing or ambiguous",
        )

    def to_pending_action(self, plan: WriteActionPlan, *, ttl_minutes: int = 30) -> PendingWriteAction:
        return PendingWriteAction(
            action_id=plan.action_id,
            route=plan.route,
            operation_level=plan.operation_level,
            target_scope=plan.target_scope,
            affected_objects=plan.affected_objects,
            confirmation_text=plan.confirmation_text,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes),
        )

    def confirmation_matches(
        self,
        *,
        pending_action: PendingWriteAction,
        confirmation_text: str,
        target_scope: dict | None = None,
    ) -> bool:
        if pending_action.expires_at is not None and pending_action.expires_at < datetime.now(timezone.utc):
            return False
        if confirmation_text.strip() != pending_action.confirmation_text:
            return False
        if target_scope is not None and target_scope != pending_action.target_scope:
            return False
        return True

    @staticmethod
    def verification_required(tool: ToolDeclaration) -> bool:
        spec = tool.spec
        return bool(spec and spec.requires_post_read_verification)

    def build_verification(
        self,
        *,
        tool: ToolDeclaration,
        success: bool,
        details: dict | None = None,
    ) -> ToolVerification:
        spec = tool.spec
        return ToolVerification(
            performed=self.verification_required(tool),
            success=success,
            method=spec.verification_tool if spec and spec.verification_tool else "post_read",
            details=details or {},
        )

    @staticmethod
    def has_explicit_scope(target_scope: dict) -> bool:
        if not target_scope:
            return False
        scope_keys = {"document_ids", "category_id", "tag_id", "report_id", "workspace_path", "query_filter"}
        return any(bool(target_scope.get(key)) for key in scope_keys)
