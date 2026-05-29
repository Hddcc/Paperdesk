"""Safe trace payloads for deterministic workspace chat operations."""

from __future__ import annotations

from typing import Any

from app.models import WorkspaceFileReadResult

from .workspace_operation_resolver import (
    WorkspaceCommandBoundary,
    WorkspaceFileOverwriteIntent,
    WorkspaceFileWriteNewIntent,
)


class WorkspaceTraceBuilder:
    """Build trace payloads without file content, storage paths, or workspace roots."""

    @staticmethod
    def file_created(workspace_file) -> dict[str, object]:
        return {
            "relative_path": workspace_file.relative_path,
            "display_name": workspace_file.display_name,
            "file_kind": workspace_file.file_kind,
            "mime_type": workspace_file.mime_type,
            "size_bytes": workspace_file.size_bytes,
            "checksum": workspace_file.checksum,
            "source_message_id": workspace_file.source_message_id,
            "source_file_count": len(workspace_file.source_file_ids),
            "source_document_count": len(workspace_file.source_document_ids),
            "status": workspace_file.status,
        }

    @staticmethod
    def file_create_skipped(
        intent: WorkspaceFileWriteNewIntent,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "relative_path": intent.relative_path,
            "display_name": intent.display_name,
            "file_kind": intent.file_kind,
            "source_message_id": intent.source_message_id,
            "source_file_count": len(intent.source_file_ids or []),
            "source_document_count": len(intent.source_document_ids or []),
            "status": status,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def command_blocked(boundary: WorkspaceCommandBoundary) -> dict[str, object]:
        return {
            "command_hint": boundary.command_hint,
            "reason": boundary.reason,
            "status": "blocked",
        }

    @staticmethod
    def overwrite_pending(pending: dict[str, Any]) -> dict[str, object]:
        diff_preview = str(pending.get("diff_preview") or "")
        return {
            "relative_path": pending.get("relative_path"),
            "file_kind": pending.get("file_kind"),
            "old_size_bytes": pending.get("old_size_bytes"),
            "new_size_bytes": pending.get("new_size_bytes"),
            "old_checksum": pending.get("old_checksum"),
            "new_checksum": pending.get("new_checksum"),
            "diff_chars": len(diff_preview),
            "diff_truncated": bool(pending.get("diff_truncated")),
            "source_message_id": pending.get("source_message_id"),
            "source_file_count": len(pending.get("source_file_ids") or []),
            "source_document_count": len(pending.get("source_document_ids") or []),
            "status": "pending",
        }

    @staticmethod
    def file_overwritten(workspace_file) -> dict[str, object]:
        return {
            "relative_path": workspace_file.relative_path,
            "file_kind": workspace_file.file_kind,
            "size_bytes": workspace_file.size_bytes,
            "checksum": workspace_file.checksum,
            "status": "completed",
        }

    @staticmethod
    def overwrite_skipped(
        intent: WorkspaceFileOverwriteIntent,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "relative_path": intent.relative_path,
            "display_name": intent.display_name,
            "file_kind": intent.file_kind,
            "source_message_id": intent.source_message_id,
            "source_file_count": len(intent.source_file_ids or []),
            "source_document_count": len(intent.source_document_ids or []),
            "status": status,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def context_injected(result: WorkspaceFileReadResult) -> dict[str, object]:
        return {
            "relative_path": result.relative_path,
            "mime_type": result.mime_type,
            "size_bytes": result.size_bytes,
            "char_count": result.char_count,
            "included_chars": result.included_chars,
            "truncated": result.truncated,
            "status": result.status,
        }
