"""Safe adapter for ToolRegistry-declared workspace file tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models import ToolObservation, ToolObservationError

from .workspace_chat_operations import WorkspaceChatOperationService
from .workspace_file_service import WorkspaceFileService
from .workspace_operation_resolver import WorkspaceFileOverwriteIntent


class WorkspaceFileToolAdapter:
    """Execute a narrow set of workspace file tools through existing services."""

    TOOL_LIST = "workspace.file.list"
    TOOL_READ = "workspace.file.read"
    TOOL_WRITE_NEW = "workspace.file.write_new"
    TOOL_OVERWRITE_PREPARE = "workspace.file.overwrite_prepare"
    SUPPORTED_TOOLS = {
        TOOL_LIST,
        TOOL_READ,
        TOOL_WRITE_NEW,
        TOOL_OVERWRITE_PREPARE,
    }

    def __init__(
        self,
        *,
        workspace_file_service: WorkspaceFileService,
        workspace_chat_operations: WorkspaceChatOperationService,
    ) -> None:
        self.workspace_file_service = workspace_file_service
        self.workspace_chat_operations = workspace_chat_operations

    def execute(
        self,
        tool_name: str,
        *,
        session_id: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolObservation:
        args = dict(arguments or {})
        if tool_name not in self.SUPPORTED_TOOLS:
            return self._error(
                tool_name,
                code="unsupported_workspace_tool",
                message="Unsupported workspace file tool",
                operation_level="query-level",
            )
        try:
            if tool_name == self.TOOL_LIST:
                return self._list(session_id=session_id, args=args)
            if tool_name == self.TOOL_READ:
                return self._read(session_id=session_id, args=args)
            if tool_name == self.TOOL_WRITE_NEW:
                return self._write_new(session_id=session_id, args=args)
            return self._overwrite_prepare(session_id=session_id, args=args)
        except Exception as exc:
            return self._error(
                tool_name,
                code="workspace_tool_failed",
                message=str(exc) or "Workspace file tool failed",
                operation_level="content-level" if tool_name in {self.TOOL_WRITE_NEW, self.TOOL_OVERWRITE_PREPARE} else "query-level",
                io_type="write" if tool_name in {self.TOOL_WRITE_NEW, self.TOOL_OVERWRITE_PREPARE} else "read",
                write_type="create" if tool_name == self.TOOL_WRITE_NEW else ("prepare_overwrite" if tool_name == self.TOOL_OVERWRITE_PREPARE else "none"),
                requires_confirmation=tool_name == self.TOOL_OVERWRITE_PREPARE,
            )

    def _list(self, *, session_id: str, args: dict[str, Any]) -> ToolObservation:
        max_entries = self._bounded_int(args.get("max_entries"), default=100, minimum=1, maximum=self.workspace_file_service.MAX_LIST_ENTRIES)
        items = self.workspace_file_service.list_workspace_files(
            session_id=session_id,
            path=str(args.get("path") or ""),
            recursive=bool(args.get("recursive", False)),
            max_entries=max_entries,
        )
        payload_items = [
            item.model_dump(mode="json", exclude_none=True)
            for item in items
        ]
        return ToolObservation(
            tool_name=self.TOOL_LIST,
            success=True,
            operation_level="query-level",
            io_type="read",
            counts={
                "count": len(payload_items),
                "max_entries": max_entries,
                "truncated": len(payload_items) >= max_entries,
            },
            data={
                "items": payload_items,
                "count": len(payload_items),
                "truncated": len(payload_items) >= max_entries,
                "max_entries": max_entries,
            },
            target_objects=[
                {
                    "type": "workspace_directory",
                    "relative_path": str(args.get("path") or ""),
                }
            ],
            message="Workspace files listed.",
        )

    def _read(self, *, session_id: str, args: dict[str, Any]) -> ToolObservation:
        path = self._required_string(args, "path")
        result = self.workspace_file_service.read_workspace_file(
            session_id=session_id,
            relative_path=path,
            max_chars=self._bounded_int(
                args.get("max_chars"),
                default=self.workspace_file_service.DEFAULT_READ_MAX_CHARS,
                minimum=1,
                maximum=self.workspace_file_service.MAX_READ_CHARS,
            ),
        )
        data = {
            "relative_path": result.relative_path,
            "content": result.content,
            "included_chars": result.included_chars,
            "char_count": result.char_count,
            "truncated": result.truncated,
            "mime_type": result.mime_type,
            "size_bytes": result.size_bytes,
        }
        return ToolObservation(
            tool_name=self.TOOL_READ,
            success=True,
            operation_level="query-level",
            io_type="read",
            target_objects=[self._file_object(result.relative_path, mime_type=result.mime_type, size_bytes=result.size_bytes)],
            counts={
                "included_chars": result.included_chars,
                "char_count": result.char_count,
                "truncated": result.truncated,
                "size_bytes": result.size_bytes,
            },
            data=data,
            message="Workspace file read.",
        )

    def _write_new(self, *, session_id: str, args: dict[str, Any]) -> ToolObservation:
        path = self._required_string(args, "path")
        content = self._required_string(args, "content")
        default_kind = self._file_kind_for_path(path)
        workspace_file = self.workspace_file_service.create_generated_file(
            session_id=session_id,
            relative_path=path,
            content=content,
            display_name=str(args.get("display_name") or Path(path).name),
            file_kind=default_kind,
            created_by="agent",
        )
        data = {
            "relative_path": workspace_file.relative_path,
            "display_name": workspace_file.display_name,
            "file_kind": workspace_file.file_kind,
            "mime_type": workspace_file.mime_type,
            "size_bytes": workspace_file.size_bytes,
            "checksum": workspace_file.checksum,
            "status": workspace_file.status,
        }
        return ToolObservation(
            tool_name=self.TOOL_WRITE_NEW,
            success=True,
            operation_level="content-level",
            io_type="write",
            write_type="create",
            affected_objects=[self._file_object(workspace_file.relative_path, mime_type=workspace_file.mime_type, size_bytes=workspace_file.size_bytes)],
            counts={"size_bytes": workspace_file.size_bytes},
            data=data,
            message="Workspace file created.",
        )

    def _overwrite_prepare(self, *, session_id: str, args: dict[str, Any]) -> ToolObservation:
        path = self._required_string(args, "path")
        new_content = self._required_string(args, "new_content")
        file_kind = self._file_kind_for_path(path)
        _reason = str(args.get("reason") or "").strip()
        intent = WorkspaceFileOverwriteIntent(
            relative_path=path,
            content=new_content,
            display_name=Path(path).name,
            file_kind=file_kind,
            reason=_reason or "Workspace overwrite prepared through ToolRegistry adapter.",
        )
        _message, status, pending, error_message = self.workspace_chat_operations.create_overwrite_pending(
            session_id=session_id,
            intent=intent,
        )
        if pending is None:
            return self._error(
                self.TOOL_OVERWRITE_PREPARE,
                code="workspace_overwrite_prepare_failed",
                message=error_message or _message,
                operation_level="content-level",
                io_type="write",
                write_type="prepare_overwrite",
                requires_confirmation=True,
            )
        diff_preview = str(pending.get("diff_preview") or "")
        data = {
            "relative_path": str(pending["relative_path"]),
            "old_checksum": str(pending["old_checksum"]),
            "new_checksum": str(pending["new_checksum"]),
            "diff_preview": diff_preview,
            "diff_truncated": bool(pending.get("diff_truncated")),
            "pending_action_created": True,
            "confirmation_required": True,
            "status": "pending",
        }
        return ToolObservation(
            tool_name=self.TOOL_OVERWRITE_PREPARE,
            success=True,
            operation_level="content-level",
            io_type="write",
            write_type="prepare_overwrite",
            target_objects=[self._file_object(str(pending["relative_path"]), mime_type=pending.get("mime_type"), size_bytes=int(pending["old_size_bytes"]))],
            counts={
                "old_size_bytes": int(pending["old_size_bytes"]),
                "new_size_bytes": int(pending["new_size_bytes"]),
                "diff_chars": len(diff_preview),
                "diff_truncated": bool(pending.get("diff_truncated")),
            },
            data=data,
            requires_followup=True,
            requires_confirmation=True,
            message=f"Workspace overwrite {status}.",
        )

    def _file_kind_for_path(self, relative_path: str) -> str:
        extension = Path(relative_path).suffix.casefold()
        allowed = self.workspace_file_service.ALLOWED_EXTENSIONS.get(extension)
        if allowed is None:
            raise ValueError("Unsupported workspace file extension")
        return allowed[0]

    @staticmethod
    def _required_string(args: dict[str, Any], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Workspace tool argument '{key}' is required")
        return value

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _file_object(relative_path: str, *, mime_type: Any = None, size_bytes: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "workspace_file",
            "relative_path": relative_path,
        }
        if mime_type is not None:
            payload["mime_type"] = mime_type
        if size_bytes is not None:
            payload["size_bytes"] = size_bytes
        return payload

    @staticmethod
    def _error(
        tool_name: str,
        *,
        code: str,
        message: str,
        operation_level: str,
        io_type: str = "read",
        write_type: str = "none",
        requires_confirmation: bool = False,
    ) -> ToolObservation:
        return ToolObservation(
            tool_name=tool_name,
            success=False,
            operation_level=operation_level,
            io_type=io_type,
            write_type=write_type,
            requires_confirmation=requires_confirmation,
            error=ToolObservationError(
                code=code,
                message=message,
                recoverable=True,
                suggested_next_action="ask_user_to_clarify",
            ),
            message=message,
        )
