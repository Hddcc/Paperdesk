"""Compatibility wrapper for workspace domain chat operations."""

from app.domains.workspace.chat_operations import (
    WorkspaceChatOperationService,
    WorkspacePendingActionAdapter,
    build_workspace_file_context_block,
    unsupported_workspace_write_extension_message,
    workspace_command_boundary_message,
    workspace_file_created_message,
    workspace_file_overwrite_boundary_message,
    workspace_file_write_new_boundary_message,
    workspace_internal_write_boundary_message,
)

__all__ = [
    "WorkspaceChatOperationService",
    "WorkspacePendingActionAdapter",
    "build_workspace_file_context_block",
    "unsupported_workspace_write_extension_message",
    "workspace_command_boundary_message",
    "workspace_file_created_message",
    "workspace_file_overwrite_boundary_message",
    "workspace_file_write_new_boundary_message",
    "workspace_internal_write_boundary_message",
]
