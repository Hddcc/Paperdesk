"""Compatibility wrapper for workspace domain operation resolution."""

from app.domains.workspace.operations import (
    WorkspaceBoundaryGuard,
    WorkspaceCommandBoundary,
    WorkspaceFileOverwriteIntent,
    WorkspaceFilePendingResponse,
    WorkspaceFileReadIntent,
    WorkspaceFileWriteNewIntent,
    WorkspaceIntentResolver,
    WorkspacePathExtractor,
    unsupported_workspace_write_extension_message,
    workspace_command_boundary_message,
    workspace_file_overwrite_boundary_message,
    workspace_file_write_new_boundary_message,
    workspace_internal_write_boundary_message,
)

__all__ = [
    "WorkspaceBoundaryGuard",
    "WorkspaceCommandBoundary",
    "WorkspaceFileOverwriteIntent",
    "WorkspaceFilePendingResponse",
    "WorkspaceFileReadIntent",
    "WorkspaceFileWriteNewIntent",
    "WorkspaceIntentResolver",
    "WorkspacePathExtractor",
    "unsupported_workspace_write_extension_message",
    "workspace_command_boundary_message",
    "workspace_file_overwrite_boundary_message",
    "workspace_file_write_new_boundary_message",
    "workspace_internal_write_boundary_message",
]
