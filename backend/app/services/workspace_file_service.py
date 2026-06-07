"""Compatibility wrapper for workspace domain file service."""

from app.domains.workspace.files import WorkspaceFileService, WorkspaceFileServiceError

__all__ = ["WorkspaceFileService", "WorkspaceFileServiceError"]
