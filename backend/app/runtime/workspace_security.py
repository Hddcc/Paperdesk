"""Compatibility wrapper for Agent Core workspace safety primitives."""

from app.agent.safety.workspace import (
    WorkspaceOperation,
    WorkspacePermissionDecision,
    WorkspacePermissionPolicy,
    WorkspaceRiskLevel,
    WorkspaceRoot,
    WorkspaceSandboxMode,
    WorkspaceSecurityError,
    is_sensitive_workspace_path,
    resolve_workspace_path,
    workspace_root_for_session,
)

__all__ = [
    "WorkspaceOperation",
    "WorkspacePermissionDecision",
    "WorkspacePermissionPolicy",
    "WorkspaceRiskLevel",
    "WorkspaceRoot",
    "WorkspaceSandboxMode",
    "WorkspaceSecurityError",
    "is_sensitive_workspace_path",
    "resolve_workspace_path",
    "workspace_root_for_session",
]
