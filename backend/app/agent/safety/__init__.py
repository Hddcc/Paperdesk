"""Safety boundary for scope checks, confirmations, and write guards."""

from __future__ import annotations

from typing import Any

from .pending import PendingActionStore
from .workspace import (
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
from .write import AgentWriteSafetyService

__all__ = [
    "AgentWriteSafetyService",
    "PendingActionStore",
    "WorkspaceOperation",
    "WorkspacePermissionDecision",
    "WorkspacePermissionPolicy",
    "WorkspaceRiskLevel",
    "WorkspaceRoot",
    "WorkspaceSandboxMode",
    "WorkspaceSecurityError",
    "WorkspaceBoundaryGuard",
    "WorkspaceIntentResolver",
    "WorkspacePathExtractor",
    "is_sensitive_workspace_path",
    "resolve_workspace_path",
    "workspace_root_for_session",
]


def __getattr__(name: str) -> Any:
    if name == "WorkspaceBoundaryGuard":
        from app.domains.workspace.operations import WorkspaceBoundaryGuard

        return WorkspaceBoundaryGuard
    if name == "WorkspaceIntentResolver":
        from app.domains.workspace.operations import WorkspaceIntentResolver

        return WorkspaceIntentResolver
    if name == "WorkspacePathExtractor":
        from app.domains.workspace.operations import WorkspacePathExtractor

        return WorkspacePathExtractor
    raise AttributeError(name)
