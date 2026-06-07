"""Workspace path and permission safety primitives.

This module is intentionally standalone: it defines the local workspace
boundary and operation risk classification, but it does not execute commands or
perform file writes for the agent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PureWindowsPath
import re


class WorkspaceSecurityError(ValueError):
    """Raised when a workspace path or operation violates the safety policy."""


class WorkspaceRiskLevel(str, Enum):
    """Risk buckets used by workspace permission decisions."""

    AUTO_ALLOW = "auto_allow"
    CONFIRM = "confirm"
    STRONG_CONFIRM = "strong_confirm"
    FORBID = "forbid"


class WorkspaceSandboxMode(str, Enum):
    """Workspace permission modes for local-agent file safety."""

    READ_ONLY = "read_only"
    SAFE_WRITE = "safe_write"
    DEV_MODE = "dev_mode"
    STRICT_CONFIRM = "strict_confirm"


class WorkspaceOperation(str, Enum):
    """Supported operation names for the workspace permission policy."""

    LIST = "list"
    READ = "read"
    WRITE = "write"
    OVERWRITE = "overwrite"
    EDIT = "edit"
    RENAME = "rename"
    DELETE = "delete"
    COMMAND = "command"


@dataclass(frozen=True)
class WorkspaceRoot:
    """Canonical session-scoped workspace root."""

    session_id: str
    root: Path


@dataclass(frozen=True)
class WorkspacePermissionDecision:
    """Result of classifying a workspace operation."""

    operation: WorkspaceOperation
    risk_level: WorkspaceRiskLevel
    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""
    matched_rules: list[str] = field(default_factory=list)


_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_SENSITIVE_EXACT_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".netrc",
    "appdata",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "keys",
    "known_hosts",
    "secrets",
    "tokens",
}
_SENSITIVE_SUFFIXES = {
    ".cer",
    ".crt",
    ".der",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
}


def workspace_root_for_session(workspace_base: Path, session_id: str) -> WorkspaceRoot:
    """Return the default local-agent root for one chat session.

    The root is always under the configured backend workspace directory:
    ``{workspace_base}/sessions/{session_id}``. The caller may create it later
    when a write-capable phase is introduced.
    """

    normalized_session_id = session_id.strip()
    if _SESSION_ID_PATTERN.fullmatch(normalized_session_id) is None:
        raise WorkspaceSecurityError("Invalid workspace session id")
    root = (workspace_base / "sessions" / normalized_session_id).resolve()
    workspace_base_resolved = workspace_base.resolve()
    try:
        root.relative_to(workspace_base_resolved)
    except ValueError as exc:
        raise WorkspaceSecurityError("Workspace root must stay under workspace base") from exc
    return WorkspaceRoot(session_id=normalized_session_id, root=root)


def resolve_workspace_path(root: Path, relative_path: str) -> Path:
    """Resolve a user-provided relative path inside a workspace root.

    The function rejects path traversal, absolute paths, Windows drive paths,
    UNC paths, hidden path components, sensitive files, and symlink escapes.
    """

    root_resolved = root.resolve()
    raw_path = _normalize_user_path(relative_path)
    _reject_disallowed_raw_path(raw_path)

    candidate = (root_resolved / raw_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise WorkspaceSecurityError("Path escapes workspace root") from exc

    relative_candidate = candidate.relative_to(root_resolved)
    _reject_sensitive_relative_path(relative_candidate)
    return candidate


def is_sensitive_workspace_path(relative_path: str | Path) -> bool:
    """Return whether a workspace-relative path is hidden or sensitive."""

    try:
        _reject_sensitive_relative_path(Path(relative_path))
    except WorkspaceSecurityError:
        return True
    return False


class WorkspacePermissionPolicy:
    """Classify workspace operations without performing them."""

    def __init__(self, mode: WorkspaceSandboxMode | str = WorkspaceSandboxMode.SAFE_WRITE) -> None:
        self.mode = WorkspaceSandboxMode(mode)

    def classify(
        self,
        operation: WorkspaceOperation | str,
        *,
        relative_path: str | Path | None = None,
        path_exists: bool = False,
        is_sensitive: bool | None = None,
        command_name: str | None = None,
    ) -> WorkspacePermissionDecision:
        op = WorkspaceOperation(operation)
        matched_rules: list[str] = []

        sensitive = is_sensitive
        if sensitive is None and relative_path is not None:
            sensitive = is_sensitive_workspace_path(relative_path)
        sensitive = bool(sensitive)
        if sensitive:
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.FORBID,
                allowed=False,
                reason="Sensitive workspace path is forbidden",
                matched_rules=["sensitive_path"],
            )

        if op in {WorkspaceOperation.LIST, WorkspaceOperation.READ}:
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.AUTO_ALLOW,
                allowed=True,
                reason="Read-only workspace operation",
                matched_rules=["read_only"],
            )

        if op == WorkspaceOperation.COMMAND:
            return self._command_decision(op, command_name=command_name)

        if self.mode == WorkspaceSandboxMode.READ_ONLY:
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.FORBID,
                allowed=False,
                reason=f"{self.mode.value} mode allows only list/read operations",
                matched_rules=["sandbox_read_only"],
            )

        if op == WorkspaceOperation.WRITE:
            if self.mode == WorkspaceSandboxMode.STRICT_CONFIRM:
                return WorkspacePermissionDecision(
                    operation=op,
                    risk_level=WorkspaceRiskLevel.CONFIRM,
                    allowed=True,
                    requires_confirmation=True,
                    reason="strict_confirm mode requires confirmation for new file writes",
                    matched_rules=["sandbox_strict_confirm", "write"],
                )
            if path_exists:
                return WorkspacePermissionDecision(
                    operation=op,
                    risk_level=WorkspaceRiskLevel.CONFIRM,
                    allowed=True,
                    requires_confirmation=True,
                    reason="Writing to an existing path requires confirmation",
                    matched_rules=["existing_path_write"],
                )
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.AUTO_ALLOW,
                allowed=True,
                reason="Creating a new workspace artifact is allowed",
                matched_rules=["new_artifact_write"],
            )

        if op in {WorkspaceOperation.OVERWRITE, WorkspaceOperation.EDIT, WorkspaceOperation.RENAME}:
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.CONFIRM,
                allowed=True,
                requires_confirmation=True,
                reason=f"{op.value} requires confirmation",
                matched_rules=[op.value],
            )

        if op == WorkspaceOperation.DELETE:
            return WorkspacePermissionDecision(
                operation=op,
                risk_level=WorkspaceRiskLevel.STRONG_CONFIRM,
                allowed=True,
                requires_confirmation=True,
                reason="Delete requires strong confirmation",
                matched_rules=["delete"],
            )

        return WorkspacePermissionDecision(
            operation=op,
            risk_level=WorkspaceRiskLevel.FORBID,
            allowed=False,
            reason="Unsupported workspace operation",
            matched_rules=matched_rules or ["unsupported_operation"],
        )

    def _command_decision(
        self,
        operation: WorkspaceOperation,
        *,
        command_name: str | None,
    ) -> WorkspacePermissionDecision:
        if self.mode == WorkspaceSandboxMode.READ_ONLY:
            return WorkspacePermissionDecision(
                operation=operation,
                risk_level=WorkspaceRiskLevel.FORBID,
                allowed=False,
                reason="Command execution is forbidden in read_only mode",
                matched_rules=["sandbox_read_only", "command_forbidden"],
            )
        if command_name and command_name.casefold() in {"git status", "git diff"}:
            return WorkspacePermissionDecision(
                operation=operation,
                risk_level=WorkspaceRiskLevel.CONFIRM,
                allowed=False,
                requires_confirmation=True,
                reason="Command execution is not enabled in phase 39.2.1",
                matched_rules=["command_disabled"],
            )
        return WorkspacePermissionDecision(
            operation=operation,
            risk_level=WorkspaceRiskLevel.FORBID,
            allowed=False,
            reason="Command execution is forbidden in phase 39.2.1",
            matched_rules=["command_forbidden"],
        )


def _normalize_user_path(relative_path: str) -> str:
    cleaned = str(relative_path or "").strip()
    if not cleaned:
        raise WorkspaceSecurityError("Workspace path is required")
    return cleaned.replace("\\", "/")


def _reject_disallowed_raw_path(relative_path: str) -> None:
    if relative_path.startswith(("/", "\\")):
        raise WorkspaceSecurityError("Absolute paths are not allowed")
    if relative_path.startswith("//") or relative_path.startswith("\\\\"):
        raise WorkspaceSecurityError("UNC paths are not allowed")
    if _WINDOWS_DRIVE_PATTERN.match(relative_path):
        raise WorkspaceSecurityError("Windows drive paths are not allowed")
    if PureWindowsPath(relative_path).drive:
        raise WorkspaceSecurityError("Windows drive paths are not allowed")

    parts = PureWindowsPath(relative_path).parts
    if any(part == ".." for part in parts):
        raise WorkspaceSecurityError("Path traversal is not allowed")
    if any(part in {"", "."} for part in parts):
        return


def _reject_sensitive_relative_path(relative_path: Path) -> None:
    for part in relative_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise WorkspaceSecurityError("Path traversal is not allowed")
        lower = part.casefold()
        if lower.startswith("."):
            raise WorkspaceSecurityError("Hidden workspace paths are not allowed")
        if lower in _SENSITIVE_EXACT_NAMES:
            raise WorkspaceSecurityError("Sensitive workspace file is not allowed")
        if _has_sensitive_name_tokens(lower):
            raise WorkspaceSecurityError("Sensitive workspace file is not allowed")
        if Path(lower).suffix in _SENSITIVE_SUFFIXES:
            raise WorkspaceSecurityError("Sensitive workspace file is not allowed")


def _has_sensitive_name_tokens(filename: str) -> bool:
    stem = Path(filename).stem.casefold()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if stem in {"apikey"}:
        return True
    if "secret" in tokens or "token" in tokens or "credential" in tokens or "credentials" in tokens:
        return True
    if {"api", "key"}.issubset(tokens):
        return True
    if {"access", "key"}.issubset(tokens):
        return True
    if {"auth", "token"}.issubset(tokens):
        return True
    if {"private", "key"}.issubset(tokens):
        return True
    return False
