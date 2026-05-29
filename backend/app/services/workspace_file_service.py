"""Create generated files inside a session workspace with safety checks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import mimetypes
from pathlib import Path
from uuid import uuid4

from app.models import WorkspaceFile, WorkspaceFileListItem, WorkspaceFileReadResult
from app.repositories import ChatRepository, WorkspaceFileRepository
from app.runtime.workspace_security import (
    WorkspaceOperation,
    WorkspacePermissionPolicy,
    WorkspaceSandboxMode,
    WorkspaceSecurityError,
    resolve_workspace_path,
    workspace_root_for_session,
)


class WorkspaceFileServiceError(ValueError):
    """Raised when a generated workspace file cannot be created safely."""


class WorkspaceFileService:
    """Persist agent/system generated text artifacts under workspace sessions."""

    ALLOWED_EXTENSIONS = {
        ".txt": ("txt", "text/plain"),
        ".md": ("md", "text/markdown"),
        ".json": ("json", "application/json"),
        ".csv": ("csv", "text/csv"),
        ".html": ("html", "text/html"),
        ".py": ("py", "text/x-python"),
        ".go": ("go", "text/x-go"),
        ".js": ("js", "text/javascript"),
        ".ts": ("ts", "text/typescript"),
        ".vue": ("vue", "text/x-vue"),
        ".css": ("css", "text/css"),
        ".java": ("java", "text/x-java-source"),
        ".cpp": ("cpp", "text/x-c++src"),
        ".c": ("c", "text/x-csrc"),
        ".rs": ("rs", "text/rust"),
        ".yaml": ("yaml", "application/yaml"),
        ".yml": ("yaml", "application/yaml"),
        ".toml": ("toml", "application/toml"),
    }
    READABLE_EXTENSIONS = {
        ".txt": ("txt", "text/plain"),
        ".md": ("md", "text/markdown"),
        ".json": ("json", "application/json"),
        ".csv": ("csv", "text/csv"),
        ".html": ("html", "text/html"),
        ".py": ("py", "text/x-python"),
        ".go": ("go", "text/x-go"),
        ".js": ("js", "text/javascript"),
        ".ts": ("ts", "text/typescript"),
        ".vue": ("vue", "text/x-vue"),
        ".css": ("css", "text/css"),
        ".java": ("java", "text/x-java-source"),
        ".cpp": ("cpp", "text/x-c++src"),
        ".c": ("c", "text/x-csrc"),
        ".rs": ("rs", "text/rust"),
        ".yaml": ("yaml", "application/yaml"),
        ".yml": ("yaml", "application/yaml"),
        ".toml": ("toml", "application/toml"),
    }
    MESSAGE_FORMAT_EXTENSIONS = {
        "md": ".md",
        "txt": ".txt",
    }
    MAX_LIST_ENTRIES = 500
    DEFAULT_READ_MAX_CHARS = 12000
    MAX_READ_CHARS = 50000

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceFileRepository,
        chat_repository: ChatRepository,
        workspace_base_dir: Path,
        max_file_bytes: int = 1024 * 1024,
        policy: WorkspacePermissionPolicy | None = None,
    ) -> None:
        self.workspace_repository = workspace_repository
        self.chat_repository = chat_repository
        self.workspace_base_dir = workspace_base_dir
        self.max_file_bytes = max_file_bytes
        self.policy = policy or WorkspacePermissionPolicy(mode=WorkspaceSandboxMode.SAFE_WRITE)

    def list_workspace_files(
        self,
        *,
        session_id: str,
        path: str = "",
        recursive: bool = False,
        max_entries: int = 100,
    ) -> list[WorkspaceFileListItem]:
        if self.chat_repository.get_session(session_id) is None:
            raise WorkspaceFileServiceError("Chat session not found")

        root = workspace_root_for_session(self.workspace_base_dir, session_id)
        root_path = root.root.resolve()

        normalized_path = str(path or "").strip()
        if normalized_path:
            try:
                target = resolve_workspace_path(root_path, normalized_path)
            except WorkspaceSecurityError as exc:
                raise WorkspaceFileServiceError(str(exc)) from exc
        else:
            target = root_path

        if not root_path.exists():
            return []

        if not target.exists():
            raise WorkspaceFileServiceError("Workspace path not found")
        if not target.is_dir():
            raise WorkspaceFileServiceError("Workspace path is not a directory")

        limit = max(1, min(int(max_entries or 100), self.MAX_LIST_ENTRIES))
        metadata = {
            item.relative_path: item
            for item in self.workspace_repository.list_by_session(root.session_id)
        }
        iterator = target.rglob("*") if recursive else target.iterdir()
        items: list[WorkspaceFileListItem] = []
        for candidate in sorted(iterator, key=lambda item: item.as_posix()):
            if len(items) >= limit:
                break
            item = self._safe_list_item(root_path, candidate, metadata)
            if item is not None:
                items.append(item)
        return items

    def read_workspace_file(
        self,
        *,
        session_id: str,
        relative_path: str,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> WorkspaceFileReadResult:
        if self.chat_repository.get_session(session_id) is None:
            raise WorkspaceFileServiceError("Chat session not found")

        root = workspace_root_for_session(self.workspace_base_dir, session_id)
        try:
            target = resolve_workspace_path(root.root, relative_path)
        except WorkspaceSecurityError as exc:
            raise WorkspaceFileServiceError(str(exc)) from exc

        if not target.exists():
            raise WorkspaceFileServiceError("Workspace file not found")
        if target.is_dir():
            raise WorkspaceFileServiceError("Workspace path is a directory")
        if not target.is_file():
            raise WorkspaceFileServiceError("Workspace path is not a regular file")

        kind, mime_type = self._readable_extension(target)
        size_bytes = target.stat().st_size
        if size_bytes > self.max_file_bytes:
            raise WorkspaceFileServiceError("Workspace file is too large to read")

        raw = target.read_bytes()
        if b"\x00" in raw[:8192]:
            raise WorkspaceFileServiceError("Binary workspace files cannot be read")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceFileServiceError("Workspace file is not valid UTF-8 text") from exc

        effective_max_chars = max(1, min(int(max_chars or self.DEFAULT_READ_MAX_CHARS), self.MAX_READ_CHARS))
        included = content[:effective_max_chars]
        workspace_relative_path = target.relative_to(root.root.resolve()).as_posix()
        metadata = self.workspace_repository.list_by_session(root.session_id)
        stored = next((item for item in metadata if item.relative_path == workspace_relative_path), None)
        return WorkspaceFileReadResult(
            relative_path=workspace_relative_path,
            display_name=stored.display_name if stored is not None else target.name,
            mime_type=stored.mime_type if stored is not None else mime_type,
            size_bytes=size_bytes,
            content=included,
            char_count=len(content),
            included_chars=len(included),
            truncated=len(included) < len(content),
            checksum=stored.checksum if stored is not None else hashlib.sha256(raw).hexdigest(),
            status=stored.status if stored is not None else "ready",
        )

    def create_from_assistant_message(
        self,
        *,
        session_id: str,
        message_id: str,
        filename: str,
        format: str | None = None,
        title: str | None = None,
    ) -> WorkspaceFile:
        session = self.chat_repository.get_session(session_id)
        if session is None:
            raise WorkspaceFileServiceError("Chat session not found")

        message = self.chat_repository.get_message(message_id)
        if message is None or message.session_id != session_id:
            raise WorkspaceFileServiceError("Chat message not found")
        if message.role != "assistant":
            raise WorkspaceFileServiceError("Only assistant messages can be saved as workspace files")
        if not message.content.strip():
            raise WorkspaceFileServiceError("Assistant message content is empty")

        normalized_filename, file_kind = self._normalize_message_filename(
            filename=filename,
            format=format,
        )
        return self.create_generated_file(
            session_id=session.id,
            relative_path=f"generated/{normalized_filename}",
            content=message.content,
            display_name=(title or normalized_filename).strip() or normalized_filename,
            file_kind=file_kind,
            source_message_id=message.id,
            source_file_ids=list(message.used_file_ids),
            source_document_ids=list(message.used_document_ids),
            created_by="agent",
        )

    def create_generated_file(
        self,
        *,
        session_id: str,
        relative_path: str,
        content: str,
        display_name: str,
        file_kind: str,
        source_message_id: str | None = None,
        source_file_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
        created_by: str = "agent",
    ) -> WorkspaceFile:
        if self.chat_repository.get_session(session_id) is None:
            raise WorkspaceFileServiceError("Chat session not found")
        if created_by not in {"agent", "user", "system"}:
            raise WorkspaceFileServiceError("Invalid workspace file creator")

        root = workspace_root_for_session(self.workspace_base_dir, session_id)
        try:
            destination = resolve_workspace_path(root.root, relative_path)
        except WorkspaceSecurityError as exc:
            raise WorkspaceFileServiceError(str(exc)) from exc

        extension = destination.suffix.casefold()
        allowed = self.ALLOWED_EXTENSIONS.get(extension)
        if allowed is None:
            raise WorkspaceFileServiceError("Unsupported workspace file extension")

        default_kind, mime_type = allowed
        normalized_kind = (file_kind or default_kind).strip() or default_kind
        if normalized_kind != default_kind:
            normalized_kind = default_kind

        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise WorkspaceFileServiceError("Workspace file is too large")
        if destination.exists():
            raise WorkspaceFileServiceError("Workspace file already exists")

        workspace_relative_path = destination.relative_to(root.root.resolve()).as_posix()
        decision = self.policy.classify(
            WorkspaceOperation.WRITE,
            relative_path=workspace_relative_path,
            path_exists=False,
        )
        if not decision.allowed or decision.requires_confirmation:
            raise WorkspaceFileServiceError(decision.reason or "Workspace write is not allowed")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

        now = datetime.now(timezone.utc)
        workspace_file = WorkspaceFile(
            id=str(uuid4()),
            session_id=root.session_id,
            source_message_id=source_message_id,
            created_by=created_by,  # type: ignore[arg-type]
            file_kind=normalized_kind,
            display_name=(display_name or destination.name).strip() or destination.name,
            relative_path=workspace_relative_path,
            storage_path=str(destination),
            mime_type=mime_type,
            size_bytes=len(encoded),
            checksum=hashlib.sha256(encoded).hexdigest(),
            status="ready",
            source_file_ids=list(source_file_ids or []),
            source_document_ids=list(source_document_ids or []),
            created_at=now,
            updated_at=now,
        )
        return self.workspace_repository.create(workspace_file)

    def inspect_existing_text_file(
        self,
        *,
        session_id: str,
        relative_path: str,
    ) -> dict[str, object]:
        target, workspace_relative_path, file_kind, mime_type, raw, content = self._load_existing_text_file(
            session_id=session_id,
            relative_path=relative_path,
        )
        return {
            "relative_path": workspace_relative_path,
            "display_name": target.name,
            "file_kind": file_kind,
            "mime_type": mime_type,
            "size_bytes": len(raw),
            "checksum": hashlib.sha256(raw).hexdigest(),
            "content": content,
        }

    def overwrite_existing_file(
        self,
        *,
        session_id: str,
        relative_path: str,
        content: str,
        old_checksum: str,
        source_message_id: str | None = None,
        source_file_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
        created_by: str = "agent",
    ) -> WorkspaceFile:
        if created_by not in {"agent", "user", "system"}:
            raise WorkspaceFileServiceError("Invalid workspace file creator")

        target, workspace_relative_path, file_kind, mime_type, raw, _old_content = self._load_existing_text_file(
            session_id=session_id,
            relative_path=relative_path,
        )
        current_checksum = hashlib.sha256(raw).hexdigest()
        if current_checksum != old_checksum:
            raise WorkspaceFileServiceError("Workspace file changed after preview")

        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise WorkspaceFileServiceError("Workspace file is too large")

        decision = self.policy.classify(
            WorkspaceOperation.OVERWRITE,
            relative_path=workspace_relative_path,
            path_exists=True,
        )
        if not decision.allowed:
            raise WorkspaceFileServiceError(decision.reason or "Workspace overwrite is not allowed")

        target.write_text(content, encoding="utf-8")
        new_checksum = hashlib.sha256(encoded).hexdigest()
        now = datetime.now(timezone.utc)
        existing = next(
            (
                item
                for item in self.workspace_repository.list_by_session(session_id)
                if item.relative_path == workspace_relative_path
            ),
            None,
        )
        if existing is not None:
            updated = self.workspace_repository.update_metadata(
                existing.id,
                source_message_id=source_message_id,
                file_kind=file_kind,
                display_name=target.name,
                relative_path=workspace_relative_path,
                storage_path=str(target),
                mime_type=mime_type,
                size_bytes=len(encoded),
                checksum=new_checksum,
                status="ready",
                source_file_ids=list(source_file_ids or []),
                source_document_ids=list(source_document_ids or []),
            )
            if updated is None:
                raise WorkspaceFileServiceError("Workspace file metadata update failed")
            return updated

        workspace_file = WorkspaceFile(
            id=str(uuid4()),
            session_id=session_id,
            source_message_id=source_message_id,
            created_by=created_by,  # type: ignore[arg-type]
            file_kind=file_kind,
            display_name=target.name,
            relative_path=workspace_relative_path,
            storage_path=str(target),
            mime_type=mime_type,
            size_bytes=len(encoded),
            checksum=new_checksum,
            status="ready",
            source_file_ids=list(source_file_ids or []),
            source_document_ids=list(source_document_ids or []),
            created_at=now,
            updated_at=now,
        )
        return self.workspace_repository.create(workspace_file)

    def _safe_list_item(
        self,
        root_path: Path,
        candidate: Path,
        metadata: dict[str, WorkspaceFile],
    ) -> WorkspaceFileListItem | None:
        try:
            relative_path = candidate.resolve().relative_to(root_path).as_posix()
            resolved = resolve_workspace_path(root_path, relative_path)
            if resolved != candidate.resolve():
                return None
        except (OSError, RuntimeError, WorkspaceSecurityError, ValueError):
            return None

        stored = metadata.get(relative_path)
        try:
            stat = candidate.stat()
        except OSError:
            return None

        is_directory = candidate.is_dir()
        if is_directory:
            file_kind = "directory"
            mime_type = None
            readable = False
            reason = "Directory"
        else:
            extension = candidate.suffix.casefold()
            read_info = self.READABLE_EXTENSIONS.get(extension)
            file_kind = stored.file_kind if stored is not None else (read_info[0] if read_info else extension.lstrip(".") or "file")
            mime_type = stored.mime_type if stored is not None else (read_info[1] if read_info else mimetypes.guess_type(candidate.name)[0])
            readable = read_info is not None
            reason = None if readable else "Unsupported workspace file extension"

        return WorkspaceFileListItem(
            id=stored.id if stored is not None else None,
            display_name=stored.display_name if stored is not None else candidate.name,
            relative_path=relative_path,
            file_kind=file_kind,
            mime_type=mime_type,
            size_bytes=0 if is_directory else stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            created_at=stored.created_at if stored is not None else None,
            source="generated" if stored is not None else "filesystem",
            is_directory=is_directory,
            status=stored.status if stored is not None else "ready",
            readable=readable,
            reason=reason,
        )

    def _readable_extension(self, path: Path) -> tuple[str, str]:
        extension = path.suffix.casefold()
        readable = self.READABLE_EXTENSIONS.get(extension)
        if readable is None:
            raise WorkspaceFileServiceError("Unsupported workspace file extension")
        return readable

    def _load_existing_text_file(
        self,
        *,
        session_id: str,
        relative_path: str,
    ) -> tuple[Path, str, str, str, bytes, str]:
        if self.chat_repository.get_session(session_id) is None:
            raise WorkspaceFileServiceError("Chat session not found")

        root = workspace_root_for_session(self.workspace_base_dir, session_id)
        try:
            target = resolve_workspace_path(root.root, relative_path)
        except WorkspaceSecurityError as exc:
            raise WorkspaceFileServiceError(str(exc)) from exc

        if not target.exists():
            raise WorkspaceFileServiceError("Workspace file not found")
        if target.is_dir():
            raise WorkspaceFileServiceError("Workspace path is a directory")
        if not target.is_file():
            raise WorkspaceFileServiceError("Workspace path is not a regular file")

        file_kind, mime_type = self._readable_extension(target)
        size_bytes = target.stat().st_size
        if size_bytes > self.max_file_bytes:
            raise WorkspaceFileServiceError("Workspace file is too large to read")

        raw = target.read_bytes()
        if b"\x00" in raw[:8192]:
            raise WorkspaceFileServiceError("Binary workspace files cannot be read")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceFileServiceError("Workspace file is not valid UTF-8 text") from exc

        workspace_relative_path = target.relative_to(root.root.resolve()).as_posix()
        return target, workspace_relative_path, file_kind, mime_type, raw, content

    @classmethod
    def _normalize_message_filename(
        cls,
        *,
        filename: str,
        format: str | None = None,
    ) -> tuple[str, str]:
        cleaned = str(filename or "").strip()
        if not cleaned:
            raise WorkspaceFileServiceError("Workspace filename is required")
        if "\x00" in cleaned:
            raise WorkspaceFileServiceError("Workspace filename is invalid")
        if "/" in cleaned or "\\" in cleaned:
            raise WorkspaceFileServiceError("Workspace filename must not include path separators")

        requested_format = (format or "").strip().casefold().lstrip(".")
        if requested_format and requested_format not in cls.MESSAGE_FORMAT_EXTENSIONS:
            raise WorkspaceFileServiceError("Unsupported workspace file format")

        suffix = Path(cleaned).suffix.casefold()
        if suffix:
            matching_format = None
            for candidate_format, candidate_suffix in cls.MESSAGE_FORMAT_EXTENSIONS.items():
                if suffix == candidate_suffix:
                    matching_format = candidate_format
                    break
            if matching_format is None:
                raise WorkspaceFileServiceError("Unsupported workspace file extension")
            if requested_format and requested_format != matching_format:
                raise WorkspaceFileServiceError("Workspace file format does not match filename extension")
            return cleaned, matching_format

        file_kind = requested_format or "md"
        return f"{cleaned}{cls.MESSAGE_FORMAT_EXTENSIONS[file_kind]}", file_kind
