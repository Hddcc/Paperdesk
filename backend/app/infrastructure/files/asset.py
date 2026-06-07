"""Upload and manage session-scoped user file assets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.models import FileAsset
from app.repositories import ChatRepository, FileAssetRepository

from .text_extractor import FileTextExtractor


class FileAssetService:
    """Handle non-library file uploads for Workbench sessions."""

    ALLOWED_EXTENSIONS = {".txt": "txt", ".md": "md", ".docx": "docx", ".pdf": "pdf"}
    ALLOWED_MIME_TYPES = {
        ".txt": {"text/plain", "application/octet-stream"},
        ".md": {"text/markdown", "text/plain", "application/octet-stream"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/octet-stream",
        },
        ".pdf": {"application/pdf", "application/octet-stream"},
    }
    PREVIEW_LIMIT = 1500
    PDF_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

    def __init__(
        self,
        *,
        file_repository: FileAssetRepository,
        chat_repository: ChatRepository,
        storage_dir: Path,
        max_upload_bytes: int,
        text_extractor: FileTextExtractor | None = None,
    ) -> None:
        self.file_repository = file_repository
        self.chat_repository = chat_repository
        self.storage_dir = storage_dir
        self.max_upload_bytes = max_upload_bytes
        self.text_extractor = text_extractor or FileTextExtractor()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload_session_file(self, session_id: str, upload: UploadFile) -> FileAsset:
        if self.chat_repository.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        original_name = Path(upload.filename).name
        extension = Path(original_name).suffix.lower()
        kind = self.ALLOWED_EXTENSIONS.get(extension)
        if kind is None:
            await upload.close()
            raise HTTPException(status_code=400, detail="Unsupported file extension")

        mime_type = upload.content_type or None
        if mime_type and mime_type not in self.ALLOWED_MIME_TYPES[extension]:
            await upload.close()
            raise HTTPException(status_code=400, detail="Unsupported file MIME type")

        content = await upload.read()
        await upload.close()
        effective_max_upload_bytes = self.PDF_MAX_UPLOAD_BYTES if kind == "pdf" else self.max_upload_bytes
        if len(content) > effective_max_upload_bytes:
            limit_mb = max(1, effective_max_upload_bytes // (1024 * 1024))
            raise HTTPException(status_code=413, detail=f"File is too large（文件超过 {limit_mb}MB，请选择更小的文件。）")

        file_id = str(uuid4())
        safe_filename = f"{file_id}{extension}"
        destination = self.storage_dir / session_id / safe_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

        now = datetime.now(timezone.utc)
        asset = FileAsset(
            id=file_id,
            filename=safe_filename,
            display_name=original_name,
            mime_type=mime_type,
            extension=extension.lstrip("."),
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            storage_path=str(destination),
            source="upload",
            scope="session",
            session_id=session_id,
            kind=kind,
            status="processing",
            text_extract_status="pending",
            created_at=now,
        )
        self.file_repository.create(asset)
        return self._extract_and_update(asset)

    def list_session_files(self, session_id: str) -> list[FileAsset]:
        return self.file_repository.list_by_session(session_id)

    def _extract_and_update(self, asset: FileAsset) -> FileAsset:
        result = self.text_extractor.extract(Path(asset.storage_path), kind=asset.kind)
        if result.status == "ready":
            return self.file_repository.update_status(
                asset.id,
                status="ready",
                text_extract_status="ready",
                preview_text=result.text[: self.PREVIEW_LIMIT],
                text_char_count=len(result.text),
                failure_reason=None,
            ) or asset
        if result.status == "skipped":
            return self.file_repository.update_status(
                asset.id,
                status="unsupported",
                text_extract_status="skipped",
                preview_text=None,
                text_char_count=0,
                failure_reason=result.failure_reason,
            ) or asset
        return self.file_repository.update_status(
            asset.id,
            status="failed",
            text_extract_status="failed",
            preview_text=None,
            text_char_count=len(result.text),
            failure_reason=result.failure_reason or "Text extraction failed",
        ) or asset
