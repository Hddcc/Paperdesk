"""Workspace application boundary."""

from __future__ import annotations

from fastapi import UploadFile

from app.domains.workspace import WorkbenchService, WorkspaceFileService
from app.infrastructure.files import FileAssetService


class WorkspaceUseCase:
    """Use case for workbench and session workspace file flows."""

    def __init__(
        self,
        *,
        workbench_service: WorkbenchService | None = None,
        workspace_file_service: WorkspaceFileService | None = None,
        file_asset_service: FileAssetService | None = None,
    ) -> None:
        self.workbench_service = workbench_service
        self.workspace_file_service = workspace_file_service
        self.file_asset_service = file_asset_service

    def get_config(self):
        return self._workbench().get_config()

    def get_capabilities(self):
        return self._workbench().get_capabilities()

    def get_file_context(self, session_id: str):
        return self._workbench().get_file_context(session_id)

    def get_message_trace_summary(self, message_id: str):
        return self._workbench().get_message_trace_summary(message_id)

    def list_workspace_files(self, *, session_id: str, path: str, recursive: bool, max_entries: int):
        return self._workspace_files().list_workspace_files(
            session_id=session_id,
            path=path,
            recursive=recursive,
            max_entries=max_entries,
        )

    def read_workspace_file(self, *, session_id: str, relative_path: str, max_chars: int):
        return self._workspace_files().read_workspace_file(
            session_id=session_id,
            relative_path=relative_path,
            max_chars=max_chars,
        )

    async def upload_session_file(self, session_id: str, file: UploadFile):
        return await self._file_assets().upload_session_file(session_id, file)

    def _workbench(self) -> WorkbenchService:
        if self.workbench_service is None:
            raise RuntimeError("WorkbenchService is required")
        return self.workbench_service

    def _workspace_files(self) -> WorkspaceFileService:
        if self.workspace_file_service is None:
            raise RuntimeError("WorkspaceFileService is required")
        return self.workspace_file_service

    def _file_assets(self) -> FileAssetService:
        if self.file_asset_service is None:
            raise RuntimeError("FileAssetService is required")
        return self.file_asset_service
