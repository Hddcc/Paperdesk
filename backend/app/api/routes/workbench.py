"""Workbench read-only routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.main import get_workbench_service, get_workspace_file_service, get_workspace_use_case
from app.application import WorkspaceUseCase
from app.domains.workspace import WorkbenchService, WorkspaceFileService, WorkspaceFileServiceError
from app.models import (
    WorkbenchCapabilitiesResponse,
    WorkbenchConfigResponse,
    WorkbenchFileContextResponse,
    WorkbenchMessageTraceSummary,
    WorkspaceFileListResponse,
)

router = APIRouter(prefix="/workbench", tags=["workbench"])


class MessageWorkspaceFileCreateRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    format: str | None = None
    title: str | None = None


class MessageWorkspaceFileExportRequest(BaseModel):
    path: str = Field(..., min_length=1)


@router.get("/config")
def get_workbench_config(use_case: WorkspaceUseCase = Depends(get_workspace_use_case)) -> dict:
    return WorkbenchConfigResponse.model_validate(use_case.get_config()).model_dump(mode="json")


@router.get("/capabilities")
def get_workbench_capabilities(use_case: WorkspaceUseCase = Depends(get_workspace_use_case)) -> dict:
    return WorkbenchCapabilitiesResponse.model_validate(use_case.get_capabilities()).model_dump(mode="json")


@router.get("/sessions/{session_id}/files")
def get_workbench_session_files(
    session_id: str,
    use_case: WorkspaceUseCase = Depends(get_workspace_use_case),
) -> dict:
    try:
        context = use_case.get_file_context(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WorkbenchFileContextResponse.model_validate(context).model_dump(mode="json")


@router.get("/sessions/{session_id}/workspace-files")
def list_workspace_files(
    session_id: str,
    path: str = Query(default=""),
    recursive: bool = Query(default=False),
    max_entries: int = Query(default=100, ge=1, le=500),
    service: WorkspaceFileService = Depends(get_workspace_file_service),
) -> dict:
    try:
        files = service.list_workspace_files(
            session_id=session_id,
            path=path,
            recursive=recursive,
            max_entries=max_entries,
        )
    except WorkspaceFileServiceError as exc:
        raise _workspace_file_http_error(exc) from exc
    return WorkspaceFileListResponse(
        session_id=session_id,
        path=path,
        recursive=recursive,
        files=files,
    ).model_dump(mode="json")


@router.get("/sessions/{session_id}/workspace-files/read")
def read_workspace_file(
    session_id: str,
    path: str = Query(..., min_length=1),
    max_chars: int = Query(default=WorkspaceFileService.DEFAULT_READ_MAX_CHARS, ge=1, le=WorkspaceFileService.MAX_READ_CHARS),
    service: WorkspaceFileService = Depends(get_workspace_file_service),
) -> dict:
    try:
        result = service.read_workspace_file(
            session_id=session_id,
            relative_path=path,
            max_chars=max_chars,
        )
    except WorkspaceFileServiceError as exc:
        raise _workspace_file_http_error(exc) from exc
    return result.model_dump(mode="json")


@router.post("/sessions/{session_id}/files/upload")
async def upload_workbench_session_file(
    session_id: str,
    file: UploadFile = File(...),
    use_case: WorkspaceUseCase = Depends(get_workspace_use_case),
) -> dict:
    return (await use_case.upload_session_file(session_id, file)).model_dump(mode="json")


@router.post("/sessions/{session_id}/messages/{message_id}/workspace-files")
def create_workspace_file_from_message(
    session_id: str,
    message_id: str,
    request: MessageWorkspaceFileCreateRequest,
    service: WorkspaceFileService = Depends(get_workspace_file_service),
) -> dict:
    try:
        workspace_file = service.create_from_assistant_message(
            session_id=session_id,
            message_id=message_id,
            filename=request.filename,
            format=request.format,
            title=request.title,
        )
    except WorkspaceFileServiceError as exc:
        message = str(exc)
        if message in {"Chat session not found", "Chat message not found"}:
            raise HTTPException(status_code=404, detail=message) from exc
        if "already exists" in message:
            raise HTTPException(status_code=409, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    return workspace_file.model_dump(mode="json")


@router.post("/sessions/{session_id}/messages/{message_id}/export-to-path")
def export_message_to_local_path(
    session_id: str,
    message_id: str,
    request: MessageWorkspaceFileExportRequest,
    service: WorkspaceFileService = Depends(get_workspace_file_service),
) -> dict:
    try:
        workspace_file = service.export_assistant_message_to_path(
            session_id=session_id,
            message_id=message_id,
            destination_path=request.path,
        )
    except WorkspaceFileServiceError as exc:
        message = str(exc)
        if message in {"Chat session not found", "Chat message not found"}:
            raise HTTPException(status_code=404, detail=message) from exc
        if "already exists" in message:
            raise HTTPException(status_code=409, detail=message) from exc
        if "too large" in message:
            raise HTTPException(status_code=413, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    return workspace_file.model_dump(mode="json")


@router.get("/sessions/{session_id}/workspace-files/{file_id}/download")
def download_workspace_file(
    session_id: str,
    file_id: str,
    service: WorkspaceFileService = Depends(get_workspace_file_service),
) -> FileResponse:
    try:
        path, workspace_file = service.get_workspace_file_for_download(
            session_id=session_id,
            file_id=file_id,
        )
    except WorkspaceFileServiceError as exc:
        raise _workspace_file_http_error(exc) from exc
    return FileResponse(
        path,
        media_type=workspace_file.mime_type or "application/octet-stream",
        filename=workspace_file.display_name,
    )


def _workspace_file_http_error(exc: WorkspaceFileServiceError) -> HTTPException:
    message = str(exc)
    if message in {"Chat session not found", "Workspace file not found", "Workspace path not found"}:
        return HTTPException(status_code=404, detail=message)
    if "too large" in message:
        return HTTPException(status_code=413, detail=message)
    return HTTPException(status_code=400, detail=message)


@router.get("/messages/{message_id}/trace")
def get_workbench_message_trace(
    message_id: str,
    use_case: WorkspaceUseCase = Depends(get_workspace_use_case),
) -> dict:
    try:
        summary = use_case.get_message_trace_summary(message_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WorkbenchMessageTraceSummary.model_validate(summary).model_dump(mode="json")
