"""Workbench read-only routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.main import get_file_asset_service, get_workbench_service
from app.models import (
    WorkbenchCapabilitiesResponse,
    WorkbenchConfigResponse,
    WorkbenchFileContextResponse,
    WorkbenchMessageTraceSummary,
)
from app.services import FileAssetService, WorkbenchService

router = APIRouter(prefix="/workbench", tags=["workbench"])


@router.get("/config")
def get_workbench_config(service: WorkbenchService = Depends(get_workbench_service)) -> dict:
    return WorkbenchConfigResponse.model_validate(service.get_config()).model_dump(mode="json")


@router.get("/capabilities")
def get_workbench_capabilities(service: WorkbenchService = Depends(get_workbench_service)) -> dict:
    return WorkbenchCapabilitiesResponse.model_validate(service.get_capabilities()).model_dump(mode="json")


@router.get("/sessions/{session_id}/files")
def get_workbench_session_files(
    session_id: str,
    service: WorkbenchService = Depends(get_workbench_service),
) -> dict:
    try:
        context = service.get_file_context(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WorkbenchFileContextResponse.model_validate(context).model_dump(mode="json")


@router.post("/sessions/{session_id}/files/upload")
async def upload_workbench_session_file(
    session_id: str,
    file: UploadFile = File(...),
    service: FileAssetService = Depends(get_file_asset_service),
) -> dict:
    return (await service.upload_session_file(session_id, file)).model_dump(mode="json")


@router.get("/messages/{message_id}/trace")
def get_workbench_message_trace(
    message_id: str,
    service: WorkbenchService = Depends(get_workbench_service),
) -> dict:
    try:
        summary = service.get_message_trace_summary(message_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WorkbenchMessageTraceSummary.model_validate(summary).model_dump(mode="json")
