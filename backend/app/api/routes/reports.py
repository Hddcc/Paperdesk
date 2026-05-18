"""Report routes."""

import os
from pathlib import Path
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.api.main import get_export_service, get_report_lifecycle_service, get_report_repository
from app.repositories import ReportRepository
from app.services import ExportService, ReportLifecycleService

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportFromMessageRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    session_id: str | None = None
    optional_title: str | None = None


@router.get("")
def list_reports(repository: ReportRepository = Depends(get_report_repository)) -> list[dict]:
    return [report.model_dump(mode="json") for report in repository.list_reports()]


@router.get("/{report_id}")
def get_report(report_id: str, repository: ReportRepository = Depends(get_report_repository)) -> dict:
    report = repository.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.model_dump(mode="json")


@router.post("/from-message")
def save_report_from_message(
    request: ReportFromMessageRequest,
    service: ReportLifecycleService = Depends(get_report_lifecycle_service),
) -> dict:
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for local chat messages")
    try:
        report = service.save_from_message(
            session_id=request.session_id,
            message_id=request.message_id,
            optional_title=request.optional_title,
        )
    except ValueError as exc:
        status_code = 400 if str(exc).startswith("Only assistant") else 404
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return report.model_dump(mode="json")


@router.get("/{report_id}/export.md")
def export_report_markdown(
    report_id: str,
    repository: ReportRepository = Depends(get_report_repository),
    export_service: ExportService = Depends(get_export_service),
) -> PlainTextResponse:
    report = repository.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    export_path = export_service.export_markdown(report)
    return PlainTextResponse(
        content=report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{export_path.name}"'},
    )


@router.delete("/{report_id}")
def delete_report(
    report_id: str,
    repository: ReportRepository = Depends(get_report_repository),
    export_service: ExportService = Depends(get_export_service),
) -> dict:
    report = repository.delete_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    export_path = export_service.get_export_path(report_id)
    _delete_report_exports(export_path)
    return report.model_dump(mode="json")


def _safe_unlink(path: Path) -> None:
    for _ in range(3):
        try:
            if path.exists():
                os.chmod(path, 0o666)
            path.unlink(missing_ok=True)
            if not path.exists():
                return
        except OSError:
            time.sleep(0.05)


def _delete_report_exports(primary_path: Path) -> None:
    candidate_paths: set[Path] = {primary_path}

    env_report_dir = os.environ.get("REPORT_DIR")
    if env_report_dir:
        candidate_paths.add(Path(env_report_dir) / primary_path.name)

    search_roots = {primary_path.parent}
    if env_report_dir:
        search_roots.add(Path(env_report_dir))
        search_roots.add(Path(env_report_dir).parent)

    for root in list(search_roots):
        if root.exists():
            for candidate in root.rglob(primary_path.name):
                candidate_paths.add(candidate)

    for candidate in candidate_paths:
        _safe_unlink(candidate)
