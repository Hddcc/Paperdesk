"""Report routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.main import get_export_service, get_report_repository
from app.repositories import ReportRepository
from app.services import ExportService

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports(repository: ReportRepository = Depends(get_report_repository)) -> list[dict]:
    return [report.model_dump(mode="json") for report in repository.list_reports()]


@router.get("/{report_id}")
def get_report(report_id: str, repository: ReportRepository = Depends(get_report_repository)) -> dict:
    report = repository.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.model_dump(mode="json")


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
    _safe_unlink(export_path)
    return report.model_dump(mode="json")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
