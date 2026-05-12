"""Report routes."""

import os
from pathlib import Path
import time

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
