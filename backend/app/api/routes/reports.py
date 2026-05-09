"""Report routes."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.main import get_repository
from app.repositories import SQLiteRepository

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports(repository: SQLiteRepository = Depends(get_repository)) -> list[dict]:
    return [report.model_dump(mode="json") for report in repository.list_reports()]


@router.get("/{report_id}")
def get_report(report_id: str, repository: SQLiteRepository = Depends(get_repository)) -> dict:
    report = repository.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.model_dump(mode="json")

