"""Markdown export route."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from app.api.main import get_export_service, get_report_repository
from app.domains.artifact import ExportService
from app.repositories import ReportRepository

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/{report_id}")
def export_report(
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
        headers={
            "Content-Disposition": f'attachment; filename="{export_path.name}"',
        },
    )
