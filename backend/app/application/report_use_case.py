"""Report application boundary."""

from __future__ import annotations

from app.repositories import ReportRepository
from app.domains.artifact import ExportService
from app.domains.paper import ReportLifecycleService


class ReportUseCase:
    """Use case for report listing, saving, export, and deletion."""

    def __init__(
        self,
        *,
        report_repository: ReportRepository | None = None,
        report_lifecycle_service: ReportLifecycleService | None = None,
        export_service: ExportService | None = None,
    ) -> None:
        self.report_repository = report_repository
        self.report_lifecycle_service = report_lifecycle_service
        self.export_service = export_service

    def list_reports(self):
        return self._repository().list_reports()

    def get_report(self, report_id: str):
        return self._repository().get_report(report_id)

    def save_from_message(self, *, session_id: str, message_id: str, optional_title: str | None = None):
        return self._lifecycle().save_from_message(
            session_id=session_id,
            message_id=message_id,
            optional_title=optional_title,
        )

    def export_markdown(self, report):
        return self._export().export_markdown(report)

    def delete_report(self, report_id: str):
        return self._repository().delete_report(report_id)

    def export_path(self, report_id: str):
        return self._export().get_export_path(report_id)

    def _repository(self) -> ReportRepository:
        if self.report_repository is None:
            raise RuntimeError("ReportRepository is required")
        return self.report_repository

    def _lifecycle(self) -> ReportLifecycleService:
        if self.report_lifecycle_service is None:
            raise RuntimeError("ReportLifecycleService is required")
        return self.report_lifecycle_service

    def _export(self) -> ExportService:
        if self.export_service is None:
            raise RuntimeError("ExportService is required")
        return self.export_service
