"""Artifact domain facade over report and export services."""

from __future__ import annotations

from app.domains.paper import ReportLifecycleService

from .export import ExportService


class ArtifactDomainFacade:
    """Named artifact boundary for report/export and future diagram outputs."""

    report_lifecycle_service: type[ReportLifecycleService] = ReportLifecycleService
    export_service: type[ExportService] = ExportService
