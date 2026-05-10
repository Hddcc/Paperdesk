"""Shared schemas for API, services, and frontend coordination."""

from .enums import EvidenceSourceType, ResearchRunStatus, TodoTaskStatus
from .library import ChunkRecord, LibraryDocument
from .paper import EvidenceItem, PaperRecord, PaperSearchRequest, PaperSearchResponse
from .report import CitationRecord, ReportListItem, ResearchReport, TaskSummary
from .research import ResearchRequest, ResearchRun, ResearchState, TodoTask

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "EvidenceItem",
    "EvidenceSourceType",
    "LibraryDocument",
    "PaperRecord",
    "PaperSearchRequest",
    "PaperSearchResponse",
    "ReportListItem",
    "ResearchReport",
    "ResearchRequest",
    "ResearchRun",
    "ResearchRunStatus",
    "ResearchState",
    "TaskSummary",
    "TodoTask",
    "TodoTaskStatus",
]
