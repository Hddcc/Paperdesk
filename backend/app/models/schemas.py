"""Backward-compatible schema imports."""

from .library import ChunkRecord, LibraryDocument
from .paper import EvidenceItem, PaperRecord, PaperSearchRequest, PaperSearchResponse
from .report import CitationRecord, ReportListItem, ResearchReport, TaskSummary
from .research import ResearchRequest, ResearchRun, TodoTask

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "EvidenceItem",
    "LibraryDocument",
    "PaperRecord",
    "PaperSearchRequest",
    "PaperSearchResponse",
    "ReportListItem",
    "ResearchReport",
    "ResearchRequest",
    "ResearchRun",
    "TaskSummary",
    "TodoTask",
]
