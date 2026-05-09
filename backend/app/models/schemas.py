"""Backward-compatible schema imports."""

from .library import ChunkRecord, LibraryDocument
from .paper import EvidenceItem, PaperRecord
from .report import CitationRecord, ReportListItem, ResearchReport, TaskSummary
from .research import ResearchRequest, ResearchRun, TodoTask

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "EvidenceItem",
    "LibraryDocument",
    "PaperRecord",
    "ReportListItem",
    "ResearchReport",
    "ResearchRequest",
    "ResearchRun",
    "TaskSummary",
    "TodoTask",
]
