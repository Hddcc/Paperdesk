"""Backward-compatible schema imports."""

from .library import ChunkRecord, LibraryDocument
from .paper import (
    EvidenceItem,
    PaperAnalysisRequest,
    PaperAnalysisResponse,
    PaperAnalysisSection,
    PaperCurationItem,
    PaperCurationRequest,
    PaperCurationResponse,
    PaperRecord,
    PaperSearchRequest,
    PaperSearchResponse,
    RagAskRequest,
    RagAskResponse,
)
from .report import CitationRecord, ReportListItem, ResearchReport, TaskSummary
from .research import ResearchRequest, ResearchRun, ResearchRunDetail, TodoTask
from .research_runtime import ResearchRuntimeState

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "EvidenceItem",
    "LibraryDocument",
    "PaperAnalysisRequest",
    "PaperAnalysisResponse",
    "PaperAnalysisSection",
    "PaperCurationItem",
    "PaperCurationRequest",
    "PaperCurationResponse",
    "PaperRecord",
    "PaperSearchRequest",
    "PaperSearchResponse",
    "RagAskRequest",
    "RagAskResponse",
    "ReportListItem",
    "ResearchReport",
    "ResearchRequest",
    "ResearchRun",
    "ResearchRunDetail",
    "ResearchRuntimeState",
    "TaskSummary",
    "TodoTask",
]
