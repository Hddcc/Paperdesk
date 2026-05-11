"""Shared schemas for API, services, and frontend coordination."""

from .enums import EvidenceSourceType, ResearchRunStatus, TodoTaskStatus
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
from .research import ResearchRequest, ResearchRun, ResearchRunDetail, ResearchState, TodoTask

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "EvidenceItem",
    "EvidenceSourceType",
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
    "ResearchRunStatus",
    "ResearchState",
    "TaskSummary",
    "TodoTask",
    "TodoTaskStatus",
]
