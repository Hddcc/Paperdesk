"""Backward-compatible schema imports."""

from .library import (
    ChunkRecord,
    DocumentCategory,
    DocumentCategoryAssignmentRequest,
    DocumentCategoryCreateRequest,
    DocumentCategoryUpdateRequest,
    LibraryDocument,
)
from .paper import (
    EvidenceQuality,
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
from .research_runtime import ResearchContextState, ResearchRuntimeState

__all__ = [
    "ChunkRecord",
    "CitationRecord",
    "DocumentCategory",
    "DocumentCategoryAssignmentRequest",
    "DocumentCategoryCreateRequest",
    "DocumentCategoryUpdateRequest",
    "EvidenceQuality",
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
    "ResearchContextState",
    "ResearchRuntimeState",
    "TaskSummary",
    "TodoTask",
]
