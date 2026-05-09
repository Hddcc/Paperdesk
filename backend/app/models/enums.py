"""Enum definitions for the PaperDesk workflow."""

from enum import Enum


class ResearchRunStatus(str, Enum):
    """State machine for a research run."""

    CREATED = "created"
    PLANNING = "planning"
    SEARCHING_ONLINE = "searching_online"
    RETRIEVING_LOCAL = "retrieving_local"
    SUMMARIZING_TASK = "summarizing_task"
    WRITING_REPORT = "writing_report"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceSourceType(str, Enum):
    """Supported evidence source categories."""

    ONLINE_PAPER = "online_paper"
    LOCAL_DOCUMENT = "local_document"

