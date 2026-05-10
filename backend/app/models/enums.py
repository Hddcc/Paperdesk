"""Enum definitions for the PaperDesk workflow."""

from enum import Enum


class ResearchRunStatus(str, Enum):
    """State machine for a research run."""

    CREATED = "created"
    PLANNING = "planning"
    RUNNING_TASK = "running_task"
    WRITING_REPORT = "writing_report"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TodoTaskStatus(str, Enum):
    """State machine for a single TODO task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def coerce_research_run_status(value: str | ResearchRunStatus) -> ResearchRunStatus:
    """Coerce persisted legacy run states into the current phase-06 state model."""

    if isinstance(value, ResearchRunStatus):
        return value

    cleaned = value.strip().lower()
    if cleaned in ResearchRunStatus._value2member_map_:
        return ResearchRunStatus(cleaned)
    if cleaned in {"searching_online", "retrieving_local", "summarizing_task"}:
        return ResearchRunStatus.RUNNING_TASK
    raise ValueError(f"Unsupported research run status: {value}")


def coerce_todo_task_status(value: str | TodoTaskStatus) -> TodoTaskStatus:
    """Coerce persisted legacy task states into the current phase-06 task state model."""

    if isinstance(value, TodoTaskStatus):
        return value

    cleaned = value.strip().lower()
    if cleaned in TodoTaskStatus._value2member_map_:
        return TodoTaskStatus(cleaned)
    if cleaned in {"created", "planning"}:
        return TodoTaskStatus.PENDING
    if cleaned in {"searching_online", "retrieving_local", "summarizing_task", "writing_report"}:
        return TodoTaskStatus.IN_PROGRESS
    if cleaned == "cancelled":
        return TodoTaskStatus.FAILED
    raise ValueError(f"Unsupported todo task status: {value}")


class EvidenceSourceType(str, Enum):
    """Supported evidence source categories."""

    ONLINE_PAPER = "online_paper"
    LOCAL_DOCUMENT = "local_document"
