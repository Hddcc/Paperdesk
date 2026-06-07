"""Lightweight memory boundary for recent context, summaries, and preferences."""

from __future__ import annotations

from typing import Any

from .context import AgentContextLifecycleService

__all__ = [
    "AgentContextLifecycleService",
    "ChatMemoryService",
    "ContextAssembler",
    "ContextBudgetService",
    "ContextCompactionService",
]


def __getattr__(name: str) -> Any:
    if name == "ChatMemoryService":
        from app.services.chat_memory_service import ChatMemoryService

        return ChatMemoryService
    if name == "ContextAssembler":
        from app.services.context_assembler import ContextAssembler

        return ContextAssembler
    if name == "ContextBudgetService":
        from app.services.context_budget_service import ContextBudgetService

        return ContextBudgetService
    if name == "ContextCompactionService":
        from app.services.context_compaction_service import ContextCompactionService

        return ContextCompactionService
    raise AttributeError(name)
