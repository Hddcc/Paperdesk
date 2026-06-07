"""Compatibility package for paper research agents.

New code should import these business agents from
`app.domains.paper.research_agents`.
"""

from app.domains.paper.research_agents import (
    LibraryRetrieverAgent,
    PaperAnalysisAgent,
    PaperSearchAgent,
    PaperSelectionAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)

__all__ = [
    "LibraryRetrieverAgent",
    "PaperAnalysisAgent",
    "PaperSearchAgent",
    "PaperSelectionAgent",
    "ReadingSummarizerAgent",
    "ReportWriterAgent",
    "TopicPlannerAgent",
]
