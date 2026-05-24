"""Product-facing research task routing models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .skill_selection import SkillSelection


class ResearchInputMode(str, Enum):
    """User-facing input forms that can enter the unified task chain."""

    PROMPT = "prompt"
    UPLOADED_FILE = "uploaded_file"
    KNOWLEDGE_BASE = "knowledge_base"


class ResearchTaskType(str, Enum):
    """Result-oriented research task types recognized before agent execution."""

    QA = "qa"
    PAPER_SUMMARY = "paper_summary"
    MULTI_PAPER_REVIEW = "multi_paper_review"
    COMPARISON = "comparison"
    METHOD_EXPLAINER = "method_explainer"
    RESEARCH_BRIEF_TASK = "research_brief"


class ResearchEvidencePolicy(str, Enum):
    """Default evidence sourcing policy selected by the task router."""

    LOCAL_FIRST = "local_first"
    ONLINE_FIRST = "online_first"
    LOCAL_ONLY = "local_only"
    ONLINE_SUPPLEMENT = "online_supplement"


class ResearchExecutionRoute(str, Enum):
    """Coarse backend path chosen for a product task."""

    KNOWLEDGE_QA = "knowledge_qa"
    SINGLE_PAPER_SUMMARY = "single_paper_summary"
    MAIN_AGENT_REVIEW = "main_agent_review"
    COMPARISON_ANALYSIS = "comparison_analysis"
    METHOD_EXPLANATION = "method_explanation"
    RESEARCH_BRIEF = "research_brief"


class ResearchArtifactProtocolType(str, Enum):
    """Stable result protocol families for final artifacts."""

    QA = "qa"
    PAPER_SUMMARY = "paper_summary"
    REVIEW = "review"
    COMPARISON = "comparison"
    METHOD_EXPLAINER = "method_explainer"
    RESEARCH_BRIEF = "research_brief"


class ResearchArtifactProtocol(BaseModel):
    """Completion contract for a final research artifact."""

    protocol_type: ResearchArtifactProtocolType
    title: str
    required_sections: list[str] = Field(default_factory=list)
    citation_required: bool = True


class ResearchTaskRoute(BaseModel):
    """Task route decision produced before the main-agent loop starts."""

    task_type: ResearchTaskType
    input_modes: list[ResearchInputMode] = Field(default_factory=lambda: [ResearchInputMode.PROMPT])
    evidence_policy: ResearchEvidencePolicy
    execution_route: ResearchExecutionRoute
    artifact_protocol: ResearchArtifactProtocol
    selected_document_ids: list[str] = Field(default_factory=list)
    needs_local_knowledge: bool = False
    needs_online_search: bool = True
    use_main_agent_loop: bool = True
    allow_single_pass: bool = False
    active_skill_id: str | None = None
    used_skills: list[SkillSelection] = Field(default_factory=list)
    rationale: str = ""
