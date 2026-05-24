"""Shared skill selection trace models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SkillSelection(BaseModel):
    """Safe trace payload describing an automatically selected skill."""

    skill_id: str
    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    triggered_by: list[str] = Field(default_factory=list)
    trigger_reason: str = ""
    matched_signals: dict[str, Any] = Field(default_factory=dict)
    source: str = "rule"
    is_primary: bool = True


class SkillSelectionResult(BaseModel):
    """Skill selection result for the current user turn."""

    primary_skill: SkillSelection | None = None
    used_skills: list[SkillSelection] = Field(default_factory=list)
