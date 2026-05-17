"""Structured self-reflection models for chat-side agent runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReflectionActionType = Literal["call_tool", "rewrite_answer", "replan", "record_lesson"]


class ReflectionImprovementAction(BaseModel):
    """Executable suggestion produced by the reflection evaluator."""

    type: ReflectionActionType
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class ReflectionResult(BaseModel):
    """Auditable scoring result for a knowledge-agent or planner answer."""

    overall_score: int = Field(..., ge=1, le=10)
    intent_score: int = Field(..., ge=1, le=10)
    tool_score: int = Field(..., ge=1, le=10)
    evidence_score: int = Field(..., ge=1, le=10)
    answer_score: int = Field(..., ge=1, le=10)
    completion_score: int = Field(..., ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    improvement_actions: list[ReflectionImprovementAction] = Field(default_factory=list)
    should_retry: bool = False
    memory_lessons: list[str] = Field(default_factory=list)
