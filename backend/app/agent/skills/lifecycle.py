"""Lifecycle skill adapter for PaperDesk Agent requests."""

from __future__ import annotations

from typing import Any

from app.models import ActiveSkillState, AgentLifecycleStage, RuntimeRequest, SkillSelectionResult


class AgentSkillLifecycleService:
    """Attach selected skill metadata to lifecycle requests."""

    def __init__(self, skill_registry: Any) -> None:
        self.skill_registry = skill_registry

    def active_skill_from_selection(self, selection_result: SkillSelectionResult | None) -> ActiveSkillState | None:
        if selection_result is None or selection_result.primary_skill is None:
            return None
        selection = selection_result.primary_skill
        manifest = next(
            (skill for skill in self.skill_registry.list_enabled() if skill.skill_id == selection.skill_id),
            None,
        )
        if manifest is None:
            return None
        return ActiveSkillState(
            skill_id=manifest.skill_id,
            name=manifest.name,
            source=str(manifest.source.value if hasattr(manifest.source, "value") else manifest.source),
            confidence=selection.confidence,
            trigger_reason=selection.trigger_reason,
            allowed_tool_ids=list(manifest.allowed_tool_ids),
            capability_ids=list(manifest.capability_ids),
        )

    def attach_active_skill(
        self,
        request: RuntimeRequest,
        selection_result: SkillSelectionResult | None,
    ) -> RuntimeRequest:
        request.active_skill = self.active_skill_from_selection(selection_result)
        request.add_trace(
            AgentLifecycleStage.SKILL,
            "active skill resolved" if request.active_skill else "no active skill selected",
            {
                "skill_id": request.active_skill.skill_id if request.active_skill else None,
                "source": request.active_skill.source if request.active_skill else None,
                "allowed_tool_count": len(request.active_skill.allowed_tool_ids) if request.active_skill else 0,
                "capability_ids": list(request.active_skill.capability_ids) if request.active_skill else [],
            },
        )
        return request
