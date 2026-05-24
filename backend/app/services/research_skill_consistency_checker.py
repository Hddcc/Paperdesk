"""Read-only consistency checks for research routes and skill triggers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import (
    ResearchArtifactProtocolType,
    ResearchExecutionRoute,
    ResearchTaskRoute,
    ResearchTaskType,
    SkillManifest,
    SkillSelectionResult,
)


class ResearchSkillConsistencyExpected(BaseModel):
    """Static expectation for one research task type."""

    skill_id: str
    execution_route: ResearchExecutionRoute
    protocol_type: ResearchArtifactProtocolType


class ResearchSkillConsistencyMismatch(BaseModel):
    """Single route/skill consistency mismatch."""

    field: str
    expected: str | list[str] | None = None
    actual: str | list[str] | None = None
    message: str


class ResearchSkillConsistencyReport(BaseModel):
    """Read-only report for route and skill consistency."""

    ok: bool
    severity: Literal["ok", "warning", "error"]
    task_type: str
    expected_skill_id: str | None = None
    active_skill_id: str | None = None
    primary_skill_id: str | None = None
    used_skill_ids: list[str] = Field(default_factory=list)
    execution_route: str | None = None
    expected_execution_route: str | None = None
    artifact_protocol_type: str | None = None
    expected_protocol_type: str | None = None
    skill_artifact_protocol_type: str | None = None
    mismatches: list[ResearchSkillConsistencyMismatch] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchSkillConsistencyChecker:
    """Compare route decisions with skill selection metadata without side effects."""

    EXPECTED: dict[ResearchTaskType, ResearchSkillConsistencyExpected] = {
        ResearchTaskType.QA: ResearchSkillConsistencyExpected(
            skill_id="qa",
            execution_route=ResearchExecutionRoute.KNOWLEDGE_QA,
            protocol_type=ResearchArtifactProtocolType.QA,
        ),
        ResearchTaskType.PAPER_SUMMARY: ResearchSkillConsistencyExpected(
            skill_id="paper_summary",
            execution_route=ResearchExecutionRoute.SINGLE_PAPER_SUMMARY,
            protocol_type=ResearchArtifactProtocolType.PAPER_SUMMARY,
        ),
        ResearchTaskType.MULTI_PAPER_REVIEW: ResearchSkillConsistencyExpected(
            skill_id="multi_paper_review",
            execution_route=ResearchExecutionRoute.MAIN_AGENT_REVIEW,
            protocol_type=ResearchArtifactProtocolType.REVIEW,
        ),
        ResearchTaskType.COMPARISON: ResearchSkillConsistencyExpected(
            skill_id="comparison",
            execution_route=ResearchExecutionRoute.COMPARISON_ANALYSIS,
            protocol_type=ResearchArtifactProtocolType.COMPARISON,
        ),
        ResearchTaskType.METHOD_EXPLAINER: ResearchSkillConsistencyExpected(
            skill_id="method_explainer",
            execution_route=ResearchExecutionRoute.METHOD_EXPLANATION,
            protocol_type=ResearchArtifactProtocolType.METHOD_EXPLAINER,
        ),
        ResearchTaskType.RESEARCH_BRIEF_TASK: ResearchSkillConsistencyExpected(
            skill_id="research_brief",
            execution_route=ResearchExecutionRoute.RESEARCH_BRIEF,
            protocol_type=ResearchArtifactProtocolType.RESEARCH_BRIEF,
        ),
    }

    def check_route_selection(
        self,
        *,
        task_route: ResearchTaskRoute,
        skill_selection_result: SkillSelectionResult,
        available_skills: list[SkillManifest],
        warning_only: bool = False,
    ) -> ResearchSkillConsistencyReport:
        """Return a consistency report without mutating the route or selection."""

        expected = self.EXPECTED.get(task_route.task_type)
        primary_skill_id = (
            skill_selection_result.primary_skill.skill_id
            if skill_selection_result.primary_skill is not None
            else None
        )
        used_skill_ids = [skill.skill_id for skill in skill_selection_result.used_skills]
        expected_skill_id = expected.skill_id if expected is not None else None
        expected_execution_route = expected.execution_route.value if expected is not None else None
        expected_protocol_type = expected.protocol_type.value if expected is not None else None
        skill_by_id = {skill.skill_id: skill for skill in available_skills}
        active_skill = skill_by_id.get(task_route.active_skill_id or "")
        skill_protocol_type = (
            active_skill.artifact_protocol.protocol_type.value
            if active_skill is not None
            else None
        )
        mismatches: list[ResearchSkillConsistencyMismatch] = []
        notes: list[str] = []

        if expected is None:
            mismatches.append(
                ResearchSkillConsistencyMismatch(
                    field="task_type",
                    expected=sorted(item.value for item in self.EXPECTED),
                    actual=task_route.task_type.value,
                    message="Task type is not covered by the static consistency expectation table.",
                )
            )
        else:
            self._append_mismatch(
                mismatches,
                field="primary_skill_id",
                expected=expected.skill_id,
                actual=primary_skill_id,
                message="Primary skill should match the expected skill for task_type.",
            )
            self._append_mismatch(
                mismatches,
                field="active_skill_id",
                expected=expected.skill_id,
                actual=task_route.active_skill_id,
                message="Active skill should match the expected skill for task_type.",
            )
            first_used_skill_id = used_skill_ids[0] if used_skill_ids else None
            self._append_mismatch(
                mismatches,
                field="used_skill_ids[0]",
                expected=primary_skill_id,
                actual=first_used_skill_id,
                message="First used skill should match the primary skill.",
            )
            self._append_mismatch(
                mismatches,
                field="execution_route",
                expected=expected.execution_route.value,
                actual=task_route.execution_route.value,
                message="Execution route should match the static expectation for task_type.",
            )
            self._append_mismatch(
                mismatches,
                field="artifact_protocol.protocol_type",
                expected=expected.protocol_type.value,
                actual=task_route.artifact_protocol.protocol_type.value,
                message="Route artifact protocol should match the static expectation for task_type.",
            )
            if active_skill is None:
                mismatches.append(
                    ResearchSkillConsistencyMismatch(
                        field="active_skill_manifest",
                        expected=expected.skill_id,
                        actual=task_route.active_skill_id,
                        message="Active skill manifest was not found in available skills.",
                    )
                )
            else:
                supported_task_types = [item.value for item in active_skill.supported_task_types]
                if task_route.task_type not in active_skill.supported_task_types:
                    mismatches.append(
                        ResearchSkillConsistencyMismatch(
                            field="manifest.supported_task_types",
                            expected=task_route.task_type.value,
                            actual=supported_task_types,
                            message="Active skill manifest should support the routed task type.",
                        )
                    )
                trigger = active_skill.trigger
                if trigger is None:
                    mismatches.append(
                        ResearchSkillConsistencyMismatch(
                            field="manifest.trigger",
                            expected="present",
                            actual=None,
                            message="Active skill manifest should include trigger metadata.",
                        )
                    )
                else:
                    trigger_task_types = [item.casefold() for item in trigger.task_types]
                    trigger_routes = [item.casefold() for item in trigger.routes]
                    if task_route.task_type.value.casefold() not in trigger_task_types:
                        mismatches.append(
                            ResearchSkillConsistencyMismatch(
                                field="manifest.trigger.task_types",
                                expected=task_route.task_type.value,
                                actual=trigger.task_types,
                                message="Active skill trigger should list the routed task type.",
                            )
                        )
                    if task_route.execution_route.value.casefold() not in trigger_routes:
                        mismatches.append(
                            ResearchSkillConsistencyMismatch(
                                field="manifest.trigger.routes",
                                expected=task_route.execution_route.value,
                                actual=trigger.routes,
                                message="Active skill trigger should list the routed execution route.",
                            )
                        )
                self._append_mismatch(
                    mismatches,
                    field="manifest.artifact_protocol.protocol_type",
                    expected=expected.protocol_type.value,
                    actual=active_skill.artifact_protocol.protocol_type.value,
                    message="Active skill artifact protocol should match the static expectation.",
                )

        if mismatches:
            notes.append("Consistency mismatches were observed in read-only checker mode.")
        else:
            notes.append("Route, skill selection, manifest trigger, and artifact protocol are aligned.")
        severity: Literal["ok", "warning", "error"] = (
            "ok" if not mismatches else "warning" if warning_only else "error"
        )
        return ResearchSkillConsistencyReport(
            ok=not mismatches,
            severity=severity,
            task_type=task_route.task_type.value,
            expected_skill_id=expected_skill_id,
            active_skill_id=task_route.active_skill_id,
            primary_skill_id=primary_skill_id,
            used_skill_ids=used_skill_ids,
            execution_route=task_route.execution_route.value,
            expected_execution_route=expected_execution_route,
            artifact_protocol_type=task_route.artifact_protocol.protocol_type.value,
            expected_protocol_type=expected_protocol_type,
            skill_artifact_protocol_type=skill_protocol_type,
            mismatches=mismatches,
            notes=notes,
        )

    @staticmethod
    def _append_mismatch(
        mismatches: list[ResearchSkillConsistencyMismatch],
        *,
        field: str,
        expected: str | list[str] | None,
        actual: str | list[str] | None,
        message: str,
    ) -> None:
        if actual != expected:
            mismatches.append(
                ResearchSkillConsistencyMismatch(
                    field=field,
                    expected=expected,
                    actual=actual,
                    message=message,
                )
            )
