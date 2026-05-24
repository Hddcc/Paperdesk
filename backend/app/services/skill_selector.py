"""Rule-based automatic skill selection for Knowledge Chat tracing."""

from __future__ import annotations

from typing import Any

from app.models import ChatAttachment, SkillManifest, SkillSelection, SkillSelectionResult


class SkillSelector:
    """Select one primary Skill for observability without changing behavior."""

    _SUMMARY_KEYWORDS = ("总结", "概括", "summary", "summarize")
    _REVIEW_KEYWORDS = ("综述", "文献回顾", "review", "overview")
    _COMPARE_KEYWORDS = ("对比", "比较", "区别", "差异", "compare", "comparison")
    _METHOD_KEYWORDS = ("方法", "解释", "讲解", "原理", "method", "explain")
    _BRIEF_KEYWORDS = ("研究路线", "路线图", "研究方向", "brief", "roadmap")
    _STRONG_KEYWORD_GROUPS = {
        "paper_summary": _SUMMARY_KEYWORDS,
        "multi_paper_review": _REVIEW_KEYWORDS,
        "comparison": _COMPARE_KEYWORDS,
        "method_explainer": _METHOD_KEYWORDS,
        "research_brief": _BRIEF_KEYWORDS,
    }
    _EXPECTED_TRIGGER_SKILL_IDS = frozenset({"qa", *_STRONG_KEYWORD_GROUPS})

    def select(
        self,
        *,
        prompt: str,
        command: str | None = None,
        intent_hint: str | None = None,
        selected_document_count: int = 0,
        attachments: list[ChatAttachment] | None = None,
        available_skills: list[SkillManifest] | None = None,
        task_type: str | None = None,
        route: str | None = None,
    ) -> SkillSelectionResult:
        skills = {
            skill.skill_id: skill
            for skill in (available_skills or [])
            if self._is_default_candidate(skill)
        }
        if not skills:
            return SkillSelectionResult()

        normalized_prompt = prompt.casefold()
        normalized_command = (command or "").strip().lstrip("/").casefold()
        normalized_intent = (intent_hint or "").strip().casefold()
        normalized_task_type = (task_type or "").strip().casefold()
        normalized_route = (route or "").strip().casefold()
        attachments = attachments or []
        library_attachment_count = sum(1 for item in attachments if item.kind == "library_document")
        effective_document_count = max(
            selected_document_count,
            sum(1 for item in attachments if item.kind in {"library_document", "uploaded_pdf"}),
        )

        selected_id, triggered_by, reason, confidence, keywords = self._select_skill_id_from_manifest(
            prompt=normalized_prompt,
            command=normalized_command,
            intent_hint=normalized_intent,
            selected_document_count=effective_document_count,
            library_attachment_count=library_attachment_count,
            attachment_kinds={item.kind for item in attachments},
            task_type=normalized_task_type,
            route=normalized_route,
            skills=skills,
        ) or self._select_skill_id(
            prompt=normalized_prompt,
            command=normalized_command,
            intent_hint=normalized_intent,
            selected_document_count=effective_document_count,
            library_attachment_count=library_attachment_count,
            task_type=normalized_task_type,
            route=normalized_route,
            available_skill_ids=set(skills),
        )
        manifest = skills.get(selected_id)
        if manifest is None:
            manifest = skills.get("qa") or next(iter(skills.values()))
            selected_id = manifest.skill_id
            triggered_by = ["fallback"]
            reason = "No matching enabled skill was available; fell back to qa-compatible skill."
            confidence = min(confidence, 0.5)
            keywords = []

        matched_signals: dict[str, Any] = {
            "command": normalized_command or None,
            "intent_hint": normalized_intent or None,
            "selected_document_count": effective_document_count,
            "library_attachment_count": library_attachment_count,
            "keywords": keywords,
            "task_type": normalized_task_type or None,
            "route": normalized_route or None,
        }
        cleaned_signals = {
            key: value
            for key, value in matched_signals.items()
            if value is not None and value != "" and value != []
        }
        selection = SkillSelection(
            skill_id=selected_id,
            name=manifest.name,
            confidence=confidence,
            triggered_by=triggered_by,
            trigger_reason=reason,
            matched_signals=cleaned_signals,
            source="rule",
            is_primary=True,
        )
        return SkillSelectionResult(primary_skill=selection, used_skills=[selection])

    def _select_skill_id(
        self,
        *,
        prompt: str,
        command: str,
        intent_hint: str,
        selected_document_count: int,
        library_attachment_count: int,
        task_type: str,
        route: str,
        available_skill_ids: set[str],
    ) -> tuple[str, list[str], str, float, list[str]]:
        if command == "summary":
            if selected_document_count >= 2:
                return (
                    "multi_paper_review",
                    ["slash_command", "selected_document_count"],
                    "/summary with multiple selected documents.",
                    0.94,
                    [],
                )
            return (
                "paper_summary",
                ["slash_command", "selected_document_count"],
                "/summary with zero or one selected document.",
                0.92,
                [],
            )
        if command == "compare":
            return ("comparison", ["slash_command"], "/compare command selected comparison skill.", 0.96, [])

        hinted_id = self._skill_from_hint(intent_hint or task_type or route)
        if hinted_id in available_skill_ids:
            return (
                hinted_id,
                ["task_type" if task_type else "route" if route else "intent_hint"],
                "Existing task type or route matched a builtin skill.",
                0.9,
                [],
            )

        if selected_document_count >= 2:
            compare_keywords = self._matched_keywords(prompt, self._COMPARE_KEYWORDS)
            if compare_keywords:
                return (
                    "comparison",
                    ["selected_document_count", "prompt_keyword"],
                    "Multiple selected documents and comparison intent.",
                    0.9,
                    compare_keywords,
                )
            review_keywords = self._matched_keywords(prompt, self._REVIEW_KEYWORDS)
            if review_keywords:
                return (
                    "multi_paper_review",
                    ["selected_document_count", "prompt_keyword"],
                    "Multiple selected documents and review intent.",
                    0.88,
                    review_keywords,
                )

        if selected_document_count <= 1:
            summary_keywords = self._matched_keywords(prompt, self._SUMMARY_KEYWORDS)
            if summary_keywords:
                return (
                    "paper_summary",
                    ["selected_document_count", "prompt_keyword"],
                    "Single selected document and summary intent.",
                    0.86,
                    summary_keywords,
                )

        method_keywords = self._matched_keywords(prompt, self._METHOD_KEYWORDS)
        if method_keywords:
            return ("method_explainer", ["prompt_keyword"], "Method explanation intent matched.", 0.84, method_keywords)

        brief_keywords = self._matched_keywords(prompt, self._BRIEF_KEYWORDS)
        if brief_keywords:
            return ("research_brief", ["prompt_keyword"], "Research roadmap or brief intent matched.", 0.84, brief_keywords)

        if library_attachment_count > 0:
            return (
                "qa",
                ["library_document_attachment"],
                "Library document attachment without a stronger task keyword.",
                0.72,
                [],
            )

        return ("qa", ["fallback"], "No explicit skill trigger matched; using qa as the safe default.", 0.55, [])

    def _select_skill_id_from_manifest(
        self,
        *,
        prompt: str,
        command: str,
        intent_hint: str,
        selected_document_count: int,
        library_attachment_count: int,
        attachment_kinds: set[str],
        task_type: str,
        route: str,
        skills: dict[str, SkillManifest],
    ) -> tuple[str, list[str], str, float, list[str]] | None:
        if not self._manifest_triggers_complete(skills):
            return None

        command_match = self._manifest_command_match(command, selected_document_count, skills)
        if command_match is not None:
            return command_match

        hinted_id = self._manifest_hint_match(intent_hint or task_type or route, skills)
        if hinted_id is not None:
            return (
                hinted_id,
                ["task_type" if task_type else "route" if route else "intent_hint"],
                "Existing task type or route matched a builtin skill.",
                0.9,
                [],
            )

        if selected_document_count >= 2:
            compare_keywords = self._manifest_matched_keywords(prompt, "comparison", skills)
            if compare_keywords:
                return (
                    "comparison",
                    ["selected_document_count", "prompt_keyword"],
                    "Multiple selected documents and comparison intent.",
                    0.9,
                    compare_keywords,
                )
            review_keywords = self._manifest_matched_keywords(prompt, "multi_paper_review", skills)
            if review_keywords:
                return (
                    "multi_paper_review",
                    ["selected_document_count", "prompt_keyword"],
                    "Multiple selected documents and review intent.",
                    0.88,
                    review_keywords,
                )

        if selected_document_count <= 1:
            summary_keywords = self._manifest_matched_keywords(prompt, "paper_summary", skills)
            if summary_keywords:
                return (
                    "paper_summary",
                    ["selected_document_count", "prompt_keyword"],
                    "Single selected document and summary intent.",
                    0.86,
                    summary_keywords,
                )

        method_keywords = self._manifest_matched_keywords(prompt, "method_explainer", skills)
        if method_keywords:
            return ("method_explainer", ["prompt_keyword"], "Method explanation intent matched.", 0.84, method_keywords)

        brief_keywords = self._manifest_matched_keywords(prompt, "research_brief", skills)
        if brief_keywords:
            return ("research_brief", ["prompt_keyword"], "Research roadmap or brief intent matched.", 0.84, brief_keywords)

        qa = skills.get("qa")
        if qa is not None and qa.trigger is not None:
            configured_attachment_kinds = {item.casefold() for item in qa.trigger.attachment_kinds}
            if library_attachment_count > 0 and "library_document" in configured_attachment_kinds:
                return (
                    "qa",
                    ["library_document_attachment"],
                    "Library document attachment without a stronger task keyword.",
                    0.72,
                    [],
                )
            if qa.trigger.fallback:
                return ("qa", ["fallback"], "No explicit skill trigger matched; using qa as the safe default.", 0.55, [])

        if any(attachment_kinds):
            return None
        return None

    def _manifest_command_match(
        self,
        command: str,
        selected_document_count: int,
        skills: dict[str, SkillManifest],
    ) -> tuple[str, list[str], str, float, list[str]] | None:
        if not command:
            return None
        matching_ids = [
            skill.skill_id
            for skill in skills.values()
            if skill.trigger is not None
            and command in {item.strip().lstrip("/").casefold() for item in skill.trigger.commands}
            and self._document_count_matches(skill, selected_document_count, allow_missing=True)
        ]
        if command == "summary":
            if selected_document_count >= 2 and "multi_paper_review" in matching_ids:
                return (
                    "multi_paper_review",
                    ["slash_command", "selected_document_count"],
                    "/summary with multiple selected documents.",
                    0.94,
                    [],
                )
            if "paper_summary" in matching_ids:
                return (
                    "paper_summary",
                    ["slash_command", "selected_document_count"],
                    "/summary with zero or one selected document.",
                    0.92,
                    [],
                )
        if command == "compare" and "comparison" in matching_ids:
            return ("comparison", ["slash_command"], "/compare command selected comparison skill.", 0.96, [])
        return None

    @classmethod
    def _manifest_triggers_complete(cls, skills: dict[str, SkillManifest]) -> bool:
        if not cls._EXPECTED_TRIGGER_SKILL_IDS.issubset(skills):
            return False
        for skill_id in cls._EXPECTED_TRIGGER_SKILL_IDS:
            trigger = skills[skill_id].trigger
            if trigger is None:
                return False
            if skill_id == "qa" and not trigger.fallback:
                return False
            if skill_id == "paper_summary" and "summary" not in {item.casefold() for item in trigger.commands}:
                return False
            if skill_id == "multi_paper_review" and "summary" not in {item.casefold() for item in trigger.commands}:
                return False
            if skill_id == "comparison" and "compare" not in {item.casefold() for item in trigger.commands}:
                return False
            if skill_id != "qa" and not trigger.keywords:
                return False
        return True

    def _manifest_hint_match(self, value: str, skills: dict[str, SkillManifest]) -> str | None:
        if not value:
            return None
        normalized = value.casefold()
        for skill in sorted(skills.values(), key=lambda item: (item.priority, item.skill_id)):
            trigger = skill.trigger
            if trigger is None:
                continue
            aliases = [
                *trigger.intent_hints,
                *trigger.task_types,
                *trigger.routes,
                *[item.value for item in skill.supported_task_types],
            ]
            if normalized in {item.casefold() for item in aliases}:
                return skill.skill_id
        for skill in sorted(skills.values(), key=lambda item: (item.priority, item.skill_id)):
            keywords = self._manifest_matched_keywords(normalized, skill.skill_id, skills)
            if keywords:
                return skill.skill_id
        return None

    def _manifest_matched_keywords(
        self,
        prompt: str,
        skill_id: str,
        skills: dict[str, SkillManifest],
    ) -> list[str]:
        skill = skills.get(skill_id)
        if skill is None or skill.trigger is None or not skill.trigger.keywords:
            return []
        return [keyword for keyword in skill.trigger.keywords if keyword.casefold() in prompt]

    @staticmethod
    def _document_count_matches(
        skill: SkillManifest,
        selected_document_count: int,
        *,
        allow_missing: bool = False,
    ) -> bool:
        constraint = skill.trigger.document_count if skill.trigger is not None else None
        if constraint is None:
            return allow_missing
        if constraint.min is not None and selected_document_count < constraint.min:
            return False
        if constraint.max is not None and selected_document_count > constraint.max:
            return False
        return True

    @staticmethod
    def _is_default_candidate(skill: SkillManifest) -> bool:
        return (
            skill.enabled
            and skill.available_by_default
            and str(skill.maturity.value if hasattr(skill.maturity, "value") else skill.maturity) == "stable"
            and str(skill.source.value if hasattr(skill.source, "value") else skill.source) == "builtin"
        )

    @classmethod
    def _skill_from_hint(cls, value: str) -> str | None:
        if not value:
            return None
        aliases = {
            "summary": "paper_summary",
            "paper_summary": "paper_summary",
            "single_paper_summary": "paper_summary",
            "review": "multi_paper_review",
            "multi_paper_review": "multi_paper_review",
            "comparison": "comparison",
            "compare": "comparison",
            "method": "method_explainer",
            "method_explainer": "method_explainer",
            "research_brief": "research_brief",
            "brief": "research_brief",
            "qa": "qa",
            "paper_qa": "qa",
            "knowledge_qa": "qa",
        }
        if value in aliases:
            return aliases[value]
        for skill_id, keywords in cls._STRONG_KEYWORD_GROUPS.items():
            if any(keyword.casefold() in value for keyword in keywords):
                return skill_id
        return None

    @staticmethod
    def _matched_keywords(prompt: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if keyword.casefold() in prompt]
