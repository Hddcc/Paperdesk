"""File-backed skill registry for backend research capabilities."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.models import SkillDefinition, SkillManifest, ResearchTaskType, SkillMaturity, SkillSource


class SkillRegistry:
    """Load, validate and select backend skills."""

    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        custom_skill_dirs: list[Path] | None = None,
    ) -> None:
        self.skills_dir = skills_dir or Path(__file__).resolve().parent.parent / "skills" / "builtin"
        self.custom_skill_dirs = list(custom_skill_dirs or [])
        self._manifests: dict[str, SkillManifest] = {}
        self._skill_dirs: dict[str, Path] = {}
        self._definitions: dict[str, SkillDefinition] = {}
        self.validation_errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._manifests.clear()
        self._skill_dirs.clear()
        self._definitions.clear()
        self.validation_errors.clear()
        self._load_skill_root(self.skills_dir, default_source=SkillSource.BUILTIN)
        for custom_dir in self.custom_skill_dirs:
            self._load_skill_root(custom_dir, default_source=SkillSource.CUSTOM)

    def register_manifest(self, manifest: SkillManifest, skill_dir: Path) -> None:
        self._manifests[manifest.skill_id] = manifest
        self._skill_dirs[manifest.skill_id] = skill_dir

    def register(self, skill: SkillDefinition) -> None:
        manifest = SkillManifest(
            skill_id=skill.skill_id,
            name=skill.name,
            enabled=skill.enabled,
            supported_task_types=skill.supported_task_types,
            default_execution_mode=skill.default_execution_mode,
            description=skill.description,
            artifact_protocol=skill.artifact_protocol,
            version=skill.version,
            priority=skill.priority,
            allowed_tool_ids=list(skill.available_tools),
        )
        self._manifests[skill.skill_id] = manifest
        self._definitions[skill.skill_id] = skill

    def list_all(self) -> list[SkillManifest]:
        return sorted(self._manifests.values(), key=lambda skill: (skill.priority, skill.skill_id))

    def list_enabled(self) -> list[SkillManifest]:
        return [
            skill
            for skill in self.list_all()
            if skill.enabled
            and skill.available_by_default
            and skill.maturity == SkillMaturity.STABLE
            and skill.source in {SkillSource.BUILTIN, SkillSource.CUSTOM}
        ]

    def candidates_for(self, task_type: ResearchTaskType) -> list[SkillManifest]:
        return [
            skill
            for skill in self.list_enabled()
            if task_type in skill.supported_task_types
        ]

    def default_for(self, task_type: ResearchTaskType) -> SkillManifest | None:
        candidates = self.candidates_for(task_type)
        if not candidates:
            return None
        return candidates[0]

    def load_definition(self, skill_id: str) -> SkillDefinition | None:
        if skill_id in self._definitions:
            return self._definitions[skill_id]
        manifest = self._manifests.get(skill_id)
        skill_dir = self._skill_dirs.get(skill_id)
        if manifest is None or skill_dir is None:
            return None
        body_path = skill_dir / manifest.skill_file
        if not self._is_safe_skill_file(skill_dir, body_path):
            self.validation_errors[manifest.skill_id] = "Skill file must stay inside its skill directory."
            return None
        if not body_path.exists():
            return None
        body = body_path.read_text(encoding="utf-8").strip()
        metadata = self._load_skill_metadata(body)
        available_tools = self._dedupe([*manifest.allowed_tool_ids, *metadata.get("available_tools", [])])
        definition = SkillDefinition(
            skill_id=manifest.skill_id,
            name=manifest.name,
            enabled=manifest.enabled,
            supported_task_types=manifest.supported_task_types,
            default_execution_mode=manifest.default_execution_mode,
            description=manifest.description,
            body=body,
            available_tools=available_tools,
            references=metadata.get("references", []),
            inputs=metadata.get("inputs", {}),
            artifact_protocol=manifest.artifact_protocol,
            version=manifest.version,
            priority=manifest.priority,
        )
        self._definitions[skill_id] = definition
        return definition

    def _load_skill_root(self, root: Path, *, default_source: SkillSource) -> None:
        if not root.exists():
            return
        for path in sorted(root.glob("*/manifest.json")):
            try:
                manifest = self._load_manifest(path, default_source=default_source)
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                self.validation_errors[str(path)] = str(exc)
                continue
            self._manifests[manifest.skill_id] = manifest
            self._skill_dirs[manifest.skill_id] = path.parent

    def _load_manifest(self, path: Path, *, default_source: SkillSource) -> SkillManifest:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("source", default_source.value)
        payload.setdefault("skill_file", "SKILL.md")
        skill_file = str(payload.get("skill_file") or "SKILL.md")
        if Path(skill_file).is_absolute() or ".." in Path(skill_file).parts:
            raise ValueError("skill_file must be a relative path inside the skill directory")
        manifest = SkillManifest.model_validate(payload)
        if default_source == SkillSource.CUSTOM and manifest.source == SkillSource.BUILTIN:
            manifest.source = SkillSource.CUSTOM
        return manifest

    @staticmethod
    def _load_skill_metadata(body: str) -> dict:
        marker = "```json"
        if marker not in body:
            return {}
        start = body.find(marker) + len(marker)
        end = body.find("```", start)
        if end < 0:
            return {}
        raw = body[start:end].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            results.append(item)
        return results

    @staticmethod
    def _is_safe_skill_file(skill_dir: Path, body_path: Path) -> bool:
        try:
            return body_path.resolve().is_relative_to(skill_dir.resolve())
        except AttributeError:
            resolved_body = body_path.resolve()
            resolved_dir = skill_dir.resolve()
            return str(resolved_body).startswith(str(resolved_dir))
