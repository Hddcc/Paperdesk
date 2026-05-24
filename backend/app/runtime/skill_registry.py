"""File-backed skill registry for backend research capabilities."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import SkillDefinition, SkillManifest, ResearchTaskType, SkillMaturity, SkillSource


class SkillRegistry:
    """Load, validate and select backend skills."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir or Path(__file__).resolve().parent.parent / "skills" / "builtin"
        self._manifests: dict[str, SkillManifest] = {}
        self._skill_dirs: dict[str, Path] = {}
        self._definitions: dict[str, SkillDefinition] = {}
        self.reload()

    def reload(self) -> None:
        self._manifests.clear()
        self._skill_dirs.clear()
        self._definitions.clear()
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.glob("*/manifest.json")):
            manifest = self._load_manifest(path)
            self._manifests[manifest.skill_id] = manifest
            self._skill_dirs[manifest.skill_id] = path.parent

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
            and skill.source == SkillSource.BUILTIN
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
        if not body_path.exists():
            return None
        body = body_path.read_text(encoding="utf-8").strip()
        metadata = self._load_skill_metadata(body)
        definition = SkillDefinition(
            skill_id=manifest.skill_id,
            name=manifest.name,
            enabled=manifest.enabled,
            supported_task_types=manifest.supported_task_types,
            default_execution_mode=manifest.default_execution_mode,
            description=manifest.description,
            body=body,
            available_tools=metadata.get("available_tools", []),
            references=metadata.get("references", []),
            inputs=metadata.get("inputs", {}),
            artifact_protocol=manifest.artifact_protocol,
            version=manifest.version,
            priority=manifest.priority,
        )
        self._definitions[skill_id] = definition
        return definition

    @staticmethod
    def _load_manifest(path: Path) -> SkillManifest:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SkillManifest.model_validate(payload)

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
