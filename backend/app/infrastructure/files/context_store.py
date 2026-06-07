"""File-backed runtime context storage for chat sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from app.config import Settings


class ContextFileStore:
    """Manage visible runtime context files used by the chat context layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.runtime_context_path
        self.runtime_dir = self.root / "runtime"
        self.sessions_dir = self.root / "sessions"
        self.project_rules_path = self.root / "CLAUDE.md"
        self.user_preferences_path = self.runtime_dir / "user.md"
        self._ensure_base_files()

    def initialize_session(self, session_id: str, title: str) -> None:
        session_dir = self._ensure_session_dirs(session_id)
        session_md = session_dir / "session.md"
        state_path = session_dir / "context_state.json"
        if not session_md.exists():
            lines = [
                "# Session Summary",
                "",
                "## Topic",
                f"- {self._normalize_line(title) or '新对话'}",
                "",
                "## Known Preferences",
                "- None yet.",
                "",
                "## Referenced Documents",
                "- None yet.",
                "",
                "## Pending Topics",
                f"- 当前会话主题：{self._normalize_line(title) or '新对话'}",
                "",
                "## Compacted History",
                "- None yet.",
                "",
            ]
            session_md.write_text("\n".join(lines), encoding="utf-8")
        if not state_path.exists():
            state_path.write_text(
                json.dumps(
                    {
                        "stage": "normal",
                        "estimated_tokens": 0,
                        "budget_tokens": 0,
                        "sources": [],
                        "last_compacted_at": None,
                        "compacted_message_ids": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def get_session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def read_project_rules(self) -> str:
        self._ensure_base_files()
        return self.project_rules_path.read_text(encoding="utf-8").strip()

    def read_user_preferences(self) -> list[str]:
        self._ensure_base_files()
        return self._read_bullets_from_markdown(self.user_preferences_path)

    def add_user_preference(self, preference: str) -> None:
        normalized = self._normalize_line(preference)
        if not normalized:
            return
        entries = self.read_user_preferences()
        if normalized not in entries:
            entries.append(normalized)
        self._write_bullets_markdown(
            self.user_preferences_path,
            title="# User Preferences",
            bullets=entries,
        )

    def read_session_summary(self, session_id: str) -> str:
        self.initialize_session(session_id, "新对话")
        session_md = self.get_session_dir(session_id) / "session.md"
        return session_md.read_text(encoding="utf-8").strip()

    def write_session_summary(
        self,
        session_id: str,
        *,
        title: str,
        user_preferences: list[str],
        references: list[str],
        pending_topics: list[str],
        compact_history: list[str],
    ) -> None:
        self._ensure_session_dirs(session_id)
        sections = {
            "Topic": [self._normalize_line(title) or "新对话"],
            "Known Preferences": self._dedupe(user_preferences),
            "Referenced Documents": self._dedupe(references),
            "Pending Topics": self._dedupe(pending_topics),
            "Compacted History": self._dedupe(compact_history),
        }
        lines = ["# Session Summary", ""]
        for section, bullets in sections.items():
            lines.append(f"## {section}")
            if bullets:
                lines.extend(f"- {item}" for item in bullets)
            else:
                lines.append("- None yet.")
            lines.append("")
        (self.get_session_dir(session_id) / "session.md").write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding="utf-8",
        )

    def list_compact_summaries(self, session_id: str, *, limit: int = 3) -> list[str]:
        compact_dir = self.get_session_dir(session_id) / "compact"
        if not compact_dir.exists():
            return []
        files = sorted(compact_dir.glob("compact-*.md"))
        return [path.read_text(encoding="utf-8").strip() for path in files[-limit:]]

    def append_compact_summary(
        self,
        session_id: str,
        *,
        title: str,
        summary_lines: list[str],
    ) -> str:
        compact_dir = self.get_session_dir(session_id) / "compact"
        compact_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(compact_dir.glob("compact-*.md"))) + 1
        filename = f"compact-{index:03d}.md"
        path = compact_dir / filename
        lines = [f"# {title}", ""]
        if summary_lines:
            lines.extend(f"- {self._normalize_line(item)}" for item in summary_lines if self._normalize_line(item))
        else:
            lines.append("- No summary generated.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return filename

    def read_context_state(self, session_id: str) -> dict[str, Any]:
        self.initialize_session(session_id, "新对话")
        path = self.get_session_dir(session_id) / "context_state.json"
        if not path.exists():
            return {
                "stage": "normal",
                "estimated_tokens": 0,
                "budget_tokens": 0,
                "sources": [],
                "last_compacted_at": None,
                "compacted_message_ids": [],
            }
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("compacted_message_ids", [])
        payload.setdefault("sources", [])
        payload.setdefault("stage", "normal")
        payload.setdefault("estimated_tokens", 0)
        payload.setdefault("budget_tokens", 0)
        payload.setdefault("last_compacted_at", None)
        return payload

    def write_context_state(self, session_id: str, payload: dict[str, Any]) -> None:
        self._ensure_session_dirs(session_id)
        data = dict(payload)
        data.setdefault("compacted_message_ids", [])
        data.setdefault("sources", [])
        data.setdefault("stage", "normal")
        data.setdefault("estimated_tokens", 0)
        data.setdefault("budget_tokens", 0)
        data.setdefault("last_compacted_at", None)
        (self.get_session_dir(session_id) / "context_state.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def mark_compacted_now(self, session_id: str) -> str:
        now = datetime.now(timezone.utc).isoformat()
        state = self.read_context_state(session_id)
        state["last_compacted_at"] = now
        self.write_context_state(session_id, state)
        return now

    def sync_session_summary(
        self,
        session_id: str,
        *,
        title: str,
        user_preferences: list[str],
        references: list[str],
        pending_topics: list[str],
    ) -> None:
        compact_summaries = self.list_compact_summaries(session_id, limit=2)
        compact_history = []
        for compact in compact_summaries:
            first_bullet = next(
                (line[2:].strip() for line in compact.splitlines() if line.startswith("- ")),
                "",
            )
            if first_bullet:
                compact_history.append(first_bullet)
        self.write_session_summary(
            session_id,
            title=title,
            user_preferences=user_preferences,
            references=references,
            pending_topics=pending_topics,
            compact_history=compact_history,
        )

    def _ensure_session_dirs(self, session_id: str) -> Path:
        session_dir = self.get_session_dir(session_id)
        (session_dir / "compact").mkdir(parents=True, exist_ok=True)
        return session_dir

    def _ensure_base_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.project_rules_path.exists():
            source = self.settings.project_root / "CLAUDE.md"
            if source.exists():
                self.project_rules_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                self.project_rules_path.write_text("# PaperDesk\n\n- Default project rules.\n", encoding="utf-8")
        if not self.user_preferences_path.exists():
            self._write_bullets_markdown(
                self.user_preferences_path,
                title="# User Preferences",
                bullets=[],
            )

    @staticmethod
    def _write_bullets_markdown(path: Path, *, title: str, bullets: list[str]) -> None:
        lines = [title, ""]
        if bullets:
            lines.extend(f"- {item}" for item in bullets)
        else:
            lines.append("- None yet.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _read_bullets_from_markdown(path: Path) -> list[str]:
        if not path.exists():
            return []
        bullets: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("- "):
                continue
            value = line[2:].strip()
            if not value or value == "None yet.":
                continue
            bullets.append(value)
        return ContextFileStore._dedupe(bullets)

    @staticmethod
    def _normalize_line(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = ContextFileStore._normalize_line(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results
