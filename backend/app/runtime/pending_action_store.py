"""File-backed storage for Knowledge pending actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.context_file_store import ContextFileStore


class PendingActionStore:
    """Own low-level pending action file IO without business semantics."""

    filename = "pending_knowledge_action.json"

    def __init__(self, file_store: ContextFileStore) -> None:
        self.file_store = file_store

    def path_for(self, session_id: str) -> Path:
        self.file_store.initialize_session(session_id, "知识库对话")
        return self.file_store.get_session_dir(session_id) / self.filename

    def write(self, session_id: str, payload: dict[str, Any]) -> None:
        self.path_for(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def clear(self, session_id: str) -> None:
        self.path_for(session_id).unlink(missing_ok=True)

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).exists()
