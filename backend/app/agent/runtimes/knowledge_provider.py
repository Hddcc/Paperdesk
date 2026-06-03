"""Capability provider around the legacy KnowledgeAgentRuntime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class KnowledgeAgentCapabilityProvider:
    """Expose reusable knowledge-agent capabilities through a narrow surface."""

    def __init__(self, runtime: Any | None) -> None:
        self.runtime = runtime

    @property
    def available(self) -> bool:
        return self.runtime is not None

    @property
    def pending_action_store(self):
        return getattr(self.runtime, "pending_action_store", None)

    def is_status_only_answer(self, content: str) -> bool:
        if self.runtime is None:
            return False
        return bool(self.runtime.is_status_only_answer(content))

    def ensure_final_answer(self, **kwargs: Any) -> str:
        if self.runtime is None:
            return str(kwargs.get("previous_content") or "")
        return self.runtime.ensure_final_answer(**kwargs)

    def conversation_referents(self, session_id: str) -> dict[str, Any]:
        if self.runtime is None:
            return {}
        return self.runtime.conversation_referents(session_id)

    def has_pending_action(self, session_id: str) -> bool:
        if self.runtime is None:
            return False
        return bool(self.runtime.has_pending_action(session_id))

    def handle(self, **kwargs: Any):
        if self.runtime is None:
            return None
        return self.runtime.handle(**kwargs)

    def run_react(self, **kwargs: Any):
        if self.runtime is None:
            return None
        return self.runtime.run_react(**kwargs)

    def build_context_lines(self) -> list[str]:
        if self.runtime is None:
            return []
        return self.runtime.build_context_lines()

    def record_fast_path_referents(
        self,
        *,
        session_id: str,
        content: str,
        document_ids: list[str],
        source: str,
    ) -> None:
        if self.runtime is None or not document_ids:
            return
        read_state = getattr(self.runtime, "_read_react_state", None)
        write_state = getattr(self.runtime, "_write_react_state", None)
        if not callable(read_state) or not callable(write_state):
            return
        ids = list(dict.fromkeys(str(document_id) for document_id in document_ids if document_id))
        if not ids:
            return
        state = read_state(session_id)
        referent = {
            "label": f"fast-path resolved {len(ids)} documents",
            "document_ids": ids,
            "source_tool": "chat.fast_path",
            "source": source,
            "count": len(ids),
        }
        state["last_document_set"] = referent
        if len(ids) == 1:
            state["last_single_document"] = {
                **referent,
                "document_id": ids[0],
                "count": 1,
            }
        else:
            state["last_multi_document_set"] = referent
        state["last_user_goal"] = content[:240]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_state(session_id, state)
