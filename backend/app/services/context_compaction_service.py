"""Compaction helpers for Claude Code-style chat context management."""

from __future__ import annotations

from app.config import Settings
from app.models import ChatAttachment, ChatMessage, EvidenceItem

from .context_file_store import ContextFileStore


class ContextCompactionService:
    """Compact evidence and history without changing upstream business ownership."""

    def __init__(self, settings: Settings, file_store: ContextFileStore) -> None:
        self.settings = settings
        self.file_store = file_store

    def compact_evidence(self, evidence_items: list[EvidenceItem]) -> list[str]:
        compacted: list[str] = []
        for item in evidence_items[: self.settings.max_evidence_items]:
            quote = (item.quote or item.snippet or "").strip()
            if len(quote) > self.settings.max_evidence_chars_per_item:
                head = quote[: self.settings.max_evidence_chars_per_item // 2].rstrip()
                tail = quote[-self.settings.max_evidence_chars_per_item // 2 :].lstrip()
                quote = f"{head} ... {tail}"
            compacted.append(
                "\n".join(
                    [
                        f"来源：{item.citation_label}",
                        f"标题：{item.title}",
                        f"页码：{item.page_number if item.page_number is not None else '未知'}",
                        f"短摘录：{quote or '无可用摘录'}",
                    ]
                )
            )
        return compacted

    def build_attachment_hints(self, attachments: list[ChatAttachment]) -> list[str]:
        hints: list[str] = []
        for attachment in attachments:
            if attachment.kind == "image":
                hints.append(f"图片：{attachment.display_name}")
                continue
            title = attachment.metadata.get("title") if attachment.metadata else None
            descriptor = title if isinstance(title, str) and title.strip() else attachment.display_name
            hints.append(f"论文：{descriptor}")
        return hints

    def compact_history(
        self,
        session_id: str,
        *,
        history: list[ChatMessage],
        already_compacted_ids: set[str],
    ) -> tuple[set[str], list[str], str | None]:
        keep_messages = max(self.settings.recent_turns_min * 2, 4)
        if len(history) <= keep_messages:
            return already_compacted_ids, [], None

        candidates = [
            message
            for message in history[:-keep_messages]
            if message.role in {"user", "assistant"} and message.id not in already_compacted_ids
        ]
        if not candidates:
            return already_compacted_ids, [], None

        summary_lines = []
        for message in candidates[-8:]:
            role_label = "用户" if message.role == "user" else "助手"
            snippet = " ".join(message.content.strip().split())
            if len(snippet) > 160:
                snippet = f"{snippet[:159]}…"
            summary_lines.append(f"{role_label}：{snippet}")

        filename = self.file_store.append_compact_summary(
            session_id,
            title="Compacted Conversation History",
            summary_lines=summary_lines,
        )
        next_ids = set(already_compacted_ids)
        next_ids.update(message.id for message in candidates)
        return next_ids, summary_lines, filename
