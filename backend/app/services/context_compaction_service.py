"""Compaction helpers for Claude Code-style chat context management."""

from __future__ import annotations

import re

from openai import OpenAI

from app.config import Settings
from app.models import ChatAttachment, ChatMessage, EvidenceItem

from .context_file_store import ContextFileStore


class ContextCompactionService:
    """Compact evidence and history without changing upstream business ownership."""

    def __init__(
        self,
        settings: Settings,
        file_store: ContextFileStore,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.settings = settings
        self.file_store = file_store
        self.model = model or settings.effective_llm_model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

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

        candidates_to_compact, payload = self._build_history_payload(candidates)
        if not candidates_to_compact:
            return already_compacted_ids, [], None

        summary_lines = self._summarize_history_with_llm(payload)
        if not summary_lines:
            summary_lines = self._fallback_summary_lines(candidates_to_compact)

        filename = self.file_store.append_compact_summary(
            session_id,
            title="Compacted Conversation History",
            summary_lines=summary_lines,
        )
        next_ids = set(already_compacted_ids)
        next_ids.update(message.id for message in candidates_to_compact)
        return next_ids, summary_lines, filename

    def _summarize_history_with_llm(self, payload: str) -> list[str]:
        if not self.api_key:
            return []
        if not payload:
            return []
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的对话记忆压缩器。"
                            "你的任务是把即将移出原始上下文的旧对话压缩成后续轮次可直接使用的记忆摘要。"
                            "只保留对继续对话有用的信息：用户目标、长期偏好、已做决定、涉及论文或标签、"
                            "未完成事项、关键证据和纠错经验。不要编造旧对话里没有的信息。"
                            "输出 4 到 10 条中文 Markdown bullet，每条不超过 80 个汉字，不要额外解释。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "以下旧对话将从原始消息窗口移入 compact 摘要。"
                            "请生成可放回系统上下文的短记忆：\n\n"
                            f"{payload}"
                        ),
                    },
                ],
            )
        except Exception:
            return []

        choices = getattr(response, "choices", None)
        if not choices:
            return []
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        text = self._content_to_text(content)
        return self._parse_summary_lines(text)

    @staticmethod
    def _build_history_payload(candidates: list[ChatMessage]) -> tuple[list[ChatMessage], str]:
        selected: list[ChatMessage] = []
        lines: list[str] = []
        total_chars = 0
        max_total_chars = 24000
        max_message_chars = 900
        for message in candidates:
            role_label = "用户" if message.role == "user" else "助手"
            content = " ".join(message.content.strip().split())
            if not content:
                continue
            if len(content) > max_message_chars:
                content = f"{content[:max_message_chars - 1]}…"
            line = f"{role_label}：{content}"
            if total_chars + len(line) > max_total_chars:
                remaining_count = len(candidates) - len(selected)
                if remaining_count > 0:
                    lines.append(f"系统提示：后续还有 {remaining_count} 条旧消息未放入摘要输入。")
                break
            lines.append(line)
            selected.append(message)
            total_chars += len(line)
        return selected, "\n".join(lines)

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
        return ""

    @classmethod
    def _parse_summary_lines(cls, text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•]\s*", "", line)
            line = re.sub(r"^\d+[.)、]\s*", "", line)
            line = " ".join(line.split())
            if not line:
                continue
            if len(line) > 160:
                line = f"{line[:159]}…"
            lines.append(line)
            if len(lines) >= 12:
                break
        if lines:
            return lines
        compact = " ".join(text.split())
        if not compact:
            return []
        return [compact[:160]]

    @staticmethod
    def _fallback_summary_lines(candidates: list[ChatMessage]) -> list[str]:
        summary_lines = []
        for message in candidates[-12:]:
            role_label = "用户" if message.role == "user" else "助手"
            snippet = " ".join(message.content.strip().split())
            if len(snippet) > 180:
                snippet = f"{snippet[:179]}…"
            summary_lines.append(f"{role_label}：{snippet}")
        return summary_lines
