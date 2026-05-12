"""Context budget estimation and threshold decisions."""

from __future__ import annotations

import math
from typing import Any

from app.config import Settings

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


class ContextBudgetService:
    """Estimate prompt size and expose PaperDesk-specific compact thresholds."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def max_context_tokens(self) -> int:
        return self.settings.effective_max_context_tokens

    @property
    def budget_tokens(self) -> int:
        return max(self.max_context_tokens - self.settings.response_reserve_tokens, 2000)

    @property
    def warn_tokens(self) -> int:
        return int(self.budget_tokens * self.settings.compact_warn_ratio)

    @property
    def force_tokens(self) -> int:
        return int(self.budget_tokens * self.settings.compact_force_ratio)

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += 4
            total += self.estimate_value(message.get("role"))
            total += self.estimate_value(message.get("content"))
        return total

    def estimate_value(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self.estimate_text(value)
        if isinstance(value, list):
            return sum(self.estimate_value(item) for item in value)
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                return 85
            return sum(self.estimate_value(item) for item in value.values())
        return self.estimate_text(str(value))

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        if tiktoken is not None:
            try:
                encoding = tiktoken.encoding_for_model(self.settings.llm_model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        estimated = math.ceil(ascii_chars / 4 + non_ascii_chars * 0.8)
        return max(estimated, 1)
