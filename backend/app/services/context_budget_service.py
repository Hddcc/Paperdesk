"""Context budget estimation and threshold decisions."""

from __future__ import annotations

import math
from typing import Any

from app.agent.memory.context import AgentContextLifecycleService
from app.config import Settings
from app.models import ContextBudgetAllocation, ContextBudgetProfile

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


class ContextBudgetService:
    """Estimate prompt size and expose PaperDesk-specific compact thresholds."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.context_policy = AgentContextLifecycleService(
            default_profile=self._profile_from_settings(),
            model_context_window=settings.effective_max_context_tokens,
        )

    @property
    def max_context_tokens(self) -> int:
        return self.allocation.effective_context_window

    @property
    def budget_tokens(self) -> int:
        return max(
            self.allocation.effective_context_window
            - self.allocation.generation_reserve
            - self.allocation.safety_reserve,
            2000,
        )

    @property
    def warn_tokens(self) -> int:
        return int(self.budget_tokens * self.settings.compact_warn_ratio)

    @property
    def force_tokens(self) -> int:
        return int(self.budget_tokens * self.settings.compact_force_ratio)

    @property
    def allocation(self) -> ContextBudgetAllocation:
        return self.context_policy.allocate_budget(
            profile=self._profile_from_settings(),
            model_context_window=self.settings.effective_max_context_tokens,
            explicit_token_budget=self.settings.max_context_tokens,
        )

    @property
    def fallback_message_cap(self) -> int:
        return self.allocation.fallback_message_cap

    @property
    def sliding_messages_budget(self) -> int:
        return self.allocation.sliding_messages_budget

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

    def _profile_from_settings(self) -> ContextBudgetProfile:
        effective = self.settings.effective_max_context_tokens
        if effective <= 8192:
            return ContextBudgetProfile.SMALL
        if effective <= 32768:
            return ContextBudgetProfile.STANDARD
        return ContextBudgetProfile.LARGE
