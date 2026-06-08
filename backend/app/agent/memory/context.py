"""Agent Core context assembly for PaperDesk runtime requests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

from app.models import (
    AgentLifecycleStage,
    ContextBudgetAllocation,
    ContextBudgetProfile,
    ContextPacket,
    ContextTruncationMetadata,
    CustomInstructionPacket,
    MemoryLayerPacket,
    RuntimeRequest,
)


PROFILE_WINDOWS: dict[ContextBudgetProfile, int] = {
    ContextBudgetProfile.SMALL: 8192,
    ContextBudgetProfile.STANDARD: 32768,
    ContextBudgetProfile.LARGE: 131072,
}


@dataclass(frozen=True)
class ContextInstructionInput:
    """Raw custom instruction input before precedence is applied."""

    global_instruction: str | None = None
    session_instruction: str | None = None


class AgentContextLifecycleService:
    """Build complete route-aware context packets for lifecycle runtimes."""

    def __init__(
        self,
        *,
        default_token_budget: int | None = None,
        default_profile: ContextBudgetProfile | str = ContextBudgetProfile.STANDARD,
        model_context_window: int | None = None,
        estimator: Any | None = None,
        summary_compressor: Any | None = None,
    ) -> None:
        self.default_profile = self._coerce_profile(default_profile)
        self.model_context_window = model_context_window
        self.estimator = estimator
        self.summary_compressor = summary_compressor
        self.default_token_budget = default_token_budget

    def build_context(
        self,
        *,
        recent_messages: Iterable[dict[str, Any]] | None = None,
        session_summary: str | None = None,
        long_term_preferences: Iterable[str] | None = None,
        memory_snapshot: dict[str, Any] | None = None,
        recent_task_state: dict[str, Any] | None = None,
        global_custom_instruction: str | None = None,
        session_custom_instruction: str | None = None,
        selected_document_ids: Iterable[str] | None = None,
        selected_file_ids: Iterable[str] | None = None,
        evidence: Iterable[dict[str, Any]] | None = None,
        tool_observations: Iterable[dict[str, Any]] | None = None,
        pending_action: dict[str, Any] | None = None,
        workspace_scope: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
        capability_scope: dict[str, Any] | None = None,
        profile: ContextBudgetProfile | str | None = None,
        model_context_window: int | None = None,
        token_budget: int | None = None,
    ) -> ContextPacket:
        allocation = self.allocate_budget(
            profile=profile,
            model_context_window=model_context_window,
            explicit_token_budget=token_budget,
        )
        messages, dropped_messages, truncation = self._trim_recent_messages(
            list(recent_messages or []),
            message_budget=allocation.sliding_messages_budget,
            fallback_message_cap=allocation.fallback_message_cap,
        )
        summary = self._clean_text(session_summary) or self._summarize_dropped_messages(dropped_messages)
        long_term = self._clean_list(long_term_preferences or [])
        custom_instructions = CustomInstructionPacket(
            global_instruction=self._clean_text(global_custom_instruction),
            session_instruction=self._clean_text(session_custom_instruction),
        )
        memory = MemoryLayerPacket(
            session_summary=summary,
            long_term_preferences=long_term,
            recent_task_state=recent_task_state or {},
            memory_snapshot=memory_snapshot or {},
        )
        return ContextPacket(
            recent_messages=messages,
            session_summary=summary,
            long_term_preferences=long_term,
            custom_instructions=custom_instructions,
            memory=memory,
            selected_document_ids=self._dedupe(selected_document_ids or []),
            selected_file_ids=self._dedupe(selected_file_ids or []),
            evidence=list(evidence or []),
            tool_observations=list(tool_observations or []),
            pending_action=pending_action,
            workspace_scope=workspace_scope or {},
            preferences=preferences or {},
            capability_scope=capability_scope or {},
            token_budget=allocation.effective_context_window,
            budget_allocation=allocation,
            truncation=truncation.model_copy(
                update={
                    "summary_used": bool(summary),
                    "estimated_tokens": truncation.estimated_tokens
                    + self._estimate_text(summary or "")
                    + self._estimate_value(long_term)
                    + self._estimate_text(custom_instructions.global_instruction or "")
                    + self._estimate_text(custom_instructions.session_instruction or ""),
                }
            ),
        )

    def allocate_budget(
        self,
        *,
        profile: ContextBudgetProfile | str | None = None,
        model_context_window: int | None = None,
        explicit_token_budget: int | None = None,
    ) -> ContextBudgetAllocation:
        selected_profile = self._coerce_profile(profile or self.default_profile)
        configured = explicit_token_budget or self.default_token_budget or PROFILE_WINDOWS[selected_profile]
        configured = max(int(configured), 4096)
        model_window = model_context_window or self.model_context_window
        effective = min(configured, model_window) if model_window else configured
        effective = max(int(effective), 4096)
        if effective <= PROFILE_WINDOWS[ContextBudgetProfile.SMALL]:
            selected_profile = ContextBudgetProfile.SMALL
        elif effective <= PROFILE_WINDOWS[ContextBudgetProfile.STANDARD]:
            selected_profile = ContextBudgetProfile.STANDARD
        else:
            selected_profile = ContextBudgetProfile.LARGE
        return self._allocate_by_window(
            profile=selected_profile,
            configured_context_window=configured,
            model_context_window=model_window,
            effective_context_window=effective,
        )

    def attach_context(self, request: RuntimeRequest, context: ContextPacket) -> RuntimeRequest:
        request.context = context
        request.add_trace(
            AgentLifecycleStage.CONTEXT,
            "lifecycle context assembled",
            {
                "recent_message_count": len(context.recent_messages),
                "selected_document_count": len(context.selected_document_ids),
                "selected_file_count": len(context.selected_file_ids),
                "evidence_count": len(context.evidence),
                "tool_observation_count": len(context.tool_observations),
                "has_pending_action": context.pending_action is not None,
                "has_workspace_scope": bool(context.workspace_scope),
                "has_preferences": bool(context.preferences),
                "has_session_summary": bool(context.session_summary),
                "long_term_preference_count": len(context.long_term_preferences),
                "has_global_custom_instruction": bool(context.custom_instructions.global_instruction),
                "has_session_custom_instruction": bool(context.custom_instructions.session_instruction),
                "context_profile": context.budget_allocation.profile.value,
                "effective_context_window": context.budget_allocation.effective_context_window,
                "token_budget": context.token_budget,
                "dropped_message_count": context.truncation.dropped_message_count,
            },
        )
        return request

    def direct_chat_context(self, request: RuntimeRequest) -> ContextPacket:
        """Return the minimal context needed for a direct chat route."""

        return self.build_context(
            recent_messages=request.context.recent_messages,
            session_summary=request.context.session_summary,
            long_term_preferences=request.context.long_term_preferences,
            memory_snapshot=request.context.memory.memory_snapshot,
            recent_task_state=request.context.memory.recent_task_state,
            global_custom_instruction=request.context.custom_instructions.global_instruction,
            session_custom_instruction=request.context.custom_instructions.session_instruction,
            selected_file_ids=request.context.selected_file_ids,
            pending_action=request.context.pending_action,
            workspace_scope=request.context.workspace_scope,
            preferences=request.context.preferences,
            tool_observations=request.context.tool_observations,
            token_budget=request.context.token_budget,
        )

    @classmethod
    def _allocate_by_window(
        cls,
        *,
        profile: ContextBudgetProfile,
        configured_context_window: int,
        model_context_window: int | None,
        effective_context_window: int,
    ) -> ContextBudgetAllocation:
        output_reserve = cls._generation_reserve(profile, effective_context_window)
        safety_reserve = cls._bucket(effective_context_window, 0.03125, minimum=512, maximum=4096)
        usable = max(effective_context_window - output_reserve - safety_reserve, 2048)
        instruction = cls._bucket(usable, 0.04, minimum=512, maximum=4096)
        long_term = cls._bucket(usable, 0.02, minimum=256, maximum=2048)
        summary = cls._bucket(usable, 0.04, minimum=512, maximum=4096)
        sliding = cls._bucket(usable, 0.14, minimum=1024, maximum=16384)
        selected_files = cls._bucket(usable, 0.14, minimum=512, maximum=16384)
        evidence = cls._bucket(usable, 0.50, minimum=1024, maximum=65536)
        tool_observations = cls._bucket(usable, 0.06, minimum=512, maximum=8192)
        allocated = (
            instruction
            + long_term
            + summary
            + sliding
            + selected_files
            + evidence
            + tool_observations
            + output_reserve
            + safety_reserve
        )
        return ContextBudgetAllocation(
            profile=profile,
            configured_context_window=configured_context_window,
            model_context_window=model_context_window,
            effective_context_window=effective_context_window,
            generation_reserve=output_reserve,
            safety_reserve=safety_reserve,
            instruction_budget=instruction,
            long_term_memory_budget=long_term,
            session_summary_budget=summary,
            sliding_messages_budget=sliding,
            selected_files_budget=selected_files,
            rag_evidence_budget=evidence,
            tool_observations_budget=tool_observations,
            overflow_reserve=max(effective_context_window - allocated, 0),
            fallback_message_cap=24 if effective_context_window <= 32768 else 48,
            estimation_method="injected_estimator" if cls is not None else "heuristic",
        )

    @staticmethod
    def _generation_reserve(profile: ContextBudgetProfile, effective_context_window: int) -> int:
        if profile == ContextBudgetProfile.SMALL:
            return min(max(1024, int(effective_context_window * 0.125)), 1024)
        if profile == ContextBudgetProfile.STANDARD:
            return min(max(4096, int(effective_context_window * 0.125)), 4096)
        return min(max(8192, int(effective_context_window * 0.0625)), 8192)

    def _trim_recent_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        message_budget: int,
        fallback_message_cap: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ContextTruncationMetadata]:
        retained: list[dict[str, Any]] = []
        total = 0
        for message in reversed(messages):
            estimated = self._estimate_message(message)
            if retained and total + estimated > message_budget:
                break
            retained.append(message)
            total += estimated
        retained.reverse()
        dropped = max(len(messages) - len(retained), 0)
        fallback_used = False
        if len(retained) > fallback_message_cap:
            retained = retained[-fallback_message_cap:]
            dropped = max(len(messages) - len(retained), 0)
            fallback_used = True
            total = self._estimate_value(retained)
        dropped_messages = messages[:dropped] if dropped else []
        truncated_sections = []
        if dropped:
            truncated_sections.append("recent_messages")
        return retained, dropped_messages, ContextTruncationMetadata(
            estimated_tokens=total,
            retained_message_count=len(retained),
            dropped_message_count=dropped,
            fallback_message_cap_used=fallback_used,
            truncated_sections=truncated_sections,
        )

    def _summarize_dropped_messages(self, messages: list[dict[str, Any]]) -> str | None:
        if not messages:
            return None
        if self.summary_compressor is not None:
            if callable(self.summary_compressor):
                summary = self.summary_compressor(messages)
            elif hasattr(self.summary_compressor, "summarize"):
                summary = self.summary_compressor.summarize(messages)
            else:
                summary = None
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        snippets: list[str] = []
        for message in messages[-8:]:
            role = str(message.get("role") or "message")
            content = " ".join(str(message.get("content") or "").split())
            if not content:
                continue
            if len(content) > 120:
                content = f"{content[:119]}..."
            snippets.append(f"{role}: {content}")
        if not snippets:
            return None
        return "Earlier conversation summary:\n" + "\n".join(f"- {item}" for item in snippets)

    def _estimate_message(self, message: dict[str, Any]) -> int:
        return 4 + self._estimate_value(message.get("role")) + self._estimate_value(message.get("content"))

    def _estimate_value(self, value: Any) -> int:
        if self.estimator is not None and hasattr(self.estimator, "estimate_value"):
            return int(self.estimator.estimate_value(value))
        if value is None:
            return 0
        if isinstance(value, str):
            return self._estimate_text(value)
        if isinstance(value, list):
            return sum(self._estimate_value(item) for item in value)
        if isinstance(value, dict):
            return sum(self._estimate_value(item) for item in value.values())
        return self._estimate_text(str(value))

    def _estimate_text(self, text: str) -> int:
        if self.estimator is not None and hasattr(self.estimator, "estimate_text"):
            return int(self.estimator.estimate_text(text))
        if not text:
            return 0
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return max(math.ceil(ascii_chars / 4 + non_ascii_chars * 0.8), 1)

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip() if value is not None else ""
            if not item or item in seen:
                continue
            seen.add(item)
            results.append(item)
        return results

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @classmethod
    def _clean_list(cls, values: Iterable[str]) -> list[str]:
        return [item for item in (cls._clean_text(value) for value in values) if item]

    @staticmethod
    def _coerce_profile(value: ContextBudgetProfile | str) -> ContextBudgetProfile:
        if isinstance(value, ContextBudgetProfile):
            return value
        normalized = str(value).strip().casefold()
        for profile in ContextBudgetProfile:
            if profile.value == normalized:
                return profile
        return ContextBudgetProfile.STANDARD

    @staticmethod
    def _bucket(window: int, ratio: float, *, minimum: int, maximum: int) -> int:
        return max(minimum, min(int(window * ratio), maximum))
