"""Prompt assembly for chat context with staged compaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models import ChatAttachment, ChatContextState, ChatMessage, ChatMessageRequest, ChatSession, EvidenceItem

from .context_budget_service import ContextBudgetService
from .context_compaction_service import ContextCompactionService
from .context_file_store import ContextFileStore


class ContextAssembler:
    """Assemble chat prompt messages using Claude Code-like layering."""

    def __init__(
        self,
        *,
        budget_service: ContextBudgetService,
        compaction_service: ContextCompactionService,
        file_store: ContextFileStore,
    ) -> None:
        self.budget_service = budget_service
        self.compaction_service = compaction_service
        self.file_store = file_store

    def assemble(
        self,
        *,
        session: ChatSession,
        history: list[ChatMessage],
        current_request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        evidence_items: list[EvidenceItem],
        knowledge_context_lines: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], ChatContextState]:
        self.file_store.initialize_session(session.id, session.title)
        state_payload = self.file_store.read_context_state(session.id)
        compacted_ids = set(state_payload.get("compacted_message_ids", []))

        project_rules = self.file_store.read_project_rules()
        user_preferences = self.file_store.read_user_preferences()
        compact_summaries = self.file_store.list_compact_summaries(session.id, limit=3)
        raw_history = [message for message in history if message.role in {"user", "assistant"}]
        recent_message_limit = 16
        visible_history = [message for message in raw_history if message.id not in compacted_ids][-recent_message_limit:]

        stage = "normal"
        evidence_lines = self._render_full_evidence(evidence_items)
        messages, sources = self._build_messages(
            session=session,
            project_rules=project_rules,
            user_preferences=user_preferences,
            compact_summaries=compact_summaries,
            visible_history=visible_history,
            current_request=current_request,
            attachments=attachments,
            evidence_lines=evidence_lines,
            knowledge_context_lines=knowledge_context_lines or [],
        )
        estimated = self.budget_service.estimate_messages(messages)

        if estimated > self.budget_service.warn_tokens and evidence_items:
            stage = "evidence_compacted"
            evidence_lines = self.compaction_service.compact_evidence(evidence_items)
            messages, sources = self._build_messages(
                session=session,
                project_rules=project_rules,
                user_preferences=user_preferences,
                compact_summaries=compact_summaries,
                visible_history=visible_history,
                current_request=current_request,
                attachments=attachments,
                evidence_lines=evidence_lines,
                knowledge_context_lines=knowledge_context_lines or [],
            )
            estimated = self.budget_service.estimate_messages(messages)

        compact_summary_lines: list[str] = []
        if estimated > self.budget_service.force_tokens:
            compacted_ids, compact_summary_lines, _ = self.compaction_service.compact_history(
                session.id,
                history=raw_history,
                already_compacted_ids=compacted_ids,
            )
            visible_history = [
                message for message in raw_history if message.id not in compacted_ids
            ][-recent_message_limit:]
            compact_summaries = self.file_store.list_compact_summaries(session.id, limit=3)
            stage = "history_compacted"
            messages, sources = self._build_messages(
                session=session,
                project_rules=project_rules,
                user_preferences=user_preferences,
                compact_summaries=compact_summaries,
                visible_history=visible_history,
                current_request=current_request,
                attachments=attachments,
                evidence_lines=evidence_lines,
                knowledge_context_lines=knowledge_context_lines or [],
            )
            estimated = self.budget_service.estimate_messages(messages)

        if estimated > self.budget_service.budget_tokens:
            min_messages = max(self.budget_service.settings.recent_turns_min * 2, 2)
            while len(visible_history) > min_messages and estimated > self.budget_service.budget_tokens:
                visible_history = visible_history[1:]
                messages, sources = self._build_messages(
                    session=session,
                    project_rules=project_rules,
                    user_preferences=user_preferences,
                    compact_summaries=compact_summaries,
                    visible_history=visible_history,
                    current_request=current_request,
                    attachments=attachments,
                    evidence_lines=evidence_lines,
                    knowledge_context_lines=knowledge_context_lines or [],
                )
                estimated = self.budget_service.estimate_messages(messages)
            stage = "truncated"

        references = self._collect_reference_lines(history, attachments, evidence_items)
        pending_topics = [self._truncate_line(current_request.content, limit=120)]
        if compact_summary_lines:
            pending_topics.append("历史长对话已压缩为 compact 摘要，并保留最近原始轮次。")

        self.file_store.sync_session_summary(
            session.id,
            title=session.title,
            user_preferences=user_preferences,
            references=references,
            pending_topics=pending_topics,
        )

        last_compacted_at = state_payload.get("last_compacted_at")
        if stage in {"history_compacted", "truncated"}:
            last_compacted_at = self.file_store.mark_compacted_now(session.id)

        next_state = {
            "stage": stage,
            "estimated_tokens": estimated,
            "budget_tokens": self.budget_service.budget_tokens,
            "sources": sources,
            "last_compacted_at": last_compacted_at,
            "compacted_message_ids": sorted(compacted_ids),
        }
        self.file_store.write_context_state(session.id, next_state)

        return messages, ChatContextState(
            stage=stage,
            estimated_tokens=estimated,
            budget_tokens=self.budget_service.budget_tokens,
            sources=sources,
            last_compacted_at=(
                datetime.fromisoformat(last_compacted_at) if last_compacted_at else None
            ),
        )

    def read_context_state(self, session_id: str) -> ChatContextState:
        payload = self.file_store.read_context_state(session_id)
        budget_tokens = payload.get("budget_tokens") or self.budget_service.budget_tokens
        return ChatContextState(
            stage=payload.get("stage", "normal"),
            estimated_tokens=int(payload.get("estimated_tokens", 0)),
            budget_tokens=int(budget_tokens),
            sources=list(payload.get("sources", [])),
            last_compacted_at=(
                datetime.fromisoformat(payload["last_compacted_at"])
                if payload.get("last_compacted_at")
                else None
            ),
        )

    def _build_messages(
        self,
        *,
        session: ChatSession,
        project_rules: str,
        user_preferences: list[str],
        compact_summaries: list[str],
        visible_history: list[ChatMessage],
        current_request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        evidence_lines: list[str],
        knowledge_context_lines: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        sources = ["system_instruction", "project_rules"]
        system_parts = [
            "你是 PaperDesk 的论文阅读与通用问答助手。",
            "默认使用中文回答。",
            "普通历史、常识、编程、写作等问题要像通用 AI 一样自然回答，不要主动声称知识库范围不足。",
            "只有当用户明确要求根据论文库、上传论文、所选论文或文献证据回答时，才把答案限定在论文库证据内；若这类证据不可用，再说明证据不足。",
            "当本轮存在可用论文证据时，把证据整合进答案；论文库相关结论不能脱离证据胡编。",
            "如果用户在同一条消息里提出多个问题，无论数量多少，都必须按问题顺序逐项回答；不要只输出检索状态、证据数量或内部过程。",
            "记忆是索引，不是真相。如果记忆摘要与当前材料冲突，以当前材料为准。",
            project_rules,
        ]

        if user_preferences:
            sources.append("user_preferences")
            system_parts.append("用户长期偏好：\n" + "\n".join(f"- {item}" for item in user_preferences))

        session_summary = self.file_store.read_session_summary(session.id)
        if session_summary:
            sources.append("session_summary")
            system_parts.append("当前会话摘要：\n" + session_summary)

        if compact_summaries:
            sources.append("compact_summary")
            system_parts.append("历史压缩摘要：\n" + "\n\n".join(compact_summaries))

        if evidence_lines:
            sources.append("rag_evidence")
            system_parts.append("本轮可引用证据：\n" + "\n\n".join(evidence_lines))

        if knowledge_context_lines:
            sources.append("knowledge_agent_context")
            system_parts.append(
                "PaperDesk 知识库运行态摘要：\n"
                + "\n".join(f"- {item}" for item in knowledge_context_lines)
            )

        messages: list[dict[str, Any]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        if visible_history:
            sources.append("recent_messages")
        for message in visible_history:
            messages.append({"role": message.role, "content": message.content})

        user_parts: list[dict[str, Any]] = [{"type": "text", "text": current_request.content.strip()}]
        attachment_hints = self.compaction_service.build_attachment_hints(attachments)
        if attachments:
            sources.append("attachments")
        for attachment in attachments:
            if attachment.kind == "image" and attachment.data_url:
                user_parts.append({"type": "image_url", "image_url": {"url": attachment.data_url}})
        if attachment_hints:
            user_parts.append(
                {
                    "type": "text",
                    "text": "本轮附加材料：\n" + "\n".join(f"- {item}" for item in attachment_hints),
                }
            )
        messages.append({"role": "user", "content": user_parts})
        return messages, sources

    @staticmethod
    def _render_full_evidence(evidence_items: list[EvidenceItem]) -> list[str]:
        return [
            "\n".join(
                [
                    f"来源：{item.citation_label}",
                    f"标题：{item.title}",
                    f"页码：{item.page_number if item.page_number is not None else '未知'}",
                    f"证据：{item.quote or item.snippet}",
                ]
            )
            for item in evidence_items
        ]

    @staticmethod
    def _collect_reference_lines(
        history: list[ChatMessage],
        attachments: list[ChatAttachment],
        evidence_items: list[EvidenceItem],
    ) -> list[str]:
        references: list[str] = []
        for message in history:
            for citation in message.citations:
                references.append(f"历史引用：{citation}")
            for attachment in message.attachments:
                if attachment.kind in {"uploaded_pdf", "library_document"}:
                    references.append(f"历史附件：{attachment.display_name}")
        for attachment in attachments:
            if attachment.kind in {"uploaded_pdf", "library_document"}:
                references.append(f"当前附件：{attachment.display_name}")
        for item in evidence_items:
            references.append(f"本轮证据：{item.citation_label}")
        deduped: list[str] = []
        seen: set[str] = set()
        for entry in references:
            if entry in seen:
                continue
            seen.add(entry)
            deduped.append(entry)
        return deduped[:12]

    @staticmethod
    def _truncate_line(text: str, *, limit: int) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 1]}…"
