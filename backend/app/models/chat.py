"""Chat-session and memory models for the knowledge workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MemoryRecordType = Literal["user", "feedback", "project", "reference"]
ChatAttachmentKind = Literal["image", "uploaded_pdf", "library_document", "session_file"]
ChatMessageRole = Literal["user", "assistant", "system"]
KnowledgeRetrievalStatus = Literal["ready", "skipped", "degraded", "unavailable"]
ContextStage = Literal["normal", "evidence_compacted", "history_compacted", "truncated"]
SlashCommandId = Literal["summary", "compare", "tag", "library", "help"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatAttachment(BaseModel):
    id: str
    kind: ChatAttachmentKind
    display_name: str
    mime_type: str | None = None
    document_id: str | None = None
    file_asset_id: str | None = None
    data_url: str | None = None
    file_path: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryHit(BaseModel):
    id: str
    memory_type: MemoryRecordType
    summary: str
    detail: str | None = None
    source_kind: str | None = None
    source_id: str | None = None
    status: str = "active"
    last_verified_at: datetime | None = None


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: ChatMessageRole
    content: str
    status: str = "completed"
    retrieval_status: KnowledgeRetrievalStatus | None = None
    warning: str | None = None
    citations: list[str] = Field(default_factory=list)
    used_document_ids: list[str] = Field(default_factory=list)
    used_file_ids: list[str] = Field(default_factory=list)
    memory_hits: list[MemoryHit] = Field(default_factory=list)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    saved_report_id: str | None = None
    agent_trace_id: str | None = None
    action_status: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_message_preview: str | None = None


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    selected_document_ids: list[str] = Field(default_factory=list)
    selected_file_ids: list[str] = Field(default_factory=list)
    agent_profile_id: str | None = None
    model_id: str | None = None
    command: str | None = None
    intent_hint: str | None = None
    deep_research: bool = False

    @model_validator(mode="after")
    def normalize_lists(self) -> "ChatMessageRequest":
        self.selected_document_ids = self._dedupe_ids(self.selected_document_ids)
        self.selected_file_ids = self._dedupe_ids(self.selected_file_ids)
        if self.agent_profile_id is not None:
            self.agent_profile_id = self.agent_profile_id.strip() or None
        if self.model_id is not None:
            self.model_id = self.model_id.strip() or None
        if self.command is not None:
            command = self.command.strip().lstrip("/").casefold()
            allowed_commands: set[SlashCommandId] = {"summary", "compare", "tag", "library", "help"}
            self.command = command if command in allowed_commands else None
        if self.intent_hint is not None:
            self.intent_hint = self.intent_hint.strip() or None
        return self

    @staticmethod
    def _dedupe_ids(values: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip() if value is not None else ""
            if not item or item in seen:
                continue
            seen.add(item)
            results.append(item)
        return results


class MemoryRecord(BaseModel):
    id: str
    memory_type: MemoryRecordType
    scope: str = "global"
    summary: str
    detail: str | None = None
    source_kind: str | None = None
    source_id: str | None = None
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_verified_at: datetime | None = None


class MemorySnapshot(BaseModel):
    items: list[MemoryHit] = Field(default_factory=list)
    refreshed_at: datetime = Field(default_factory=utc_now)


class ChatContextState(BaseModel):
    stage: ContextStage = "normal"
    estimated_tokens: int = 0
    budget_tokens: int = 0
    context_profile: str | None = None
    effective_context_window: int = 0
    retained_message_count: int = 0
    dropped_message_count: int = 0
    truncated_sections: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    last_compacted_at: datetime | None = None


class ChatSessionDetail(BaseModel):
    session: ChatSession
    messages: list[ChatMessage] = Field(default_factory=list)
    memory_snapshot: MemorySnapshot = Field(default_factory=MemorySnapshot)
    context_state: ChatContextState = Field(default_factory=ChatContextState)


class ChatSendResponse(BaseModel):
    session: ChatSession
    user_message: ChatMessage
    assistant_message: ChatMessage
    memory_snapshot: MemorySnapshot = Field(default_factory=MemorySnapshot)
    context_state: ChatContextState = Field(default_factory=ChatContextState)
    library_mutated: bool = False
