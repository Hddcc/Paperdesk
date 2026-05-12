"""Chat-session and memory models for the knowledge workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MemoryRecordType = Literal["user", "feedback", "project", "reference"]
ChatAttachmentKind = Literal["image", "uploaded_pdf", "library_document"]
ChatMessageRole = Literal["user", "assistant", "system"]
KnowledgeRetrievalStatus = Literal["ready", "skipped", "degraded", "unavailable"]
ContextStage = Literal["normal", "evidence_compacted", "history_compacted", "truncated"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatAttachment(BaseModel):
    id: str
    kind: ChatAttachmentKind
    display_name: str
    mime_type: str | None = None
    document_id: str | None = None
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
    memory_hits: list[MemoryHit] = Field(default_factory=list)
    attachments: list[ChatAttachment] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def normalize_lists(self) -> "ChatMessageRequest":
        self.selected_document_ids = [item for item in self.selected_document_ids if item]
        return self


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
