"""Claude Code-style runtime for the chat knowledge agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import json
import re
import time
from typing import Any
from uuid import uuid4

from openai import OpenAI

from app.models import (
    AgentTask,
    AgentTaskStatus,
    ChatAttachment,
    ChatMessageRequest,
    ChatSession,
    EvidenceItem,
    LibraryDocument,
    ResearchRunStatus,
    SubagentProfile,
    TaskNotification,
    ToolPolicy,
    ToolObservation,
    ToolObservationError,
    ToolSpec,
    ToolVerification,
    TraceEventType,
)
from app.repositories import CategoryRepository, ResearchRepository, RuntimeRepository
from app.services.skill_context_builder import SkillContextBuilder
from app.services.context_file_store import ContextFileStore
from app.services.document_library_service import DocumentLibraryService
from app.services.rag_service import RagService
from app.vectorstores import AbstractVectorStore

from .message_bus import MessageBus
from .pending_action_store import PendingActionStore


@dataclass(slots=True)
class KnowledgeAgentResult:
    """Result returned to the chat service after a knowledge-agent run."""

    content: str
    retrieval_status: str = "skipped"
    warning: str | None = None
    citations: list[str] = field(default_factory=list)
    used_document_ids: list[str] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    action_status: str | None = None
    agent_trace_id: str | None = None
    library_mutated: bool = False


@dataclass(slots=True)
class _TaskOutcome:
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _DraftResult:
    answer: str
    llm_draft_success: bool
    fallback_used: bool = False
    drafting_error: str | None = None


@dataclass(slots=True)
class _ReactAction:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    task_intent: dict[str, Any] = field(default_factory=dict)
    action_plan: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class _ReactObservation:
    tool: str
    status: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _AnswerObligation:
    key: str
    description: str
    required_tools: tuple[str, ...]
    target: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _StepState:
    step_id: str
    intent: str
    operation_level: str = "none"
    source_entities: list[dict[str, Any]] = field(default_factory=list)
    target_entities: list[dict[str, Any]] = field(default_factory=list)
    target_scope: str = ""
    depends_on: list[str] = field(default_factory=list)
    inherit_rules: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "read_only"
    resolved_tool_name: str = ""
    resolved_tool_args: dict[str, Any] = field(default_factory=dict)
    preview_result: dict[str, Any] = field(default_factory=dict)
    before_snapshot: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    verification_result: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    observation_index: int | None = None
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PlanState:
    plan_id: str
    original_user_prompt: str
    steps: list[_StepState] = field(default_factory=list)
    current_step_index: int = 0
    completed_steps: list[str] = field(default_factory=list)
    pending_confirmation_step: str | None = None
    failed_steps: list[str] = field(default_factory=list)
    global_context: dict[str, Any] = field(default_factory=dict)
    final_status: str = "pending"


@dataclass(frozen=True, slots=True)
class _ToolRiskPolicy:
    risk_level: str
    required_args: tuple[str, ...] = ()
    scope_type: str = "none"
    operation_level: str = "query-level"
    write_type: str = "read"
    target_type: str = "none"
    destructive: bool = False
    requires_confirmation: bool = False
    requires_verification: bool = False
    destructive_kind: str = "none"


@dataclass(slots=True)
class _WritePreview:
    operation: str
    risk_level: str
    tool_name: str
    tool_args: dict[str, Any]
    target_entity: dict[str, Any]
    affected_count: int
    affected_entities: list[dict[str, Any]]
    expected_scope: str
    before_snapshot: dict[str, Any]
    confirmation_phrase: str


@dataclass(slots=True)
class _ResolvedAction:
    intent_type: str
    operation: str
    target_type: str
    scope_type: str
    document_ids: list[str] = field(default_factory=list)
    label_name: str | None = None
    category_name: str | None = None
    risk_level: str = "read"
    requires_tool: bool = True
    requires_confirmation: bool = False
    tool_name: str = ""
    confidence: float = 1.0
    reason: str = ""
    clarification_needed: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "operation": self.operation,
            "target_type": self.target_type,
            "scope_type": self.scope_type,
            "document_ids": self.document_ids,
            "label_name": self.label_name,
            "category_name": self.category_name,
            "risk_level": self.risk_level,
            "requires_tool": self.requires_tool,
            "requires_confirmation": self.requires_confirmation,
            "tool_name": self.tool_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "clarification_needed": self.clarification_needed,
        }


class KnowledgeAgentRuntime:
    """Main runtime for Knowledge Chat tool and evidence workflows.

    This class currently owns the ReAct loop, tool execution, write guardrails,
    pending actions, observation wrapping, evidence merging, and answer
    synthesis. Keep future slimming incremental and behavior-preserving.
    """

    _CONFIRM_MARKERS = ("确认", "是的", "继续", "执行", "同意", "可以", "confirm", "yes")
    _DELETE_MARKERS = ("删除", "移除", "清空", "删掉", "delete", "remove", "clear")
    _SUMMARY_MARKERS = ("总结", "综述", "概述", "对比", "比较", "review", "summary", "summarize", "compare")
    _INTERNAL_DEGRADED_TOOL = "agent.intent.degraded"
    _INTERNAL_CATEGORY_CONFLICT_TOOL = "agent.category_semantics_conflict"
    _TOOL_RISK_REGISTRY: dict[str, _ToolRiskPolicy] = {
        "library.operator.create_category": _ToolRiskPolicy(
            risk_level="safe_write",
            required_args=("category_name",),
            scope_type="single_entity",
            operation_level="entity-level",
            write_type="create",
            target_type="category",
            requires_verification=True,
            destructive_kind="none",
        ),
        "library.operator.assign_category": _ToolRiskPolicy(
            risk_level="scoped_write",
            required_args=("category_name",),
            scope_type="documents",
            operation_level="relation-level",
            write_type="append",
            target_type="paper-category relation",
            requires_verification=True,
            destructive_kind="append_relation",
        ),
        "library.operator.rename_category": _ToolRiskPolicy(
            risk_level="scoped_write",
            required_args=("source_category_name", "target_category_name"),
            scope_type="single_entity",
            operation_level="entity-level",
            write_type="update",
            target_type="category",
            requires_verification=True,
            destructive_kind="rename_or_merge_entity",
        ),
        "library.operator.delete_unused_categories": _ToolRiskPolicy(
            risk_level="destructive",
            required_args=("selector",),
            scope_type="category_entities_with_zero_documents",
            operation_level="entity-level",
            write_type="delete",
            target_type="category",
            destructive=True,
            requires_confirmation=True,
            requires_verification=True,
            destructive_kind="delete_unused_category_entities",
        ),
        "library.operator.clear_categories": _ToolRiskPolicy(
            risk_level="destructive",
            required_args=("operation",),
            scope_type="explicit",
            operation_level="relation-level",
            write_type="clear",
            target_type="paper-category relation",
            destructive=True,
            requires_confirmation=True,
            requires_verification=True,
            destructive_kind="remove_or_clear_relations",
        ),
        "memory.write": _ToolRiskPolicy(
            risk_level="safe_write",
            required_args=("summary",),
            scope_type="session_memory",
            operation_level="content-level",
            write_type="append",
            target_type="memory",
            requires_verification=False,
            destructive_kind="none",
        ),
    }
    _READ_TOOL_SPECS: dict[str, ToolSpec] = {
        "tool.registry.list": ToolSpec(
            name="tool.registry.list",
            display_name="List PaperDesk chat tools",
            description="Read the available chat runtime tools and safety metadata.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=[],
            output_observation_type="tool_registry_observation",
            available_by_default=True,
        ),
        "library.explorer.stats": ToolSpec(
            name="library.explorer.stats",
            display_name="Read library statistics",
            description="Read paper counts and processing status from the local PaperDesk library.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["paper"],
            output_observation_type="paper_list_observation",
            available_by_default=True,
        ),
        "library.explorer.category_stats": ToolSpec(
            name="library.explorer.category_stats",
            display_name="Read tag/category statistics",
            description="Read tag/category list and per-category paper counts.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["category"],
            output_observation_type="category_list_observation",
            available_by_default=True,
        ),
        "library.explorer.find_documents": ToolSpec(
            name="library.explorer.find_documents",
            display_name="Resolve documents or tag collections",
            description="Resolve papers by selected IDs, title/filename, or exact tag/category names.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["paper", "category"],
            output_observation_type="paper_list_observation",
            available_by_default=True,
        ),
        "library.explorer.document_metadata": ToolSpec(
            name="library.explorer.document_metadata",
            display_name="Read document metadata",
            description="Read metadata for resolved documents.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["paper"],
            output_observation_type="paper_list_observation",
            available_by_default=True,
        ),
        "library.explorer.document_categories": ToolSpec(
            name="library.explorer.document_categories",
            display_name="Read document tags/categories",
            description="Read current tag/category links for resolved documents.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["paper-category relation"],
            output_observation_type="category_relation_observation",
            available_by_default=True,
        ),
        "evidence.retriever.search": ToolSpec(
            name="evidence.retriever.search",
            display_name="Retrieve document evidence",
            description="Retrieve local RAG evidence from ready papers.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["paper", "chunk"],
            output_observation_type="rag_search_observation",
            available_by_default=True,
        ),
        "evidence.retriever.search_by_category": ToolSpec(
            name="evidence.retriever.search_by_category",
            display_name="Retrieve evidence by tag/category",
            description="Retrieve grouped RAG evidence for papers under tag/category entities.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["category", "chunk"],
            output_observation_type="rag_search_observation",
            available_by_default=True,
        ),
        "report.drafter.write": ToolSpec(
            name="report.drafter.write",
            display_name="Synthesize grounded answer",
            description="Generate the final user-facing answer from resolved documents and retrieved evidence.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["report", "paper"],
            output_observation_type="report_observation",
            available_by_default=True,
        ),
        "report.drafter.write_by_category": ToolSpec(
            name="report.drafter.write_by_category",
            display_name="Synthesize grouped tag/category answer",
            description="Generate final grouped summaries from tag/category evidence groups.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["report", "category"],
            output_observation_type="report_observation",
            available_by_default=True,
        ),
        "memory.read": ToolSpec(
            name="memory.read",
            display_name="Read chat memory",
            description="Read user preferences and prior reflection notes.",
            scope="knowledge",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            input_object_types=["memory"],
            output_observation_type="memory_observation",
            available_by_default=True,
        ),
    }

    def __init__(
        self,
        *,
        document_library_service: DocumentLibraryService,
        category_repository: CategoryRepository,
        research_repository: ResearchRepository,
        runtime_repository: RuntimeRepository,
        rag_service: RagService,
        vectorstore: AbstractVectorStore,
        file_store: ContextFileStore,
        model: str,
        api_key: str | None,
        base_url: str | None,
        enable_subagent_execution: bool = False,
        enable_skill_context_prompt_injection: bool = True,
        enable_skill_context_paper_qa_lightweight_only: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.document_library_service = document_library_service
        self.category_repository = category_repository
        self.research_repository = research_repository
        self.runtime_repository = runtime_repository
        self.rag_service = rag_service
        self.vectorstore = vectorstore
        self.file_store = file_store
        self.pending_action_store = PendingActionStore(file_store)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.enable_subagent_execution = enable_subagent_execution
        self.enable_skill_context_prompt_injection = enable_skill_context_prompt_injection
        self.enable_skill_context_paper_qa_lightweight_only = enable_skill_context_paper_qa_lightweight_only
        self.timeout = timeout
        self.message_bus = MessageBus(runtime_repository)

    def build_context_lines(self) -> list[str]:
        """Build a compact library capability summary for the chat prompt."""

        documents = self.document_library_service.list_documents()
        ready_documents = [document for document in documents if document.status == "ready"]
        category_stats = self._category_stats_payload(limit_documents=0)
        preview = documents[:8]
        lines = [
            f"论文库：共 {len(documents)} 篇，已可用 {len(ready_documents)} 篇；"
            f"有标签 {category_stats['tagged_document_count']} 篇，无标签 {category_stats['untagged_document_count']} 篇。",
            "知识库聊天采用 ReAct 主 Agent：先选择工具、读取观察结果，再继续行动或回答。",
            "可用工具：library.explorer.*、library.operator.*、evidence.retriever.search、report.drafter.write、memory.read/write。",
            "权限策略：只读自动允许；新建分类、追加标签等非破坏写操作经校验后执行；删除论文/分类必须先确认。",
            "重要经验：标签统计必须读取真实分类关联；复合命令要拆成多步，不能把后半句当标签名。",
        ]
        if category_stats["categories"]:
            lines.append("现有分类：" + "、".join(category["name"] for category in category_stats["categories"][:12]))
        if preview:
            lines.append(
                "论文预览："
                + "；".join(
                    f"{document.display_name or document.filename}({document.status})"
                    for document in preview
                )
            )
        return lines

    def conversation_referents(self, session_id: str) -> dict[str, Any]:
        """Expose current conversation referents for the orchestrator."""

        return self._read_react_state(session_id)

    def has_pending_action(self, session_id: str) -> bool:
        """Return whether a protected action is awaiting user confirmation."""

        return self._read_pending_action(session_id) is not None

    def handle(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        content = request.content.strip()
        if not content:
            return None

        pending_result = self._maybe_execute_pending_action(session, content, trace_id=trace_id)
        if pending_result is not None:
            return pending_result

        clarification = self._ambiguous_write_clarification(content, trace_id=trace_id)
        if clarification is not None:
            return clarification

        all_library_assign_result = self._try_all_library_assign_preview(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            trace_id=trace_id,
        )
        if all_library_assign_result is not None:
            return all_library_assign_result

        read_then_write_result = self._try_read_then_write_preview(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            trace_id=trace_id,
        )
        if read_then_write_result is not None:
            return read_then_write_result

        relation_scope_clarification = self._ambiguous_relation_write_scope_clarification(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            observations=[],
            trace_id=trace_id,
        )
        if relation_scope_clarification is not None:
            return relation_scope_clarification

        if self._is_destructive_intent(content) and not self._should_plan_before_destructive_confirmation(
            content,
            selected_document_ids,
            attachments,
        ):
            return self._request_confirmation(session, content, trace_id=trace_id)

        if not self._should_handle_with_react(content, selected_document_ids, attachments):
            return None

        lightweight_result = self._try_paper_qa_lightweight(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
        )
        if lightweight_result is not None:
            return lightweight_result

        return self._run_react_agent(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
        )

    def run_react(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
        runtime_label: str = "knowledge_react_execution",
    ) -> KnowledgeAgentResult:
        """Run the ReAct executor after an upstream orchestrator selected it."""

        content = request.content.strip()
        pending_result = self._maybe_execute_pending_action(session, content, trace_id=trace_id)
        if pending_result is not None:
            return pending_result
        clarification = self._ambiguous_write_clarification(content, trace_id=trace_id)
        if clarification is not None:
            return clarification
        all_library_assign_result = self._try_all_library_assign_preview(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            trace_id=trace_id,
        )
        if all_library_assign_result is not None:
            return all_library_assign_result
        read_then_write_result = self._try_read_then_write_preview(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            trace_id=trace_id,
        )
        if read_then_write_result is not None:
            return read_then_write_result
        relation_scope_clarification = self._ambiguous_relation_write_scope_clarification(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            observations=[],
            trace_id=trace_id,
        )
        if relation_scope_clarification is not None:
            return relation_scope_clarification
        if self._is_destructive_intent(content) and not self._should_plan_before_destructive_confirmation(
            content,
            selected_document_ids,
            attachments,
        ):
            return self._request_confirmation(session, content, trace_id=trace_id)

        lightweight_result = self._try_paper_qa_lightweight(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
            runtime_label=runtime_label,
        )
        if lightweight_result is not None:
            return lightweight_result

        return self._run_react_agent(
            session=session,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            trace_id=trace_id,
            runtime_label=runtime_label,
        )

    def _try_paper_qa_lightweight(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
        runtime_label: str = "knowledge_react_execution",
    ) -> KnowledgeAgentResult | None:
        content = request.content.strip()
        scope = self._paper_qa_lightweight_scope(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
        )
        if scope is None:
            return None
        scope_type, document = scope
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        if not owns_run:
            self._append_react_trace(
                run_id=run_id,
                status=f"{runtime_label}_started",
                payload={"session_id": session.id, "topic": content, "runtime_variant": "paper_qa_lightweight"},
            )
        self._append_react_trace(
            run_id=run_id,
            status="paper_qa_lightweight_started",
            payload={
                "session_id": session.id,
                "scope_type": scope_type,
                "document_id": document.id,
                "used_document_ids": [document.id],
                "question": content[:240],
            },
        )
        if document.status != "ready" or (
            document.parser_status and document.parser_status not in {"indexed", "completed", "ready"}
        ):
            answer = (
                f"这篇论文当前状态是 {document.status}"
                f"{f'/{document.parser_status}' if document.parser_status else ''}，还没有可用于正文问答的 ready/indexed 证据。"
            )
            self._append_react_trace(
                run_id=run_id,
                status="paper_qa_lightweight_answer",
                payload={
                    "used_document_ids": [document.id],
                    "evidence_count": 0,
                    "llm_called": False,
                    "answer_length": len(answer),
                    "fallback_used": True,
                    "action_status": "needs_clarification",
                },
            )
            if owns_run:
                self._finish_run(run_id)
            else:
                self._append_react_trace(
                    run_id=run_id,
                    status=f"{runtime_label}_finished",
                    payload={"final_status": "needs_clarification", "evidence_count": 0, "final_content_length": len(answer)},
                )
            return KnowledgeAgentResult(
                content=answer,
                retrieval_status="skipped",
                used_document_ids=[document.id],
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )

        action = _ReactAction(
            "evidence.retriever.search",
            {"question": content, "document_ids": [document.id]},
            "轻量论文问答先检索单篇论文证据。",
        )
        self._append_react_trace(
            run_id=run_id,
            status="retrieval_tool_started",
            payload={"step": 1, "tool": action.tool, "arguments": action.arguments, "runtime_variant": "paper_qa_lightweight"},
        )
        started_at = time.perf_counter()
        observation = self._finalize_tool_observation(
            self._tool_retrieve_evidence(
                run_id,
                session,
                content,
                action.arguments,
                [],
            )
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        payload_view = self._react_observation_payload(observation)
        evidence_items = self._merge_evidence_items([], payload_view)[:6]
        evidence_count = len(evidence_items)
        self._append_react_trace(
            run_id=run_id,
            status="retrieval_tool_finished",
            payload={
                "step": 1,
                "tool": observation.tool,
                "status": observation.status,
                "evidence_count": evidence_count,
                "used_document_count": 1,
                "duration_ms": duration_ms,
                "runtime_variant": "paper_qa_lightweight",
            },
        )
        tool_spec = self._spec_for_tool(observation.tool)
        verification = observation.observation.get("verification") if isinstance(observation.observation, dict) else None
        self._append_react_trace(
            run_id=run_id,
            status="tool_call_log",
            payload={
                "route_id": "paper_qa_lightweight",
                "tool_name": observation.tool,
                "operation_level": tool_spec.operation_level if tool_spec else "query-level",
                "io_type": tool_spec.io_type if tool_spec else "read",
                "write_type": tool_spec.write_type if tool_spec else "none",
                "target_objects": observation.observation.get("target_objects") if isinstance(observation.observation, dict) else [],
                "requires_confirmation": False,
                "success": observation.status == "completed",
                "verification_success": verification.get("success") if isinstance(verification, dict) else None,
                "duration_ms": duration_ms,
                "used_document_ids": [document.id],
                "evidence_count": evidence_count,
            },
        )
        self._append_react_trace(
            run_id=run_id,
            status="react_observation",
            payload={
                "step": 1,
                "tool": observation.tool,
                "status": observation.status,
                "summary": observation.summary,
                "payload": self._safe_trace_payload(observation.payload),
                "observation": self._safe_trace_payload(observation.observation),
                "runtime_variant": "paper_qa_lightweight",
            },
        )
        observations = [observation]
        if observation.status in {"needs_clarification", "validation_failed", "failed"}:
            final_status = observation.status
            answer = observation.summary
            retrieval_status = "degraded" if observation.status == "failed" else "skipped"
            warning = observation.summary if observation.status == "failed" else None
            fallback_used = True
            llm_called = False
        elif not evidence_items:
            final_status = "needs_clarification"
            retrieval_status = "skipped"
            warning = "没有检索到足够正文片段，本轮未生成基于论文内容的回答。"
            answer = (
                "本轮没有检索到可引用的论文正文证据，因此我不能只根据文件名、标题或元数据回答这篇论文内容。"
                "请确认所选论文已完成入库、正文 chunk 与向量/关键词索引可用后，再重新发送问题。"
            )
            fallback_used = True
            llm_called = False
        else:
            active_skill_context = self._active_skill_context_for_paper_qa_lightweight(run_id, evidence_items=evidence_items)
            draft = self._draft_paper_qa_short_answer(
                content,
                document,
                evidence_items,
                active_skill_context=active_skill_context,
            )
            answer = draft.answer
            fallback_used = draft.fallback_used
            llm_called = bool(self.api_key)
            final_status = "completed" if draft.llm_draft_success else "degraded"
            retrieval_status = "ready" if draft.llm_draft_success else "degraded"
            warning = draft.drafting_error if draft.fallback_used else None
        self._append_react_trace(
            run_id=run_id,
            status="paper_qa_lightweight_answer",
            payload={
                "used_document_ids": [document.id],
                "evidence_count": evidence_count,
                "llm_called": llm_called,
                "answer_length": len(answer),
                "fallback_used": fallback_used,
                "action_status": final_status,
                "retrieval_status": retrieval_status,
            },
        )
        self._record_react_reflection(session=session, user_goal=content, observations=observations, status=final_status)
        self._update_react_state(session=session, user_goal=content, observations=observations, status=final_status)
        if owns_run:
            self._finish_run(run_id, ResearchRunStatus.COMPLETED if final_status != "failed" else ResearchRunStatus.FAILED)
        else:
            self._append_react_trace(
                run_id=run_id,
                status=f"{runtime_label}_finished",
                payload={
                    "final_status": final_status,
                    "observation_count": 1,
                    "evidence_count": evidence_count,
                    "final_content_length": len(answer),
                    "runtime_variant": "paper_qa_lightweight",
                },
            )
        return KnowledgeAgentResult(
            content=answer.strip(),
            retrieval_status=retrieval_status,
            warning=warning,
            citations=self._collect_citations(evidence_items),
            used_document_ids=[document.id],
            evidence_items=evidence_items,
            action_status=final_status,
            agent_trace_id=run_id,
            library_mutated=False,
        )

    def _paper_qa_lightweight_scope(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> tuple[str, LibraryDocument] | None:
        if not self._is_lightweight_paper_qa_request(content):
            return None
        if self._is_report_like_request(content):
            return None
        if self._is_compare_like_request(content):
            return None
        if self._mentions_all_library(content):
            return None
        if self._is_staged_mixed_request(content):
            return None
        if self._looks_like_read_then_write_assignment(content):
            return None
        if (
            self._is_assignment_intent(content)
            or self._is_create_category_intent(content)
            or self._is_clear_categories_intent(content)
            or self._is_destructive_intent(content)
            or self._is_document_label_relation_assignment(content)
        ):
            return None
        if self._is_grouped_category_summary_request(content):
            return None
        if self._is_labeled_document_collection_request(content):
            return None
        if self._is_metadata_query(content, selected_document_ids, attachments) or self._is_document_category_query(content):
            return None

        selected_ids = self._real_document_ids(selected_document_ids)
        attachment_ids = self._real_document_ids([attachment.document_id for attachment in attachments if attachment.document_id])
        if len(selected_ids) > 1 or len(attachment_ids) > 1:
            return None
        explicit_ids = list(dict.fromkeys([*selected_ids, *attachment_ids]))
        if len(explicit_ids) > 1:
            return None
        scope_type = "current_selection"
        if not explicit_ids:
            if not self._mentions_lightweight_recent_single(content):
                return None
            explicit_ids = self._recent_scope_document_ids(session.id, singular=True)
            if len(explicit_ids) != 1:
                return None
            scope_type = "recent_selection"
        document_id = explicit_ids[0] if explicit_ids else ""
        document = next((item for item in self.document_library_service.list_documents() if item.id == document_id), None)
        if document is None:
            return None
        return scope_type, document

    @staticmethod
    def _mentions_lightweight_recent_single(content: str) -> bool:
        return KnowledgeAgentRuntime._mentions_single_current_document(content) or any(
            marker in content
            for marker in (
                "刚才那篇",
                "刚刚那篇",
                "上次那篇",
                "前面那篇",
                "那篇论文",
                "那篇文章",
                "上一篇论文",
                "上篇论文",
            )
        )

    @staticmethod
    def _is_lightweight_paper_qa_request(content: str) -> bool:
        normalized = content.casefold()
        markers = (
            "讲什么",
            "讲了什么",
            "主要讲",
            "大致研究什么",
            "研究方向",
            "主要面向",
            "面向",
            "核心任务",
            "关注",
            "方法",
            "解决",
            "关系",
            "创新",
            "贡献",
            "实验",
            "结论",
            "结果",
            "局限",
            "不足",
            "优缺点",
            "优点",
            "缺点",
            "是什么意思",
            "说明",
            "解释",
            "概括",
            "概览",
            "总结",
            "一句话",
            "一段话",
            "两句话",
            "摘要一下",
            "what",
            "method",
            "problem",
            "contribution",
            "novelty",
            "experiment",
            "result",
            "conclusion",
            "summarize",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _is_report_like_request(content: str) -> bool:
        normalized = content.casefold()
        if re.search(r"(?:[12]000|one thousand|two thousand)\s*(?:字|词|word|words)", normalized):
            return True
        return any(
            marker in normalized
            for marker in (
                "写报告",
                "写一份报告",
                "写一篇报告",
                "生成报告",
                "分析报告",
                "阅读报告",
                "完整报告",
                "完整阅读报告",
                "报告生成",
                "写综述",
                "写一份综述",
                "生成综述",
                "系统综述",
                "综述报告",
                "生成 markdown",
                "生成markdown",
                "保存 markdown",
                "保存markdown",
                "保存报告",
                "导出报告",
                "可导出",
                "按章节",
                "分章节",
                "章节写",
                "三点式阅读摘要",
                "write a report",
                "review report",
                "reading report",
                "save report",
                "export report",
                "markdown",
                "systematic review",
                "survey",
            )
        )

    @staticmethod
    def _is_compare_like_request(content: str) -> bool:
        normalized = content.casefold()
        return any(
            marker in normalized
            for marker in (
                "对比",
                "比较",
                "相同点",
                "不同点",
                "区别",
                "分别",
                "各自",
                "compare",
                "comparison",
                "versus",
                " vs ",
            )
        )

    @staticmethod
    def _is_staged_mixed_request(content: str) -> bool:
        normalized = content.casefold()
        if re.search(r"先.+(?:再|然后|接着|并且)", content):
            return True
        return any(
            marker in normalized
            for marker in (
                "顺便",
                "再补充",
                "再用一句话",
                "then explain",
                "and explain",
            )
        )

    def _draft_paper_qa_short_answer(
        self,
        question: str,
        document: LibraryDocument,
        evidence_items: list[EvidenceItem],
        *,
        active_skill_context: str = "",
    ) -> _DraftResult:
        fallback = self._draft_paper_qa_short_fallback(question, document, evidence_items)
        if not evidence_items:
            return _DraftResult(answer=fallback, llm_draft_success=False, fallback_used=True, drafting_error="insufficient_evidence")
        if not self.api_key:
            return _DraftResult(answer=fallback, llm_draft_success=False, fallback_used=True, drafting_error="llm_not_configured")
        evidence_block = "\n\n".join(
            "\n".join(
                [
                    f"[{index}] 来源：{item.citation_label}",
                    f"页码：{item.page_number if item.page_number is not None else '未知'}",
                    f"证据：{self._compact_evidence_text(item.quote or item.snippet)}",
                ]
            )
            for index, item in enumerate(evidence_items[:6], start=1)
        )
        length_rule = "如果用户要求一句话，只输出一句话；如果要求两句话，只输出两句话；其他情况控制在 1-3 段。"
        user_sections = [
            f"用户问题：{question}",
            (
                "论文元数据："
                f"标题={document.title or document.display_name or document.filename}；"
                f"文件名={document.display_name or document.filename}；页数={document.page_count}"
            ),
            "证据：\n" + evidence_block,
            length_rule,
        ]
        if active_skill_context:
            user_sections.insert(0, active_skill_context)
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的论文短答助手。只基于给定证据回答，证据不足时说明边界。"
                            "用中文自然表达，保留论文标题、简称和关键术语，避免长报告结构。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(user_sections),
                    },
                ],
            )
        except Exception as exc:
            return _DraftResult(
                answer=fallback,
                llm_draft_success=False,
                fallback_used=True,
                drafting_error=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
        answer = self._extract_message_text(response)
        if not answer:
            return _DraftResult(answer=fallback, llm_draft_success=False, fallback_used=True, drafting_error="empty_llm_response")
        return _DraftResult(answer=self._ensure_short_answer_title(answer, document), llm_draft_success=True)

    def _active_skill_context_for_paper_qa_lightweight(
        self,
        run_id: str,
        *,
        evidence_items: list[EvidenceItem],
    ) -> str:
        if not self.enable_skill_context_prompt_injection:
            self._record_skill_context_injection_skip(run_id, reason="disabled")
            return ""
        if not self.enable_skill_context_paper_qa_lightweight_only:
            self._record_skill_context_injection_skip(run_id, reason="paper_qa_lightweight_disabled")
            return ""
        if not evidence_items:
            self._record_skill_context_injection_skip(run_id, reason="no_evidence")
            return ""
        if not self.api_key:
            self._record_skill_context_injection_skip(run_id, reason="llm_not_configured")
            return ""
        summary = self._skill_context_summary_from_trace(run_id)
        if not summary:
            self._record_skill_context_injection_skip(run_id, reason="no_summary")
            return ""
        if summary.get("skill_id") not in {"paper_summary", "qa", "method_explainer"}:
            self._record_skill_context_injection_skip(run_id, reason="unsupported_skill")
            return ""
        rendered = SkillContextBuilder.render_active_skill_context(summary)
        if not rendered:
            self._record_skill_context_injection_skip(run_id, reason="empty_context")
            return ""
        self._append_react_trace(
            run_id=run_id,
            status="skill_context_injected",
            payload={
                "skill_id": summary.get("skill_id"),
                "path": "paper_qa_lightweight",
                "char_count": len(rendered),
                "injected": True,
                "reason": "paper_qa_lightweight_short_answer_llm_draft",
            },
        )
        return rendered

    def _skill_context_summary_from_trace(self, run_id: str) -> dict[str, Any] | None:
        try:
            traces = self.runtime_repository.list_traces(run_id)
        except Exception:
            return None
        for trace in traces:
            if trace.status != "agent_mode_selected" or not isinstance(trace.payload, dict):
                continue
            summary = trace.payload.get("skill_context_summary")
            if isinstance(summary, dict):
                return summary
        return None

    def _record_skill_context_injection_skip(self, run_id: str, *, reason: str) -> None:
        self._append_react_trace(
            run_id=run_id,
            status="skill_context_injection_skipped",
            payload={
                "path": "paper_qa_lightweight",
                "injected": False,
                "reason": reason,
            },
        )

    @staticmethod
    def _compact_evidence_text(text: str, *, limit: int = 700) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact if len(compact) <= limit else compact[:limit].rstrip() + "..."

    @staticmethod
    def _ensure_short_answer_title(answer: str, document: LibraryDocument) -> str:
        text = answer.strip()
        title = (document.title or document.display_name or document.filename or "").strip()
        if not title:
            return text
        lowered = text.casefold()
        if title.casefold() in lowered:
            return text
        return f"《{title}》：{text}"

    def _draft_paper_qa_short_fallback(
        self,
        question: str,
        document: LibraryDocument,
        evidence_items: list[EvidenceItem],
    ) -> str:
        title = document.title or document.display_name or document.filename
        snippets = [self._compact_evidence_text(item.quote or item.snippet, limit=220) for item in evidence_items[:3]]
        if not snippets:
            return (
                "本轮没有检索到可引用的论文正文证据，因此我不能只根据文件名、标题或元数据回答这篇论文内容。"
                "请确认所选论文已完成入库、正文 chunk 与向量/关键词索引可用后，再重新发送问题。"
            )
        if "一句话" in question:
            return f"根据检索到的正文证据，{title} 主要涉及：{snippets[0]}"
        lines = [f"根据检索到的正文证据，{title} 可以简要概括如下："]
        lines.extend(f"- {snippet}" for snippet in snippets)
        lines.append("上述内容只覆盖本轮检索到的证据片段。")
        return "\n".join(lines)

    def _ambiguous_write_clarification(
        self,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        if not self._is_ambiguous_category_delete_request(content):
            return None
        if self._is_clear_categories_intent(content) and self._mentions_previous_referent(content):
            return None
        return KnowledgeAgentResult(
            content=(
                "这个写操作目标还不够明确：你是要删除论文实体、删除空标签/分类实体，"
                "还是只移除论文和标签/分类之间的关系？请明确目标后我再继续。"
            ),
            action_status="needs_clarification",
            agent_trace_id=trace_id,
            library_mutated=False,
        )

    def _try_all_library_assign_preview(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return None
        if not self._is_assignment_intent(content) or not self._mentions_all_library(content):
            return None
        if self._needs_untagged_assignment(content) or self._has_downstream_report_generation_request(content):
            return None
        category_names = self._extract_category_names_from_request(content)
        category_name = category_names[0] if category_names else self._extract_category_name_from_request(content)
        if not category_name:
            return KnowledgeAgentResult(
                content="我识别到你想给所有论文打标签，但没有可靠识别出目标标签名。请明确要添加哪个标签。",
                action_status="needs_clarification",
                agent_trace_id=trace_id,
                library_mutated=False,
            )
        ready_document_ids = [
            document.id
            for document in self.document_library_service.list_documents()
            if document.status == "ready"
        ]
        if not ready_document_ids:
            return KnowledgeAgentResult(
                content="当前没有 ready 状态的论文，因此没有生成全库打标签预览，也没有改动论文库。",
                action_status="completed",
                used_document_ids=[],
                agent_trace_id=trace_id,
                library_mutated=False,
            )
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        preview = self._preview_assign_category_from_documents(
            category_name=category_name,
            document_ids=ready_document_ids,
            source_goal=content,
            risk_level="batch_write",
            expected_scope="all_library",
        )
        self._write_pending_tool_action(session.id, preview, source_goal=content)
        self._append_react_trace(
            run_id=run_id,
            status="write_preview_created",
            payload={
                "tool": "library.operator.assign_category",
                "operation": preview.operation,
                "operation_level": "relation-level",
                "write_type": "append",
                "target_type": "paper-category relation",
                "risk_level": preview.risk_level,
                "affected_count": preview.affected_count,
                "target_count": preview.affected_count,
                "targets": preview.affected_entities[:12],
                "expected_scope": preview.expected_scope,
                "requires_confirmation": True,
                "confirmation_phrase": preview.confirmation_phrase,
                "library_mutated": False,
            },
        )
        if owns_run:
            self._finish_run(run_id)
        return KnowledgeAgentResult(
            content=self._preview_confirmation_text(preview),
            action_status="confirmation_required",
            used_document_ids=ready_document_ids,
            agent_trace_id=run_id,
            library_mutated=False,
        )

    def _try_read_then_write_preview(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return None
        if self._state_document_ids(self._read_react_state(session.id)):
            return None
        plan = self._build_read_then_write_plan(content)
        if plan is None:
            return None
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        self._append_react_trace(
            run_id=run_id,
            status="read_then_write_plan_created",
            payload=self._safe_trace_payload(plan),
        )
        if plan["kind"] == "clarify":
            if owns_run:
                self._finish_run(run_id)
            return KnowledgeAgentResult(
                content=str(plan["message"]),
                action_status="needs_clarification",
                agent_trace_id=run_id,
                library_mutated=False,
            )

        read_step = plan["steps"][0]
        write_step = plan["steps"][1]
        read_result = self._execute_read_then_write_read_step(read_step)
        self._append_react_trace(
            run_id=run_id,
            status="read_then_write_read_completed",
            payload=self._safe_trace_payload(read_result),
        )
        if read_result["status"] == "needs_clarification":
            if owns_run:
                self._finish_run(run_id)
            return KnowledgeAgentResult(
                content=str(read_result["message"]),
                action_status="needs_clarification",
                agent_trace_id=run_id,
                library_mutated=False,
            )

        document_ids = [str(item) for item in read_result.get("document_ids") or [] if item]
        documents = [item for item in read_result.get("documents") or [] if isinstance(item, dict)]
        if not document_ids:
            if owns_run:
                self._finish_run(run_id)
            return KnowledgeAgentResult(
                content="没有符合条件的论文，因此没有生成写入预览，也没有改动论文库。",
                action_status="completed",
                used_document_ids=[],
                agent_trace_id=run_id,
                library_mutated=False,
            )

        category_name = str(write_step.get("category_name") or "").strip()
        preview = self._preview_assign_category_from_documents(
            category_name=category_name,
            document_ids=document_ids,
            source_goal=content,
        )
        self._write_pending_tool_action(
            session.id,
            preview,
            source_goal=content,
            plan_state=None,
            step_state=None,
        )
        self._append_react_trace(
            run_id=run_id,
            status="write_preview_created",
            payload={
                "tool": "library.operator.assign_category",
                "operation": preview.operation,
                "operation_level": "relation-level",
                "write_type": "append",
                "target_type": "paper-category relation",
                "risk_level": preview.risk_level,
                "affected_count": preview.affected_count,
                "target_count": preview.affected_count,
                "targets": preview.affected_entities[:12],
                "expected_scope": preview.expected_scope,
                "requires_confirmation": True,
                "confirmation_phrase": preview.confirmation_phrase,
                "read_step": read_step,
                "write_step": {**write_step, "document_ids": document_ids},
            },
        )
        if owns_run:
            self._finish_run(run_id)
        names = "、".join(str(item.get("name") or item.get("filename") or item.get("id")) for item in documents[:8])
        suffix = "等" if len(documents) > 8 else ""
        return KnowledgeAgentResult(
            content=(
                f"我先按只读步骤找到 {len(document_ids)} 篇符合条件的论文：{names}{suffix}。\n"
                f"写入预览：将给这些论文添加标签/分类「{category_name}」。"
                f"请回复「{preview.confirmation_phrase}」后我再执行；确认前不会改动论文库。"
            ),
            action_status="confirmation_required",
            used_document_ids=document_ids,
            agent_trace_id=run_id,
            library_mutated=False,
        )

    def _build_read_then_write_plan(self, content: str) -> dict[str, Any] | None:
        if self._has_downstream_report_generation_request(content):
            return None
        read_operation = self._read_then_write_read_operation(content)
        if (
            read_operation is not None
            and self._is_assignment_intent(content)
            and self._has_read_then_write_safety_marker(content)
        ):
            return {
                "kind": "clarify",
                "message": "我会按你的要求只做只读查询，本轮不会生成写入预览，也不会改动论文库。",
            }
        if not self._looks_like_read_then_write_assignment(content):
            if (
                self._is_assignment_intent(content)
                and any(marker in content for marker in ("这些论文", "这些文章"))
                and not read_operation
            ):
                return {
                    "kind": "clarify",
                    "message": "我没有找到本轮可绑定的只读结果，不能把“这些论文”默认扩大到全库。请先说明要查哪一批论文。",
                }
            return None
        if read_operation is None:
            return None
        if self._has_read_then_write_safety_marker(content):
            return {
                "kind": "clarify",
                "message": "我会按你的要求只做只读查询，本轮不会生成写入预览，也不会改动论文库。",
            }
        category_names = self._read_then_write_target_category_names(content)
        if not category_names:
            return {
                "kind": "clarify",
                "message": "我已经识别到这是读后写请求，但没有可靠识别出要添加的标签/分类名称。请明确要加哪个标签或分类。",
            }
        category_name = category_names[0]
        read_step = {
            "step_id": "read-1",
            "kind": "read",
            "operation": read_operation["operation"],
            "target_type": read_operation["target_type"],
            "scope_type": read_operation["scope_type"],
            "document_ids": [],
            "category_name": read_operation.get("category_name"),
            "depends_on": [],
            "output_binding": "document_ids",
            "requires_confirmation": False,
            "risk_level": "read_only",
            "tool_name": read_operation["tool_name"],
            "reason": read_operation["reason"],
        }
        write_step = {
            "step_id": "write-1",
            "kind": "write",
            "operation": "assign_label",
            "target_type": "paper-category relation",
            "scope_type": "read_step_result",
            "document_ids": [],
            "category_name": category_name,
            "depends_on": ["read-1"],
            "output_binding": "",
            "requires_confirmation": True,
            "risk_level": "scoped_write",
            "tool_name": "library.operator.assign_category",
            "reason": "把只读步骤返回的 document_ids 作为唯一写入目标。",
        }
        return {"kind": "read_then_write", "steps": [read_step, write_step]}

    def _execute_read_then_write_read_step(self, step: dict[str, Any]) -> dict[str, Any]:
        operation = str(step.get("operation") or "")
        if operation == "find_untagged_documents":
            payload = self._category_stats_payload(limit_documents=None)
            documents = [item for item in payload.get("untagged_documents") or [] if isinstance(item, dict)]
            return {
                "status": "completed",
                "operation": operation,
                "document_ids": [str(item) for item in payload.get("untagged_document_ids") or [] if item],
                "documents": documents,
            }
        if operation == "find_documents_by_category":
            category_name = str(step.get("category_name") or "").strip()
            lookup = self._category_lookup_payload([category_name]) if category_name else {}
            if lookup.get("missing_names"):
                candidates = "、".join(str(item) for item in lookup.get("candidate_names") or [])
                suffix = f"；相近标签/分类：{candidates}" if candidates else ""
                return {"status": "needs_clarification", "message": f"没有找到标签/分类「{category_name}」{suffix}。"}
            if lookup.get("ambiguous_names"):
                candidates = "、".join(str(item) for item in lookup.get("candidate_names") or [])
                return {"status": "needs_clarification", "message": f"标签/分类「{category_name}」匹配到多个候选，请确认：{candidates}"}
            matched_names = [str(item) for item in (lookup.get("matched_names") or [category_name]) if str(item).strip()]
            documents = [
                self._document_payload(document)
                for document in self.document_library_service.list_documents()
                if document.status == "ready"
                and any(category.name in set(matched_names) for category in document.categories)
            ]
            return {
                "status": "completed",
                "operation": operation,
                "category_name": matched_names[0] if matched_names else category_name,
                "document_ids": [str(item["id"]) for item in documents],
                "documents": documents,
            }
        return {"status": "needs_clarification", "message": "我还不能可靠识别这类读后写请求的只读范围。"}

    def _looks_like_read_then_write_assignment(self, content: str) -> bool:
        if not self._is_assignment_intent(content):
            return False
        if self._read_then_write_read_operation(content) is None:
            return False
        read_markers = ("几篇", "哪些", "找出", "列出", "查", "查一下", "看看", "有几篇")
        bind_markers = ("这些论文", "这些文章", "它们", "这批", "这几篇", "然后", "再帮", "再给", "并且")
        return any(marker in content for marker in read_markers) and any(marker in content for marker in bind_markers)

    def _read_then_write_read_operation(self, content: str) -> dict[str, Any] | None:
        read_segment = self._read_then_write_read_segment(content)
        untagged_markers = ("不带标签", "没有标签", "无标签", "未打标签", "没标签", "没有分类", "未分类")
        if any(marker in read_segment for marker in untagged_markers):
            return {
                "operation": "find_untagged_documents",
                "target_type": "paper",
                "scope_type": "untagged",
                "tool_name": "library.explorer.category_stats",
                "reason": "确定性读取当前没有标签/分类的论文集合。",
            }
        category_name = self._extract_existing_category_mention(read_segment)
        if category_name and any(marker in read_segment for marker in ("标签", "分类", "标签下", "分类下", "下面", "下的")):
            return {
                "operation": "find_documents_by_category",
                "target_type": "paper",
                "scope_type": "category",
                "category_name": category_name,
                "tool_name": "library.explorer.find_documents",
                "reason": "确定性读取指定标签/分类下的 ready 论文集合。",
            }
        return None

    @staticmethod
    def _read_then_write_read_segment(content: str) -> str:
        separators = (
            "然后",
            "再帮",
            "再给",
            "并且",
            "同时",
            "接着",
            "帮我把这些",
            "给这些",
            "把这些",
            "；",
            ";",
        )
        indexes = [content.find(separator) for separator in separators if content.find(separator) > 0]
        if not indexes:
            return content
        return content[: min(indexes)]

    def _read_then_write_write_segment(self, content: str) -> str:
        read_segment = self._read_then_write_read_segment(content)
        return content[len(read_segment) :] if len(read_segment) < len(content) else content

    def _read_then_write_target_category_names(self, content: str) -> list[str]:
        segment = self._read_then_write_write_segment(content)
        names: list[str] = []

        def add(value: str) -> None:
            candidate = self._clean_category_name(value)
            if not candidate or candidate in {"某个", "某个分类", "某个标签", "这个标签", "该标签", "此标签"}:
                return
            if not self._category_name_validation_error(candidate) and candidate not in names:
                names.append(candidate)

        for value in re.findall(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", segment):
            add(value)
        if names:
            return names
        for pattern in (
            r"(?:加上|打上|添加|补上|归类到|设置成|设为|设成|标为)\s*(?:一个|1个)?\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})\s*(?:标签|分类)",
            r"(?:标签|分类)\s*[：:]\s*([^，。！？；;\n]{1,40})",
        ):
            match = re.search(pattern, segment)
            if match:
                add(match.group(1))
        return names

    @staticmethod
    def _has_read_then_write_safety_marker(content: str) -> bool:
        normalized = content.casefold()
        return any(
            marker in normalized
            for marker in (
                "不要修改",
                "不要改",
                "只读",
                "只读看看",
                "不要打标签",
                "不要加标签",
                "不要添加标签",
                "不要添加分类",
                "do not modify",
                "read only",
            )
        )

    @staticmethod
    def _has_downstream_report_generation_request(content: str) -> bool:
        normalized = content.casefold()
        return any(
            marker in normalized
            for marker in (
                "写一份总结",
                "写个总结",
                "写总结",
                "生成总结",
                "按标签分别写",
                "分别写一份总结",
                "写一篇报告",
                "写报告",
                "生成报告",
                "分析报告",
                "analysis report",
                "write a report",
                "summary",
            )
        )

    def _ambiguous_relation_write_scope_clarification(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        if not self._is_relation_level_write_intent(content):
            return None
        if self._is_tag_category_semantics_conflict(content):
            return None
        if not self._ambiguous_relation_write_scope_without_context(
            session=session,
            content=content,
            action_arguments={},
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            observations=observations,
        ):
            return None
        return KnowledgeAgentResult(
            content="我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签/移除标签操作。",
            action_status="needs_clarification",
            agent_trace_id=trace_id,
            library_mutated=False,
        )

    @staticmethod
    def _is_relation_level_write_intent(content: str) -> bool:
        return KnowledgeAgentRuntime._is_assignment_intent(content) or KnowledgeAgentRuntime._is_clear_categories_intent(content)

    @staticmethod
    def _mentions_ambiguous_plural_document_referent(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "这些论文",
                "这些文章",
                "这些文档",
                "这几篇",
                "这批",
                "它们",
                "他们",
                "上面这些",
                "上述这些",
                "刚刚这些",
                "刚才这些",
                "刚刚这几篇",
                "刚才这几篇",
            )
        )

    @staticmethod
    def _mentions_explicit_category_document_scope(content: str) -> bool:
        lowered = content.casefold()
        if any(
            marker in content
            for marker in (
                "标签下的论文",
                "分类下的论文",
                "标签下面的论文",
                "分类下面的论文",
                "标签里的论文",
                "分类里的论文",
                "标签中的论文",
                "分类中的论文",
                "标签下",
                "分类下",
            )
        ):
            return True
        return bool(
            re.search(r"(?:under|in)\s+.+\s+(?:tag|category).*(?:paper|papers|document|documents)", lowered)
            or re.search(r"(?:paper|papers|document|documents).*(?:under|in)\s+.+\s+(?:tag|category)", lowered)
        )

    def _ambiguous_relation_write_scope_without_context(
        self,
        *,
        session: ChatSession,
        content: str,
        action_arguments: dict[str, Any],
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> bool:
        if not self._mentions_ambiguous_plural_document_referent(content):
            return False
        if self._needs_untagged_assignment(content) or str(action_arguments.get("scope") or "") == "untagged":
            return False
        if self._mentions_all_library(content):
            return False
        if self._mentions_explicit_category_document_scope(content):
            return False
        if self._real_document_ids(selected_document_ids):
            return False
        attachment_ids = self._real_document_ids([attachment.document_id for attachment in attachments if attachment.document_id])
        if attachment_ids:
            return False
        state_ids = self._state_document_ids(self._read_react_state(session.id))
        if state_ids:
            return False
        if self._real_document_ids(self._document_ids_from_observations(observations)):
            return False
        category_filter_ids = self._category_filter_document_ids_from_observations(observations)
        if category_filter_ids is not None:
            return False
        return True

    def _should_handle_with_react(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return True

        lowered = content.casefold()
        library_markers = (
            "论文库",
            "文献库",
            "本地论文",
            "库里",
            "标签",
            "分类",
            "打标签",
            "归类",
            "无标签",
            "没有标签",
        )
        drafting_markers = ("总结", "综述", "概述", "对比", "比较", "brief", "review", "summary", "summarize", "compare")
        if any(marker in content for marker in library_markers):
            return True
        if any(marker in lowered for marker in drafting_markers) and self._extract_document_tokens(content):
            return True
        if "论文" in content and any(marker in content for marker in ("几篇", "多少", "哪些", "什么", "写一篇")):
            return True
        return False

    def _should_plan_before_destructive_confirmation(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        """Allow ReAct to handle safe subgoals before confirming a destructive subgoal."""

        if not self._is_destructive_intent(content):
            return False
        if self._is_delete_unused_categories_intent(content):
            return True
        if self._is_clear_categories_intent(content):
            return True
        safe_obligations = [
            obligation
            for obligation in self._answer_obligations(content, selected_document_ids, attachments)
            if not obligation.key.endswith("_verified")
        ]
        if safe_obligations:
            return True
        return self._has_final_literal_output_request(content)

    def _run_react_agent(
        self,
        *,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        trace_id: str | None = None,
        runtime_label: str = "knowledge_react_execution",
    ) -> KnowledgeAgentResult:
        content = request.content.strip()
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        if not owns_run:
            self._append_react_trace(
                run_id=run_id,
                status=f"{runtime_label}_started",
                payload={"session_id": session.id, "topic": content},
            )
        observations: list[_ReactObservation] = []
        final_text = ""
        final_status = "completed"
        used_document_ids: list[str] = []
        evidence_items: list[EvidenceItem] = []
        retrieval_status = "skipped"
        warning: str | None = None
        library_mutated = False
        plan_state = self._build_initial_plan_state(content, selected_document_ids, attachments)

        try:
            self._append_react_trace(
                run_id=run_id,
                status="plan_state_created",
                payload=self._plan_payload(plan_state),
            )
            for step in range(12):
                obligations = self._answer_obligations(content, selected_document_ids, attachments)
                action = self._next_react_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
                self._maybe_promote_plan_from_action(plan_state, action)
                self._append_react_trace(
                    run_id=run_id,
                    status="react_action_planned",
                    payload={
                        "step": step + 1,
                        "tool": action.tool,
                        "arguments": self._safe_trace_payload(action.arguments),
                        "rationale": action.rationale,
                        "planning_source": "llm" if action.task_intent else "fallback_rule",
                        "task_intent": self._safe_trace_payload(action.task_intent),
                        "action_plan": self._safe_trace_payload(action.action_plan),
                        "confidence": action.confidence,
                        "answer_obligations": self._obligations_payload(obligations, observations),
                        "plan_state": self._plan_payload(plan_state),
                    },
                )

                if action.tool == "final.answer":
                    repair_action = self._next_plan_repair_action(
                        plan_state=plan_state,
                        session=session,
                        content=content,
                        selected_document_ids=selected_document_ids,
                        attachments=attachments,
                        observations=observations,
                    )
                    if repair_action is None:
                        final_text = self._user_visible_final_answer(
                            content,
                            observations,
                            str(action.arguments.get("content") or ""),
                        )
                        break
                    action = repair_action
                    self._append_react_trace(
                        run_id=run_id,
                        status="plan_repair_action_selected",
                        payload={
                            "step": step + 1,
                            "tool": action.tool,
                            "arguments": self._safe_trace_payload(action.arguments),
                            "plan_state": self._plan_payload(plan_state),
                        },
                    )

                step_state = self._bind_action_to_plan_step(plan_state, action, observations)
                if step_state.observation_index is None:
                    step_state.observation_index = len(observations)
                self._append_react_trace(
                    run_id=run_id,
                    status="plan_step_started",
                    payload={"step": self._step_payload(step_state), "plan_id": plan_state.plan_id},
                )
                if action.tool != "final.answer":
                    tool_started_at = time.perf_counter()
                    self._append_react_trace(
                        run_id=run_id,
                        status="retrieval_tool_started" if action.tool.startswith("evidence.retriever.") else "tool_started",
                        payload={
                            "step": step + 1,
                            "tool": action.tool,
                            "arguments": self._safe_trace_payload(action.arguments),
                            "plan_step_id": step_state.step_id,
                        },
                    )
                else:
                    tool_started_at = time.perf_counter()
                action, observation = self._safe_execute_plan(
                    run_id=run_id,
                    session=session,
                    content=content,
                    action=action,
                    observations=observations,
                    plan_state=plan_state,
                    step_state=step_state,
                )
                duration_ms = int((time.perf_counter() - tool_started_at) * 1000)
                self._append_react_trace(
                    run_id=run_id,
                    status="retrieval_tool_finished" if observation.tool.startswith("evidence.retriever.") else "tool_finished",
                    payload={
                        "step": step + 1,
                        "tool": observation.tool,
                        "status": observation.status,
                        "evidence_count": len(observation.payload.get("evidence_items") or []),
                        "used_document_count": len(observation.payload.get("document_ids") or []),
                        "plan_step_id": step_state.step_id,
                    },
                )
                tool_spec = self._spec_for_tool(observation.tool)
                verification = observation.observation.get("verification") if isinstance(observation.observation, dict) else None
                self._append_react_trace(
                    run_id=run_id,
                    status="tool_call_log",
                    payload={
                        "route_id": runtime_label,
                        "tool_name": observation.tool,
                        "operation_level": (tool_spec.operation_level if tool_spec else self._operation_level_for_tool(observation.tool)),
                        "io_type": tool_spec.io_type if tool_spec else ("write" if observation.payload.get("library_mutated") else "read"),
                        "write_type": tool_spec.write_type if tool_spec else "none",
                        "target_objects": observation.observation.get("target_objects") if isinstance(observation.observation, dict) else [],
                        "requires_confirmation": observation.observation.get("requires_confirmation") if isinstance(observation.observation, dict) else False,
                        "success": observation.observation.get("success") if isinstance(observation.observation, dict) else observation.status == "completed",
                        "verification_success": verification.get("success") if isinstance(verification, dict) else None,
                        "duration_ms": duration_ms,
                    },
                )

                observations.append(observation)
                step_state.observation_index = len(observations) - 1
                self._append_react_trace(
                    run_id=run_id,
                    status="react_observation",
                    payload={
                        "step": step + 1,
                        "tool": observation.tool,
                        "status": observation.status,
                        "summary": observation.summary,
                        "payload": self._safe_trace_payload(observation.payload),
                        "observation": self._safe_trace_payload(observation.observation),
                        "plan_step": self._step_payload(step_state),
                    },
                )

                used_document_ids = self._merge_document_ids(used_document_ids, observation.payload)
                evidence_items = self._merge_evidence_items(evidence_items, observation.payload)
                if evidence_items:
                    retrieval_status = "ready"
                evidence_quality = observation.payload.get("evidence_quality")
                if isinstance(evidence_quality, dict) and "vector_unavailable" in (evidence_quality.get("warnings") or []):
                    warning = (
                        "Milvus 向量服务当前不可用，本轮已改用已入库论文正文的关键词检索；"
                        "结论仍基于检索到的正文片段，但语义召回能力会弱于向量检索。"
                    )
                if observation.payload.get("library_mutated"):
                    library_mutated = True
                if observation.status == "degraded":
                    retrieval_status = "degraded"
                    warning = observation.summary
                if observation.status == "failed" and observation.tool.startswith("evidence.retriever."):
                    retrieval_status = "degraded"
                if observation.status in {"needs_clarification", "confirmation_required", "validation_failed", "failed", "degraded"}:
                    final_status = observation.status
                    if observation.tool in {"report.drafter.write", "report.drafter.write_by_category"} and observation.payload.get("answer"):
                        final_text = str(observation.payload.get("answer") or "").strip()
                    else:
                        final_text = self._synthesize_react_answer(content, observations)
                    break
                if observation.tool in {"report.drafter.write", "report.drafter.write_by_category"}:
                    final_text = str(observation.payload.get("answer") or "").strip()
                    break
            else:
                final_text = self._synthesize_react_answer(content, observations)

            if not final_text:
                final_text = self._synthesize_react_answer(content, observations)
            final_text = self.ensure_final_answer(
                user_prompt=content,
                original_request=request,
                runtime_mode=runtime_label,
                evidence_items=evidence_items,
                citations=self._collect_citations(evidence_items),
                used_document_ids=used_document_ids,
                tool_observations=observations,
                previous_content=final_text,
                trace_digest={"final_status": final_status, "observation_count": len(observations)},
                trace_id=run_id,
                action_status=final_status,
            )
            final_text, final_status, completion_payload = self._enforce_plan_completion_gate(
                plan_state,
                observations,
                final_text,
                final_status,
            )
            self._append_react_trace(
                run_id=run_id,
                status="plan_completion_checked",
                payload={
                    **self._plan_completion_payload(content, selected_document_ids, attachments, observations),
                    "hard_gate": completion_payload,
                },
            )
            deferred_destructive = None
            if final_status == "completed":
                deferred_destructive = self._defer_destructive_part_after_safe_plan(
                    session=session,
                    content=content,
                    observations=observations,
                    current_answer=final_text,
                    run_id=run_id,
                )
            if deferred_destructive is not None:
                final_text = deferred_destructive
                final_status = "confirmation_required"
            self._record_react_reflection(
                session=session,
                user_goal=content,
                observations=observations,
                status=final_status,
            )
            self._update_react_state(
                session=session,
                user_goal=content,
                observations=observations,
                status=final_status,
            )
            if owns_run:
                self._finish_run(run_id)
            else:
                self._append_react_trace(
                    run_id=run_id,
                    status=f"{runtime_label}_finished",
                    payload={
                        "final_status": final_status,
                        "observation_count": len(observations),
                        "evidence_count": len(evidence_items),
                        "final_content_length": len(final_text),
                    },
                )
            return KnowledgeAgentResult(
                content=final_text,
                retrieval_status=retrieval_status,
                warning=warning,
                citations=self._collect_citations(evidence_items),
                used_document_ids=used_document_ids,
                evidence_items=evidence_items,
                action_status=final_status,
                agent_trace_id=run_id,
                library_mutated=library_mutated,
            )
        except Exception as exc:
            self._append_react_trace(
                run_id=run_id,
                status="react_failed",
                payload={"error": str(exc)},
            )
            if owns_run:
                self._finish_run(run_id, ResearchRunStatus.FAILED)
            return KnowledgeAgentResult(
                content="知识库 Agent 执行时遇到异常，已经停止本轮工具调用，避免对论文库产生不可靠修改。",
                retrieval_status="degraded",
                warning=str(exc),
                action_status="failed",
                agent_trace_id=run_id,
            )

    def _next_react_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction:
        if self._is_tag_category_semantics_conflict(content):
            return _ReactAction(
                self._INTERNAL_CATEGORY_CONFLICT_TOOL,
                {},
                "用户请求把标签和分类当作两套字段处理，但当前系统中二者是同一字段。",
            )
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        unmet_obligations = self._unmet_obligations(obligations, observations)
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if self.api_key:
            try:
                llm_action = self._next_react_action_with_llm(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
            except Exception:
                llm_action = None
            if llm_action is not None and self._clamp_confidence(llm_action.confidence) < 0.55:
                llm_action = None
            if llm_action is not None:
                if (
                    self._is_delete_unused_categories_intent(content)
                    and llm_action.tool == "library.operator.clear_categories"
                ):
                    llm_action = _ReactAction(
                        "library.operator.delete_unused_categories",
                        {"selector": "unused"},
                        "Corrected operation-level mismatch: unused category cleanup is an entity delete, not document tag clearing.",
                        task_intent={
                            **llm_action.task_intent,
                            "task_type": "delete_unused_categories",
                            "operation_level": "entity-level",
                        },
                        action_plan=llm_action.action_plan,
                        confidence=llm_action.confidence,
                    )
                if (
                    llm_action.tool == "final.answer"
                    and unmet_obligations
                ):
                    repair_action = self._action_for_unmet_obligation(
                        session=session,
                        content=content,
                        selected_document_ids=selected_document_ids,
                        attachments=attachments,
                        observations=observations,
                        obligations=unmet_obligations,
                    )
                    if repair_action is not None:
                        return repair_action
                    return _ReactAction(
                        self._INTERNAL_DEGRADED_TOOL,
                        {
                            "reason": (
                                "Configured LLM returned final before satisfying required answer obligations: "
                                + ", ".join(obligation.key for obligation in unmet_obligations)
                            )
                        },
                        "LLM returned a premature final answer; no safe next tool could be inferred.",
                    )
                if llm_action.tool == "final.answer":
                    return llm_action
                if observations and llm_action.tool in completed_tools and not self._can_repeat_react_tool(llm_action):
                    if unmet_obligations:
                        repair_action = self._action_for_unmet_obligation(
                            session=session,
                            content=content,
                            selected_document_ids=selected_document_ids,
                            attachments=attachments,
                            observations=observations,
                            obligations=unmet_obligations,
                        )
                        if repair_action is not None:
                            return repair_action
                    return _ReactAction(
                        "final.answer",
                        {"content": self._synthesize_react_answer(content, observations)},
                        "The model repeated an already completed tool after obligations were satisfied; answer from the first successful observation.",
                    )
                if llm_action.tool != "final.answer" or observations or not self._requires_tool_observation(
                    content,
                    selected_document_ids,
                    attachments,
                ):
                    return llm_action
                return _ReactAction(
                    self._INTERNAL_DEGRADED_TOOL,
                    {
                        "reason": (
                            "Configured LLM returned a final answer before reading required "
                            "library/tool observations."
                        )
                        },
                    "LLM tool planning returned a premature final answer; degrading explicitly.",
                )
            if observations and not unmet_obligations:
                return _ReactAction(
                    "final.answer",
                    {"content": self._synthesize_react_answer(content, observations)},
                    "All answer obligations are satisfied by tool observations; synthesize the final answer.",
                )
            if unmet_obligations:
                repair_action = self._action_for_unmet_obligation(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                    obligations=unmet_obligations,
                )
                if repair_action is not None:
                    return repair_action
            return self._fallback_next_react_action(
                session=session,
                content=content,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
            )
        if observations:
            if unmet_obligations:
                repair_action = self._action_for_unmet_obligation(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                    obligations=unmet_obligations,
                )
                if repair_action is not None:
                    return repair_action
            else:
                return _ReactAction(
                    "final.answer",
                    {"content": self._synthesize_react_answer(content, observations)},
                    "All answer obligations are satisfied by tool observations; synthesize the final answer.",
                )
        return self._fallback_next_react_action(
            session=session,
            content=content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            observations=observations,
        )

    def _next_react_action_with_llm(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 ReAct 主 Agent。你只能输出一个 JSON 对象，不能输出 Markdown。"
                            "你的工作方式是 Reason -> Tool Action -> Observation -> Next Action -> Final。"
                            "如果需要知道论文库、标签、可用工具或论文正文信息，必须先选择工具；不要凭记忆猜数据库内容。"
                            "复杂需求必须拆成多步工具链，例如先查标签覆盖，再补标签，再RAG检索，再写总结。"
                            "你会收到 task_state.answer_obligations：它是本轮必须回答或执行的任务清单。"
                            "只有当 task_state.unmet_obligations 为空时，才允许输出 final；否则必须选择一个能推进未满足任务的工具。"
                            "不要把 task_state 当成固定流程；你仍要按用户意图、当前 observations 和工具能力选择下一步。"
                            "用户说把某个标签删掉/替换成另一个标签时，如果语义是换名或保留论文关联，必须用 library.operator.rename_category；"
                            "用户说清除标签或分类时，标签和分类是同一字段，必须用 library.operator.clear_categories；"
                            "如果用户要求清除标签但保留分类，必须停止并说明当前系统中标签和分类是同一字段，不能这样执行。"
                            "用户说用所有某标签论文写报告/对比/总结时，必须先用 category_name 精确读取该标签下真实论文。"
                            "用户问每篇论文对应什么标签、每篇文章的标签、所有论文分别是什么标签时，"
                            "必须调用 library.explorer.document_categories，不能只用 category_stats 或运行态摘要。"
                            "会话上下文可以、也应该用于理解“另外几篇”“刚才那些论文”等指代；"
                            "但当前库状态、计数和写入结果必须来自工具 observation。"
                            "写操作只能通过 library.operator.* 工具提出，后端会再次校验；删除论文或分类必须交给 confirmation_required 流程。"
                            "最终回答要直接给用户结论，不要输出“根据对话记录/根据上下文/根据运行态摘要”等过程话术。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_goal": content,
                                "context_snapshot": self._build_react_snapshot(
                                    session.id,
                                    selected_document_ids,
                                    attachments,
                                ),
                                "task_state": self._react_task_state(
                                    content,
                                    selected_document_ids,
                                    attachments,
                                    observations,
                                ),
                                "available_tools": self._react_tool_specs(),
                                "observations": [
                                    {
                                        "tool": observation.tool,
                                        "status": observation.status,
                                        "summary": observation.summary,
                                        "payload": self._safe_trace_payload(observation.payload),
                                    }
                                    for observation in observations
                                ],
                                "tool_scheduling_contract": {
                                    "llm_role": "understand intent, identify entities, choose the next real tool, or finish from observations",
                                    "database_truth_source": "only tool observations and repository-backed payloads",
                                    "guardrails": [
                                        "destructive operations require confirmation outside this planner",
                                        "library.operator writes must be followed by verified state in observations",
                                        "selected documents or document IDs require grounded tool observations",
                                        "do not use report templates unless requested_output is report or analysis_report",
                                    ],
                                    "fallback_note": "task_state obligations are safety/fallback hints, not the primary semantic brain",
                                },
                                "output_schema": {
                                    "task_intent": {
                                        "user_intent": "brief goal summary",
                            "task_type": "general_chat|metadata_query|document_qa|tag_query|tag_write|tag_rename|delete_unused_categories|category_entity_cleanup|category_query|category_write|collection_analysis|report_generation|other",
                            "operation_level": "query-level|entity-level|relation-level|content-level",
                                        "operation": "ask|summarize|compare|assign_label|remove_label|clear_labels|rename_label|delete_empty_labels|save_report|other",
                                        "target_type": "paper|paper_label_relation|label_entity|report|evidence|general_chat|unknown",
                                        "scope_hint": "current_selection|recent_selection|explicit_documents|explicit_label|all_library|unknown",
                                        "referenced_documents": ["document id, title, or filename mentioned by user"],
                                        "label_or_category": "explicit label/category name if any",
                                        "suggested_tool": "available tool name if a tool is needed",
                                        "requires_confirmation": False,
                                        "clarification_needed": False,
                                        "risk_level": "safe|read_only|write|destructive",
                                        "requested_output": "answer|list|count|summary|comparison|analysis_report|operation_result|other",
                                        "entities": [
                                            {
                                                "text": "entity text",
                                                "type": "tag|category|document|collection|metadata|report|unknown",
                                                "role": "source|target|filter|object",
                                                "confidence": 0.0,
                                            }
                                        ],
                                        "needs_verification": False,
                                        "required_capabilities": ["capability"],
                                    },
                                    "action_plan": [
                                        {
                                            "tool": "available tool name",
                                            "arguments": {},
                                            "purpose": "short purpose",
                                            "requires_verification_after": False,
                                        }
                                    ],
                                    "steps": [
                                        {
                                            "step_id": "1",
                                            "intent": "sub-goal intent",
                                            "tool": "available tool name or pending_confirmation",
                                            "arguments": {},
                                            "operation_level": "query-level|entity-level|relation-level|content-level",
                                            "risk_level": "read_only|safe_write|scoped_write|destructive|critical",
                                            "requires_confirmation": False,
                                            "needs_verification": True,
                                        }
                                    ],
                                    "confidence": 0.0,
                                    "tool_call": {
                                        "type": "tool",
                                        "tool": "one available tool name",
                                        "arguments": {},
                                        "rationale": "brief visible planning summary",
                                    },
                                    "final": {"type": "final", "content": "answer based only on observations"},
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
        except Exception:
            return None
        message = self._extract_message_text(response)
        if not message:
            return None
        payload = self._extract_json_payload(message)
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("tool_call"), dict):
            envelope = payload
            payload = payload["tool_call"]
        elif isinstance(payload.get("final"), dict):
            envelope = payload
            payload = payload["final"]
        else:
            envelope = payload
        task_intent = envelope.get("task_intent") if isinstance(envelope.get("task_intent"), dict) else {}
        action_plan = envelope.get("action_plan") if isinstance(envelope.get("action_plan"), list) else []
        if not action_plan and isinstance(envelope.get("steps"), list):
            action_plan = envelope["steps"]
        confidence = self._clamp_confidence(envelope.get("confidence", payload.get("confidence", 1.0)))
        if payload.get("type") == "final":
            return _ReactAction(
                tool="final.answer",
                arguments={"content": str(payload.get("content") or "")},
                rationale=str(payload.get("rationale") or "final"),
                task_intent=task_intent,
                action_plan=[item for item in action_plan if isinstance(item, dict)],
                confidence=confidence,
            )
        if payload.get("type") == "tool":
            arguments = payload.get("arguments")
            action = _ReactAction(
                tool=str(payload.get("tool") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                rationale=str(payload.get("rationale") or ""),
                task_intent=task_intent,
                action_plan=[item for item in action_plan if isinstance(item, dict)],
                confidence=confidence,
            )
            return self._normalize_llm_tool_action(action)
        return None

    def _normalize_llm_tool_action(self, action: _ReactAction) -> _ReactAction:
        """Adapt LLM intent into existing tool arguments without inventing DB state."""

        entities = [
            item for item in action.task_intent.get("entities") or []
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        category_names = [
            self._clean_category_name(str(item.get("text") or ""))
            for item in entities
            if str(item.get("type") or "").casefold() in {"tag", "category", "collection"}
            and str(item.get("role") or "").casefold() in {"source", "filter", "object", ""}
        ]
        category_names = [
            name for name in dict.fromkeys(category_names)
            if name and not self._category_name_validation_error(name)
        ]
        target_entities = [
            self._clean_category_name(str(item.get("text") or ""))
            for item in entities
            if str(item.get("role") or "").casefold() == "target"
            and str(item.get("type") or "").casefold() in {"tag", "category", "unknown"}
        ]
        target_entities = [
            name for name in dict.fromkeys(target_entities)
            if name and not self._category_name_validation_error(name)
        ]

        if action.tool in {
            "library.explorer.find_documents",
            "evidence.retriever.search_by_category",
        } and category_names:
            action.arguments.setdefault("category_names", category_names)
            action.arguments.setdefault("category_name", category_names[0])
        if action.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            names = target_entities or category_names
            if names:
                action.arguments.setdefault("category_name", names[0])
                action.arguments.setdefault("category_names", names)
        if action.tool == "library.operator.rename_category":
            source = action.arguments.get("source_category_name") or (category_names[0] if category_names else "")
            target = action.arguments.get("target_category_name") or (target_entities[0] if target_entities else "")
            if source:
                action.arguments["source_category_name"] = source
            if target:
                action.arguments["target_category_name"] = target
        if action.tool == "library.operator.clear_categories":
            names = category_names or target_entities
            if names:
                action.arguments["category_name"] = names[0]
                action.arguments.setdefault("operation", "remove_single_category_link")
                action.arguments.pop("scope", None)
        if action.tool == "library.operator.delete_unused_categories":
            action.arguments["selector"] = "unused"
        return action

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    @staticmethod
    def _can_repeat_react_tool(action: _ReactAction) -> bool:
        if action.tool.startswith("library.operator."):
            return False
        return action.tool in {
            "evidence.retriever.search",
            "evidence.retriever.search_by_category",
            "report.drafter.write",
            "report.drafter.write_by_category",
            "memory.read",
        }

    def _force_llm_write_tool_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        """Ask the configured LLM again for a write tool call, never a natural-language final."""

        if not self.api_key:
            return None
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的工具调用规划器，只输出 JSON。"
                            "当前用户请求涉及修改论文库标签/分类，不能输出 final，必须选择一个写库工具。"
                            "只能从 available_write_tools 里选择工具。"
                            "如果用户要求给没有标签的论文加标签，应使用 library.operator.assign_category，"
                            "arguments 里使用 scope=untagged 和用户指定的 category_name。"
                            "如果用户要求清空带有某标签的论文的标签，应使用 library.operator.clear_categories，"
                            "arguments 里使用 category_name=该标签名。"
                            "如果已经有成功写库 observation，可以输出 final。"
                            "除此之外，不要解释、不要说正在执行、不要输出 Markdown。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_goal": content,
                                "context_snapshot": self._build_react_snapshot(
                                    session.id,
                                    selected_document_ids,
                                    attachments,
                                ),
                                "available_write_tools": [
                                    tool
                                    for tool in self._react_tool_specs()
                                    if str(tool.get("name", "")).startswith("library.operator.")
                                ],
                                "observations": [
                                    {
                                        "tool": observation.tool,
                                        "status": observation.status,
                                        "summary": observation.summary,
                                        "payload": self._safe_trace_payload(observation.payload),
                                    }
                                    for observation in observations
                                ],
                                "output_schema": {
                                    "type": "tool",
                                    "tool": "library.operator.assign_category",
                                    "arguments": {"category_name": "标签名", "scope": "untagged"},
                                    "rationale": "brief tool-selection reason",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
        except Exception:
            return None
        message = self._extract_message_text(response)
        if not message:
            return None
        payload = self._extract_json_payload(message)
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("tool_call"), dict):
            payload = payload["tool_call"]
        if payload.get("type") != "tool":
            return None
        tool = str(payload.get("tool") or "")
        if not tool.startswith("library.operator."):
            return None
        arguments = payload.get("arguments")
        return _ReactAction(
            tool=tool,
            arguments=arguments if isinstance(arguments, dict) else {},
            rationale=str(payload.get("rationale") or "forced write tool planning"),
        )

    def _answer_obligations(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> list[_AnswerObligation]:
        obligations: list[_AnswerObligation] = []

        def add(key: str, description: str, tools: tuple[str, ...], **target: Any) -> None:
            if any(item.key == key for item in obligations):
                return
            obligations.append(_AnswerObligation(key, description, tools, target))

        labeled_category_names = self._category_entity_names_for_request(content)
        if self._needs_library_stats(content) and not labeled_category_names:
            add(
                "library_stats",
                "回答论文库总量、可用数量或处理状态。",
                ("library.explorer.stats",),
            )
        if self._needs_category_stats(content) and not labeled_category_names:
            add(
                "category_stats",
                "回答标签/分类数量、有标签论文数、无标签论文数或标签覆盖统计。",
                ("library.explorer.category_stats",),
            )
        if self._requires_document_category_observation(content, selected_document_ids) and not labeled_category_names:
            add(
                "document_category_details",
                "回答逐篇论文对应的真实标签/分类明细。",
                ("library.explorer.document_categories",),
                scope="selected" if selected_document_ids else "all_or_tagged",
            )
        if (
            self._is_metadata_query(content, selected_document_ids, attachments)
            and not self._is_assignment_intent(content)
            and not self._is_clear_categories_intent(content)
        ):
            add(
                "document_metadata_details",
                "回答论文题名、作者、期刊/会议、发表时间或年份等元数据字段。",
                ("library.explorer.document_metadata",),
                requested_fields=self._requested_metadata_fields(content),
            )
        if (
            labeled_category_names
            and not selected_document_ids
            and not self._is_assignment_intent(content)
            and not self._is_clear_categories_intent(content)
            and not self._is_rename_category_intent(content)
        ):
            if self._is_labeled_document_analysis_request(content):
                add(
                    "resolved_document_set",
                    "先按用户提到的标签/分类实体读取真实论文集合。",
                    ("library.explorer.find_documents",),
                    category_names=labeled_category_names,
                )
                add(
                    "evidence_retrieval",
                    "检索标签/分类实体下真实论文集合的正文或元数据证据。",
                    ("evidence.retriever.search",),
                )
                add(
                    "report",
                    "基于标签/分类实体下真实论文集合的证据生成最终分析、总结、对比或报告。",
                    ("report.drafter.write",),
                )
            else:
                add(
                    "labeled_document_set",
                    "读取用户提到的标签/分类/分组实体及其真实关联论文集合。",
                    ("library.explorer.find_documents",),
                    category_names=labeled_category_names,
                )
        rename_pair = self._extract_category_rename_request(content)
        if self._is_delete_unused_categories_intent(content):
            add(
                "delete_unused_categories_verified",
                "Delete unused tag/category entities with document_count=0 after preview and confirmation; do not modify document-category links.",
                ("library.operator.delete_unused_categories",),
                selector="unused",
                operation_level="entity-level",
                scope="categories_with_zero_documents",
                expected_effect={
                    "delete_category_entities": True,
                    "affected_document_count": 0,
                    "should_not_modify_document_category_relations": True,
                },
            )
        document_label_relation_assignment = self._is_document_label_relation_assignment(content)
        if (self._is_rename_category_intent(content) or rename_pair) and not document_label_relation_assignment:
            add(
                "rename_category_verified",
                "执行并校验标签/分类重命名或合并。",
                ("library.operator.rename_category",),
            )
            if self._needs_category_stats(content):
                add(
                    "post_write_category_stats",
                    "写操作完成后重新读取标签/分类统计，回答复合命令中的统计子目标。",
                    ("library.explorer.category_stats",),
                    category_name=rename_pair[1] if rename_pair else "",
                )
        if self._is_clear_categories_intent(content):
            add(
                "clear_categories_verified",
                "执行并校验清空论文标签/分类关系。",
                ("library.operator.clear_categories",),
            )
        assignment_category_names = self._extract_category_names_from_request(content)
        if self._is_assignment_intent(content):
            if not (
                self._needs_untagged_assignment(content)
                or selected_document_ids
                or any(attachment.document_id for attachment in attachments)
                or self._mentions_previous_referent(content)
                or self._mentions_single_current_document(content)
                or self._mentions_plural_current_documents(content)
                or self._mentions_all_library(content)
                or self._extract_document_tokens(content)
            ):
                add(
                    "resolved_document_set",
                    "先定位用户提到的论文集合。",
                    ("library.explorer.find_documents",),
                )
            add(
                "assign_category_verified",
                "执行并校验给目标论文追加标签/分类。",
                ("library.operator.assign_category",),
                category_names=assignment_category_names,
            )
        if self._is_create_category_intent(content):
            add(
                "create_category_verified",
                "执行并校验新建标签/分类。",
                ("library.operator.create_category",),
                content=content,
                category_names=assignment_category_names or self._extract_category_names_from_request(content),
            )
        if (
            self._is_document_category_query(content)
            and not labeled_category_names
            and not self._is_assignment_intent(content)
            and not self._is_clear_categories_intent(content)
            and not self._is_summary_request(content, selected_document_ids)
            and not self._requires_document_category_observation(content, selected_document_ids)
            and not self._is_metadata_query(content, selected_document_ids, attachments)
        ):
            add(
                "resolved_document_set",
                "先定位用户提到的论文集合。",
                ("library.explorer.find_documents",),
            )
            add(
                "document_category_details",
                "回答已定位论文对应的真实标签/分类明细。",
                ("library.explorer.document_categories",),
                scope="resolved",
            )
        if (
            not self._is_metadata_query(content, selected_document_ids, attachments)
            and (
                self._is_summary_request(content, selected_document_ids) or self._is_selected_document_answer_request(
                    content,
                    selected_document_ids,
                    attachments,
                )
            )
        ):
            if self._is_grouped_category_summary_request(content):
                add(
                    "category_evidence",
                    "按标签/分类分组检索论文证据。",
                    ("evidence.retriever.search_by_category",),
                )
                add(
                    "category_report",
                    "基于分组证据生成总结或对比。",
                    ("report.drafter.write_by_category",),
                )
            else:
                if not selected_document_ids and not attachments and self._extract_document_tokens(content):
                    add(
                        "resolved_document_set",
                        "先定位用户提到的论文集合。",
                        ("library.explorer.find_documents",),
                    )
                add(
                    "evidence_retrieval",
                    "检索论文正文或元数据证据。",
                    ("evidence.retriever.search",),
                )
                add(
                    "report",
                    "基于真实检索证据生成总结或对比回答。",
                    ("report.drafter.write",),
                )
        if (
            not self._is_metadata_query(content, selected_document_ids, attachments)
            and self._is_compound_question_request(content)
            and self._should_answer_compound_with_evidence(
                content,
                selected_document_ids,
                attachments,
            )
        ):
            if not selected_document_ids and not attachments and self._extract_document_tokens(content):
                add(
                    "resolved_document_set",
                    "先定位用户提到的论文集合。",
                    ("library.explorer.find_documents",),
                )
            add(
                "evidence_retrieval",
                "为同一条消息中的多个问题检索可回答的论文证据。",
                ("evidence.retriever.search",),
            )
            add(
                "report",
                "按用户问题顺序逐项生成回答，而不是只返回检索状态。",
                ("report.drafter.write",),
            )
        return obligations

    def _react_task_state(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> dict[str, Any]:
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        return {
            "answer_obligations": self._obligations_payload(obligations, observations),
            "satisfied_obligations": [
                item.key for item in obligations if self._obligation_satisfied(item, observations)
            ],
            "unmet_obligations": [
                item.key for item in obligations if not self._obligation_satisfied(item, observations)
            ],
            "planning_rule": "Use ReAct: choose tools until every unmet obligation is satisfied, then final.",
        }

    def _build_initial_plan_state(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> _PlanState:
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        steps = [
            self._step_from_obligation(index, obligation)
            for index, obligation in enumerate(obligations, start=1)
        ]
        goal_clauses = self._goal_clauses(content)
        return _PlanState(
            plan_id=f"plan-{uuid4().hex[:8]}",
            original_user_prompt=content,
            steps=steps,
            global_context={
                "plan_source": "answer_obligations",
                "goal_clause_count": len(goal_clauses),
                "selected_document_ids": selected_document_ids,
                "attachment_document_ids": [
                    attachment.document_id for attachment in attachments if attachment.document_id
                ],
                "attachment_count": len(attachments),
            },
        )

    def _step_from_obligation(self, index: int, obligation: _AnswerObligation) -> _StepState:
        tool = obligation.required_tools[0] if obligation.required_tools else ""
        args = self._default_args_for_obligation(obligation)
        return _StepState(
            step_id=f"step-{index}",
            intent=obligation.description,
            operation_level=self._operation_level_for_tool(tool),
            target_entities=self._entities_from_obligation(obligation),
            target_scope=str(obligation.target.get("scope") or obligation.target.get("expected") or ""),
            depends_on=[],
            risk_level=self._risk_level_for_tool(tool),
            resolved_tool_name=tool,
            resolved_tool_args=args,
            metadata={
                "source": "answer_obligation",
                "obligation_key": obligation.key,
                "required_tools": list(obligation.required_tools),
                "target": dict(obligation.target),
            },
        )

    def _default_args_for_obligation(self, obligation: _AnswerObligation) -> dict[str, Any]:
        key = obligation.key
        target = dict(obligation.target)
        if key in {"resolved_document_set", "labeled_document_set"}:
            args: dict[str, Any] = {"expected": "many"}
            if target.get("category_names"):
                args["category_names"] = list(target.get("category_names") or [])
            return args
        if key == "document_metadata_details":
            return {"requested_fields": list(target.get("requested_fields") or [])}
        if key == "create_category_verified":
            names = [str(item) for item in target.get("category_names") or [] if str(item).strip()]
            return {"category_name": names[0], "category_names": names} if names else {}
        if key == "assign_category_verified":
            names = [str(item) for item in target.get("category_names") or [] if str(item).strip()]
            args = {"category_name": names[0], "category_names": names} if names else {}
            return args
        if key == "delete_unused_categories_verified":
            return {"selector": "unused"}
        return {}

    @staticmethod
    def _entities_from_obligation(obligation: _AnswerObligation) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for name in obligation.target.get("category_names") or []:
            if str(name).strip():
                entities.append({"type": "category", "role": "filter", "text": str(name).strip()})
        category_name = obligation.target.get("category_name")
        if isinstance(category_name, str) and category_name.strip():
            entities.append({"type": "category", "role": "filter", "text": category_name.strip()})
        return entities

    def _risk_level_for_tool(self, tool_name: str) -> str:
        policy = self._TOOL_RISK_REGISTRY.get(tool_name)
        if policy is not None:
            return policy.risk_level
        if tool_name.startswith("library.explorer.") or tool_name.startswith("evidence.retriever."):
            return "read_only"
        if tool_name.startswith("report.drafter.") or tool_name == "memory.read":
            return "read_only"
        return "read_only"

    @classmethod
    def _spec_for_tool(cls, tool_name: str) -> ToolSpec | None:
        policy = cls._TOOL_RISK_REGISTRY.get(tool_name)
        if policy is not None:
            return ToolSpec(
                name=tool_name,
                display_name=tool_name,
                description=policy.destructive_kind or tool_name,
                scope="experimental" if tool_name == "memory.write" else "knowledge",
                operation_level=policy.operation_level,
                io_type="write",
                write_type=policy.write_type,
                destructive=policy.destructive,
                requires_confirmation=policy.requires_confirmation,
                input_object_types=[] if policy.target_type == "none" else [policy.target_type],
                output_observation_type=(
                    "category_delete_observation"
                    if tool_name == "library.operator.delete_unused_categories"
                    else "write_action_observation"
                ),
                requires_post_read_verification=policy.requires_verification,
                verification_tool=(
                    "library.explorer.category_stats"
                    if policy.target_type == "category"
                    else "library.explorer.document_categories"
                    if policy.target_type == "paper-category relation"
                    else None
                ),
                available_by_default=tool_name != "memory.write",
            )
        return cls._READ_TOOL_SPECS.get(tool_name)

    @staticmethod
    def _normalize_operation_level(value: Any) -> str:
        operation_level = str(value or "").casefold().strip()
        aliases = {
            "none": "query-level",
            "read": "query-level",
            "query": "query-level",
            "entity": "entity-level",
            "relation": "relation-level",
            "document": "entity-level",
            "global": "relation-level",
            "content": "content-level",
        }
        operation_level = aliases.get(operation_level, operation_level)
        allowed = {"query-level", "entity-level", "relation-level", "content-level"}
        return operation_level if operation_level in allowed else "query-level"

    def _operation_level_for_tool(self, tool_name: str) -> str:
        spec = self._spec_for_tool(tool_name)
        if spec is not None:
            return spec.operation_level
        if tool_name in {"library.explorer.find_documents", "evidence.retriever.search_by_category"}:
            return "query-level"
        if tool_name == "library.explorer.document_categories":
            return "query-level"
        return "query-level"

    def _maybe_promote_plan_from_action(
        self,
        plan_state: _PlanState,
        action: _ReactAction,
    ) -> None:
        if plan_state.global_context.get("llm_plan_loaded"):
            return
        raw_steps = [item for item in action.action_plan if isinstance(item, dict)]
        if not raw_steps:
            if not plan_state.steps and action.tool != "final.answer":
                plan_state.steps.append(self._step_from_action(action, 1, source="single_tool_action"))
            return
        llm_steps = [
            self._step_from_plan_item(item, index)
            for index, item in enumerate(raw_steps, start=1)
        ]
        llm_steps = [step for step in llm_steps if step.resolved_tool_name]
        plan_state.global_context["llm_plan_loaded"] = True
        if not llm_steps:
            return
        goal_clause_count = len(self._goal_clauses(plan_state.original_user_prompt))
        if goal_clause_count > len(llm_steps) and self._requires_library_write_observation(plan_state.original_user_prompt):
            plan_state.global_context["llm_plan_undercovered"] = True
            plan_state.global_context["goal_clause_count"] = goal_clause_count
            plan_state.global_context["llm_step_count"] = len(llm_steps)
        if not plan_state.steps or len(llm_steps) >= len(plan_state.steps):
            plan_state.steps = llm_steps
            plan_state.current_step_index = 0
            plan_state.global_context["plan_source"] = "llm_action_plan"
            return
        plan_state.global_context["llm_plan_undercovered"] = True

    def _step_from_plan_item(self, item: dict[str, Any], index: int) -> _StepState:
        tool = str(item.get("tool") or item.get("resolved_tool_name") or "").strip()
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        source_entities = item.get("source_entities") if isinstance(item.get("source_entities"), list) else []
        target_entities = item.get("target_entities") if isinstance(item.get("target_entities"), list) else []
        if not source_entities and isinstance(item.get("source_entity"), dict):
            source_entities = [item["source_entity"]]
        if not target_entities and isinstance(item.get("target_entity"), dict):
            target_entities = [item["target_entity"]]
        depends_on = [
            str(value)
            for value in item.get("depends_on") or []
            if str(value).strip()
        ] if isinstance(item.get("depends_on"), list) else []
        return _StepState(
            step_id=str(item.get("step_id") or f"step-{index}"),
            intent=str(item.get("intent") or item.get("purpose") or item.get("rationale") or tool),
            operation_level=self._normalize_operation_level(item.get("operation_level") or item.get("level") or self._operation_level_for_tool(tool)),
            source_entities=[entity for entity in source_entities if isinstance(entity, dict)],
            target_entities=[entity for entity in target_entities if isinstance(entity, dict)],
            target_scope=str(item.get("target_scope") or item.get("scope") or ""),
            depends_on=depends_on,
            inherit_rules=item.get("inherit_rules") if isinstance(item.get("inherit_rules"), dict) else {},
            risk_level=str(item.get("risk_level") or self._risk_level_for_tool(tool)),
            resolved_tool_name=tool,
            resolved_tool_args=dict(args),
            metadata={"source": "llm_action_plan"},
        )

    def _step_from_action(self, action: _ReactAction, index: int, *, source: str) -> _StepState:
        return _StepState(
            step_id=f"step-{index}",
            intent=action.rationale or action.tool,
            operation_level=self._operation_level_for_tool(action.tool),
            source_entities=self._entities_for_action(action, role="source"),
            target_entities=self._entities_for_action(action, role="target"),
            target_scope=str(action.arguments.get("scope") or ""),
            depends_on=[f"step-{item}" for item in range(1, index)],
            risk_level=self._risk_level_for_tool(action.tool),
            resolved_tool_name=action.tool,
            resolved_tool_args=dict(action.arguments),
            metadata={"source": source},
        )

    @staticmethod
    def _entities_for_action(action: _ReactAction, *, role: str) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        if action.tool == "library.operator.rename_category":
            if role == "source" and action.arguments.get("source_category_name"):
                entities.append({"type": "category", "role": "source", "text": str(action.arguments["source_category_name"])})
            if role == "target" and action.arguments.get("target_category_name"):
                entities.append({"type": "category", "role": "target", "text": str(action.arguments["target_category_name"])})
        elif action.tool in {"library.operator.create_category", "library.operator.assign_category"} and role == "target":
            for name in action.arguments.get("category_names") or [action.arguments.get("category_name")]:
                if str(name or "").strip():
                    entities.append({"type": "category", "role": "target", "text": str(name).strip()})
        elif action.tool in {"library.explorer.find_documents", "evidence.retriever.search_by_category"}:
            if role in {"source", "target"}:
                for name in action.arguments.get("category_names") or [action.arguments.get("category_name")]:
                    if str(name or "").strip():
                        entities.append({"type": "category", "role": "filter", "text": str(name).strip()})
        return entities

    def _plan_payload(self, plan_state: _PlanState) -> dict[str, Any]:
        return {
            "plan_id": plan_state.plan_id,
            "final_status": plan_state.final_status,
            "current_step_index": plan_state.current_step_index,
            "completed_steps": list(plan_state.completed_steps),
            "pending_confirmation_step": plan_state.pending_confirmation_step,
            "failed_steps": list(plan_state.failed_steps),
            "global_context": self._safe_trace_payload(plan_state.global_context),
            "steps": [self._step_payload(step) for step in plan_state.steps],
        }

    def _step_payload(self, step: _StepState) -> dict[str, Any]:
        return {
            "step_id": step.step_id,
            "intent": step.intent,
            "operation_level": step.operation_level,
            "source_entities": self._safe_trace_payload(step.source_entities),
            "target_entities": self._safe_trace_payload(step.target_entities),
            "target_scope": step.target_scope,
            "depends_on": list(step.depends_on),
            "inherit_rules": self._safe_trace_payload(step.inherit_rules),
            "risk_level": step.risk_level,
            "resolved_tool_name": step.resolved_tool_name,
            "resolved_tool_args": self._safe_trace_payload(step.resolved_tool_args),
            "status": step.status,
            "observation_index": step.observation_index,
            "issues": list(step.issues),
            "metadata": self._safe_trace_payload(step.metadata),
        }

    def _bind_action_to_plan_step(
        self,
        plan_state: _PlanState,
        action: _ReactAction,
        observations: list[_ReactObservation],
    ) -> _StepState:
        self._refresh_plan_state_from_observations(plan_state, observations)
        for step in plan_state.steps:
            if step.status in {"completed", "failed", "pending_confirmation", "skipped"}:
                continue
            if not self._step_dependencies_satisfied(step, plan_state):
                continue
            if step.resolved_tool_name == action.tool:
                if self._action_matches_step_arguments(action, step) or not step.resolved_tool_args:
                    step.resolved_tool_name = action.tool
                    if action.arguments:
                        step.resolved_tool_args = dict(action.arguments)
                    return step
        for step in plan_state.steps:
            if step.status == "pending" and self._step_dependencies_satisfied(step, plan_state):
                if not step.resolved_tool_name or step.resolved_tool_name == action.tool:
                    step.resolved_tool_name = action.tool
                    step.resolved_tool_args = dict(action.arguments)
                    return step
        step = self._step_from_action(action, len(plan_state.steps) + 1, source="runtime_action")
        plan_state.steps.append(step)
        return step

    @staticmethod
    def _step_dependencies_satisfied(step: _StepState, plan_state: _PlanState) -> bool:
        if not step.depends_on:
            return True
        completed = set(plan_state.completed_steps)
        return all(str(item) in completed for item in step.depends_on)

    def _action_matches_step_arguments(self, action: _ReactAction, step: _StepState) -> bool:
        if action.tool != step.resolved_tool_name:
            return False
        if action.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            action_names = set(self._category_names_from_arguments(action.arguments))
            step_names = set(self._step_target_category_names(step))
            return not step_names or not action_names or action_names == step_names
        if action.tool == "library.operator.rename_category":
            action_source = str(action.arguments.get("source_category_name") or "")
            action_target = str(action.arguments.get("target_category_name") or "")
            step_source = self._step_source_category_name(step)
            step_target = self._step_target_category_name(step)
            return (not step_source or not action_source or step_source == action_source) and (
                not step_target or not action_target or step_target == action_target
            )
        if action.tool == "library.operator.clear_categories":
            action_category = str(action.arguments.get("category_name") or "")
            step_category = self._step_target_category_name(step) or self._step_source_category_name(step)
            return not step_category or not action_category or step_category == action_category
        return True

    def _apply_step_argument_isolation(
        self,
        plan_state: _PlanState,
        step: _StepState,
        action: _ReactAction,
    ) -> tuple[_ReactAction, str | None]:
        arguments = dict(action.arguments)
        if action.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            step_names = self._step_target_category_names(step)
            action_names = self._category_names_from_arguments(arguments)
            if step_names and action_names and set(step_names) != set(action_names):
                return action, "Step tool arguments do not match the step target labels; execution was blocked."
            if step_names and not action_names:
                arguments["category_name"] = step_names[0]
                arguments["category_names"] = step_names
            if len(plan_state.steps) > 1 and not self._category_names_from_arguments(arguments):
                return action, "Compound write step lacks an isolated target label; execution was blocked."
        if action.tool == "library.operator.rename_category":
            step_source = self._step_source_category_name(step)
            step_target = self._step_target_category_name(step)
            action_source = self._clean_category_name(str(arguments.get("source_category_name") or ""))
            action_target = self._clean_category_name(str(arguments.get("target_category_name") or ""))
            if step_source and action_source and step_source != action_source:
                return action, "Rename source label does not match the current step; execution was blocked."
            if step_target and action_target and step_target != action_target:
                return action, "Rename target label does not match the current step; execution was blocked."
            if step_source and not action_source:
                arguments["source_category_name"] = step_source
            if step_target and not action_target:
                arguments["target_category_name"] = step_target
        if action.tool == "library.operator.clear_categories":
            step_category = self._step_target_category_name(step) or self._step_source_category_name(step)
            action_category = self._clean_category_name(str(arguments.get("category_name") or ""))
            if step_category and action_category and step_category != action_category:
                return action, "Clear-category target does not match the current step; execution was blocked."
            if step_category and not action_category and str(arguments.get("operation") or "") == "remove_single_category_link":
                arguments["category_name"] = step_category
        return _ReactAction(
            tool=action.tool,
            arguments=arguments,
            rationale=action.rationale,
            task_intent=action.task_intent,
            action_plan=action.action_plan,
            confidence=action.confidence,
        ), None

    @staticmethod
    def _step_entity_names(step: _StepState, *, role: str | None = None) -> list[str]:
        entities = [*step.source_entities, *step.target_entities]
        names: list[str] = []
        for entity in entities:
            if role and str(entity.get("role") or "").casefold() != role:
                continue
            entity_type = str(entity.get("type") or entity.get("entity_type") or "").casefold()
            if entity_type and entity_type not in {"tag", "category", "label", "collection", "unknown"}:
                continue
            text = str(entity.get("text") or entity.get("name") or "").strip()
            if text and text not in names:
                names.append(text)
        return names

    def _step_target_category_names(self, step: _StepState) -> list[str]:
        names = self._step_entity_names(step, role="target")
        if not names:
            names = [
                self._clean_category_name(str(item))
                for item in step.resolved_tool_args.get("category_names") or []
                if self._clean_category_name(str(item))
            ]
        if not names and step.resolved_tool_args.get("category_name"):
            names = [self._clean_category_name(str(step.resolved_tool_args.get("category_name")))]
        if not names and step.resolved_tool_args.get("target_category_name"):
            names = [self._clean_category_name(str(step.resolved_tool_args.get("target_category_name")))]
        return [name for name in names if name]

    def _step_target_category_name(self, step: _StepState) -> str | None:
        names = self._step_target_category_names(step)
        return names[0] if names else None

    def _step_source_category_name(self, step: _StepState) -> str | None:
        names = self._step_entity_names(step, role="source")
        if not names and step.resolved_tool_args.get("source_category_name"):
            names = [self._clean_category_name(str(step.resolved_tool_args.get("source_category_name")))]
        if not names and step.resolved_tool_args.get("category_name"):
            names = [self._clean_category_name(str(step.resolved_tool_args.get("category_name")))]
        return names[0] if names else None

    def _safe_execute_plan(
        self,
        *,
        run_id: str,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
        plan_state: _PlanState,
        step_state: _StepState,
    ) -> tuple[_ReactAction, _ReactObservation]:
        step_state.status = "in_progress"
        plan_state.current_step_index = max(0, plan_state.steps.index(step_state))
        isolated_action, isolation_error = self._apply_step_argument_isolation(plan_state, step_state, action)
        if isolation_error:
            observation = _ReactObservation(
                tool=action.tool,
                status="validation_failed",
                summary=isolation_error,
                payload={
                    "tool": action.tool,
                    "plan_id": plan_state.plan_id,
                    "step_id": step_state.step_id,
                    "library_mutated": False,
                },
            )
            observation = self._finalize_tool_observation(observation)
            self._update_plan_step_after_observation(plan_state, step_state, observation)
            return action, observation
        step_state.resolved_tool_name = isolated_action.tool
        step_state.resolved_tool_args = dict(isolated_action.arguments)
        resolved_action = self._resolve_action_intent(
            session=session,
            content=content,
            action=isolated_action,
            selected_document_ids=self._selected_document_ids_from_plan(plan_state),
            attachments=self._attachments_from_plan(plan_state),
            observations=observations,
        )
        if resolved_action is not None:
            isolated_action.arguments["_resolved_action"] = resolved_action.as_payload()
            step_state.resolved_tool_args = dict(isolated_action.arguments)
        if (
            isolated_action.tool == "library.operator.rename_category"
            and self._is_document_label_relation_assignment(content)
        ):
            observation = _ReactObservation(
                tool=isolated_action.tool,
                status="validation_failed",
                summary="这类请求是修改论文与标签的关系，不是重命名全局标签实体；已拦截 rename_category。",
                payload={
                    "tool": isolated_action.tool,
                    "plan_id": plan_state.plan_id,
                    "step_id": step_state.step_id,
                    "_resolved_action": isolated_action.arguments.get("_resolved_action"),
                    "library_mutated": False,
                },
            )
            observation = self._finalize_tool_observation(observation)
            self._update_plan_step_after_observation(plan_state, step_state, observation)
            return isolated_action, observation
        validation_error = self._validate_react_action(
            isolated_action,
            observations,
            content=content,
            plan_step=step_state,
            plan_state=plan_state,
        )
        if validation_error:
            observation = _ReactObservation(
                tool=isolated_action.tool,
                status="validation_failed",
                summary=validation_error,
                payload={
                    "tool": isolated_action.tool,
                    "plan_id": plan_state.plan_id,
                    "step_id": step_state.step_id,
                    "library_mutated": False,
                },
            )
        else:
            observation = self._execute_react_action(
                run_id=run_id,
                session=session,
                content=content,
                action=isolated_action,
                observations=observations,
                plan_state=plan_state,
                step_state=step_state,
            )
        observation = self._finalize_tool_observation(observation)
        self._update_plan_step_after_observation(plan_state, step_state, observation)
        return isolated_action, observation

    @staticmethod
    def _selected_document_ids_from_plan(plan_state: _PlanState) -> list[str]:
        ids = plan_state.global_context.get("selected_document_ids")
        return [str(item) for item in ids if item] if isinstance(ids, list) else []

    @staticmethod
    def _attachments_from_plan(plan_state: _PlanState) -> list[ChatAttachment]:
        ids = plan_state.global_context.get("attachment_document_ids")
        if not isinstance(ids, list):
            return []
        return [
            ChatAttachment(id=str(item), kind="library_document", display_name=str(item), document_id=str(item))
            for item in ids
            if item
        ]

    def _finalize_tool_observation(self, observation: _ReactObservation) -> _ReactObservation:
        if observation.observation:
            return observation
        if observation.tool == "library.operator.assign_category":
            observation.payload.setdefault("category_names", [])
            observation.payload.setdefault("category_name", "")
        standard = self._build_tool_observation(observation)
        observation.observation = standard.model_dump(mode="json")
        return observation

    def _build_tool_observation(self, observation: _ReactObservation) -> ToolObservation:
        spec = self._spec_for_tool(observation.tool)
        if spec is None:
            spec = ToolSpec(
                name=observation.tool,
                display_name=observation.tool,
                description=observation.tool,
                scope="experimental",
                operation_level="query-level",
                io_type="read",
                write_type="none",
                input_object_types=[],
                output_observation_type="tool_observation",
                available_by_default=False,
            )
        success = observation.status == "completed" and not bool(observation.payload.get("verification_error"))
        return ToolObservation(
            tool_name=observation.tool,
            success=success,
            operation_level=spec.operation_level,
            io_type=spec.io_type,
            write_type=spec.write_type,
            target_objects=self._observation_target_objects(spec, observation.payload),
            affected_objects=self._observation_affected_objects(observation.tool, observation.payload),
            counts=self._observation_counts(observation.tool, observation.payload),
            data=self._observation_data(observation.tool, observation.payload),
            evidence=self._observation_evidence(observation.payload),
            requires_followup=observation.status in {"needs_clarification", "confirmation_required", "degraded"},
            requires_confirmation=(
                spec.requires_confirmation
                or observation.status == "confirmation_required"
                or bool(observation.payload.get("requires_confirmation"))
            ),
            verification=self._observation_verification(spec, observation),
            error=None if success else self._observation_error(observation),
            message=observation.summary,
        )

    @staticmethod
    def _react_observation_payload(observation: _ReactObservation) -> dict[str, Any]:
        """Return a payload-compatible view with standard data layered on top."""

        payload_view = dict(observation.payload or {})
        standard = observation.observation if isinstance(observation.observation, dict) else {}
        data = standard.get("data")
        if isinstance(data, dict):
            payload_view.update(data)
        return payload_view

    @staticmethod
    def _react_observation_evidence(observation: _ReactObservation) -> list[Any]:
        standard = observation.observation if isinstance(observation.observation, dict) else {}
        evidence = standard.get("evidence")
        if isinstance(evidence, list) and evidence:
            return evidence
        payload_evidence = observation.payload.get("evidence_items")
        return payload_evidence if isinstance(payload_evidence, list) else []

    @staticmethod
    def _react_observation_verification(observation: _ReactObservation) -> dict[str, Any] | None:
        standard = observation.observation if isinstance(observation.observation, dict) else {}
        verification = standard.get("verification")
        if isinstance(verification, dict):
            verification_view = dict(verification)
        else:
            payload_verification = observation.payload.get("verification")
            verification_view = dict(payload_verification) if isinstance(payload_verification, dict) else {}

        for key in ("verified_state", "verification_error", "rollback_error"):
            if key in observation.payload:
                verification_view.setdefault(key, observation.payload[key])
        return verification_view or None

    @staticmethod
    def _observation_error(observation: _ReactObservation) -> ToolObservationError | None:
        if observation.status == "completed":
            return None
        code_by_status = {
            "needs_clarification": "AMBIGUOUS_TARGET",
            "confirmation_required": "CONFIRMATION_REQUIRED",
            "validation_failed": "VALIDATION_FAILED",
            "failed": "VERIFICATION_FAILED" if observation.payload.get("verification_error") else "TOOL_UNAVAILABLE",
        }
        return ToolObservationError(
            code=code_by_status.get(observation.status, "TOOL_UNAVAILABLE"),
            message=observation.summary,
            recoverable=observation.status != "failed",
            suggested_next_action=(
                "ask_user_to_confirm"
                if observation.status == "confirmation_required"
                else "ask_user_to_clarify"
                if observation.status in {"needs_clarification", "validation_failed"}
                else "manual_check_required"
            ),
        )

    @staticmethod
    def _observation_verification(spec: ToolSpec, observation: _ReactObservation) -> ToolVerification | None:
        if (
            not spec.requires_post_read_verification
            and "verified_state" not in observation.payload
            and "verification_error" not in observation.payload
        ):
            return None
        error = observation.payload.get("verification_error")
        details: dict[str, Any] = {}
        if "verified_state" in observation.payload:
            details["verified_state"] = observation.payload.get("verified_state")
        if error:
            details["error"] = error
        return ToolVerification(
            performed=True,
            success=not bool(error) and observation.status == "completed",
            method="post_read",
            details=details,
        )

    @staticmethod
    def _observation_target_objects(spec: ToolSpec, payload: dict[str, Any]) -> list[dict[str, Any]]:
        targets = payload.get("targets") or payload.get("affected_entities") or []
        if isinstance(targets, list) and targets:
            return [item for item in targets if isinstance(item, dict)]
        if payload.get("document_ids"):
            return [{"type": "paper", "id": str(item)} for item in payload.get("document_ids") or [] if item]
        if payload.get("category_names"):
            return [{"type": "category", "name": str(item)} for item in payload.get("category_names") or [] if item]
        if payload.get("category_name"):
            return [{"type": "category", "name": str(payload.get("category_name"))}]
        return [{"type": item} for item in spec.input_object_types]

    @staticmethod
    def _observation_affected_objects(tool_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if tool_name == "library.operator.delete_unused_categories":
            return [
                {"type": "category", "id": str(category_id), "name": str(name)}
                for category_id, name in zip(payload.get("deleted_category_ids") or [], payload.get("deleted_category_names") or [])
            ]
        if tool_name == "library.operator.clear_categories":
            return [
                {"type": "paper-category relation", "paper_id": str(document_id)}
                for document_id in payload.get("affected_document_ids") or payload.get("document_ids") or []
            ]
        if tool_name == "library.operator.assign_category":
            return [
                {"type": "paper-category relation", "paper_id": str(document_id)}
                for document_id in payload.get("document_ids") or []
            ]
        if tool_name == "library.operator.create_category":
            return [
                {"type": "category", "id": str(item.get("id")), "name": str(item.get("name"))}
                for item in payload.get("categories") or []
                if isinstance(item, dict)
            ]
        return []

    @staticmethod
    def _observation_counts(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        counts: dict[str, Any] = {}
        if "category_count" in payload:
            categories = [item for item in payload.get("categories") or [] if isinstance(item, dict)]
            counts.update(
                {
                    "total": payload.get("category_count"),
                    "empty_count": sum(1 for item in categories if int(item.get("document_count") or item.get("paper_count") or 0) == 0),
                    "tagged_documents": payload.get("tagged_document_count"),
                    "untagged_documents": payload.get("untagged_document_count"),
                }
            )
        if "deleted_count" in payload:
            counts["deleted_categories"] = payload.get("deleted_count")
            counts.setdefault("modified_relations", payload.get("affected_document_count", 0))
        if "updated_count" in payload:
            counts["updated_relations"] = payload.get("updated_count")
        if "evidence_items" in payload:
            counts["evidence_items"] = len(payload.get("evidence_items") or [])
        return counts

    @staticmethod
    def _observation_data(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        if tool_name == "library.explorer.category_stats":
            categories = []
            for item in payload.get("categories") or []:
                if not isinstance(item, dict):
                    continue
                paper_count = int(item.get("paper_count") or item.get("document_count") or 0)
                categories.append({**item, "paper_count": paper_count})
            data["categories"] = categories
            data["total"] = payload.get("category_count", len(categories))
            data["empty_count"] = sum(1 for item in categories if int(item.get("paper_count") or 0) == 0)
        return data

    @staticmethod
    def _observation_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = []
        for item in payload.get("evidence_items") or []:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            evidence.append(
                {
                    "paper_id": item.get("document_id") or item.get("source_id") or metadata.get("document_id"),
                    "paper_title": item.get("title") or metadata.get("filename"),
                    "chunk_id": item.get("evidence_id") or item.get("id") or metadata.get("chunk_id"),
                    "page": item.get("page_number"),
                    "score": item.get("rerank_score") if item.get("rerank_score") is not None else item.get("score"),
                    "text": item.get("quote") or item.get("snippet") or "",
                    "source_type": "local_paper" if str(item.get("source_type")) == "local_document" else item.get("source_type"),
                }
            )
        return evidence

    def _update_plan_step_after_observation(
        self,
        plan_state: _PlanState,
        step: _StepState,
        observation: _ReactObservation,
    ) -> None:
        step.execution_result = self._safe_trace_payload(observation.payload)
        step.observation_index = step.observation_index if step.observation_index is not None else -1
        if observation.status == "completed":
            step.status = "completed"
            if step.step_id not in plan_state.completed_steps:
                plan_state.completed_steps.append(step.step_id)
            if step.step_id in plan_state.failed_steps:
                plan_state.failed_steps.remove(step.step_id)
        elif observation.status == "confirmation_required":
            step.status = "pending_confirmation"
            plan_state.pending_confirmation_step = step.step_id
            step.preview_result = self._safe_trace_payload(observation.payload)
        elif observation.status in {"failed", "validation_failed", "degraded", "needs_clarification"}:
            step.status = "failed"
            step.issues.append(observation.summary)
            if step.step_id not in plan_state.failed_steps:
                plan_state.failed_steps.append(step.step_id)
        else:
            step.status = observation.status or "pending"
        if observation.payload.get("verification_error"):
            step.verification_result = {
                "status": "failed",
                "error": observation.payload.get("verification_error"),
            }
        elif observation.payload.get("verified_state") or observation.status == "completed":
            step.verification_result = {"status": "verified" if observation.status == "completed" else observation.status}
        self._refresh_plan_state_from_observations(plan_state, [])

    def _refresh_plan_state_from_observations(
        self,
        plan_state: _PlanState,
        observations: list[_ReactObservation],
    ) -> None:
        for index, observation in enumerate(observations):
            for step in plan_state.steps:
                if step.status == "completed":
                    continue
                if self._observation_matches_step(step, observation):
                    step.status = "completed"
                    step.observation_index = index
                    step.execution_result = self._safe_trace_payload(observation.payload)
                    if step.step_id not in plan_state.completed_steps:
                        plan_state.completed_steps.append(step.step_id)
        if plan_state.failed_steps:
            plan_state.final_status = "failed"
        elif plan_state.pending_confirmation_step:
            plan_state.final_status = "pending_confirmation"
        elif plan_state.steps and all(step.status == "completed" for step in plan_state.steps):
            plan_state.final_status = "completed"
        elif plan_state.steps:
            plan_state.final_status = "pending"
        else:
            plan_state.final_status = "completed"

    def _observation_matches_step(self, step: _StepState, observation: _ReactObservation) -> bool:
        obligation_key = str(step.metadata.get("obligation_key") or "")
        if observation.status == "completed" and obligation_key == "library_stats" and observation.tool == "library.explorer.category_stats":
            return True
        if observation.status == "completed" and obligation_key == "create_category_verified" and observation.tool in {
            "library.operator.create_category",
            "library.operator.assign_category",
        }:
            expected = set(self._step_target_category_names(step))
            observed = set(str(item) for item in observation.payload.get("category_names") or [] if item)
            if observation.payload.get("category_name"):
                observed.add(str(observation.payload.get("category_name")))
            return not expected or expected.issubset(observed)
        if observation.status == "completed" and obligation_key == "post_write_category_stats" and observation.tool in {
            "library.explorer.category_stats",
            "evidence.retriever.search_by_category",
            "report.drafter.write_by_category",
        }:
            return True
        if observation.status == "completed" and obligation_key == "document_category_details" and observation.tool in {
            "library.explorer.category_stats",
            "evidence.retriever.search_by_category",
            "report.drafter.write_by_category",
        }:
            return True
        if observation.status == "completed" and obligation_key == "report" and observation.tool == "evidence.retriever.search":
            return False
        if observation.status != "completed" or observation.tool != step.resolved_tool_name:
            return False
        if observation.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            expected = set(self._step_target_category_names(step))
            observed = set(str(item) for item in observation.payload.get("category_names") or [] if item)
            if observation.payload.get("category_name"):
                observed.add(str(observation.payload.get("category_name")))
            return not expected or expected.issubset(observed)
        if observation.tool == "library.operator.rename_category":
            source = self._step_source_category_name(step)
            target = self._step_target_category_name(step)
            return (
                (not source or source == observation.payload.get("source_category_name"))
                and (not target or target == observation.payload.get("target_category_name"))
            )
        if observation.tool == "library.operator.clear_categories":
            expected = self._step_target_category_name(step) or self._step_source_category_name(step)
            observed = observation.payload.get("category_name")
            return not expected or expected == observed
        return True

    def _next_plan_repair_action(
        self,
        *,
        plan_state: _PlanState,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        self._refresh_plan_state_from_observations(plan_state, observations)
        for step in plan_state.steps:
            if step.status in {"completed", "failed", "pending_confirmation", "skipped"}:
                continue
            if not self._step_dependencies_satisfied(step, plan_state):
                continue
            action = self._action_for_plan_step(
                step=step,
                session=session,
                content=content,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
            )
            if action is not None:
                return action
        return None

    def _action_for_plan_step(
        self,
        *,
        step: _StepState,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        if step.metadata.get("source") == "answer_obligation":
            obligation = _AnswerObligation(
                key=str(step.metadata.get("obligation_key") or ""),
                description=step.intent,
                required_tools=tuple(str(item) for item in step.metadata.get("required_tools") or []),
                target=dict(step.metadata.get("target") or {}),
            )
            if (
                obligation.key == "resolved_document_set"
                and self._is_assignment_intent(content)
                and (
                    selected_document_ids
                    or any(attachment.document_id for attachment in attachments)
                    or self._mentions_previous_referent(content)
                    or self._mentions_all_library(content)
                )
            ):
                step.status = "skipped"
                return None
            return self._action_for_unmet_obligation(
                session=session,
                content=content,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
                obligations=[obligation],
            )
        if step.resolved_tool_name and step.resolved_tool_name != "final.answer":
            return _ReactAction(
                tool=step.resolved_tool_name,
                arguments=dict(step.resolved_tool_args),
                rationale=step.intent,
            )
        return None

    def _plan_completion_check(
        self,
        plan_state: _PlanState,
        observations: list[_ReactObservation],
    ) -> dict[str, Any]:
        self._refresh_plan_state_from_observations(plan_state, observations)
        pending = [
            step.step_id
            for step in plan_state.steps
            if step.status not in {"completed", "skipped"}
        ]
        failed = [step.step_id for step in plan_state.steps if step.status == "failed"]
        pending_confirmation = [
            step.step_id for step in plan_state.steps if step.status == "pending_confirmation"
        ]
        expected_goal_count = int(plan_state.global_context.get("goal_clause_count") or 0)
        uncovered = bool(plan_state.global_context.get("llm_plan_undercovered")) and (
            expected_goal_count <= 0 or len(plan_state.steps) < expected_goal_count
        )
        has_final_report = any(
            observation.status == "completed"
            and observation.tool in {"report.drafter.write", "report.drafter.write_by_category"}
            for observation in observations
        )
        has_pending_report = any(
            step.metadata.get("obligation_key") == "report"
            and step.status not in {"completed", "skipped"}
            for step in plan_state.steps
        )
        has_evidence = any(
            observation.status == "completed"
            and observation.tool == "evidence.retriever.search"
            and observation.payload.get("evidence_items")
            for observation in observations
        )
        if has_pending_report and has_evidence and not has_final_report:
            pending = list(dict.fromkeys([*pending, *[
                step.step_id
                for step in plan_state.steps
                if step.metadata.get("obligation_key") == "report"
            ]]))
        if has_final_report and not failed and not pending_confirmation:
            pending_steps = [step for step in plan_state.steps if step.step_id in pending]
            if pending_steps and all(step.risk_level == "read_only" for step in pending_steps):
                pending = []
                uncovered = False
        completed = not pending and not failed and not uncovered
        return {
            "completed": completed,
            "pending_steps": pending,
            "failed_steps": failed,
            "pending_confirmation_steps": pending_confirmation,
            "uncovered_subgoals": uncovered,
            "plan": self._plan_payload(plan_state),
        }

    def _enforce_plan_completion_gate(
        self,
        plan_state: _PlanState,
        observations: list[_ReactObservation],
        final_text: str,
        final_status: str,
    ) -> tuple[str, str, dict[str, Any]]:
        check = self._plan_completion_check(plan_state, observations)
        if check["completed"]:
            return final_text, final_status, check
        if final_status in {"confirmation_required", "needs_clarification", "validation_failed", "failed", "degraded"}:
            return final_text, final_status, check
        if any(
            observation.tool in {"report.drafter.write", "report.drafter.write_by_category"}
            and observation.status == "completed"
            and observation.payload.get("answer")
            for observation in observations
        ):
            return final_text, final_status, check
        if check["pending_confirmation_steps"]:
            return self._compose_partial_plan_answer(plan_state, observations), "confirmation_required", check
        if check["failed_steps"]:
            return self._compose_partial_plan_answer(plan_state, observations), "failed", check
        return self._compose_partial_plan_answer(plan_state, observations), "validation_failed", check

    def _compose_partial_plan_answer(
        self,
        plan_state: _PlanState,
        observations: list[_ReactObservation],
    ) -> str:
        lines = ["任务没有整体完成，已阻止把部分结果当作成功回复。"]
        if plan_state.global_context.get("llm_plan_undercovered"):
            lines.append("当前执行计划没有覆盖原始请求中的全部子目标，需要重新规划或补充步骤。")
        completed = self._completed_write_step_summaries(observations)
        for index, summary in enumerate(completed, start=1):
            lines.append(f"已完成第 {index} 步：{summary}")
        for step in plan_state.steps:
            if step.status == "pending_confirmation":
                detail = step.preview_result.get("confirmation_phrase") or step.intent
                lines.append(f"等待确认：{detail}")
            elif step.status == "failed":
                detail = step.issues[-1] if step.issues else step.intent
                lines.append(f"失败步骤 {step.step_id}：{detail}")
            elif step.status == "pending":
                lines.append(f"未执行步骤 {step.step_id}：{step.intent}")
        return "\n".join(lines)

    def _obligations_payload(
        self,
        obligations: list[_AnswerObligation],
        observations: list[_ReactObservation],
    ) -> list[dict[str, Any]]:
        return [
            {
                "key": obligation.key,
                "description": obligation.description,
                "required_tools": list(obligation.required_tools),
                "target": obligation.target,
                "status": "satisfied" if self._obligation_satisfied(obligation, observations) else "pending",
            }
            for obligation in obligations
        ]

    def _plan_completion_payload(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> dict[str, Any]:
        obligations = self._answer_obligations(content, selected_document_ids, attachments)
        statuses = self._obligations_payload(obligations, observations)
        return {
            "obligations": statuses,
            "completed": [item["key"] for item in statuses if item["status"] == "satisfied"],
            "pending": [item["key"] for item in statuses if item["status"] != "satisfied"],
            "last_observation_status": observations[-1].status if observations else "none",
        }

    def _unmet_obligations(
        self,
        obligations: list[_AnswerObligation],
        observations: list[_ReactObservation],
    ) -> list[_AnswerObligation]:
        return [
            obligation
            for obligation in obligations
            if not self._obligation_satisfied(obligation, observations)
        ]

    def _obligation_satisfied(
        self,
        obligation: _AnswerObligation,
        observations: list[_ReactObservation],
    ) -> bool:
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if obligation.key == "document_category_details":
            return self._has_document_category_observation(observations)
        if obligation.key == "labeled_document_set":
            return any(
                observation.tool == "library.explorer.find_documents"
                and observation.status in {"completed", "needs_clarification"}
                and (
                    observation.payload.get("category_name")
                    or observation.payload.get("category_names")
                    or observation.payload.get("category_lookup")
                )
                for observation in observations
            )
        if obligation.key == "create_category_verified":
            category_names = [
                str(item)
                for item in obligation.target.get("category_names") or []
                if str(item).strip()
            ]
            if not category_names:
                category_name = self._extract_category_name_from_request(
                    str(obligation.target.get("content") or "")
                )
                category_names = [category_name] if category_name else []
            if not category_names:
                return False
            observed_names: set[str] = set()
            for observation in observations:
                if observation.status != "completed":
                    continue
                if observation.tool == "library.operator.create_category":
                    observed_names.update(str(item) for item in observation.payload.get("category_names") or [] if item)
                    category_name = observation.payload.get("category_name")
                    if isinstance(category_name, str) and category_name:
                        observed_names.add(category_name)
                if observation.tool == "library.operator.assign_category" and observation.payload.get("category_name"):
                    observed_names.add(str(observation.payload.get("category_name")))
                if observation.tool == "library.operator.assign_category":
                    observed_names.update(str(item) for item in observation.payload.get("category_names") or [] if item)
            return all(name in observed_names or self._category_exists(name) for name in category_names)
        if obligation.key == "assign_category_verified":
            category_names = [
                str(item)
                for item in obligation.target.get("category_names") or []
                if str(item).strip()
            ]
            if not category_names:
                return any(tool in completed_tools for tool in obligation.required_tools)
            assigned_names: set[str] = set()
            for observation in observations:
                if observation.status != "completed" or observation.tool != "library.operator.assign_category":
                    continue
                assigned_names.update(str(item) for item in observation.payload.get("category_names") or [] if item)
                category_name = observation.payload.get("category_name")
                if isinstance(category_name, str) and category_name:
                    assigned_names.add(category_name)
            return all(name in assigned_names for name in category_names)
        if obligation.key.endswith("_verified"):
            return any(tool in completed_tools for tool in obligation.required_tools)
        if obligation.key == "report":
            report = self._latest_payload(observations, "report.drafter.write")
            return bool(report and report.get("answer"))
        if obligation.key == "category_report":
            report = self._latest_payload(observations, "report.drafter.write_by_category")
            return bool(report and report.get("answer"))
        if obligation.key == "post_write_category_stats":
            return "library.explorer.category_stats" in completed_tools
        return any(tool in completed_tools for tool in obligation.required_tools)

    def _action_for_unmet_obligation(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
        obligations: list[_AnswerObligation],
    ) -> _ReactAction | None:
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        for obligation in obligations:
            if (
                obligation.key == "resolved_document_set"
                and self._is_assignment_intent(content)
                and (
                    selected_document_ids
                    or any(attachment.document_id for attachment in attachments)
                    or self._mentions_previous_referent(content)
                    or self._mentions_all_library(content)
                )
            ):
                continue
            if obligation.key == "library_stats":
                return _ReactAction("library.explorer.stats", {}, "Read library statistics for an unmet answer obligation.")
            if obligation.key == "category_stats":
                return _ReactAction("library.explorer.category_stats", {}, "Read category statistics for an unmet answer obligation.")
            if obligation.key == "delete_unused_categories_verified":
                return _ReactAction(
                    "library.operator.delete_unused_categories",
                    {"selector": "unused"},
                    "Preview and, after confirmation, delete unused tag/category entities only.",
                )
            if obligation.key == "document_metadata_details":
                document_ids = selected_document_ids or self._document_ids_from_observations(observations)
                if not document_ids:
                    document_ids = self._document_category_target_ids(content, observations)
                return _ReactAction(
                    "library.explorer.document_metadata",
                    {
                        "document_ids": document_ids,
                        "requested_fields": obligation.target.get("requested_fields") or [],
                    },
                    "Read document metadata fields for an unmet answer obligation.",
                )
            if obligation.key == "document_category_details":
                document_ids = selected_document_ids or self._document_ids_from_observations(observations)
                if not document_ids:
                    document_ids = self._document_category_target_ids(content, observations)
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": document_ids},
                    "Read per-document category links for an unmet answer obligation.",
                )
            if obligation.key == "resolved_document_set":
                category_names = [
                    str(item)
                    for item in obligation.target.get("category_names") or []
                    if str(item).strip()
                ]
                category_name = category_names[0] if category_names else self._category_name_from_request_or_observations(content, observations)
                arguments: dict[str, Any] = {"query": content, "expected": "many"}
                if category_names:
                    arguments["category_names"] = category_names
                if category_name and self._category_exists(category_name):
                    arguments["category_name"] = category_name
                elif self._is_assignment_intent(content) and not self._extract_document_tokens(content):
                    return None
                return _ReactAction(
                    "library.explorer.find_documents",
                    arguments,
                    "Resolve the paper set required by the remaining task.",
                )
            if obligation.key == "labeled_document_set":
                category_names = [
                    str(item)
                    for item in obligation.target.get("category_names") or self._category_entity_names_for_request(content)
                    if str(item).strip()
                ]
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"category_names": category_names, "expected": "many"},
                    "Resolve labeled/category document collection before answering.",
                )
            if obligation.key == "create_category_verified":
                category_names = [
                    str(item)
                    for item in obligation.target.get("category_names") or self._extract_category_names_from_request(content)
                    if str(item).strip()
                ]
                category_name = category_names[0] if category_names else self._extract_category_name_from_request(content) or ""
                return _ReactAction(
                    "library.operator.create_category",
                    {"category_name": category_name, "category_names": category_names},
                    "Create the requested category before answering.",
                )
            if obligation.key == "assign_category_verified":
                if (
                    not self._needs_untagged_assignment(content)
                    and not selected_document_ids
                    and not any(attachment.document_id for attachment in attachments)
                    and not self._mentions_previous_referent(content)
                    and not self._mentions_single_current_document(content)
                    and not self._mentions_plural_current_documents(content)
                    and not self._mentions_all_library(content)
                ):
                    document_ids = self._document_ids_from_observations(observations)
                    if not document_ids:
                        return None
                if "library.operator.create_category" in completed_tools and self._needs_untagged_assignment(content):
                    category_names = self._extract_category_names_from_request(content) or self._category_names_from_observations(observations)
                    category_name = category_names[0] if category_names else self._extract_category_name_from_request(content) or self._category_name_from_observations(observations) or ""
                    return _ReactAction(
                        "library.operator.assign_category",
                        {"category_name": category_name, "category_names": category_names, "scope": "untagged"},
                        "Assign the created category to the requested untagged documents.",
                    )
                can_target_untagged_from_context = self._should_target_untagged_from_context(
                    content,
                    self._read_react_state(session.id),
                )
                if (
                    not self._needs_untagged_assignment(content)
                    and not can_target_untagged_from_context
                    and not selected_document_ids
                    and not any(attachment.document_id for attachment in attachments)
                    and not self._mentions_previous_referent(content)
                    and not self._mentions_single_current_document(content)
                    and not self._mentions_plural_current_documents(content)
                    and not self._mentions_all_library(content)
                    and not self._document_ids_from_observations(observations)
                ):
                    return None
                forced_action = self._force_llm_write_tool_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
                if forced_action is not None:
                    return forced_action
                return self._fallback_next_react_action(
                    session=session,
                    content=content,
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                )
            if obligation.key == "rename_category_verified":
                rename_pair = self._extract_category_rename_request(content)
                return _ReactAction(
                    "library.operator.rename_category",
                    {
                        "source_category_name": rename_pair[0] if rename_pair else "",
                        "target_category_name": rename_pair[1] if rename_pair else "",
                    },
                    "Execute the requested category rename/merge before answering.",
                )
            if obligation.key == "post_write_category_stats":
                return _ReactAction(
                    "library.explorer.category_stats",
                    {},
                    "Read category statistics after the verified write step.",
                )
            if obligation.key == "clear_categories_verified":
                arguments = self._clear_categories_arguments_for_request(
                    content,
                    selected_document_ids=selected_document_ids,
                    observations=observations,
                )
                return _ReactAction(
                    "library.operator.clear_categories",
                    arguments,
                    "Execute the requested category clearing before answering.",
                )
            if obligation.key == "category_evidence":
                category_names = [
                    str(item)
                    for item in obligation.target.get("category_names") or []
                    if str(item).strip()
                ]
                category_name = self._category_name_from_request_or_observations(content, observations)
                return _ReactAction(
                    "evidence.retriever.search_by_category",
                    {"question": content, "category_names": category_names or ([category_name] if category_name else [])},
                    "Retrieve grouped evidence for the requested category summary.",
                )
            if obligation.key == "category_report":
                return _ReactAction(
                    "report.drafter.write_by_category",
                    {"question": content},
                    "Draft the category-grouped answer from retrieved observations.",
                )
            if obligation.key == "evidence_retrieval":
                document_ids = selected_document_ids or self._document_ids_from_observations(observations)
                if not document_ids:
                    category_name = self._category_name_from_request_or_observations(content, observations)
                    if category_name and self._category_exists(category_name):
                        return _ReactAction(
                            "library.explorer.find_documents",
                            {"category_name": category_name, "expected": "many"},
                            "Resolve category documents before evidence retrieval.",
                        )
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"query": content, "expected": "many"},
                        "Resolve documents before evidence retrieval.",
                    )
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": document_ids},
                    "Retrieve evidence for the requested report.",
                )
            if obligation.key == "report":
                return _ReactAction(
                    "report.drafter.write",
                    {"question": content, "document_ids": self._document_ids_from_observations(observations)},
                    "Draft the final answer from retrieved evidence.",
                )
        return None

    def _fallback_next_react_action(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ReactAction:
        completed_tools = [observation.tool for observation in observations if observation.status == "completed"]
        document_ids = self._document_ids_from_observations(observations) or selected_document_ids
        category_name = self._category_name_from_request_or_observations(content, observations)
        category_names = self._extract_category_names_from_request(content) or self._category_names_from_observations(observations)
        assignment_category_name = self._extract_category_name_from_request(content) or category_name
        rename_pair = self._extract_category_rename_request(content)
        referent_state = self._read_react_state(session.id)
        referent_ids = self._state_document_ids(referent_state)

        if not observations:
            if self._is_assignment_intent(content):
                if self._needs_untagged_assignment(content):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "scope": "untagged",
                        },
                        "给当前无标签论文追加指定标签。",
                    )
                if self._should_target_untagged_from_context(content, referent_state):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "scope": "untagged",
                        },
                        "根据上下文指代定位到剩余未打标签论文，并追加指定标签。",
                    )
                scoped_referent_ids = self._resolve_document_scope(
                    session=session,
                    content=content,
                    action_arguments={},
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                    allow_all_library=False,
                )[1]
                if (
                    self._mentions_previous_referent(content)
                    or self._mentions_single_current_document(content)
                    or self._mentions_plural_current_documents(content)
                ) and scoped_referent_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "document_ids": scoped_referent_ids,
                            "scope": "documents",
                        },
                        "把上一轮对话指代的论文集合解析为真实 document ids 后追加标签。",
                    )
                if selected_document_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "document_ids": selected_document_ids,
                        },
                        "给当前选中论文追加指定标签。",
                    )
                if self._mentions_all_library(content):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "document_ids": [document.id for document in self.document_library_service.list_documents() if document.status == "ready"],
                        },
                        "用户明确提到全库批量打标签，交给写入护栏确认或拒绝。",
                    )
                return _ReactAction(
                    "library.operator.assign_category",
                    {
                        "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                        "category_names": category_names,
                        "document_ids": [],
                    },
                    "缺少明确论文范围，交给统一写入护栏追问。",
                )
            if self._is_delete_unused_categories_intent(content):
                return _ReactAction(
                    "library.operator.delete_unused_categories",
                    {"selector": "unused"},
                    "删除 count=0 的标签/分类实体；这是 entity-level 操作，不清空任何论文标签关系。",
                )
            if rename_pair and self._is_clear_categories_intent(content) and not self._is_assignment_intent(content):
                return _ReactAction(
                    "library.operator.rename_category",
                    {
                        "source_category_name": rename_pair[0],
                        "target_category_name": rename_pair[1],
                    },
                    "先执行复合写操作中的标签重命名/迁移步骤，再处理后续清空步骤。",
                )
            if self._is_clear_categories_intent(content):
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.operator.clear_categories",
                        {"operation": "remove_single_category_link", "category_name": category_name},
                        "按用户指定标签定位论文，再清空这些论文的分类/标签关系。",
                    )
                return _ReactAction(
                    "library.operator.clear_categories",
                    {
                        "operation": "clear_all_categories" if self._mentions_all_library(content) else "clear_document_categories",
                        "scope": "all" if self._mentions_all_library(content) else "documents",
                        "document_ids": selected_document_ids,
                    },
                    "清除论文的分类/标签关系；标签和分类在系统中是同一字段。",
                )
            if rename_pair and not self._is_assignment_intent(content):
                return _ReactAction(
                    "library.operator.rename_category",
                    {
                        "source_category_name": rename_pair[0],
                        "target_category_name": rename_pair[1],
                    },
                    "将用户的标签替换意图解析为保留关联的安全重命名/合并。",
                )
            if self._is_labeled_document_collection_request(content):
                category_names = self._category_entity_names_for_request(content)
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"category_names": category_names, "expected": "many"},
                    "按用户提到的标签/分类实体读取真实论文集合。",
                )
            if self._needs_library_stats(content):
                return _ReactAction("library.explorer.stats", {}, "读取论文库统计。")
            if self._needs_category_stats(content):
                return _ReactAction("library.explorer.category_stats", {}, "读取标签与分类统计。")
            if self._is_metadata_query(content, selected_document_ids, attachments):
                return _ReactAction(
                    "library.explorer.document_metadata",
                    {
                        "document_ids": selected_document_ids,
                        "requested_fields": self._requested_metadata_fields(content),
                    },
                    "读取论文元数据字段。",
                )
            if self._requires_document_category_observation(content, selected_document_ids):
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": selected_document_ids},
                    "读取每篇论文和真实标签/分类的关联。",
                )
            if self._is_create_category_intent(content):
                return _ReactAction(
                    "library.operator.create_category",
                    {"category_name": assignment_category_name or (category_names[0] if category_names else ""), "category_names": category_names},
                    "先创建用户明确指定的标签/分类。",
                )
            if category_name and self._category_exists(category_name) and self._is_summary_request(content, selected_document_ids):
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"category_name": category_name, "expected": "many"},
                    "按用户指定的标签读取真实论文集合。",
                )
            if self._is_assignment_intent(content):
                if self._needs_untagged_assignment(content):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "scope": "untagged",
                        },
                        "给当前无标签论文追加指定标签。",
                    )
                if self._should_target_untagged_from_context(content, referent_state):
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "scope": "untagged",
                        },
                        "根据上下文指代定位到剩余未打标签论文，并追加指定标签。",
                    )
                scoped_referent_ids = self._resolve_document_scope(
                    session=session,
                    content=content,
                    action_arguments={},
                    selected_document_ids=selected_document_ids,
                    attachments=attachments,
                    observations=observations,
                    allow_all_library=False,
                )[1]
                if (
                    self._mentions_previous_referent(content)
                    or self._mentions_single_current_document(content)
                    or self._mentions_plural_current_documents(content)
                ) and scoped_referent_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "document_ids": scoped_referent_ids,
                            "scope": "documents",
                        },
                        "把上一轮对话指代的论文集合解析为真实 document ids 后追加标签。",
                    )
                if selected_document_ids:
                    return _ReactAction(
                        "library.operator.assign_category",
                        {
                            "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                            "category_names": category_names,
                            "document_ids": selected_document_ids,
                        },
                        "给当前选中论文追加指定标签。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "many"},
                    "先定位需要打标签的论文。",
                )
            if self._is_document_category_query(content):
                if selected_document_ids:
                    return _ReactAction(
                        "library.explorer.document_categories",
                        {"document_ids": selected_document_ids},
                        "读取选中论文的标签。",
                    )
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"category_name": category_name, "expected": "many"},
                        "按用户指定的标签读取真实论文集合。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "one"},
                    "先根据标题片段定位论文。",
                )
            if self._is_grouped_category_summary_request(content):
                return _ReactAction(
                    "evidence.retriever.search_by_category",
                    {"question": content, "category_names": [category_name] if category_name else []},
                    "按标签分别检索正文证据并评估每组覆盖情况。",
                )
            if selected_document_ids:
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": selected_document_ids},
                    "用户已选中论文，先检索本地证据。",
                )
            if self._is_summary_request(content, selected_document_ids):
                if category_name and self._category_exists(category_name):
                    return _ReactAction(
                        "library.explorer.find_documents",
                        {"category_name": category_name, "expected": "many"},
                        "按用户指定标签读取真实论文集合。",
                    )
                if selected_document_ids:
                    return _ReactAction(
                        "evidence.retriever.search",
                        {"question": content, "document_ids": selected_document_ids},
                        "先检索选中论文的证据。",
                    )
                return _ReactAction(
                    "library.explorer.find_documents",
                    {"query": content, "expected": "many"},
                    "先模糊匹配用户提到的论文。",
                )
            return _ReactAction(
                "final.answer",
                {"content": "我需要先知道你希望我查论文库、改标签，还是基于论文生成总结。"},
                "无法安全选择工具。",
            )

        if "library.explorer.stats" in completed_tools and self._needs_category_stats(content) and "library.explorer.category_stats" not in completed_tools:
            return _ReactAction("library.explorer.category_stats", {}, "继续读取标签覆盖情况。")
        if (
            "library.explorer.category_stats" in completed_tools
            and self._requires_document_category_observation(content, selected_document_ids)
            and "library.explorer.document_categories" not in completed_tools
        ):
            return _ReactAction(
                "library.explorer.document_categories",
                {"document_ids": self._document_category_target_ids(content, observations)},
                "继续读取每篇论文和真实标签/分类的关联，补全复合查询的明细部分。",
            )
        if "library.operator.create_category" in completed_tools and self._needs_untagged_assignment(content) and "library.operator.assign_category" not in completed_tools:
            return _ReactAction(
                "library.operator.assign_category",
                {
                    "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                    "category_names": category_names,
                    "scope": "untagged",
                },
                "把刚创建的标签追加给无标签论文。",
            )
        if "library.operator.assign_category" in completed_tools and rename_pair and "library.operator.rename_category" not in completed_tools:
            return _ReactAction(
                "library.operator.rename_category",
                {"source_category_name": rename_pair[0], "target_category_name": rename_pair[1]},
                "继续执行同一复合命令里的标签重命名/合并。",
            )
        if (
            "library.operator.assign_category" in completed_tools
            and "library.explorer.category_stats" not in completed_tools
            and (self._needs_untagged_assignment(content) or self._needs_category_stats(content))
        ):
            return _ReactAction("library.explorer.category_stats", {}, "写操作后读取标签统计，二次验证结果。")
        if (
            "library.operator.assign_category" in completed_tools
            and "library.explorer.document_categories" not in completed_tools
            and self._requires_document_category_observation(content, selected_document_ids)
        ):
            return _ReactAction(
                "library.explorer.document_categories",
                {"document_ids": self._document_ids_from_observations(observations)},
                "写操作后读取受影响论文标签，二次验证结果。",
            )
        if (
            "library.operator.rename_category" in completed_tools
            and self._is_clear_categories_intent(content)
            and "library.operator.clear_categories" not in completed_tools
        ):
            return _ReactAction(
                "library.operator.clear_categories",
                self._clear_categories_arguments_for_request(
                    content,
                    selected_document_ids=selected_document_ids,
                    observations=observations,
                ),
                "继续执行同一复合命令里的清空/移除标签子任务。",
            )
        if (
            "library.operator.rename_category" in completed_tools
            and self._needs_category_stats(content)
            and "library.explorer.category_stats" not in completed_tools
        ):
            return _ReactAction("library.explorer.category_stats", {}, "写操作后读取标签统计，完成复合统计子目标。")
        if "library.operator.rename_category" in completed_tools:
            return _ReactAction(
                "final.answer",
                {"content": self._synthesize_react_answer(content, observations)},
                "标签重命名/合并已完成，生成最终回答。",
            )
        if "library.explorer.find_documents" in completed_tools:
            if self._has_labeled_document_set_observation(observations):
                if self._is_labeled_document_analysis_request(content) and "evidence.retriever.search" not in completed_tools:
                    document_ids = self._document_ids_from_observations(observations)
                    if document_ids:
                        return _ReactAction(
                            "evidence.retriever.search",
                            {"question": content, "document_ids": document_ids},
                            "检索标签/分类实体下已定位论文的证据。",
                        )
                if not self._is_labeled_document_analysis_request(content):
                    return _ReactAction(
                        "final.answer",
                        {"content": self._synthesize_react_answer(content, observations)},
                        "标签/分类实体下的论文集合已经解析，生成最终回答。",
                    )
            if self._is_document_category_query(content) and "library.explorer.document_categories" not in completed_tools:
                return _ReactAction(
                    "library.explorer.document_categories",
                    {"document_ids": document_ids},
                    "读取已定位论文的标签。",
                )
            if self._is_metadata_query(content, selected_document_ids, attachments) and "library.explorer.document_metadata" not in completed_tools:
                return _ReactAction(
                    "library.explorer.document_metadata",
                    {"document_ids": document_ids, "requested_fields": self._requested_metadata_fields(content)},
                    "读取已定位论文的元数据字段。",
                )
            if self._is_assignment_intent(content) and "library.operator.assign_category" not in completed_tools:
                return _ReactAction(
                    "library.operator.assign_category",
                    {
                        "category_name": assignment_category_name or (category_names[0] if category_names else ""),
                        "category_names": category_names,
                        "document_ids": document_ids,
                    },
                    "给已定位论文追加标签。",
                )
            if self._is_clear_categories_intent(content) and "library.operator.clear_categories" not in completed_tools:
                return _ReactAction(
                    "library.operator.clear_categories",
                    {"operation": "clear_document_categories", "document_ids": document_ids},
                    "清空已定位论文的分类/标签关系。",
                )
            if self._is_summary_request(content, selected_document_ids) and "evidence.retriever.search" not in completed_tools:
                return _ReactAction(
                    "evidence.retriever.search",
                    {"question": content, "document_ids": document_ids},
                    "检索已定位论文的证据。",
                )
        if "evidence.retriever.search" in completed_tools and "report.drafter.write" not in completed_tools:
            return _ReactAction(
                "report.drafter.write",
                {"question": content, "document_ids": document_ids},
                "基于检索观察生成回答。",
            )
        if "evidence.retriever.search_by_category" in completed_tools and "report.drafter.write_by_category" not in completed_tools:
            return _ReactAction(
                "report.drafter.write_by_category",
                {"question": content},
                "基于每个标签的证据覆盖情况生成分组总结。",
            )
        return _ReactAction(
            "final.answer",
            {"content": self._synthesize_react_answer(content, observations)},
            "已有足够观察，生成最终回答。",
        )

    def _execute_react_action(
        self,
        *,
        run_id: str,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
        plan_state: _PlanState | None = None,
        step_state: _StepState | None = None,
    ) -> _ReactObservation:
        tool_map = {
            self._INTERNAL_DEGRADED_TOOL: self._tool_intent_degraded,
            self._INTERNAL_CATEGORY_CONFLICT_TOOL: self._tool_category_semantics_conflict,
            "tool.registry.list": self._tool_registry_list,
            "library.explorer.stats": self._tool_library_stats,
            "library.explorer.category_stats": self._tool_category_stats,
            "library.explorer.find_documents": self._tool_find_documents,
            "library.explorer.document_metadata": self._tool_document_metadata,
            "library.explorer.document_categories": self._tool_document_categories,
            "library.operator.create_category": self._tool_create_category,
            "library.operator.assign_category": self._tool_assign_category,
            "library.operator.rename_category": self._tool_rename_category,
            "library.operator.delete_unused_categories": self._tool_delete_unused_categories,
            "library.operator.clear_categories": self._tool_clear_categories,
            "evidence.retriever.search": self._tool_retrieve_evidence,
            "evidence.retriever.search_by_category": self._tool_retrieve_evidence_by_category,
            "report.drafter.write": self._tool_draft_report,
            "report.drafter.write_by_category": self._tool_draft_report_by_category,
            "memory.read": self._tool_memory_read,
            "memory.write": self._tool_memory_write,
        }
        executor = tool_map.get(action.tool)
        if executor is None:
            return _ReactObservation(
                tool=action.tool,
                status="validation_failed",
                summary=f"未知工具：{action.tool}",
            )
        target_scope = self._report_compare_target_scope(
            session=session,
            content=content,
            action=action,
            observations=observations,
            plan_state=plan_state,
        )
        if target_scope is not None:
            self._append_react_trace(
                run_id=run_id,
                status="report_compare_target_scope",
                payload={
                    "tool": action.tool,
                    "document_ids": target_scope["document_ids"],
                    "scope_source": target_scope["scope_source"],
                    "request_type": target_scope["request_type"],
                },
            )
        if action.tool == "evidence.retriever.search_by_category" and target_scope is not None:
            if not self._mentions_explicit_category_document_scope(content):
                target_ids = list(target_scope["document_ids"])
                self._append_react_trace(
                    run_id=run_id,
                    status="target_doc_leak_prevented",
                    payload={
                        "tool": action.tool,
                        "reason": "category_search_constrained_to_report_compare_target_scope",
                        "target_document_ids": target_ids,
                        "scope_source": target_scope["scope_source"],
                    },
                )
                action = _ReactAction(
                    "evidence.retriever.search",
                    {"question": str(action.arguments.get("question") or content), "document_ids": target_ids},
                    "Constrain category evidence retrieval to the explicit report/compare target scope.",
                    task_intent=action.task_intent,
                    action_plan=action.action_plan,
                    confidence=action.confidence,
                )
                executor = tool_map[action.tool]
        if action.tool == "evidence.retriever.search" and target_scope is not None:
            requested_ids = self._real_document_ids(action.arguments.get("document_ids") or [])
            target_ids = list(target_scope["document_ids"])
            if requested_ids and set(requested_ids) != set(target_ids):
                self._append_react_trace(
                    run_id=run_id,
                    status="target_doc_leak_prevented",
                    payload={
                        "tool": action.tool,
                        "reason": "evidence_search_scope_constrained_to_report_compare_target_scope",
                        "requested_document_ids": requested_ids,
                        "target_document_ids": target_ids,
                        "scope_source": target_scope["scope_source"],
                    },
                )
                action.arguments["document_ids"] = target_ids
        if action.tool == "report.drafter.write" and target_scope is not None:
            requested_ids = self._real_document_ids(action.arguments.get("document_ids") or [])
            target_ids = list(target_scope["document_ids"])
            if set(requested_ids) != set(target_ids):
                self._append_react_trace(
                    run_id=run_id,
                    status="target_doc_leak_prevented",
                    payload={
                        "tool": action.tool,
                        "reason": "report_drafter_scope_constrained_to_report_compare_target_scope",
                        "requested_document_ids": requested_ids,
                        "target_document_ids": target_ids,
                        "scope_source": target_scope["scope_source"],
                    },
                )
                action.arguments["document_ids"] = target_ids
        if action.tool == "evidence.retriever.search":
            reused = self._evidence_reuse_observation(
                run_id=run_id,
                content=content,
                action=action,
                observations=observations,
                target_scope=target_scope,
            )
            if reused is not None:
                return reused
        if action.tool in self._TOOL_RISK_REGISTRY:
            return self._safe_execute_tool(
                run_id=run_id,
                session=session,
                content=content,
                action=action,
                observations=observations,
                executor=executor,
                plan_state=plan_state,
                step_state=step_state,
            )
        observation = executor(run_id, session, content, action.arguments, observations)
        if observation.payload.get("library_mutated"):
            observation.payload.setdefault("verified_state", self._category_stats_payload())
            observation.payload.setdefault(
                "affected_document_ids",
                [item for item in observation.payload.get("document_ids") or [] if isinstance(item, str)],
            )
            observation.payload.setdefault("operation_summary", observation.summary)
            self._append_react_trace(
                run_id=run_id,
                status="library_write_verified",
                payload={
                    "tool": observation.tool,
                    "affected_document_count": len(observation.payload.get("affected_document_ids") or []),
                    "category_names": observation.payload.get("category_names") or [],
                    "verified_state": self._safe_trace_payload(observation.payload.get("verified_state") or {}),
                },
            )
        return observation

    def _safe_execute_tool(
        self,
        *,
        run_id: str,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
        executor: Callable[[str, ChatSession, str, dict[str, Any], list[_ReactObservation]], _ReactObservation],
        plan_state: _PlanState | None = None,
        step_state: _StepState | None = None,
    ) -> _ReactObservation:
        """Runtime guardrail for model-planned write tools.

        LLM output may propose a tool and arguments, but this layer owns risk
        classification, scope validation, confirmation, snapshots, and
        verification before any database mutation is treated as successful.
        """

        policy = self._TOOL_RISK_REGISTRY[action.tool]
        arguments = dict(action.arguments)
        validation_error = self._validate_write_tool_scope(action.tool, arguments, observations, content=content)
        if validation_error:
            return _ReactObservation(
                tool=action.tool,
                status="validation_failed",
                summary=validation_error,
                payload={
                    "tool": action.tool,
                    "risk_level": policy.risk_level,
                    "guardrail": "write_scope_validation",
                    "_resolved_action": arguments.get("_resolved_action"),
                    "library_mutated": False,
                },
            )

        if action.tool == "library.operator.clear_categories":
            preview_result = self._preview_clear_categories(content, arguments, observations)
            if isinstance(preview_result, str):
                return _ReactObservation(
                    tool=action.tool,
                    status="validation_failed",
                    summary=preview_result,
                    payload={
                        "tool": action.tool,
                        "risk_level": policy.risk_level,
                        "guardrail": "destructive_preview",
                        "library_mutated": False,
                    },
                )
            self._write_pending_tool_action(
                session.id,
                preview_result,
                source_goal=content,
                plan_state=plan_state,
                step_state=step_state,
            )
            self._append_react_trace(
                run_id=run_id,
                status="write_preview_created",
                payload={
                    "tool": action.tool,
                    "operation": preview_result.operation,
                    "operation_level": "relation-level",
                    "write_type": "clear" if preview_result.operation != "remove_single_category_link" else "remove",
                    "target_type": "paper-category relation",
                    "risk_level": preview_result.risk_level,
                    "destructive": True,
                    "affected_count": preview_result.affected_count,
                    "target_count": preview_result.affected_count,
                    "targets": preview_result.affected_entities[:12],
                    "expected_scope": preview_result.expected_scope,
                    "will_delete_entities": False,
                    "will_modify_relations": True,
                    "requires_confirmation": True,
                    "confirmation_phrase": preview_result.confirmation_phrase,
                },
            )
            return _ReactObservation(
                tool=action.tool,
                status="confirmation_required",
                summary=self._preview_confirmation_text(preview_result),
                payload={
                    "operation": preview_result.operation,
                    "operation_level": "relation-level",
                    "write_type": "clear" if preview_result.operation != "remove_single_category_link" else "remove",
                    "target_type": "paper-category relation",
                    "risk_level": preview_result.risk_level,
                    "destructive": True,
                    "affected_count": preview_result.affected_count,
                    "target_count": preview_result.affected_count,
                    "affected_entities": preview_result.affected_entities[:12],
                    "targets": preview_result.affected_entities[:12],
                    "expected_scope": preview_result.expected_scope,
                    "will_delete_entities": False,
                    "will_modify_relations": True,
                    "requires_confirmation": True,
                    "confirmation_phrase": preview_result.confirmation_phrase,
                    "pending_action": True,
                    "library_mutated": False,
                },
            )

        if action.tool == "library.operator.delete_unused_categories":
            preview_result = self._preview_delete_unused_categories(arguments)
            if isinstance(preview_result, str):
                return _ReactObservation(
                    tool=action.tool,
                    status="validation_failed",
                    summary=preview_result,
                    payload={
                        "tool": action.tool,
                        "risk_level": policy.risk_level,
                        "operation_level": policy.operation_level,
                        "write_type": policy.write_type,
                        "target_type": policy.target_type,
                        "destructive": policy.destructive,
                        "guardrail": "destructive_preview",
                        "library_mutated": False,
                    },
                )
            if preview_result.affected_count <= 0:
                return _ReactObservation(
                    tool=action.tool,
                    status="completed",
                    summary="当前没有 count=0 的空标签/分类实体，无需删除。",
                    payload={
                        "operation": preview_result.operation,
                        "operation_level": "entity-level",
                        "write_type": "delete",
                        "target_type": "category",
                        "selector": "unused",
                        "destructive": False,
                        "deleted_count": 0,
                        "deleted_category_names": [],
                        "library_mutated": False,
                        "verified_state": self._category_stats_payload(),
                        "operation_summary": "no_unused_category_entities",
                    },
                )
            self._write_pending_tool_action(
                session.id,
                preview_result,
                source_goal=content,
                plan_state=plan_state,
                step_state=step_state,
            )
            self._append_react_trace(
                run_id=run_id,
                status="write_preview_created",
                payload={
                    "tool": action.tool,
                    "operation": preview_result.operation,
                    "operation_level": "entity-level",
                    "write_type": "delete",
                    "target_type": "category",
                    "risk_level": preview_result.risk_level,
                    "destructive": True,
                    "affected_count": preview_result.affected_count,
                    "target_count": preview_result.affected_count,
                    "affected_category_names": [
                        str(item.get("name"))
                        for item in preview_result.affected_entities
                        if isinstance(item, dict)
                    ],
                    "targets": preview_result.affected_entities[:20],
                    "expected_scope": preview_result.expected_scope,
                    "will_delete_entities": True,
                    "will_modify_relations": False,
                    "requires_confirmation": True,
                    "confirmation_phrase": preview_result.confirmation_phrase,
                },
            )
            return _ReactObservation(
                tool=action.tool,
                status="confirmation_required",
                summary=self._preview_confirmation_text(preview_result),
                payload={
                    "operation": preview_result.operation,
                    "operation_level": "entity-level",
                    "write_type": "delete",
                    "target_type": "category",
                    "risk_level": preview_result.risk_level,
                    "destructive": True,
                    "affected_count": preview_result.affected_count,
                    "target_count": preview_result.affected_count,
                    "affected_entities": preview_result.affected_entities[:20],
                    "targets": preview_result.affected_entities[:20],
                    "expected_scope": preview_result.expected_scope,
                    "will_delete_entities": True,
                    "will_modify_relations": False,
                    "requires_confirmation": True,
                    "confirmation_phrase": preview_result.confirmation_phrase,
                    "pending_action": True,
                    "library_mutated": False,
                },
            )

        before_snapshot = self._category_snapshot() if policy.requires_verification else {}
        guarded_action = _ReactAction(
            tool=action.tool,
            arguments={**arguments, "__before_snapshot": before_snapshot},
            rationale=action.rationale,
            task_intent=action.task_intent,
            action_plan=action.action_plan,
            confidence=action.confidence,
        )
        observation = executor(run_id, session, content, guarded_action.arguments, observations)
        if policy.requires_verification and observation.payload.get("library_mutated"):
            verification_error = self._verify_write_observation(guarded_action, observation, before_snapshot)
            if verification_error:
                self._rollback_category_snapshot(before_snapshot)
                self._append_react_trace(
                    run_id=run_id,
                    status="library_write_verification_failed",
                    payload={
                        "tool": action.tool,
                        "risk_level": policy.risk_level,
                        "reason": verification_error,
                        "rolled_back": True,
                    },
                )
                return _ReactObservation(
                    tool=action.tool,
                    status="failed",
                    summary=f"写操作后的二次校验失败，已尝试按执行前快照回滚：{verification_error}",
                    payload={
                        **observation.payload,
                        "library_mutated": False,
                        "verification_failed": True,
                        "rollback_attempted": True,
                        "verification_error": verification_error,
                    },
                )
        if observation.payload.get("library_mutated"):
            observation.payload.setdefault("verified_state", self._category_stats_payload())
            observation.payload.setdefault(
                "affected_document_ids",
                [item for item in observation.payload.get("document_ids") or [] if isinstance(item, str)],
            )
            observation.payload.setdefault("operation_summary", observation.summary)
            self._append_react_trace(
                run_id=run_id,
                status="library_write_verified",
                payload={
                    "tool": observation.tool,
                    "risk_level": policy.risk_level,
                    "affected_document_count": len(observation.payload.get("affected_document_ids") or []),
                    "category_names": observation.payload.get("category_names") or [],
                    "verified_state": self._safe_trace_payload(observation.payload.get("verified_state") or {}),
                },
            )
        return observation

    def _validate_write_tool_scope(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
        *,
        content: str = "",
    ) -> str | None:
        policy = self._TOOL_RISK_REGISTRY[tool_name]
        if tool_name == "memory.write":
            return None
        if tool_name in {"library.operator.create_category", "library.operator.assign_category"}:
            category_names = self._category_names_from_arguments(arguments)
            if not category_names:
                return "写操作缺少明确的标签/分类名称，已拒绝执行。"
            error = self._category_names_validation_error(category_names)
            if error:
                return error
            if tool_name == "library.operator.assign_category":
                scope = str(arguments.get("scope") or "")
                resolved = arguments.get("_resolved_action") if isinstance(arguments.get("_resolved_action"), dict) else {}
                document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
                if scope and scope not in {"untagged", "last_referenced", "documents"}:
                    return "打标签操作的 scope 无效，已拒绝执行。"
                if resolved.get("clarification_needed") or resolved.get("scope_type") == "unknown":
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if self._mentions_ambiguous_plural_document_referent(content) and document_ids and not resolved:
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if (
                    self._mentions_ambiguous_plural_document_referent(content)
                    and document_ids
                    and resolved.get("scope_type") == "explicit_documents"
                    and not (
                        self._needs_untagged_assignment(content)
                        or self._mentions_all_library(content)
                        or self._mentions_explicit_category_document_scope(content)
                    )
                ):
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if resolved.get("scope_type") == "all_library":
                    return "批量给所有论文打标签必须先经过确认或先筛选出明确论文集合，本轮没有改动论文库。"
                if scope == "documents" and not document_ids and not self._document_ids_from_observations(observations):
                    return "打标签操作缺少明确的论文范围，已拒绝执行。"
            return None
        if tool_name == "library.operator.rename_category":
            source_name = self._clean_category_name(str(arguments.get("source_category_name") or ""))
            target_name = self._clean_category_name(str(arguments.get("target_category_name") or ""))
            if not source_name or not target_name:
                return "重命名/合并标签缺少明确的源标签或目标标签，已拒绝执行。"
            return self._category_name_validation_error(source_name) or self._category_name_validation_error(target_name)
        if tool_name == "library.operator.delete_unused_categories":
            selector = str(arguments.get("selector") or "").strip().casefold()
            if selector != "unused":
                return "删除空标签/分类实体必须显式提供 selector=unused；不允许空 selector 或 all selector。"
            if "category_ids" in arguments:
                category_ids = [str(item) for item in arguments.get("category_ids") or [] if str(item).strip()]
                if not category_ids:
                    return "删除标签/分类实体的 category_ids 不能为空；本轮没有改动论文库。"
            return None
        if tool_name == "library.operator.clear_categories":
            operation = self._infer_clear_categories_operation(arguments)
            category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
            document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
            scope = str(arguments.get("scope") or "")
            resolved = arguments.get("_resolved_action") if isinstance(arguments.get("_resolved_action"), dict) else {}
            if operation == "remove_single_category_link" and not category_name:
                return "移除标签关联必须提供明确的 category_name，已拒绝执行。"
            if resolved.get("clarification_needed") or resolved.get("scope_type") == "unknown":
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if self._mentions_ambiguous_plural_document_referent(content) and document_ids and not resolved:
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if (
                self._mentions_ambiguous_plural_document_referent(content)
                and document_ids
                and resolved.get("scope_type") == "explicit_documents"
                and not (
                    self._mentions_all_library(content)
                    or self._mentions_all_categories(content)
                    or self._mentions_explicit_category_document_scope(content)
                )
            ):
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if operation == "remove_single_category_link" and scope not in {"all", "tagged"} and not document_ids:
                return "移除标签关联缺少明确论文范围，已拒绝执行；模糊指代不会默认扩大到全库。"
            if operation == "clear_document_categories" and not document_ids:
                return "清空指定论文标签必须提供非空 document_ids，已拒绝执行。"
            if operation == "clear_all_categories" and scope not in {"all", "tagged"}:
                return "全量清空标签必须显式提供 scope=all 或 scope=tagged，已拒绝执行。"
            if operation not in {"remove_single_category_link", "clear_document_categories", "clear_all_categories"}:
                return "清空/移除标签操作缺少明确 operation，已拒绝执行。"
            return None
        if policy.required_args:
            missing = [name for name in policy.required_args if not arguments.get(name)]
            if missing:
                return f"写操作缺少必填参数：{', '.join(missing)}。"
        return None

    def _evidence_reuse_observation(
        self,
        *,
        run_id: str,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
        target_scope: dict[str, Any] | None = None,
    ) -> _ReactObservation | None:
        requested_document_ids = self._real_document_ids(action.arguments.get("document_ids") or [])
        if not requested_document_ids or not self._can_reuse_evidence_for_request(content, requested_document_ids):
            return None
        target_document_ids = self._real_document_ids((target_scope or {}).get("document_ids") or [])
        if target_document_ids:
            requested_set = set(requested_document_ids)
            target_set = set(target_document_ids)
            if not requested_set.issubset(target_set):
                self._append_evidence_reuse_scope_mismatch_trace(
                    run_id=run_id,
                    requested_document_ids=requested_document_ids,
                    target_document_ids=target_document_ids,
                    source_document_ids=[],
                    reason="requested_scope_outside_report_compare_target",
                )
                return None
            if self._requires_full_target_evidence(content, target_document_ids) and requested_set != target_set:
                self._append_evidence_reuse_scope_mismatch_trace(
                    run_id=run_id,
                    requested_document_ids=requested_document_ids,
                    target_document_ids=target_document_ids,
                    source_document_ids=[],
                    reason="requested_scope_does_not_cover_full_report_compare_target",
                )
                return None
        source = self._latest_evidence_observation_for_scope(
            requested_document_ids=requested_document_ids,
            observations=observations,
        )
        if source is None:
            return None
        source_index, source_observation, source_payload, evidence_payload = source
        source_document_ids = self._document_ids_from_evidence_payload(evidence_payload)
        if target_document_ids:
            source_set = set(source_document_ids)
            target_set = set(target_document_ids)
            if not source_set or not source_set.issubset(target_set):
                self._append_evidence_reuse_scope_mismatch_trace(
                    run_id=run_id,
                    requested_document_ids=requested_document_ids,
                    target_document_ids=target_document_ids,
                    source_document_ids=source_document_ids,
                    reason="source_evidence_outside_report_compare_target",
                )
                return None
            if self._requires_full_target_evidence(content, target_document_ids) and not target_set.issubset(source_set):
                self._append_evidence_reuse_scope_mismatch_trace(
                    run_id=run_id,
                    requested_document_ids=requested_document_ids,
                    target_document_ids=target_document_ids,
                    source_document_ids=source_document_ids,
                    reason="source_evidence_does_not_cover_full_report_compare_target",
                )
                return None
        question = str(action.arguments.get("question") or content)
        reuse_scope_key = self._evidence_scope_key(requested_document_ids)
        payload = {
            "document_ids": requested_document_ids,
            "evidence_items": evidence_payload,
            "evidence_quality": source_payload.get("evidence_quality"),
            "cache_hit": bool(source_payload.get("cache_hit")),
            "retrieval_strategy": source_payload.get("retrieval_strategy"),
            "question": question,
            "evidence_reused": True,
            "reused_from_observation_index": source_index,
            "reused_source_tool": source_observation.tool,
            "reuse_scope_key": reuse_scope_key,
        }
        self._append_react_trace(
            run_id=run_id,
            status="evidence_reused",
            payload={
                "tool": action.tool,
                "document_ids": requested_document_ids,
                "evidence_count": len(evidence_payload),
                "reused_from_observation_index": source_index,
                "reuse_scope_key": reuse_scope_key,
            },
        )
        return _ReactObservation(
            tool="evidence.retriever.search",
            status="completed",
            summary=f"Reused {len(evidence_payload)} evidence item(s) from an earlier search in this run.",
            payload=payload,
        )

    def _append_evidence_reuse_scope_mismatch_trace(
        self,
        *,
        run_id: str,
        requested_document_ids: list[str],
        target_document_ids: list[str],
        source_document_ids: list[str],
        reason: str,
    ) -> None:
        self._append_react_trace(
            run_id=run_id,
            status="evidence_reuse_rejected_scope_mismatch",
            payload={
                "requested_document_ids": requested_document_ids,
                "target_document_ids": target_document_ids,
                "source_document_ids": source_document_ids,
                "reason": reason,
            },
        )

    def _requires_full_target_evidence(self, content: str, target_document_ids: list[str]) -> bool:
        return len(target_document_ids) > 1 and (
            self._is_compare_like_request(content) or self._is_report_like_request(content)
        )

    def _report_compare_target_scope(
        self,
        *,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        observations: list[_ReactObservation],
        plan_state: _PlanState | None = None,
    ) -> dict[str, Any] | None:
        if not (self._is_report_like_request(content) or self._is_compare_like_request(content)):
            return None
        request_type = "compare" if self._is_compare_like_request(content) else "report"

        def scope(ids: list[str], source: str) -> dict[str, Any] | None:
            real_ids = self._real_document_ids(ids)
            return {"document_ids": real_ids, "scope_source": source, "request_type": request_type} if real_ids else None

        if plan_state is not None:
            selected_ids = self._selected_document_ids_from_plan(plan_state)
            if selected_ids:
                return scope(selected_ids, "selected_document_ids")
            attachment_ids = [attachment.document_id for attachment in self._attachments_from_plan(plan_state) if attachment.document_id]
            if attachment_ids:
                return scope(attachment_ids, "library_document_attachments")

        action_ids = [str(item) for item in action.arguments.get("document_ids") or [] if item]
        if action_ids:
            return scope(action_ids, "resolved_action_document_ids")

        if self._mentions_explicit_category_document_scope(content):
            category_filter_ids = self._category_filter_document_ids_from_observations(observations)
            if category_filter_ids is not None:
                return scope(category_filter_ids, "tag_filter_document_ids")

        if self._mentions_plural_current_documents(content) or self._mentions_previous_referent(content):
            recent_ids = self._recent_scope_document_ids(session.id, singular=False)
            if recent_ids:
                return scope(recent_ids, "recent_multi_documents")
        if self._mentions_single_current_document(content):
            recent_ids = self._recent_scope_document_ids(session.id, singular=True)
            if recent_ids:
                return scope(recent_ids, "recent_single_document")
        return None

    def _latest_evidence_observation_for_scope(
        self,
        *,
        requested_document_ids: list[str],
        observations: list[_ReactObservation],
    ) -> tuple[int, _ReactObservation, dict[str, Any], list[Any]] | None:
        requested_set = set(requested_document_ids)
        if not requested_set:
            return None
        for index in range(len(observations) - 1, -1, -1):
            observation = observations[index]
            if observation.tool != "evidence.retriever.search" or observation.status != "completed":
                continue
            payload = self._react_observation_payload(observation)
            evidence_payload = observation.payload.get("evidence_items")
            if not isinstance(evidence_payload, list) or not evidence_payload:
                evidence_payload = self._react_observation_evidence(observation)
            if not evidence_payload:
                continue
            observed_ids = self._real_document_ids(payload.get("document_ids") or [])
            if not observed_ids:
                observed_ids = self._document_ids_from_evidence_payload(evidence_payload)
            if set(observed_ids) != requested_set:
                continue
            evidence_document_ids = set(self._document_ids_from_evidence_payload(evidence_payload))
            if requested_set and not requested_set.issubset(evidence_document_ids):
                continue
            return index, observation, payload, evidence_payload
        return None

    @staticmethod
    def _evidence_scope_key(document_ids: list[str]) -> str:
        return ",".join(sorted(str(item) for item in document_ids if item))

    def _document_ids_from_evidence_payload(self, evidence_payload: list[Any]) -> list[str]:
        document_ids: list[str] = []
        for item in evidence_payload:
            if isinstance(item, EvidenceItem):
                document_id = item.document_id or item.source_id
            elif isinstance(item, dict):
                document_id = item.get("document_id") or item.get("source_id")
            else:
                document_id = None
            if isinstance(document_id, str) and document_id not in document_ids:
                document_ids.append(document_id)
        return self._real_document_ids(document_ids)

    def _can_reuse_evidence_for_request(self, content: str, document_ids: list[str]) -> bool:
        if not document_ids:
            return False
        return (
            self._is_report_like_request(content)
            or self._is_compare_like_request(content)
            or self._is_summary_request(content, document_ids)
            or self._is_selected_document_answer_request(content, document_ids, [])
            or self._mentions_previous_referent(content)
            or self._mentions_single_current_document(content)
            or self._mentions_plural_current_documents(content)
        )

    def _resolve_action_intent(
        self,
        *,
        session: ChatSession,
        content: str,
        action: _ReactAction,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
    ) -> _ResolvedAction | None:
        if action.tool == "library.operator.rename_category" and self._is_document_label_relation_assignment(content):
            category_names = self._extract_category_names_from_request(content)
            category_name = category_names[0] if category_names else self._extract_category_name_from_request(content) or None
            scope_type, document_ids = self._resolve_document_scope(
                session=session,
                content=content,
                action_arguments={},
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
                allow_all_library=False,
            )
            action.tool = "library.operator.assign_category"
            action.arguments = {
                "category_name": category_name or "",
                "category_names": category_names or ([category_name] if category_name else []),
            }
            if document_ids:
                action.arguments["document_ids"] = document_ids
                action.arguments["scope"] = "documents"
            else:
                action.arguments["scope"] = "documents"
            return _ResolvedAction(
                intent_type="write",
                operation="assign_label",
                target_type="paper_label_relation",
                scope_type=scope_type,
                document_ids=document_ids,
                label_name=category_name,
                category_name=category_name,
                risk_level="safe_write",
                requires_confirmation=False,
                tool_name=action.tool,
                reason="“这篇/这些论文的标签改成 X”是论文-标签关系操作，不是全局标签实体重命名。",
                clarification_needed=scope_type == "unknown",
            )
        if action.tool not in {
            "library.operator.assign_category",
            "library.operator.clear_categories",
            "library.operator.rename_category",
            "library.operator.delete_unused_categories",
            "evidence.retriever.search",
            "report.drafter.write",
        }:
            return None
        if action.tool == "library.operator.assign_category":
            category_names = self._category_names_from_arguments(action.arguments)
            if not category_names:
                category_names = self._extract_category_names_from_request(content)
            category_name = category_names[0] if category_names else self._extract_category_name_from_request(content) or None
            if str(action.arguments.get("scope") or "") == "untagged" or self._needs_untagged_assignment(content):
                action.arguments["scope"] = "untagged"
                action.arguments.pop("document_ids", None)
                if category_name:
                    action.arguments.setdefault("category_name", category_name)
                    action.arguments.setdefault("category_names", category_names or [category_name])
                return _ResolvedAction(
                    intent_type="write",
                    operation="assign_label",
                    target_type="paper_label_relation",
                    scope_type="explicit_documents",
                    document_ids=[],
                    label_name=category_name,
                    category_name=category_name,
                    risk_level="safe_write",
                    requires_confirmation=False,
                    tool_name=action.tool,
                    reason="统一解析无标签论文补标签操作；scope=untagged 不进入文档模糊匹配。",
                    clarification_needed=False,
                )
            if self._ambiguous_relation_write_scope_without_context(
                session=session,
                content=content,
                action_arguments=action.arguments,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
            ):
                action.arguments.pop("document_ids", None)
                action.arguments["scope"] = "documents"
                if category_name:
                    action.arguments.setdefault("category_name", category_name)
                    action.arguments.setdefault("category_names", category_names or [category_name])
                return _ResolvedAction(
                    intent_type="write",
                    operation="assign_label",
                    target_type="paper_label_relation",
                    scope_type="unknown",
                    document_ids=[],
                    label_name=category_name,
                    category_name=category_name,
                    risk_level="safe_write",
                    requires_confirmation=False,
                    tool_name=action.tool,
                    reason="模糊复数指代缺少 selected/recent/read-result 论文范围，不能使用模型生成的 document_ids。",
                    clarification_needed=True,
                )
            scope_type, document_ids = self._resolve_document_scope(
                session=session,
                content=content,
                action_arguments=action.arguments,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
                allow_all_library=self._mentions_all_library(content),
            )
            if scope_type == "all_library":
                action.arguments["scope"] = "all"
            elif scope_type in {"current_selection", "recent_selection", "explicit_documents"}:
                action.arguments["document_ids"] = document_ids
                action.arguments["scope"] = "documents"
            elif scope_type == "unknown" and str(action.arguments.get("scope") or "") != "untagged":
                action.arguments.pop("document_ids", None)
                action.arguments["scope"] = "documents"
            if category_name:
                action.arguments.setdefault("category_name", category_name)
                action.arguments.setdefault("category_names", category_names or [category_name])
            return _ResolvedAction(
                intent_type="write",
                operation="assign_label",
                target_type="paper_label_relation",
                scope_type=scope_type if str(action.arguments.get("scope") or "") != "untagged" else "explicit_documents",
                document_ids=document_ids,
                label_name=category_name,
                category_name=category_name,
                risk_level="batch_write" if scope_type == "all_library" else "safe_write",
                requires_confirmation=scope_type == "all_library",
                tool_name=action.tool,
                reason="统一解析打标签关系写操作的标签名和论文范围。",
                clarification_needed=scope_type == "unknown" and str(action.arguments.get("scope") or "") != "untagged",
            )
        if action.tool == "library.operator.clear_categories":
            operation = self._infer_clear_categories_operation(action.arguments)
            category_name = (
                self._clean_category_name(str(action.arguments.get("category_name") or ""))
                or self._category_name_from_clear_clause(content)
                or self._category_name_from_request_or_observations(content, observations)
            )
            explicit_label_scope = bool(category_name) and (
                self._mentions_all_library(content)
                or self._mentions_all_categories(content)
                or any(marker in content for marker in ("带有", "带着", "包含", "含有", "这个标签", "该标签", "标签下"))
            )
            if self._ambiguous_relation_write_scope_without_context(
                session=session,
                content=content,
                action_arguments=action.arguments,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
            ):
                action.arguments.pop("document_ids", None)
                action.arguments["scope"] = "documents"
                if category_name:
                    action.arguments["category_name"] = category_name
                return _ResolvedAction(
                    intent_type="write",
                    operation="remove_label" if operation == "remove_single_category_link" else "clear_labels",
                    target_type="paper_label_relation",
                    scope_type="unknown",
                    document_ids=[],
                    label_name=category_name or None,
                    category_name=category_name or None,
                    risk_level="destructive",
                    requires_confirmation=True,
                    tool_name=action.tool,
                    reason="模糊复数指代缺少 selected/recent/read-result 论文范围，不能执行标签关系移除或清空。",
                    clarification_needed=True,
                )
            scope_type, document_ids = self._resolve_document_scope(
                session=session,
                content=content,
                action_arguments=action.arguments,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
                allow_all_library=self._mentions_all_library(content) and not explicit_label_scope,
            )
            if operation == "remove_single_category_link":
                if category_name:
                    action.arguments["category_name"] = category_name
                if explicit_label_scope:
                    action.arguments["scope"] = "tagged"
                    action.arguments.pop("document_ids", None)
                    scope_type = "explicit_label"
                    document_ids = []
                elif scope_type in {"current_selection", "recent_selection", "explicit_documents"} and document_ids:
                    action.arguments["document_ids"] = document_ids
                    action.arguments["scope"] = "documents"
                elif scope_type == "all_library":
                    action.arguments["scope"] = "all"
                else:
                    action.arguments.pop("document_ids", None)
                    action.arguments["scope"] = "documents"
            elif operation == "clear_document_categories":
                if scope_type in {"current_selection", "recent_selection", "explicit_documents"}:
                    action.arguments["document_ids"] = document_ids
                    action.arguments["scope"] = "documents"
                elif scope_type == "all_library":
                    action.arguments["operation"] = "clear_all_categories"
                    action.arguments["scope"] = "all"
                else:
                    action.arguments["document_ids"] = []
                    action.arguments["scope"] = "documents"
            return _ResolvedAction(
                intent_type="write",
                operation="remove_label" if operation == "remove_single_category_link" else "clear_labels",
                target_type="paper_label_relation",
                scope_type=scope_type,
                document_ids=document_ids,
                label_name=category_name or None,
                category_name=category_name or None,
                risk_level="batch_destructive" if scope_type == "all_library" else "destructive",
                requires_confirmation=True,
                tool_name=action.tool,
                reason="统一解析清空/移除标签关系写操作的作用域。",
                clarification_needed=scope_type == "unknown",
            )
        if action.tool == "library.operator.delete_unused_categories":
            action.arguments["selector"] = "unused"
            return _ResolvedAction(
                intent_type="manage",
                operation="delete_empty_labels",
                target_type="label_entity",
                scope_type="explicit_label",
                risk_level="destructive",
                requires_confirmation=True,
                tool_name=action.tool,
                reason="删除空标签分类是 entity-level 操作，不解析为论文关系操作。",
            )
        if action.tool == "library.operator.rename_category":
            return _ResolvedAction(
                intent_type="manage",
                operation="rename_label",
                target_type="label_entity",
                scope_type="explicit_label",
                risk_level="safe_write",
                requires_confirmation=False,
                tool_name=action.tool,
                reason="标签重命名/合并是 entity-level 操作。",
            )
        if action.tool in {"evidence.retriever.search", "report.drafter.write"}:
            scope_type, document_ids = self._resolve_document_scope(
                session=session,
                content=content,
                action_arguments=action.arguments,
                selected_document_ids=selected_document_ids,
                attachments=attachments,
                observations=observations,
                allow_all_library=False,
            )
            category_filter_ids = self._category_filter_document_ids_from_observations(observations)
            if category_filter_ids is not None:
                scope_type = "category_filter"
                document_ids = category_filter_ids
            if document_ids:
                action.arguments["document_ids"] = document_ids
            return _ResolvedAction(
                intent_type="analyze",
                operation="summarize" if self._is_summary_request(content, document_ids) else "ask",
                target_type="paper",
                scope_type=scope_type,
                document_ids=document_ids,
                risk_level="read",
                requires_confirmation=False,
                tool_name=action.tool,
                reason="统一解析论文问答/总结的文档范围。",
                clarification_needed=scope_type == "unknown",
            )
        return None

    def _resolve_document_scope(
        self,
        *,
        session: ChatSession,
        content: str,
        action_arguments: dict[str, Any],
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        observations: list[_ReactObservation],
        allow_all_library: bool,
    ) -> tuple[str, list[str]]:
        explicit_arg_ids = self._real_document_ids(action_arguments.get("document_ids") or [])
        if explicit_arg_ids:
            current_ids = self._real_document_ids(selected_document_ids)
            attachment_ids = self._real_document_ids([attachment.document_id for attachment in attachments if attachment.document_id])
            if current_ids and set(explicit_arg_ids) == set(current_ids):
                return "current_selection", explicit_arg_ids
            if attachment_ids and set(explicit_arg_ids) == set(attachment_ids):
                return "current_selection", explicit_arg_ids
            if (
                self._mentions_previous_referent(content)
                or self._mentions_single_current_document(content)
                or self._mentions_plural_current_documents(content)
            ):
                state_ids = self._recent_scope_document_ids(session.id, singular=self._mentions_single_current_document(content))
                if state_ids and set(explicit_arg_ids) == set(state_ids):
                    return "recent_selection", explicit_arg_ids
            ready_ids = {
                document.id for document in self.document_library_service.list_documents()
                if document.status == "ready"
            }
            if (
                allow_all_library
                and self._mentions_all_library(content)
                and ready_ids
                and set(explicit_arg_ids) >= ready_ids
            ):
                return "all_library", explicit_arg_ids
            return "explicit_documents", explicit_arg_ids
        current_ids = self._real_document_ids(selected_document_ids)
        if current_ids:
            return "current_selection", current_ids
        attachment_ids = self._real_document_ids([attachment.document_id for attachment in attachments if attachment.document_id])
        if attachment_ids:
            return "current_selection", attachment_ids
        if self._mentions_single_current_document(content):
            recent_single_ids = self._recent_scope_document_ids(session.id, singular=True)
            if recent_single_ids:
                return "recent_selection", recent_single_ids
            observation_ids = self._real_document_ids(self._document_ids_from_observations(observations))
            if len(observation_ids) == 1:
                return "recent_selection", observation_ids
        if self._mentions_plural_current_documents(content):
            recent_multi_ids = self._recent_scope_document_ids(session.id, singular=False)
            if recent_multi_ids:
                return "recent_selection", recent_multi_ids
            observation_ids = self._real_document_ids(self._document_ids_from_observations(observations))
            if len(observation_ids) > 1:
                return "recent_selection", observation_ids
        category_filter_ids = self._category_filter_document_ids_from_observations(observations)
        if category_filter_ids is not None:
            return "category_filter", category_filter_ids
        explicit_documents = self._resolve_documents(content, [], allow_all=False)
        if explicit_documents:
            return "explicit_documents", [document.id for document in explicit_documents]
        if self._mentions_previous_referent(content) or self._mentions_single_current_document(content):
            state_ids = self._recent_scope_document_ids(session.id, singular=self._mentions_single_current_document(content))
            if state_ids:
                return "recent_selection", state_ids
            observation_ids = self._real_document_ids(self._document_ids_from_observations(observations))
            if observation_ids:
                return "recent_selection", observation_ids
        observation_ids = self._real_document_ids(self._document_ids_from_observations(observations))
        if observation_ids:
            return "recent_selection", observation_ids
        if allow_all_library and self._mentions_all_library(content):
            return "all_library", [document.id for document in self.document_library_service.list_documents() if document.status == "ready"]
        return "unknown", []

    def _real_document_ids(self, document_ids: list[Any]) -> list[str]:
        known = {document.id for document in self.document_library_service.list_documents()}
        result: list[str] = []
        for item in document_ids:
            document_id = str(item) if item else ""
            if document_id and document_id in known and document_id not in result:
                result.append(document_id)
        return result

    @staticmethod
    def _mentions_single_current_document(content: str) -> bool:
        return any(marker in content for marker in ("这篇", "这篇论文", "这篇文章", "该论文", "该文章", "这一个", "刚刚这篇", "刚才这篇", "上面这篇", "上述这篇"))

    @staticmethod
    def _mentions_plural_current_documents(content: str) -> bool:
        return any(marker in content for marker in ("这几篇", "这四篇", "这些论文", "这些文章", "这些", "这几个", "刚刚这几篇", "刚才这几篇", "上面几篇", "上述几篇", "这批"))

    @staticmethod
    def _is_document_label_relation_assignment(content: str) -> bool:
        has_document_referent = KnowledgeAgentRuntime._mentions_single_current_document(content) or KnowledgeAgentRuntime._mentions_plural_current_documents(content)
        has_label_word = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        has_relation_verb = any(marker in content for marker in ("改成", "改为", "换成", "换为", "设为", "设成", "设置成", "打上", "打标签", "加上", "添加"))
        return has_document_referent and has_label_word and has_relation_verb

    @staticmethod
    def _infer_clear_categories_operation(arguments: dict[str, Any]) -> str:
        operation = str(arguments.get("operation") or "").strip()
        category_name = str(arguments.get("category_name") or "").strip()
        document_ids = [item for item in arguments.get("document_ids") or [] if item]
        scope = str(arguments.get("scope") or "").strip()
        normalized_operation = operation.casefold()
        if normalized_operation in {"remove", "delete", "unlink", "remove_category", "remove_label"} and category_name:
            return "remove_single_category_link"
        if normalized_operation in {"clear", "clear_categories", "clear_labels"}:
            if category_name:
                return "remove_single_category_link"
            if document_ids:
                return "clear_document_categories"
            if scope in {"all", "tagged"}:
                return "clear_all_categories"
        if operation:
            return operation
        if category_name:
            return "remove_single_category_link"
        if document_ids:
            return "clear_document_categories"
        if scope in {"all", "tagged"}:
            return "clear_all_categories"
        return ""

    def _clear_categories_arguments_for_request(
        self,
        content: str,
        *,
        selected_document_ids: list[str],
        observations: list[_ReactObservation],
    ) -> dict[str, Any]:
        relative_document_ids = self._relative_clear_document_ids(content, observations)
        if relative_document_ids is not None:
            return {
                "operation": "clear_document_categories",
                "document_ids": relative_document_ids,
                "scope": "documents",
                "relative_scope": "other_tagged_documents",
            }
        if self._has_relative_clear_scope(content):
            return {
                "operation": "clear_document_categories",
                "document_ids": [],
                "scope": "documents",
                "relative_scope": "other_tagged_documents",
            }
        category_name = self._category_name_from_clear_clause(content) or self._category_name_from_request_or_observations(content, observations)
        if category_name:
            return {"operation": "remove_single_category_link", "category_name": category_name}
        document_ids = selected_document_ids or self._document_ids_from_observations(observations)
        if document_ids:
            return {"operation": "clear_document_categories", "document_ids": document_ids, "scope": "documents"}
        if self._mentions_all_library(content) or self._mentions_all_categories(content):
            return {"operation": "clear_all_categories", "scope": "all"}
        return {"operation": "clear_document_categories", "document_ids": [], "scope": "documents"}

    def _relative_clear_document_ids(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> list[str] | None:
        if not self._is_clear_categories_intent(content):
            return None
        if not self._has_relative_clear_scope(content):
            return None
        excluded_document_ids: set[str] = set()
        for observation in observations:
            if observation.status != "completed" or not observation.tool.startswith("library.operator."):
                continue
            excluded_document_ids.update(
                str(item)
                for item in observation.payload.get("affected_document_ids")
                or observation.payload.get("document_ids")
                or []
                if item
            )
        if not excluded_document_ids:
            return None
        return [
            document.id
            for document in self.document_library_service.list_documents()
            if document.categories and document.id not in excluded_document_ids
        ]

    @staticmethod
    def _has_relative_clear_scope(content: str) -> bool:
        return any(marker in content for marker in ("其他", "其它", "剩下", "其余", "除", "except", "other", "remaining"))

    def _category_name_from_clear_clause(self, content: str) -> str | None:
        for clause in self._goal_clauses(content):
            if not self._is_clear_categories_clause(clause):
                continue
            for category in self.category_repository.list_categories():
                if category.name and category.name in clause:
                    return category.name
        return None

    def _preview_clear_categories(
        self,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _WritePreview | str:
        operation = self._infer_clear_categories_operation(arguments)
        arguments = dict(arguments)
        arguments["operation"] = operation
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        scope = str(arguments.get("scope") or "")
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        requested_category = self._category_name_from_request_or_observations(content, observations)
        if scope in {"all", "tagged"} and requested_category and not category_name:
            return "模型给出了全量 scope，但用户请求中存在明确标签目标；已拦截，未执行。"
        before_snapshot = self._category_snapshot()
        documents = self.document_library_service.list_documents()
        categories = self.category_repository.list_categories()
        if operation == "clear_all_categories" and (category_name or requested_category):
            return "全量清空 scope 与单个标签目标不一致；已拦截，未执行。"

        if operation == "remove_single_category_link":
            if not category_name:
                return "移除标签关联缺少明确标签名，未执行。"
            category = next((item for item in categories if item.name == category_name), None)
            if category is None:
                return f"没有找到名为「{category_name}」的标签/分类，未执行。"
            selected = set(document_ids)
            if selected and scope not in {"all", "tagged"}:
                target_documents = [
                    document
                    for document in documents
                    if document.id in selected and any(item.id == category.id for item in document.categories)
                ]
                expected_scope = "explicit_documents"
            elif scope in {"all", "tagged"}:
                target_documents = [
                    document
                    for document in documents
                    if any(item.id == category.id for item in document.categories)
                ]
                expected_scope = "documents_with_category"
            elif self._mentions_all_library(content) or self._mentions_all_categories(content):
                target_documents = [
                    document
                    for document in documents
                    if any(item.id == category.id for item in document.categories)
                ]
                expected_scope = "documents_with_category"
                scope = "tagged"
            else:
                return "移除标签关联缺少明确论文范围，未执行；模糊指代不会默认扩大到全库。"
            affected_entities = [
                {
                    "document_id": document.id,
                    "name": document.display_name or document.filename,
                    "before_categories": [item.name for item in document.categories],
                    "after_categories": [item.name for item in document.categories if item.id != category.id],
                }
                for document in target_documents
            ]
            return _WritePreview(
                operation=operation,
                risk_level="destructive",
                tool_name="library.operator.clear_categories",
                tool_args={**arguments, "category_name": category.name, "document_ids": document_ids},
                target_entity={"type": "category", "id": category.id, "name": category.name},
                affected_count=len(target_documents),
                affected_entities=affected_entities,
                expected_scope=expected_scope,
                before_snapshot=before_snapshot,
                confirmation_phrase=f"确认移除{category.name}标签",
            )

        if operation == "clear_document_categories":
            if not document_ids:
                return "清空指定论文标签缺少明确 document_ids，未执行。"
            selected = set(document_ids)
            target_documents = [
                document
                for document in documents
                if document.id in selected and document.categories
            ]
            affected_entities = [
                {
                    "document_id": document.id,
                    "name": document.display_name or document.filename,
                    "before_categories": [item.name for item in document.categories],
                    "after_categories": [],
                }
                for document in target_documents
            ]
            relative_other_scope = arguments.get("relative_scope") == "other_tagged_documents"
            return _WritePreview(
                operation=operation,
                risk_level="critical" if relative_other_scope else "destructive",
                tool_name="library.operator.clear_categories",
                tool_args={**arguments, "document_ids": document_ids},
                target_entity={"type": "documents", "document_ids": document_ids},
                affected_count=len(target_documents),
                affected_entities=affected_entities,
                expected_scope="other_tagged_documents" if relative_other_scope else "explicit_documents",
                before_snapshot=before_snapshot,
                confirmation_phrase="确认清空其他有标签论文的标签" if relative_other_scope else "确认清空这些论文标签",
            )

        if operation == "clear_all_categories":
            if scope not in {"all", "tagged"}:
                return "全量清空标签缺少显式 scope=all/tagged，未执行。"
            target_documents = [
                document for document in documents if document.categories
            ] if scope == "tagged" else documents
            target_documents = [document for document in target_documents if document.categories]
            affected_entities = [
                {
                    "document_id": document.id,
                    "name": document.display_name or document.filename,
                    "before_categories": [item.name for item in document.categories],
                    "after_categories": [],
                }
                for document in target_documents
            ]
            return _WritePreview(
                operation=operation,
                risk_level="critical",
                tool_name="library.operator.clear_categories",
                tool_args={**arguments, "scope": scope},
                target_entity={"type": "all_category_links", "scope": scope},
                affected_count=len(target_documents),
                affected_entities=affected_entities,
                expected_scope=f"scope:{scope}",
                before_snapshot=before_snapshot,
                confirmation_phrase="确认清空所有标签",
            )

        return "清空/移除标签操作缺少明确 operation，未执行。"

    def _preview_assign_category_from_documents(
        self,
        *,
        category_name: str,
        document_ids: list[str],
        source_goal: str,
        risk_level: str = "scoped_write",
        expected_scope: str = "read_step_document_ids",
    ) -> _WritePreview:
        documents = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids)
        ]
        affected_entities = [
            {
                "document_id": document.id,
                "name": document.display_name or document.filename,
                "title": document.title,
                "before_categories": [category.name for category in document.categories],
                "after_categories": list(
                    dict.fromkeys([*[category.name for category in document.categories], category_name])
                ),
            }
            for document in documents
        ]
        return _WritePreview(
            operation="assign_category",
            risk_level=risk_level,
            tool_name="library.operator.assign_category",
            tool_args={
                "category_name": category_name,
                "category_names": [category_name],
                "document_ids": [document.id for document in documents],
                "scope": "documents",
            },
            target_entity={
                "type": "paper-category relation",
                "name": category_name,
                "operation_level": "relation-level",
                "source_goal": source_goal[:300],
            },
            affected_count=len(documents),
            affected_entities=affected_entities,
            expected_scope=expected_scope,
            before_snapshot=self._category_snapshot(),
            confirmation_phrase=f"确认给{len(documents)}篇论文添加{category_name}标签",
        )

    def _preview_delete_unused_categories(self, arguments: dict[str, Any]) -> _WritePreview | str:
        selector = str(arguments.get("selector") or "").strip().casefold()
        if selector != "unused":
            return "删除空标签/分类实体必须显式提供 selector=unused；本轮没有改动论文库。"
        stats = self._category_stats_payload()
        unused = [
            item
            for item in stats.get("categories") or []
            if isinstance(item, dict) and int(item.get("document_count") or 0) == 0
        ]
        unused_by_id = {str(item.get("id")): item for item in unused if item.get("id")}
        requested_ids = [str(item) for item in arguments.get("category_ids") or [] if str(item).strip()]
        if requested_ids:
            missing_or_linked = [category_id for category_id in requested_ids if category_id not in unused_by_id]
            if missing_or_linked:
                return (
                    "确认前复核发现目标标签/分类已经不存在，或不再是 count=0 的空实体；"
                    "已停止执行，请重新发起预览。"
                )
            targets = [unused_by_id[category_id] for category_id in requested_ids]
        else:
            targets = unused
        before_snapshot = self._category_snapshot()
        target_names = [str(item.get("name")) for item in targets if item.get("name")]
        target_ids = [str(item.get("id")) for item in targets if item.get("id")]
        return _WritePreview(
            operation="delete_unused_categories",
            risk_level="destructive",
            tool_name="library.operator.delete_unused_categories",
            tool_args={
                "selector": "unused",
                "category_ids": target_ids,
                "category_names": target_names,
            },
            target_entity={
                "type": "category_entities",
                "operation_level": "entity-level",
                "write_type": "delete",
                "target_type": "category",
                "will_delete_entities": True,
                "will_modify_relations": False,
                "selector": "unused",
                "count": len(targets),
                "names": target_names,
            },
            affected_count=len(targets),
            affected_entities=[
                {
                    "type": "category",
                    "id": str(item.get("id")),
                    "name": str(item.get("name")),
                    "document_count": int(item.get("document_count") or 0),
                }
                for item in targets
            ],
            expected_scope="categories_with_zero_documents",
            before_snapshot=before_snapshot,
            confirmation_phrase="确认删除空标签分类",
        )

    def _write_pending_tool_action(
        self,
        session_id: str,
        preview: _WritePreview,
        *,
        source_goal: str = "",
        plan_state: _PlanState | None = None,
        step_state: _StepState | None = None,
    ) -> None:
        is_assign = preview.operation == "assign_category"
        is_delete_unused = preview.operation == "delete_unused_categories"
        payload = {
            "type": "tool_action",
            "pending_plan_id": plan_state.plan_id if plan_state is not None else None,
            "operation": preview.operation,
            "operation_level": preview.target_entity.get("operation_level")
            or ("entity-level" if is_delete_unused else "relation-level"),
            "write_type": "append"
            if is_assign
            else "delete"
            if is_delete_unused
            else ("remove" if preview.operation == "remove_single_category_link" else "clear"),
            "target_type": "category" if is_delete_unused else "paper-category relation",
            "target_count": preview.affected_count,
            "targets": preview.affected_entities,
            "will_delete_entities": is_delete_unused,
            "will_modify_relations": not is_delete_unused,
            "requires_confirmation": True,
            "risk_level": preview.risk_level,
            "tool_name": preview.tool_name,
            "tool_args": preview.tool_args,
            "resolved_target_entity": preview.target_entity,
            "expected_affected_count": preview.affected_count,
            "expected_scope": preview.expected_scope,
            "affected_entities": preview.affected_entities,
            "before_snapshot": preview.before_snapshot,
            "confirmation_phrase": preview.confirmation_phrase,
            "source_goal": source_goal[:500],
            "original_user_prompt": source_goal[:500],
            "blocked_step": self._step_payload(step_state) if step_state is not None else None,
            "completed_steps": [
                self._step_payload(step)
                for step in (plan_state.steps if plan_state is not None else [])
                if step.status == "completed"
            ],
            "remaining_steps": [
                self._step_payload(step)
                for step in (plan_state.steps if plan_state is not None else [])
                if step_state is None or step.step_id != step_state.step_id
            ],
            "step_context": self._safe_trace_payload(plan_state.global_context) if plan_state is not None else {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_pending_action(session_id, payload)

    @staticmethod
    def _preview_confirmation_text(preview: _WritePreview) -> str:
        if preview.operation == "delete_unused_categories":
            names = "、".join(
                str(item.get("name"))
                for item in preview.affected_entities
                if isinstance(item, dict) and item.get("name")
            )
            return (
                f"需要确认：我找到了 {preview.affected_count} 个 count=0 的空标签/分类实体"
                f"{'：' + names if names else ''}。"
                "该操作只会删除这些标签/分类实体，不会修改任何论文的标签关系。"
                f"请回复「{preview.confirmation_phrase}」后我再执行。"
            )
        if preview.operation == "assign_category":
            names = "、".join(
                str(item.get("name"))
                for item in preview.affected_entities[:8]
                if isinstance(item, dict) and item.get("name")
            )
            suffix = "等" if preview.affected_count > 8 else ""
            target = preview.target_entity.get("name") or "目标标签"
            return (
                f"需要确认：我将给 {preview.affected_count} 篇论文添加标签/分类「{target}」"
                f"{'：' + names + suffix if names else ''}。"
                f"请回复「{preview.confirmation_phrase}」后我再执行。"
            )
        target = preview.target_entity.get("name") or preview.target_entity.get("scope") or preview.operation
        risk_note = "高风险" if preview.risk_level == "critical" else "破坏性"
        return (
            f"需要确认：这是{risk_note}写操作，目标是「{target}」，"
            f"预计影响 {preview.affected_count} 个对象，范围为 {preview.expected_scope}。"
            f"请回复「{preview.confirmation_phrase}」后我再执行。"
        )

    def _category_snapshot(self) -> dict[str, Any]:
        categories = self.category_repository.list_categories()
        documents = self.document_library_service.list_documents()
        return {
            "categories": [category.model_dump(mode="json") for category in categories],
            "documents": [
                {
                    "id": document.id,
                    "name": document.display_name or document.filename,
                    "category_ids": [category.id for category in document.categories],
                    "category_names": [category.name for category in document.categories],
                }
                for document in documents
            ],
        }

    def _rollback_category_snapshot(self, snapshot: dict[str, Any], document_ids: set[str] | None = None) -> None:
        if not snapshot:
            return
        existing_by_name = {category.name: category for category in self.category_repository.list_categories()}
        for category_payload in snapshot.get("categories") or []:
            name = str(category_payload.get("name") or "").strip()
            if not name or name in existing_by_name:
                continue
            try:
                created = self.category_repository.create_category(name, category_payload.get("color"))
                existing_by_name[created.name] = created
            except Exception:
                continue
        existing_by_name = {category.name: category for category in self.category_repository.list_categories()}
        for document_payload in snapshot.get("documents") or []:
            document_id = str(document_payload.get("id") or "")
            if not document_id or (document_ids is not None and document_id not in document_ids):
                continue
            category_ids: list[str] = []
            for name in document_payload.get("category_names") or []:
                category = existing_by_name.get(str(name))
                if category is not None:
                    category_ids.append(category.id)
            self.category_repository.replace_document_categories(document_id, category_ids)

    def _verify_write_observation(
        self,
        action: _ReactAction,
        observation: _ReactObservation,
        before_snapshot: dict[str, Any],
    ) -> str | None:
        if observation.status != "completed":
            return None
        if action.tool == "library.operator.create_category":
            names = self._category_names_from_arguments(action.arguments)
            existing = {category.name for category in self.category_repository.list_categories()}
            missing = [name for name in names if name not in existing]
            return f"标签/分类未创建成功：{', '.join(missing)}" if missing else None
        if action.tool == "library.operator.assign_category":
            names = self._category_names_from_arguments(action.arguments)
            affected_ids = {
                str(item)
                for item in observation.payload.get("affected_document_ids")
                or observation.payload.get("document_ids")
                or []
                if item
            }
            if not names or not affected_ids:
                return None
            documents = {
                document.id: document
                for document in self.document_library_service.list_documents()
                if document.id in affected_ids
            }
            for document_id in affected_ids:
                document = documents.get(document_id)
                if document is None:
                    return f"无法二次读取受影响论文：{document_id}"
                category_names = {category.name for category in document.categories}
                missing = [name for name in names if name not in category_names]
                if missing:
                    return f"论文 {document.display_name or document.filename} 未包含预期标签：{', '.join(missing)}"
            return None
        if action.tool == "library.operator.rename_category":
            source_name = str(action.arguments.get("source_category_name") or "")
            target_name = str(action.arguments.get("target_category_name") or "")
            categories = self.category_repository.list_categories()
            if source_name and any(category.name == source_name for category in categories):
                return f"源标签仍然存在：{source_name}"
            if target_name and not any(category.name == target_name for category in categories):
                return f"目标标签不存在：{target_name}"
            before_by_doc = {
                str(item.get("id")): [str(name) for name in item.get("category_names") or []]
                for item in before_snapshot.get("documents") or []
                if item.get("id")
            }
            after_by_doc = {
                document.id: [category.name for category in document.categories]
                for document in self.document_library_service.list_documents()
            }
            for document_id, before_names in before_by_doc.items():
                after_names = after_by_doc.get(document_id, [])
                if source_name in before_names:
                    expected = [name for name in before_names if name not in {source_name, target_name}]
                    if target_name:
                        expected.append(target_name)
                    if set(after_names) != set(expected):
                        return f"璁烘枃 {document_id} 鐨勯噸鍛藉悕/鍚堝苟缁撴灉瓒呭嚭棰勬湡鑼冨洿"
                elif set(after_names) != set(before_names):
                    return f"闈炴簮鏍囩璁烘枃 {document_id} 鐨勬爣绛惧彂鐢熶簡鍙樺寲"
        return None

    def _verify_clear_categories_effect(
        self,
        *,
        operation: str,
        category_name: str,
        affected_document_ids: set[str],
        before_snapshot: dict[str, Any],
        expected_count: Any,
    ) -> str | None:
        if expected_count is not None:
            try:
                expected_number = int(expected_count)
            except (TypeError, ValueError):
                expected_number = None
            if expected_number is not None and len(affected_document_ids) != expected_number:
                return f"影响数量不一致：预期 {expected_number}，实际 {len(affected_document_ids)}"
        before_by_doc = {
            str(item.get("id")): [str(name) for name in item.get("category_names") or []]
            for item in before_snapshot.get("documents") or []
            if item.get("id")
        }
        after_by_doc = {
            str(item.get("id")): [str(name) for name in item.get("category_names") or []]
            for item in self._category_snapshot().get("documents") or []
            if item.get("id")
        }
        if operation == "remove_single_category_link":
            scoped_ids = set(affected_document_ids)
            for document_id, before_names in before_by_doc.items():
                after_names = after_by_doc.get(document_id, [])
                should_remove = category_name in before_names and (not scoped_ids or document_id in scoped_ids)
                expected_names = [name for name in before_names if name != category_name] if should_remove else before_names
                if should_remove:
                    if after_names != expected_names:
                        return f"论文 {document_id} 的标签变化超出目标标签范围"
                elif after_names != before_names:
                    return f"非目标论文 {document_id} 的标签发生了变化"
            return None
        if operation == "clear_document_categories":
            for document_id, before_names in before_by_doc.items():
                after_names = after_by_doc.get(document_id, [])
                if document_id in affected_document_ids:
                    if after_names:
                        return f"论文 {document_id} 未被清空标签"
                elif after_names != before_names:
                    return f"非目标论文 {document_id} 的标签发生了变化"
            return None
        if operation == "clear_all_categories":
            for document_id, after_names in after_by_doc.items():
                if after_names:
                    return f"论文 {document_id} 仍存在标签，清空结果未通过校验"
            return None
        return f"未知清空标签 operation：{operation}"

    def _verify_delete_category_effect(self, category_name: str, before_snapshot: dict[str, Any]) -> str | None:
        if not before_snapshot:
            return None
        after_snapshot = self._category_snapshot()
        after_categories = {str(item.get("name")) for item in after_snapshot.get("categories") or []}
        before_categories = {str(item.get("name")) for item in before_snapshot.get("categories") or []}
        unexpected_missing = sorted(name for name in before_categories - {category_name} if name not in after_categories)
        if unexpected_missing:
            return f"非目标标签被删除：{', '.join(unexpected_missing)}"
        if category_name in after_categories:
            return f"目标标签仍然存在：{category_name}"
        before_by_doc = {
            str(item.get("id")): [str(name) for name in item.get("category_names") or []]
            for item in before_snapshot.get("documents") or []
            if item.get("id")
        }
        after_by_doc = {
            str(item.get("id")): [str(name) for name in item.get("category_names") or []]
            for item in after_snapshot.get("documents") or []
            if item.get("id")
        }
        for document_id, before_names in before_by_doc.items():
            expected_names = [name for name in before_names if name != category_name]
            if after_by_doc.get(document_id, []) != expected_names:
                return f"论文 {document_id} 的标签变化超出目标标签范围"
        return None

    def _verify_delete_unused_categories_effect(
        self,
        *,
        deleted_category_ids: set[str],
        deleted_category_names: set[str],
        before_snapshot: dict[str, Any],
        expected_count: Any,
    ) -> str | None:
        if expected_count is not None:
            try:
                expected_number = int(expected_count)
            except (TypeError, ValueError):
                expected_number = None
            if expected_number is not None and len(deleted_category_ids) != expected_number:
                return f"删除数量不一致：预期 {expected_number}，实际 {len(deleted_category_ids)}"
        before_categories = [
            item
            for item in before_snapshot.get("categories") or []
            if isinstance(item, dict) and item.get("id")
        ]
        before_counts: dict[str, int] = {str(item.get("id")): 0 for item in before_categories}
        before_by_doc: dict[str, list[str]] = {}
        for item in before_snapshot.get("documents") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            document_id = str(item.get("id"))
            category_ids = [str(value) for value in item.get("category_ids") or [] if value]
            category_names = [str(value) for value in item.get("category_names") or [] if value]
            before_by_doc[document_id] = category_names
            for category_id in category_ids:
                before_counts[category_id] = before_counts.get(category_id, 0) + 1
        non_empty_before_ids = {
            category_id
            for category_id, count in before_counts.items()
            if count > 0
        }
        after_snapshot = self._category_snapshot()
        after_categories = {
            str(item.get("id")): str(item.get("name"))
            for item in after_snapshot.get("categories") or []
            if isinstance(item, dict) and item.get("id")
        }
        for category_id in deleted_category_ids:
            if category_id in after_categories:
                return f"目标空标签/分类实体仍然存在：{after_categories[category_id]}"
        missing_non_empty = [
            category_id
            for category_id in non_empty_before_ids
            if category_id not in after_categories
        ]
        if missing_non_empty:
            return "非空标签/分类实体被误删，已阻止标记成功。"
        after_by_doc = {
            str(item.get("id")): [str(name) for name in item.get("category_names") or [] if name]
            for item in after_snapshot.get("documents") or []
            if isinstance(item, dict) and item.get("id")
        }
        if set(before_by_doc) != set(after_by_doc):
            return "论文集合发生变化，删除空标签实体不应修改论文。"
        for document_id, before_names in before_by_doc.items():
            if after_by_doc.get(document_id, []) != before_names:
                return f"论文 {document_id} 的标签关系发生变化；删除空标签实体不应修改论文-标签关系。"
        if deleted_category_names:
            after_names = set(after_categories.values())
            still_present = sorted(name for name in deleted_category_names if name in after_names)
            if still_present:
                return f"目标空标签/分类实体仍然存在：{', '.join(still_present)}"
        return None

    def _tool_intent_degraded(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        reason = str(arguments.get("reason") or "Configured LLM did not produce a usable tool plan.")
        return _ReactObservation(
            tool=self._INTERNAL_DEGRADED_TOOL,
            status="degraded",
            summary=(
                "模型意图识别/工具规划没有返回可执行的结构化结果，本轮已停止自动工具调用；"
                "没有用规则分支冒充模型理解。"
            ),
            payload={"reason": reason, "llm_configured": bool(self.api_key), "fallback_used": False},
        )

    def _tool_category_semantics_conflict(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        return _ReactObservation(
            tool=self._INTERNAL_CATEGORY_CONFLICT_TOOL,
            status="needs_clarification",
            summary="当前系统中标签和分类是同一字段，无法同时清除标签又保留分类；请确认是否要清空分类。",
            payload={
                "library_mutated": False,
                "operation_summary": "tag_category_same_field_conflict",
                "verified_state": self._category_stats_payload(),
            },
        )

    def _tool_registry_list(
        self,
        _run_id: str,
        _session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        tools = self._react_tool_specs()
        return _ReactObservation(
            tool="tool.registry.list",
            status="completed",
            summary=f"已读取 {len(tools)} 个可用工具及其权限说明。",
            payload={"tools": tools},
        )

    def _tool_library_stats(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        def worker() -> _TaskOutcome:
            payload = self._library_stats_payload()
            return _TaskOutcome(
                summary=f"Found {payload['total']} library documents.",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read PaperDesk library statistics from SQLite.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "tool": "library.explorer.stats"},
            worker=worker,
        )
        payload = outcome.payload
        return _ReactObservation(
            tool="library.explorer.stats",
            status="completed",
            summary=f"SQLite 中共有 {payload['total']} 篇论文，{payload['ready']} 篇可用。",
            payload=payload,
        )

    def _tool_category_stats(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        def worker() -> _TaskOutcome:
            payload = self._category_stats_payload()
            return _TaskOutcome(
                summary=(
                    f"Found {payload['category_count']} categories, "
                    f"{payload['tagged_document_count']} tagged documents."
                ),
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read PaperDesk category and tag coverage statistics.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "tool": "library.explorer.category_stats"},
            worker=worker,
        )
        payload = outcome.payload
        return _ReactObservation(
            tool="library.explorer.category_stats",
            status="completed",
            summary=(
                f"SQLite 中有 {payload['tagged_document_count']} 篇有标签论文，"
                f"{payload['untagged_document_count']} 篇无标签论文，"
                f"{payload['category_count']} 类标签。"
            ),
            payload=payload,
        )

    def _tool_find_documents(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        query = str(arguments.get("query") or content)
        expected = str(arguments.get("expected") or "many")
        selected_ids = [str(item) for item in arguments.get("selected_document_ids") or [] if item]
        allow_all = bool(arguments.get("allow_all")) or self._mentions_all_library(content)
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        category_names = [
            self._clean_category_name(str(item))
            for item in arguments.get("category_names") or []
            if self._clean_category_name(str(item))
        ]
        if category_name and category_name not in category_names:
            category_names = [category_name, *category_names]
        if not category_names:
            category_names = self._category_entity_names_for_request(content)
        if self._is_assignment_intent(content) or self._is_clear_categories_intent(content):
            category_names = []
        if category_names and not category_name:
            category_name = category_names[0]
        if not category_name and self._is_summary_request(content, selected_ids):
            mentioned_category = self._extract_existing_category_mention(content)
            if mentioned_category:
                category_name = mentioned_category
                category_names = [mentioned_category]

        def worker() -> _TaskOutcome:
            category_lookup = self._category_lookup_payload(category_names)
            if category_names:
                documents = [
                    document
                    for document in self.document_library_service.list_documents()
                    if document.status == "ready"
                    and any(category.name in set(category_lookup["matched_names"]) for category in document.categories)
                ]
                missing_names = category_lookup["missing_names"]
                ambiguous_names = category_lookup["ambiguous_names"]
            elif category_name:
                documents = self._documents_for_category(category_name)
                missing_names = []
                ambiguous_names = []
            else:
                documents = self._resolve_documents(query, selected_ids, allow_all=allow_all)
                missing_names = []
                ambiguous_names = []
            payload = {
                "documents": [self._document_payload(document) for document in documents],
                "document_ids": [document.id for document in documents],
                "candidates": self._candidate_titles() if not category_names else category_lookup["candidate_names"],
                "category_name": category_name or None,
                "category_names": category_names,
                "category_lookup": category_lookup if category_names else None,
                "missing_category_names": missing_names,
                "ambiguous_category_names": ambiguous_names,
            }
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s).",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Fuzzy match library documents from the user request.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "query": query, "tool": "library.explorer.find_documents"},
            worker=worker,
        )
        documents = outcome.payload.get("documents") or []
        category_lookup = outcome.payload.get("category_lookup")
        if isinstance(category_lookup, dict):
            if category_lookup.get("missing_names"):
                missing = "、".join(str(item) for item in category_lookup.get("missing_names") or [])
                candidates = "、".join(str(item) for item in category_lookup.get("candidate_names") or [])
                suffix = f"；现有相近标签/分类：{candidates}" if candidates else ""
                return _ReactObservation(
                    tool="library.explorer.find_documents",
                    status="needs_clarification",
                    summary=f"没有找到标签/分类「{missing}」{suffix}。",
                    payload=outcome.payload,
                )
            if category_lookup.get("ambiguous_names"):
                ambiguous = "、".join(str(item) for item in category_lookup.get("ambiguous_names") or [])
                candidates = "、".join(str(item) for item in category_lookup.get("candidate_names") or [])
                return _ReactObservation(
                    tool="library.explorer.find_documents",
                    status="needs_clarification",
                    summary=f"标签/分类「{ambiguous}」匹配到多个候选，请确认：{candidates}",
                    payload=outcome.payload,
                )
            if not documents:
                matched = "、".join(str(item) for item in category_lookup.get("matched_names") or outcome.payload.get("category_names") or [])
                return _ReactObservation(
                    tool="library.explorer.find_documents",
                    status="completed",
                    summary=f"已识别到标签/分类「{matched}」，但当前没有关联 ready 论文。",
                    payload=outcome.payload,
                )
        if not documents:
            return _ReactObservation(
                tool="library.explorer.find_documents",
                status="needs_clarification",
                summary="没有在论文库中唯一定位到用户提到的论文。",
                payload=outcome.payload,
            )
        if expected == "one" and len(documents) > 1:
            return _ReactObservation(
                tool="library.explorer.find_documents",
                status="needs_clarification",
                summary="匹配到多篇候选论文，需要用户确认具体是哪一篇。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="library.explorer.find_documents",
            status="completed",
            summary=(
                f"已识别到标签/分类「{'、'.join(outcome.payload.get('category_names') or [category_name])}」，"
                f"下面共有 {len(documents)} 篇 ready 论文。"
                if isinstance(category_lookup, dict)
                else f"已定位 {len(documents)} 篇论文。"
            ),
            payload=outcome.payload,
        )

    def _tool_document_metadata(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            document_ids = self._document_ids_from_observations(observations)
        requested_fields = [
            str(item)
            for item in arguments.get("requested_fields") or self._requested_metadata_fields(content)
            if str(item)
        ]

        def worker() -> _TaskOutcome:
            documents = [
                document
                for document in self.document_library_service.list_documents()
                if document.id in set(document_ids)
            ]
            payload = {
                "documents": [
                    self._document_metadata_payload(document, requested_fields)
                    for document in documents
                ],
                "document_ids": [document.id for document in documents],
                "requested_fields": requested_fields,
            }
            return _TaskOutcome(
                summary=f"Read metadata for {len(documents)} document(s).",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read selected document metadata fields.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "document_ids": document_ids, "requested_fields": requested_fields},
            worker=worker,
        )
        documents = outcome.payload.get("documents") or []
        if not documents:
            return _ReactObservation(
                tool="library.explorer.document_metadata",
                status="needs_clarification",
                summary="没有找到可读取元数据的论文。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="library.explorer.document_metadata",
            status="completed",
            summary=f"已读取 {len(documents)} 篇论文的元数据。",
            payload=outcome.payload,
        )

    def _tool_document_categories(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            document_ids = self._document_ids_from_observations(observations)
        if not document_ids and self._requests_all_document_categories(content):
            document_ids = self._document_category_target_ids(content, observations)

        def worker() -> _TaskOutcome:
            documents = [document for document in self.document_library_service.list_documents() if document.id in set(document_ids)]
            payload = {
                "documents": [
                    {
                        **self._document_payload(document),
                        "categories": [category.name for category in document.categories],
                    }
                    for document in documents
                ],
                "document_ids": [document.id for document in documents],
            }
            return _TaskOutcome(
                summary=f"Read categories for {len(documents)} document(s).",
                payload=payload,
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read selected document categories as tags.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "document_ids": document_ids},
            worker=worker,
        )
        documents = outcome.payload.get("documents") or []
        if not documents:
            return _ReactObservation(
                tool="library.explorer.document_categories",
                status="needs_clarification",
                summary="没有找到可读取标签的论文。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="library.explorer.document_categories",
            status="completed",
            summary=f"已读取 {len(documents)} 篇论文的标签。",
            payload=outcome.payload,
        )

    def _tool_create_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        category_names = self._category_names_from_arguments(arguments)
        if not category_names:
            category_names = self._extract_category_names_from_request(content)
        if not category_names:
            category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
            if not category_name:
                category_name = self._extract_category_name_from_request(content) or ""
            category_names = [category_name] if category_name else []
        validation_error = self._category_names_validation_error(category_names)
        if validation_error:
            return _ReactObservation(
                tool="library.operator.create_category",
                status="validation_failed",
                summary=validation_error,
                payload={"category_names": category_names, "category_name": category_names[0] if category_names else ""},
            )

        def worker() -> _TaskOutcome:
            categories = []
            created_names = []
            existing_names = []
            for category_name in category_names:
                category, created = self._get_or_create_category(category_name)
                categories.append(category)
                if created:
                    created_names.append(category.name)
                else:
                    existing_names.append(category.name)
            return _TaskOutcome(
                summary=f"Categories ready: {', '.join(category.name for category in categories)}.",
                payload={
                    "categories": [category.model_dump(mode="json") for category in categories],
                    "category": categories[0].model_dump(mode="json"),
                    "category_names": [category.name for category in categories],
                    "category_name": categories[0].name,
                    "created_names": created_names,
                    "existing_names": existing_names,
                    "created": bool(created_names),
                    "library_mutated": bool(created_names),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Create a non-destructive category/tag if it does not already exist.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_names": category_names},
            worker=worker,
        )
        names = "、".join(outcome.payload.get("category_names") or [])
        created_names = outcome.payload.get("created_names") or []
        return _ReactObservation(
            tool="library.operator.create_category",
            status="completed",
            summary=f"已确认标签/分类：{names}。新建 {len(created_names)} 个。",
            payload=outcome.payload,
        )

    def _tool_assign_category(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        category_names = self._category_names_from_arguments(arguments)
        if not category_names:
            category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
            if not category_name or category_name in {"这个标签", "该标签", "此标签"}:
                if self._needs_untagged_assignment(content):
                    category_names = self._extract_category_names_from_request(content)
                else:
                    category_names = self._category_names_from_observations(observations) or self._extract_category_names_from_request(content)
            else:
                category_names = [category_name]
        validation_error = self._category_names_validation_error(category_names)
        if validation_error:
            return _ReactObservation(
                tool="library.operator.assign_category",
                status="validation_failed",
                summary=validation_error,
                payload={"category_names": category_names, "category_name": category_names[0] if category_names else ""},
            )

        scope = str(arguments.get("scope") or "")
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if self._needs_untagged_assignment(content):
            document_ids = []
            scope = "untagged"
        if not document_ids and self._should_target_untagged_from_context(content, self._read_react_state(session.id)):
            document_ids = []
            scope = "untagged"
        if scope == "last_referenced" or (
            not document_ids and scope != "untagged" and self._mentions_previous_referent(content)
        ):
            document_ids = self._state_document_ids(self._read_react_state(session.id))
            scope = "last_referenced"
        if not document_ids and scope != "untagged":
            document_ids = self._document_ids_from_observations(observations)

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            if scope == "untagged":
                target_documents = [document for document in documents if not document.categories]
            else:
                selected = set(document_ids)
                target_documents = [document for document in documents if document.id in selected]
            target_categories = []
            created_names = []
            for category_name in category_names:
                category, created = self._get_or_create_category(category_name)
                target_categories.append(category)
                if created:
                    created_names.append(category.name)
            updated = []
            for document in target_documents:
                existing_ids = [item.id for item in document.categories]
                next_ids = list(dict.fromkeys([*existing_ids, *[category.id for category in target_categories]]))
                updated_categories = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [item.name for item in updated_categories],
                    }
                )
            verified_documents = self.document_library_service.list_documents()
            verified_by_category = {
                category_name: [
                    document.id
                    for document in verified_documents
                    if any(category_item.name == category_name for category_item in document.categories)
                ]
                for category_name in category_names
            }
            return _TaskOutcome(
                summary=f"Assigned {', '.join(category_names)} to {len(updated)} document(s).",
                payload={
                    "categories": [category.model_dump(mode="json") for category in target_categories],
                    "category": target_categories[0].model_dump(mode="json"),
                    "category_names": category_names,
                    "category_name": category_names[0],
                    "_resolved_action": arguments.get("_resolved_action"),
                    "created_names": created_names,
                    "scope": scope or "documents",
                    "updated_count": len(updated),
                    "documents": updated,
                    "document_ids": [item["id"] for item in updated],
                    "verified_tagged_document_ids": verified_by_category.get(category_names[0], []),
                    "verified_by_category": verified_by_category,
                    "verified_state": self._category_stats_payload(),
                    "library_mutated": bool(updated),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Append a category/tag to matched library documents.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_names": category_names, "scope": scope, "document_ids": document_ids},
            worker=worker,
        )
        payload = outcome.payload
        if payload["updated_count"] == 0 and scope != "untagged":
            return _ReactObservation(
                tool="library.operator.assign_category",
                status="needs_clarification",
                summary="没有定位到需要打标签的论文。",
                payload=payload,
            )
        return _ReactObservation(
            tool="library.operator.assign_category",
            status="completed",
            summary=f"已把标签/分类「{'、'.join(payload.get('category_names') or [payload['category_name']])}」追加到 {payload['updated_count']} 篇论文。",
            payload=payload,
        )

    def _tool_rename_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        source_name = self._clean_category_name(str(arguments.get("source_category_name") or ""))
        target_name = self._clean_category_name(str(arguments.get("target_category_name") or ""))
        if not source_name or not target_name:
            extracted = self._extract_category_rename_request(content)
            if extracted:
                source_name = source_name or extracted[0]
                target_name = target_name or extracted[1]
        source_error = self._category_name_validation_error(source_name)
        target_error = self._category_name_validation_error(target_name)
        if source_error or target_error:
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="validation_failed",
                summary=source_error or target_error or "标签名称无效，本轮没有改动论文库。",
                payload={"source_category_name": source_name, "target_category_name": target_name},
            )
        if source_name == target_name:
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="completed",
                summary=f"源标签和目标标签同为「{source_name}」，无需改动。",
                payload={
                    "source_category_name": source_name,
                    "target_category_name": target_name,
                    "updated_count": 0,
                    "document_ids": [],
                    "merged": False,
                    "library_mutated": False,
                },
            )

        def worker() -> _TaskOutcome:
            categories = self.category_repository.list_categories()
            source = next((category for category in categories if category.name == source_name), None)
            if source is None:
                return _TaskOutcome(
                    summary=f"Source category not found: {source_name}.",
                    payload={
                        "source_category_name": source_name,
                        "target_category_name": target_name,
                        "source_found": False,
                        "updated_count": 0,
                        "document_ids": [],
                        "library_mutated": False,
                    },
                )
            target = next((category for category in categories if category.name == target_name), None)
            if target is None:
                target = self.category_repository.create_category(target_name, source.color)
                merged = False
            else:
                merged = True

            documents = self.document_library_service.list_documents()
            updated_documents = []
            for document in documents:
                if not any(category.id == source.id for category in document.categories):
                    continue
                next_ids = [
                    category.id
                    for category in document.categories
                    if category.id != source.id and category.id != target.id
                ]
                next_ids.append(target.id)
                categories_after = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated_documents.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [category.name for category in categories_after],
                    }
                )
            self.category_repository.delete_category(source.id)
            return _TaskOutcome(
                summary=f"Renamed category {source_name} to {target_name} for {len(updated_documents)} documents.",
                payload={
                    "source_category_name": source_name,
                    "target_category_name": target_name,
                    "source_category_id": source.id,
                    "target_category_id": target.id,
                    "source_found": True,
                    "merged": merged,
                    "operation_type": "merge_existing_target" if merged else "rename_new_target",
                    "updated_count": len(updated_documents),
                    "documents": updated_documents,
                    "document_ids": [document["id"] for document in updated_documents],
                    "library_mutated": True,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Safely rename or merge a category while preserving document-category links.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"source_category_name": source_name, "target_category_name": target_name},
            worker=worker,
        )
        payload = outcome.payload
        if not payload.get("source_found", True):
            return _ReactObservation(
                tool="library.operator.rename_category",
                status="needs_clarification",
                summary=f"没有找到名为「{source_name}」的源标签，因此没有改动论文库。",
                payload=payload,
            )
        verb = "合并到已有标签" if payload.get("merged") else "重命名为"
        return _ReactObservation(
            tool="library.operator.rename_category",
            status="completed",
            summary=(
                f"已将标签/分类「{source_name}」{verb}「{target_name}」，"
                f"保留并迁移了 {payload.get('updated_count', 0)} 篇论文的标签关联。"
            ),
            payload=payload,
        )

    def _tool_delete_unused_categories(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        _ = session, content, observations
        if not arguments.get("__confirmed_pending"):
            return _ReactObservation(
                tool="library.operator.delete_unused_categories",
                status="confirmation_required",
                summary="删除空标签/分类实体尚未经过 preview + 用户确认，已拒绝直接执行。",
                payload={"library_mutated": False, "guardrail": "confirmation_required", "operation_level": "entity-level"},
            )
        selector = str(arguments.get("selector") or "").strip().casefold()
        if selector != "unused":
            return _ReactObservation(
                tool="library.operator.delete_unused_categories",
                status="validation_failed",
                summary="删除空标签/分类实体必须使用 selector=unused；本轮没有改动论文库。",
                payload={"library_mutated": False, "operation_level": "entity-level"},
            )
        category_ids = [str(item) for item in arguments.get("category_ids") or [] if str(item).strip()]
        before_snapshot = arguments.get("__before_snapshot") if isinstance(arguments.get("__before_snapshot"), dict) else self._category_snapshot()
        expected_count = arguments.get("__expected_affected_count")

        def worker() -> _TaskOutcome:
            stats = self._category_stats_payload()
            unused_by_id = {
                str(item.get("id")): item
                for item in stats.get("categories") or []
                if isinstance(item, dict) and item.get("id") and int(item.get("document_count") or 0) == 0
            }
            targets = [unused_by_id[category_id] for category_id in category_ids if category_id in unused_by_id]
            if len(targets) != len(category_ids):
                return _TaskOutcome(
                    summary="Unused category quick recheck failed; no category entities were deleted.",
                    payload={
                        "operation": "delete_unused_categories",
                        "operation_level": "entity-level",
                        "selector": "unused",
                        "deleted_count": 0,
                        "library_mutated": False,
                        "verification_error": "quick_recheck_failed",
                    },
                )
            deleted_names: list[str] = []
            deleted_ids: list[str] = []
            for target in targets:
                category_id = str(target.get("id"))
                category_name = str(target.get("name") or "")
                deleted = self.category_repository.delete_category(category_id)
                if deleted is not None:
                    deleted_ids.append(category_id)
                    deleted_names.append(category_name)
            verified_state = self._category_stats_payload()
            verification_error = self._verify_delete_unused_categories_effect(
                deleted_category_ids=set(deleted_ids),
                deleted_category_names=set(deleted_names),
                before_snapshot=before_snapshot,
                expected_count=expected_count,
            )
            if verification_error:
                self._rollback_category_snapshot(before_snapshot)
                verified_state = self._category_stats_payload()
            return _TaskOutcome(
                summary=f"Deleted {len(deleted_names)} unused category entities.",
                payload={
                    "operation": "delete_unused_categories",
                    "operation_level": "entity-level",
                    "selector": "unused",
                    "deleted_count": len(deleted_names),
                    "deleted_category_ids": deleted_ids,
                    "deleted_category_names": deleted_names,
                    "affected_document_count": 0,
                    "affected_document_ids": [],
                    "library_mutated": bool(deleted_names) and not verification_error,
                    "verified_state": verified_state,
                    "operation_summary": "delete_unused_category_entities",
                    "verification_error": verification_error,
                    "rollback_attempted": bool(verification_error),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Delete only unused tag/category entities with zero linked documents.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"selector": selector, "category_ids": category_ids, "operation_level": "entity-level"},
            worker=worker,
        )
        payload = outcome.payload
        if payload.get("verification_error"):
            return _ReactObservation(
                tool="library.operator.delete_unused_categories",
                status="failed",
                summary=f"删除空标签/分类实体后的校验失败，已尝试回滚：{payload.get('verification_error')}",
                payload=payload,
            )
        return _ReactObservation(
            tool="library.operator.delete_unused_categories",
            status="completed",
            summary=(
                f"已删除 {payload.get('deleted_count', 0)} 个空标签/分类实体："
                f"{'、'.join(payload.get('deleted_category_names') or [])}。"
                "已验证没有修改任何论文的标签关系。"
            ),
            payload=payload,
        )

    def _tool_clear_categories(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        if self._is_tag_category_semantics_conflict(content):
            return self._tool_category_semantics_conflict(run_id, session, content, arguments, observations)
        if not arguments.get("__confirmed_pending"):
            return _ReactObservation(
                tool="library.operator.clear_categories",
                status="confirmation_required",
                summary="该清空/移除标签操作尚未经过 preview + 用户确认，已拒绝直接执行。",
                payload={"library_mutated": False, "guardrail": "confirmation_required"},
            )
        operation = self._infer_clear_categories_operation(arguments)
        scope = str(arguments.get("scope") or "")
        category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        before_snapshot = arguments.get("__before_snapshot") if isinstance(arguments.get("__before_snapshot"), dict) else self._category_snapshot()
        expected_count = arguments.get("__expected_affected_count")

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            categories = self.category_repository.list_categories()
            target_category = next((item for item in categories if item.name == category_name), None) if category_name else None
            if operation == "remove_single_category_link":
                if target_category is None:
                    return _TaskOutcome(
                        summary=f"Target category not found: {category_name}.",
                        payload={
                            "operation": operation,
                            "category_name": category_name,
                            "updated_count": 0,
                            "library_mutated": False,
                            "verification_error": "target_category_missing",
                        },
                    )
                selected = set(document_ids)
                if selected and scope not in {"all", "tagged"}:
                    target_documents = [
                        document
                        for document in documents
                        if document.id in selected and any(category.id == target_category.id for category in document.categories)
                    ]
                elif scope in {"all", "tagged"}:
                    target_documents = [
                        document
                        for document in documents
                        if any(category.id == target_category.id for category in document.categories)
                    ]
                elif self._mentions_all_library(content) or self._mentions_all_categories(content):
                    target_documents = [
                        document
                        for document in documents
                        if any(category.id == target_category.id for category in document.categories)
                    ]
                else:
                    return _TaskOutcome(
                        summary="Missing scoped document ids for relation removal.",
                        payload={
                            "operation": operation,
                            "category_name": category_name,
                            "updated_count": 0,
                            "library_mutated": False,
                            "verification_error": "missing_document_scope",
                        },
                    )
            elif operation == "clear_document_categories":
                selected = set(document_ids)
                target_documents = [document for document in documents if document.id in selected]
            elif operation == "clear_all_categories":
                if scope == "all":
                    target_documents = [document for document in documents if document.categories]
                elif scope == "tagged":
                    target_documents = [document for document in documents if document.categories]
                else:
                    return _TaskOutcome(
                        summary="Invalid critical clear scope.",
                        payload={"operation": operation, "scope": scope, "library_mutated": False},
                    )
            else:
                return _TaskOutcome(
                    summary="Invalid clear category operation.",
                    payload={"operation": operation, "library_mutated": False},
                )
            updated_documents = []
            for document in target_documents:
                if not document.categories:
                    continue
                if operation == "remove_single_category_link":
                    next_ids = [
                        category.id
                        for category in document.categories
                        if target_category is not None and category.id != target_category.id
                    ]
                else:
                    next_ids = []
                categories_after = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated_documents.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "title": document.title,
                        "categories": [category.name for category in categories_after],
                        "before_categories": [category.name for category in document.categories],
                    }
                )
            verified_state = self._category_stats_payload()
            verification_error = self._verify_clear_categories_effect(
                operation=operation,
                category_name=category_name,
                affected_document_ids={document["id"] for document in updated_documents},
                before_snapshot=before_snapshot,
                expected_count=expected_count,
            )
            if verification_error:
                self._rollback_category_snapshot(before_snapshot)
                verified_state = self._category_stats_payload()
            return _TaskOutcome(
                summary=f"Updated category links for {len(updated_documents)} document(s).",
                payload={
                    "operation": operation,
                    "scope": scope,
                    "category_name": category_name or None,
                    "_resolved_action": arguments.get("_resolved_action"),
                    "updated_count": len(updated_documents),
                    "documents": updated_documents,
                    "document_ids": [document["id"] for document in updated_documents],
                    "affected_document_ids": [document["id"] for document in updated_documents],
                    "library_mutated": bool(updated_documents) and not verification_error,
                    "verified_state": verified_state,
                    "operation_summary": operation,
                    "verification_error": verification_error,
                    "rollback_attempted": bool(verification_error),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Clear document category/tag links. Tags and categories are the same field in PaperDesk.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={
                "scope": scope,
                "category_name": category_name or None,
                "document_ids": document_ids,
                "operation": operation,
                "tag_category_same_field": True,
            },
            worker=worker,
        )
        payload = outcome.payload
        if payload.get("verification_error"):
            return _ReactObservation(
                tool="library.operator.clear_categories",
                status="failed",
                summary=f"写操作后的范围校验失败，已尝试回滚：{payload.get('verification_error')}",
                payload=payload,
            )
        if operation == "remove_single_category_link":
            summary = f"已从 {payload.get('updated_count', 0)} 篇论文中移除标签/分类「{category_name}」，其他标签已保留并二次校验。"
        elif operation == "clear_all_categories":
            summary = f"已按确认范围清空 {payload.get('updated_count', 0)} 篇论文的分类/标签关系，并二次校验。"
        else:
            summary = f"已按确认范围清空 {payload.get('updated_count', 0)} 篇论文的分类/标签关系，并二次校验。"
        return _ReactObservation(
            tool="library.operator.clear_categories",
            status="completed",
            summary=summary,
            payload=payload,
        )

    def _tool_retrieve_evidence(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            category_filter_ids = self._category_filter_document_ids_from_observations(observations)
            document_ids = (
                category_filter_ids
                if category_filter_ids is not None
                else self._document_ids_from_observations(observations)
            )
        if not document_ids:
            category_name = self._category_name_from_request_or_observations(content, observations)
            if category_name and self._category_exists(category_name):
                document_ids = [document.id for document in self._documents_for_category(category_name)]
        documents = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids) and document.status == "ready"
        ]

        def worker() -> _TaskOutcome:
            evidence = self.rag_service.retrieve_evidence(
                question=str(arguments.get("question") or content),
                documents=documents,
                top_k=min(10, max(4, len(documents) * 3)),
            )
            return _TaskOutcome(
                summary=f"Retrieved {len(evidence)} evidence item(s).",
                payload={
                    "document_ids": [document.id for document in documents],
                    "evidence_items": [item.model_dump(mode="json") for item in evidence],
                    "evidence_quality": self.rag_service.last_evidence_quality.model_dump(mode="json"),
                    "cache_hit": self.rag_service.last_retrieval_cache_hit,
                    "retrieval_strategy": self.rag_service.last_retrieval_strategy,
                    "question": str(arguments.get("question") or content),
                },
            )

        if not documents:
            return _ReactObservation(
                tool="evidence.retriever.search",
                status="needs_clarification",
                summary=self._no_ready_documents_message(document_ids),
                payload={"document_ids": document_ids},
            )
        try:
            outcome = self._run_subagent(
                run_id=run_id,
                role="evidence-retriever",
                profile=SubagentProfile.EXPLORE,
                goal="Retrieve local evidence for the ReAct answer.",
                tool_policy=ToolPolicy(read_only=True),
                context_bundle={"question": content, "document_ids": document_ids},
                worker=worker,
            )
        except Exception as exc:
            return _ReactObservation(
                tool="evidence.retriever.search",
                status="failed",
                summary=self._retrieval_service_unavailable_message(),
                payload={
                    "document_ids": document_ids,
                    "error_type": exc.__class__.__name__,
                    "requires_service": "milvus",
                },
            )
        retrieval_strategy = str(outcome.payload.get("retrieval_strategy") or "")
        evidence_count = len(outcome.payload.get("evidence_items") or [])
        summary = (
            f"Milvus 向量服务不可用，已通过论文正文关键词检索到 {evidence_count} 条证据。"
            if retrieval_strategy == "keyword_only_vector_unavailable"
            else f"已检索到 {evidence_count} 条证据。"
        )
        return _ReactObservation(
            tool="evidence.retriever.search",
            status="completed",
            summary=summary,
            payload=outcome.payload,
        )

    def _tool_retrieve_evidence_by_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        question = str(arguments.get("question") or content)
        requested_categories = {
            str(item).strip()
            for item in arguments.get("category_names") or self._category_entity_names_for_request(content)
            if str(item).strip()
        }
        lookup = self._category_lookup_payload(list(requested_categories)) if requested_categories else {}
        if lookup.get("missing_names"):
            missing = "、".join(str(item) for item in lookup.get("missing_names") or [])
            candidates = "、".join(str(item) for item in lookup.get("candidate_names") or [])
            suffix = f"；现有相近标签/分类：{candidates}" if candidates else ""
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="needs_clarification",
                summary=f"没有找到标签/分类「{missing}」{suffix}。",
                payload={
                    "category_lookup": lookup,
                    "category_names": sorted(requested_categories),
                    "category_groups": [],
                    "evidence_items": [],
                    "document_ids": [],
                },
            )
        if lookup.get("ambiguous_names"):
            ambiguous = "、".join(str(item) for item in lookup.get("ambiguous_names") or [])
            candidates = "、".join(str(item) for item in lookup.get("candidate_names") or [])
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="needs_clarification",
                summary=f"标签/分类「{ambiguous}」匹配到多个候选，请确认：{candidates}",
                payload={
                    "category_lookup": lookup,
                    "category_names": sorted(requested_categories),
                    "category_groups": [],
                    "evidence_items": [],
                    "document_ids": [],
                },
            )
        if lookup.get("matched_names"):
            requested_categories = set(str(item) for item in lookup["matched_names"])

        def worker() -> _TaskOutcome:
            documents = [document for document in self.document_library_service.list_documents() if document.status == "ready"]
            groups: dict[str, list[LibraryDocument]] = {}
            for document in documents:
                for category in document.categories:
                    if requested_categories and category.name not in requested_categories:
                        continue
                    groups.setdefault(category.name, []).append(document)

            category_groups = []
            flat_evidence = []
            for category_name, group_documents in groups.items():
                retrieval = self.rag_service.retrieve_evidence_with_quality(
                    question=f"{question}\n请优先检索与「{category_name}」标签相关的证据。",
                    documents=group_documents,
                    top_k=min(12, max(4, len(group_documents) * 4)),
                )
                evidence = retrieval.evidence_items
                evidence_payload = [item.model_dump(mode="json") for item in evidence]
                flat_evidence.extend(evidence_payload)
                boundary = None
                if not evidence_payload:
                    boundary = "该标签下论文存在，但正文证据不足。"
                category_groups.append(
                    {
                        "category_name": category_name,
                        "document_ids": [document.id for document in group_documents],
                        "documents": [self._document_payload(document) for document in group_documents],
                        "evidence_items": evidence_payload,
                        "evidence_quality": retrieval.evidence_quality.model_dump(mode="json"),
                        "cache_hit": retrieval.cache_hit,
                        "retrieval_strategy": retrieval.retrieval_strategy,
                        "evidence_boundary": boundary,
                    }
                )
            return _TaskOutcome(
                summary=f"Retrieved grouped evidence for {len(category_groups)} category group(s).",
                payload={
                    "category_groups": category_groups,
                    "evidence_items": flat_evidence,
                    "document_ids": list(
                        dict.fromkeys(
                            document_id
                            for group in category_groups
                            for document_id in group["document_ids"]
                        )
                    ),
                    "category_lookup": lookup if requested_categories else None,
                    "category_names": sorted(requested_categories),
                },
            )

        try:
            outcome = self._run_subagent(
                run_id=run_id,
                role="evidence-retriever",
                profile=SubagentProfile.EXPLORE,
                goal="Retrieve local evidence grouped by category/tag.",
                tool_policy=ToolPolicy(read_only=True),
                context_bundle={
                    "question": content,
                    "category_names": sorted(requested_categories),
                    "tool": "evidence.retriever.search_by_category",
                },
                worker=worker,
            )
        except Exception as exc:
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="failed",
                summary=self._retrieval_service_unavailable_message(grouped=True),
                payload={
                    "error_type": exc.__class__.__name__,
                    "requires_service": "milvus",
                },
            )
        groups = outcome.payload.get("category_groups") or []
        if not groups:
            if requested_categories:
                names = "、".join(sorted(requested_categories))
                all_category_groups = self._category_groups_from_library(sorted(requested_categories))
                return _ReactObservation(
                    tool="evidence.retriever.search_by_category",
                    status="completed",
                    summary=f"已识别到标签/分类「{names}」，但当前没有关联 ready 论文。",
                    payload={**outcome.payload, "category_groups": all_category_groups},
                )
            return _ReactObservation(
                tool="evidence.retriever.search_by_category",
                status="needs_clarification",
                summary="没有找到带标签且可用于检索的 ready 论文分组。",
                payload=outcome.payload,
            )
        return _ReactObservation(
            tool="evidence.retriever.search_by_category",
            status="completed",
            summary=f"已按 {len(groups)} 个标签分组检索证据。",
            payload=outcome.payload,
        )

    @staticmethod
    def _retrieval_service_unavailable_message(*, grouped: bool = False) -> str:
        scope = "按标签分组的论文正文检索" if grouped else "论文正文向量检索"
        return (
            f"{scope}服务当前不可用，无法完成基于所选论文正文的分析。"
            "请先启动 Milvus 向量服务并确认论文状态为 ready，然后重新发送问题。"
            "本轮不会改用普通聊天或仅凭论文元数据生成分析结论。"
        )

    def _no_ready_documents_message(self, document_ids: list[str]) -> str:
        selected = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids)
        ]
        if not selected:
            return "没有可用于检索的 ready 论文。"
        status_text = "、".join(
            f"{document.display_name or document.filename}（{document.status}）"
            for document in selected[:6]
        )
        suffix = "等" if len(selected) > 6 else ""
        return f"没有可用于检索的 ready 论文；当前选中的 {status_text}{suffix} 尚不能用于正文分析。"

    def _tool_draft_report(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        document_ids = [str(item) for item in arguments.get("document_ids") or [] if item]
        if not document_ids:
            category_filter_ids = self._category_filter_document_ids_from_observations(observations)
            document_ids = (
                category_filter_ids
                if category_filter_ids is not None
                else self._document_ids_from_observations(observations)
            )
        documents = [document for document in self.document_library_service.list_documents() if document.id in set(document_ids)]
        evidence_items = self._evidence_items_from_observations(observations)
        target_scope = self._report_compare_target_scope(
            session=_session,
            content=content,
            action=_ReactAction("report.drafter.write", arguments, "Resolve report target scope."),
            observations=observations,
        )
        target_document_ids = self._real_document_ids((target_scope or {}).get("document_ids") or [])
        if target_document_ids:
            target_set = set(target_document_ids)
            original_evidence_count = len(evidence_items)
            original_document_ids = [document.id for document in documents]
            evidence_items = [
                item
                for item in evidence_items
                if (item.document_id or item.source_id) in target_set
            ]
            documents = [document for document in documents if document.id in target_set]
            document_ids = [document.id for document in documents]
            filtered_document_ids = sorted(set(original_document_ids) - set(document_ids))
            filtered_evidence_count = original_evidence_count - len(evidence_items)
            if filtered_document_ids or filtered_evidence_count:
                self._append_react_trace(
                    run_id=run_id,
                    status="evidence_filtered_to_target_scope",
                    payload={
                        "tool": "report.drafter.write",
                        "target_document_ids": target_document_ids,
                        "filtered_document_ids": filtered_document_ids,
                        "filtered_evidence_count": filtered_evidence_count,
                        "remaining_evidence_count": len(evidence_items),
                    },
                )
                self._append_react_trace(
                    run_id=run_id,
                    status="target_doc_leak_prevented",
                    payload={
                        "tool": "report.drafter.write",
                        "reason": "report_evidence_filtered_to_target_scope",
                        "target_document_ids": target_document_ids,
                        "filtered_document_ids": filtered_document_ids,
                        "filtered_evidence_count": filtered_evidence_count,
                    },
                )
        resolved_from_library_query = any(
            observation.tool == "library.explorer.find_documents" and observation.status == "completed"
            for observation in observations
        )
        if not evidence_items and not resolved_from_library_query:
            return _ReactObservation(
                tool="report.drafter.write",
                status="needs_clarification",
                summary=(
                    "本轮没有检索到可引用的论文正文证据，因此我不能只根据文件名、标题或元数据生成综述报告。"
                    "请确认所选论文已完成入库、正文 chunk 与向量/关键词索引可用后，再重新发送总结请求。"
                ),
                payload={
                    "answer": "",
                    "document_ids": [document.id for document in documents],
                    "evidence_items": [],
                    "evidence_boundary": "insufficient_evidence",
                },
            )
        self._append_react_trace(
            run_id=run_id,
            status="final_answer_synthesis_started",
            payload={
                "runtime_mode": "react_report_drafter",
                "evidence_count": len(evidence_items),
                "used_document_count": len(document_ids),
                "previous_content_length": 0,
            },
        )

        def worker() -> _TaskOutcome:
            draft = self._draft_with_llm(str(arguments.get("question") or content), documents, evidence_items)
            return _TaskOutcome(
                summary="Drafted final ReAct answer from observations.",
                payload={
                    "answer": draft.answer,
                    "document_ids": [document.id for document in documents],
                    "evidence_items": [item.model_dump(mode="json") for item in evidence_items],
                    "llm_draft_success": draft.llm_draft_success,
                    "fallback_used": draft.fallback_used,
                    "drafting_error": draft.drafting_error,
                    "evidence_count": len(evidence_items),
                    "used_document_count": len(documents),
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Write a Markdown answer grounded in retrieved observations.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "document_ids": document_ids, "evidence_count": len(evidence_items)},
            worker=worker,
        )
        self._append_react_trace(
            run_id=run_id,
            status="final_answer_synthesis_finished",
            payload={
                "runtime_mode": "react_report_drafter",
                "synthesis_used_evidence_count": len(evidence_items),
                "final_content_length": len(str(outcome.payload.get("answer") or "")),
                "fallback_used": bool(outcome.payload.get("fallback_used")),
                "llm_draft_success": bool(outcome.payload.get("llm_draft_success")),
                "drafting_error": outcome.payload.get("drafting_error"),
            },
        )
        llm_draft_success = bool(outcome.payload.get("llm_draft_success"))
        fallback_used = bool(outcome.payload.get("fallback_used"))
        status = "completed" if llm_draft_success and not fallback_used else "degraded"
        return _ReactObservation(
            tool="report.drafter.write",
            status=status,
            summary="已生成基于观察结果的回答。" if status == "completed" else "LLM 综述撰写失败，已返回降级草稿。",
            payload=outcome.payload,
        )

    def _tool_draft_report_by_category(
        self,
        run_id: str,
        _session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        target_chars = int(arguments.get("target_chars") or arguments.get("target_words") or 600)
        category_groups = self._category_groups_from_observations(observations)
        if not category_groups:
            category_name = self._category_name_from_request_or_observations(content, observations)
            category_groups = self._category_groups_from_library([category_name] if category_name else None)

        def worker() -> _TaskOutcome:
            answer = self._draft_category_summaries_with_llm(
                question=str(arguments.get("question") or content),
                category_groups=category_groups,
                target_chars=target_chars,
            )
            return _TaskOutcome(
                summary=f"Drafted summaries for {len(category_groups)} category group(s).",
                payload={
                    "answer": answer,
                    "category_groups": category_groups,
                    "document_ids": list(
                        dict.fromkeys(
                            document_id
                            for group in category_groups
                            for document_id in group.get("document_ids", [])
                        )
                    ),
                    "target_chars": target_chars,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Write grouped Markdown summaries by category/tag.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={
                "question": content,
                "category_count": len(category_groups),
                "target_chars": target_chars,
            },
            worker=worker,
        )
        return _ReactObservation(
            tool="report.drafter.write_by_category",
            status="completed",
            summary=f"已按标签生成 {len(category_groups)} 份总结。",
            payload=outcome.payload,
        )

    def _tool_memory_read(
        self,
        _run_id: str,
        session: ChatSession,
        _content: str,
        _arguments: dict[str, Any],
        _observations: list[_ReactObservation],
    ) -> _ReactObservation:
        payload = self._react_memory_payload(session.id)
        return _ReactObservation(
            tool="memory.read",
            status="completed",
            summary="已读取项目规则、用户偏好和历史反思。",
            payload=payload,
        )

    def _tool_memory_write(
        self,
        _run_id: str,
        session: ChatSession,
        content: str,
        arguments: dict[str, Any],
        observations: list[_ReactObservation],
    ) -> _ReactObservation:
        summary = str(arguments.get("summary") or "").strip()
        if not summary:
            summary = self._reflection_summary(content, observations, "completed")
        if summary:
            self.file_store.add_user_preference(summary[:160])
        return _ReactObservation(
            tool="memory.write",
            status="completed",
            summary="已写入一条轻量反思记忆。",
            payload={"summary": summary[:160]},
        )

    def _validate_react_action(
        self,
        action: _ReactAction,
        observations: list[_ReactObservation],
        *,
        content: str = "",
        plan_step: _StepState | None = None,
        plan_state: _PlanState | None = None,
    ) -> str | None:
        tool_names = {item["name"] for item in self._react_tool_specs()} | {"final.answer"}
        if action.tool in {self._INTERNAL_DEGRADED_TOOL, self._INTERNAL_CATEGORY_CONFLICT_TOOL}:
            return None
        if action.tool not in tool_names:
            return f"模型请求了未注册工具「{action.tool}」，本轮已拦截。"
        if action.tool in {"library.operator.create_category", "library.operator.assign_category"}:
            category_names = self._category_names_from_arguments(action.arguments)
            if not category_names and plan_step is not None:
                category_names = self._step_target_category_names(plan_step)
            if not category_names and not (plan_state is not None and len(plan_state.steps) > 1):
                category_names = self._category_names_from_observations(observations)
            if not category_names:
                return "我没有可靠识别出要使用的标签/分类名称，因此没有改动论文库。"
            error = self._category_names_validation_error(category_names)
            if error:
                return error
            if action.tool == "library.operator.assign_category":
                scope = str(action.arguments.get("scope") or "")
                document_ids = action.arguments.get("document_ids") or []
                resolved = action.arguments.get("_resolved_action") if isinstance(action.arguments.get("_resolved_action"), dict) else {}
                if resolved.get("clarification_needed") or resolved.get("scope_type") == "unknown":
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if self._mentions_ambiguous_plural_document_referent(content) and document_ids and not resolved:
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if (
                    self._mentions_ambiguous_plural_document_referent(content)
                    and document_ids
                    and resolved.get("scope_type") == "explicit_documents"
                    and not (
                        self._needs_untagged_assignment(content)
                        or self._mentions_all_library(content)
                        or self._mentions_explicit_category_document_scope(content)
                    )
                ):
                    return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行打标签操作。"
                if resolved.get("scope_type") == "all_library":
                    return "批量给所有论文打标签必须先经过确认或先筛选出明确论文集合，本轮没有改动论文库。"
                if (
                    scope not in {"untagged", "last_referenced"}
                    and not document_ids
                    and not self._document_ids_from_observations(observations)
                ):
                    return "我没有可靠定位到要打标签的论文，因此没有改动论文库。"
            action.arguments["category_name"] = category_names[0]
            action.arguments["category_names"] = category_names
        if action.tool == "library.operator.rename_category":
            source_name = self._clean_category_name(str(action.arguments.get("source_category_name") or ""))
            target_name = self._clean_category_name(str(action.arguments.get("target_category_name") or ""))
            if not source_name or not target_name:
                return "我没有可靠识别出源标签和目标标签，因此没有改动论文库。"
            source_error = self._category_name_validation_error(source_name)
            target_error = self._category_name_validation_error(target_name)
            if source_error or target_error:
                return source_error or target_error
        if action.tool == "library.operator.clear_categories":
            scope = str(action.arguments.get("scope") or "")
            document_ids = action.arguments.get("document_ids") or []
            category_name = self._clean_category_name(str(action.arguments.get("category_name") or ""))
            operation = self._infer_clear_categories_operation(action.arguments)
            resolved = action.arguments.get("_resolved_action") if isinstance(action.arguments.get("_resolved_action"), dict) else {}
            if scope not in {"all", "tagged", "last_referenced", "documents", ""}:
                return "清空分类/标签的范围无效，因此没有改动论文库。"
            if resolved.get("clarification_needed") or resolved.get("scope_type") == "unknown":
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if self._mentions_ambiguous_plural_document_referent(content) and document_ids and not resolved:
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if (
                self._mentions_ambiguous_plural_document_referent(content)
                and document_ids
                and resolved.get("scope_type") == "explicit_documents"
                and not (
                    self._mentions_all_library(content)
                    or self._mentions_all_categories(content)
                    or self._mentions_explicit_category_document_scope(content)
                )
            ):
                return "我还不能确定“这些论文”具体指哪些论文。请先选择论文，或先让我查出一批论文后再执行移除/清空标签操作。"
            if category_name:
                if not self._category_exists(category_name):
                    return f"没有找到名为「{category_name}」的标签/分类，因此没有改动论文库。"
                if operation == "remove_single_category_link" and scope not in {"all", "tagged"} and not document_ids:
                    if not (self._mentions_all_library(content) or self._mentions_all_categories(content)):
                        return "我没有可靠定位到要移除该标签的论文，因此没有改动论文库。"
            elif scope in {"documents", ""} and not document_ids and not self._document_ids_from_observations(observations):
                return "我没有可靠定位到要清空分类/标签的论文，因此没有改动论文库。"
        if action.tool == "report.drafter.write":
            document_ids = action.arguments.get("document_ids") or self._document_ids_from_observations(observations)
            if not document_ids:
                return "我没有可靠定位到要总结的论文，因此无法生成报告。"
        return None

    def _plan_needs_report_after_evidence(self, plan_state: _PlanState, observations: list[_ReactObservation]) -> bool:
        if not any(
            step.metadata.get("obligation_key") == "report" and step.status not in {"completed", "skipped"}
            for step in plan_state.steps
        ):
            return False
        if any(observation.tool == "report.drafter.write" and observation.status in {"completed", "degraded"} for observation in observations):
            return False
        return any(
            observation.tool == "evidence.retriever.search"
            and observation.status == "completed"
            and observation.payload.get("evidence_items")
            for observation in observations
        )

    def _should_draft_after_evidence(self, content: str, observations: list[_ReactObservation]) -> bool:
        if not any(
            observation.tool == "evidence.retriever.search"
            and observation.status == "completed"
            and observation.payload.get("evidence_items")
            for observation in observations
        ):
            return False
        if any(
            observation.tool == "report.drafter.write"
            and observation.status in {"completed", "degraded"}
            for observation in observations
        ):
            return False
        return self._is_selected_document_answer_request(content, self._document_ids_from_observations(observations), [])

    def _react_tool_specs(self) -> list[dict[str, Any]]:
        raw_tools = [
            {
                "name": "tool.registry.list",
                "description": "List available PaperDesk tools, input expectations, and permission level.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.stats",
                "description": "Read paper counts and processing status from SQLite.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.category_stats",
                "description": "Read category/tag counts, tagged papers, untagged papers, and per-category counts.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "library.explorer.find_documents",
                "description": "Resolve papers by selected IDs/title/filename or by exact tag/category/group/collection entity. When the user says 'X tag/category/group/collection' or 'papers under X', pass category_name/category_names first and do not treat X as a paper title unless category lookup fails.",
                "read_only": True,
                "arguments": {"query": "string", "expected": "one|many", "allow_all": "boolean", "category_name": "string optional", "category_names": "list[string] optional"},
            },
            {
                "name": "library.explorer.document_metadata",
                "description": "Read per-document metadata fields such as title, authors, venue/journal/conference, published time/year, and tags.",
                "read_only": True,
                "arguments": {"document_ids": "list[string]", "requested_fields": "list[string]"},
            },
            {
                "name": "library.explorer.document_categories",
                "description": "Read tags/categories for resolved document IDs.",
                "read_only": True,
                "arguments": {"document_ids": "list[string]"},
            },
            {
                "name": "library.operator.create_category",
                "description": "Create a category/tag. Non-destructive write. Requires a short explicit category_name.",
                "read_only": False,
                "risk_level": "safe_write",
                "required_args": ["category_name"],
                "scope_type": "single_entity",
                "operation_level": "entity-level",
                "write_type": "create",
                "target_type": "category",
                "destructive": False,
                "requires_confirmation": False,
                "requires_verification": True,
                "operation_type": "create_entity",
                "arguments": {"category_name": "string", "category_names": "list[string] optional"},
            },
            {
                "name": "library.operator.assign_category",
                "description": "Append one or more categories/tags to document_ids or to scope=untagged. Does not overwrite existing tags.",
                "read_only": False,
                "risk_level": "scoped_write",
                "required_args": ["category_name"],
                "scope_type": "documents|untagged|last_referenced",
                "operation_level": "relation-level",
                "write_type": "append",
                "target_type": "paper-category relation",
                "destructive": False,
                "requires_confirmation": False,
                "requires_verification": True,
                "operation_type": "append_relation",
                "arguments": {"category_name": "string", "category_names": "list[string] optional", "scope": "untagged|last_referenced|documents", "document_ids": "list[string]"},
            },
            {
                "name": "library.operator.rename_category",
                "description": "Safely rename or merge a category/tag while preserving all paper links. Use for replace/rename semantics, not destructive delete.",
                "read_only": False,
                "risk_level": "scoped_write",
                "required_args": ["source_category_name", "target_category_name"],
                "scope_type": "single_entity",
                "operation_level": "entity-level",
                "write_type": "update",
                "target_type": "category",
                "destructive": False,
                "requires_confirmation": False,
                "requires_verification": True,
                "operation_type": "rename_or_merge_entity",
                "arguments": {"source_category_name": "string", "target_category_name": "string"},
            },
            {
                "name": "library.operator.delete_unused_categories",
                "description": "Delete category/tag entities with document_count=0 only. Entity-level destructive cleanup: it must not clear document tags or modify document-category links. Requires selector=unused, preview, confirmation, and verification.",
                "read_only": False,
                "risk_level": "destructive",
                "required_args": ["selector"],
                "scope_type": "category_entities_with_zero_documents",
                "operation_level": "entity-level",
                "write_type": "delete",
                "target_type": "category",
                "destructive": True,
                "requires_confirmation": True,
                "requires_verification": True,
                "operation_type": "delete_unused_category_entities",
                "arguments": {"selector": "unused"},
            },
            {
                "name": "library.operator.clear_categories",
                "description": "Destructive tag/category relation tool. Must include explicit operation: remove_single_category_link, clear_document_categories, or clear_all_categories. Runtime will preview and require confirmation before execution.",
                "read_only": False,
                "risk_level": "destructive|critical",
                "required_args": ["operation"],
                "scope_type": "explicit",
                "operation_level": "relation-level",
                "write_type": "clear",
                "target_type": "paper-category relation",
                "destructive": True,
                "requires_confirmation": True,
                "requires_verification": True,
                "operation_type": "remove_or_clear_relations",
                "arguments": {"operation": "remove_single_category_link|clear_document_categories|clear_all_categories", "scope": "all|tagged|documents", "category_name": "string optional", "document_ids": "list[string]"},
            },
            {
                "name": "evidence.retriever.search",
                "description": "Retrieve local evidence from ready papers.",
                "read_only": True,
                "arguments": {"question": "string", "document_ids": "list[string]"},
            },
            {
                "name": "evidence.retriever.search_by_category",
                "description": "Retrieve RAG evidence grouped by exact category/tag names for labeled document analysis, comparison, summary, or report tasks.",
                "read_only": True,
                "arguments": {"question": "string", "category_names": "list[string] optional"},
            },
            {
                "name": "report.drafter.write",
                "description": "Write a Markdown answer from resolved documents and retrieved evidence.",
                "read_only": True,
                "arguments": {"question": "string", "document_ids": "list[string]"},
            },
            {
                "name": "report.drafter.write_by_category",
                "description": "Write one Markdown summary per category/tag from grouped RAG observations.",
                "read_only": True,
                "arguments": {"question": "string", "target_chars": "integer, default 600"},
            },
            {
                "name": "memory.read",
                "description": "Read project rules, user preferences, and prior reflection notes.",
                "read_only": True,
                "arguments": {},
            },
            {
                "name": "memory.write",
                "description": "Write a short reflection note after a task.",
                "read_only": False,
                "risk_level": "safe_write",
                "required_args": [],
                "scope_type": "session_memory",
                "operation_level": "content-level",
                "write_type": "append",
                "target_type": "memory",
                "destructive": False,
                "requires_confirmation": False,
                "requires_verification": False,
                "arguments": {"summary": "string"},
            },
        ]
        tools = []
        for tool in raw_tools:
            name = str(tool.get("name") or "")
            spec = self._spec_for_tool(name)
            if (
                spec is None
                or not spec.available_by_default
                or spec.scope == "experimental"
                or spec.maturity != "stable"
            ):
                continue
            enriched = {
                **tool,
                "scope": spec.scope,
                "operation_level": spec.operation_level,
                "io_type": spec.io_type,
                "write_type": spec.write_type,
                "input_object_types": list(spec.input_object_types),
                "output_observation_type": spec.output_observation_type,
                "requires_post_read_verification": spec.requires_post_read_verification,
                "verification_tool": spec.verification_tool,
                "available_by_default": spec.available_by_default,
                "maturity": spec.maturity,
                "feature_flag": spec.feature_flag,
                "requires_confirmation": spec.requires_confirmation,
                "destructive": spec.destructive,
            }
            tools.append(enriched)
        return tools

    def _build_react_snapshot(
        self,
        session_id: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> dict[str, Any]:
        library_stats = self._library_stats_payload()
        category_stats = self._category_stats_payload(limit_documents=6)
        documents = self.document_library_service.list_documents()[:8]
        return {
            "library": library_stats,
            "categories": {
                "category_count": category_stats["category_count"],
                "tagged_document_count": category_stats["tagged_document_count"],
                "untagged_document_count": category_stats["untagged_document_count"],
                "category_names": [item["name"] for item in category_stats["categories"]],
            },
            "document_preview": [self._document_payload(document) for document in documents],
            "selected_document_ids": selected_document_ids,
            "attachment_document_ids": [attachment.document_id for attachment in attachments if attachment.document_id],
            "conversation_referents": self._read_react_state(session_id),
            "memory": self._react_memory_payload(session_id),
            "guardrails": [
                "有几篇有标签/几类标签必须调用 category_stats，不要只回答论文总数。",
                "新增 hdcc 标签并给无标签论文打这个标签必须拆成 create_category + assign_category(scope=untagged)。",
                "标签名只能来自引号、冒号后的短名称，或“新增一个X标签”中的 X。",
                "用户只问期刊、会议、发表时间、年份、作者、题名或标签时，调用 document_metadata/document_categories，不能套报告模板或输出摘要。",
                "删除A标签并换成B标签属于保留关联的重命名/合并，必须用 rename_category。",
                "用所有X标签论文写报告时，先按 category_name 精确读取 X 标签下的论文。",
                "标签和分类是同一字段；不能声称清除标签但保留分类。",
                "会话上下文可用于指代解析，但当前库状态、计数和写入结果必须以最新 observation 为准。",
                "最终回答不要出现“根据对话记录/根据上下文/根据运行态摘要”等过程话术，直接回答用户。",
            ],
        }

    def _library_stats_payload(self) -> dict[str, Any]:
        documents = self.document_library_service.list_documents()
        return {
            "total": len(documents),
            "ready": sum(1 for document in documents if document.status == "ready"),
            "processing": sum(1 for document in documents if document.status == "processing"),
            "failed": sum(1 for document in documents if document.status == "failed"),
            "referent_document_ids": [document.id for document in documents],
            "referent_label": "全部论文",
        }

    def _category_stats_payload(self, *, limit_documents: int | None = None) -> dict[str, Any]:
        documents = self.document_library_service.list_documents()
        categories = self.category_repository.list_categories()
        counts_by_category_id = {category.id: 0 for category in categories}
        tagged_documents = []
        untagged_documents = []
        for document in documents:
            if document.categories:
                tagged_documents.append(
                    {
                        **self._document_payload(document),
                        "categories": [category.name for category in document.categories],
                    }
                )
                for category in document.categories:
                    counts_by_category_id[category.id] = counts_by_category_id.get(category.id, 0) + 1
            else:
                untagged_documents.append(self._document_payload(document))

        def maybe_limit(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if limit_documents is None:
                return items
            return items[:limit_documents]

        return {
            "category_count": len(categories),
            "categories": [
                {
                    "id": category.id,
                    "name": category.name,
                    "color": category.color,
                    "document_count": counts_by_category_id.get(category.id, 0),
                }
                for category in categories
            ],
            "total_documents": len(documents),
            "tagged_document_count": len(tagged_documents),
            "untagged_document_count": len(untagged_documents),
            "tagged_document_ids": [document["id"] for document in tagged_documents],
            "untagged_document_ids": [document["id"] for document in untagged_documents],
            "tagged_documents": maybe_limit(tagged_documents),
            "untagged_documents": maybe_limit(untagged_documents),
        }

    @staticmethod
    def _document_payload(document: LibraryDocument) -> dict[str, Any]:
        return {
            "id": document.id,
            "name": document.display_name or document.filename,
            "filename": document.filename,
            "title": document.title,
            "status": document.status,
            "page_count": document.page_count,
        }

    def _document_metadata_payload(self, document: LibraryDocument, requested_fields: list[str]) -> dict[str, Any]:
        metadata = self._extract_document_metadata(document)
        payload = {
            **self._document_payload(document),
            "title": document.title or metadata.get("title"),
            "authors": metadata.get("authors") or [],
            "venue": metadata.get("venue"),
            "published": metadata.get("published"),
            "year": metadata.get("year"),
            "tags": [category.name for category in document.categories],
            "requested_fields": requested_fields,
        }
        return payload

    def _extract_document_metadata(self, document: LibraryDocument) -> dict[str, Any]:
        chunks = []
        try:
            chunks = self.rag_service.chunk_repository.list_chunks(document_ids=[document.id])
        except Exception:
            chunks = []
        metadata: dict[str, Any] = {}
        for chunk in chunks[:8]:
            for key, value in (chunk.metadata or {}).items():
                normalized = str(key).casefold()
                if normalized in {"venue", "journal", "conference", "source", "publication", "container_title"} and value:
                    metadata.setdefault("venue", str(value).strip())
                elif normalized in {"published", "published_at", "publication_date", "date"} and value:
                    metadata.setdefault("published", str(value).strip())
                elif normalized in {"year", "publication_year"} and value:
                    metadata.setdefault("year", str(value).strip())
                elif normalized in {"authors", "author"} and value:
                    metadata.setdefault("authors", value if isinstance(value, list) else [str(value).strip()])
                elif normalized == "title" and value:
                    metadata.setdefault("title", str(value).strip())
        if not metadata.get("venue") or not metadata.get("published") or not metadata.get("year"):
            first_text = "\n".join((chunk.content or chunk.text or "")[:1000] for chunk in chunks[:3])
            if not metadata.get("venue"):
                metadata["venue"] = self._extract_metadata_text(first_text, ("Journal", "Venue", "Conference", "会议", "期刊"))
            if not metadata.get("published"):
                metadata["published"] = self._extract_metadata_text(first_text, ("Published", "Publication Date", "发表时间", "发表日期"))
            if not metadata.get("year"):
                year_match = re.search(r"\b(19|20)\d{2}\b", str(metadata.get("published") or "") or first_text)
                if year_match:
                    metadata["year"] = year_match.group(0)
        return metadata

    @staticmethod
    def _extract_metadata_text(text: str, labels: tuple[str, ...]) -> str | None:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[:：]\s*([^\n\r；;。]{{1,120}})"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def ensure_final_answer(
        self,
        *,
        user_prompt: str,
        original_request: ChatMessageRequest | None = None,
        runtime_mode: str,
        evidence_items: list[EvidenceItem],
        citations: list[str] | None = None,
        used_document_ids: list[str] | None = None,
        tool_observations: list[_ReactObservation] | None = None,
        previous_content: str,
        trace_digest: dict[str, Any] | None = None,
        trace_id: str | None = None,
        action_status: str | None = None,
    ) -> str:
        """Post-condition guardrail: evidence/tool chains must return an answer, not a status line."""

        _ = original_request, citations
        observations = tool_observations or []
        evidence = evidence_items or self._evidence_items_from_observations(observations)
        has_grounding = bool(evidence) or any(
            observation.status == "completed"
            and (
                observation.tool.startswith("evidence.retriever.")
                or observation.tool in {"report.drafter.write", "report.drafter.write_by_category"}
            )
            for observation in observations
        )
        if not has_grounding:
            return previous_content
        if action_status in {"needs_clarification", "confirmation_required", "validation_failed", "failed"}:
            return previous_content
        if self._has_completed_library_write(observations):
            return previous_content
        if not self.is_status_only_answer(previous_content):
            return previous_content

        document_ids = list(dict.fromkeys([*(used_document_ids or []), *self._document_ids_from_observations(observations)]))
        if not document_ids:
            document_ids = [
                item.document_id or item.source_id
                for item in evidence
                if (item.document_id or item.source_id)
            ]
            document_ids = list(dict.fromkeys(document_ids))
        documents = [
            document
            for document in self.document_library_service.list_documents()
            if document.id in set(document_ids)
        ]
        trace_payload = {
            "runtime_mode": runtime_mode,
            "evidence_count": len(evidence),
            "used_document_count": len(document_ids),
            "previous_content_length": len(previous_content or ""),
            "trace_digest": trace_digest or {},
        }
        if trace_id:
            try:
                self._append_react_trace(
                    run_id=trace_id,
                    status="final_answer_synthesis_started",
                    payload=trace_payload,
                )
            except Exception:
                pass
        fallback_used = False
        try:
            draft = self._draft_with_llm(user_prompt, documents, evidence)
            answer = draft.answer
            fallback_used = draft.fallback_used
        except Exception:
            answer = ""
        if self.is_status_only_answer(answer):
            fallback_used = True
            answer = self._draft_grounded_fallback_answer(user_prompt, documents, evidence, observations)
        if self.is_status_only_answer(answer):
            fallback_used = True
            answer = (
                "我已经检索到相关证据，但最终回答生成失败。"
                "为避免把检索状态伪装成答案，本轮只说明边界：请重新发送问题，或缩小到具体论文/段落后我再基于证据回答。"
            )
        if trace_id:
            try:
                self._append_react_trace(
                    run_id=trace_id,
                    status="final_answer_synthesis_finished",
                    payload={
                        **trace_payload,
                        "synthesis_used_evidence_count": len(evidence),
                        "final_content_length": len(answer),
                        "fallback_used": fallback_used or not self.api_key,
                    },
                )
            except Exception:
                pass
        return answer.strip()

    @classmethod
    def is_status_only_answer(cls, content: str | None) -> bool:
        text = (content or "").strip()
        if not text:
            return True
        normalized = re.sub(r"\s+", " ", text).strip().casefold()
        if len(normalized) <= 12 and normalized in {"done", "completed", "ready", "ok", "success", "完成", "已完成"}:
            return True
        if len(normalized) <= 30 and any(marker in normalized for marker in ("检索完成", "调用成功", "工具成功", "操作完成", "任务完成", "所有操作已完成")):
            return True
        status_patterns = (
            r"^已检索到\s*\d+\s*条证据[。.!！]?$",
            r"^检索到\s*\d+\s*条证据[。.!！]?$",
            r"^retrieved\s+\d+\s+evidence\s+items?[。.!！]?$",
            r"^retrieval\s+(ready|completed|complete|done)[。.!！]?$",
            r"^(检索完成|工具调用成功|已找到相关内容|已完成检索|已完成|完成|成功)[。.!！]?$",
            r"^(done|completed|ready|tool call succeeded|tool completed)[。.!！]?$",
        )
        return any(re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in status_patterns)

    def _draft_grounded_fallback_answer(
        self,
        question: str,
        documents: list[LibraryDocument],
        evidence_items: list[EvidenceItem],
        observations: list[_ReactObservation],
    ) -> str:
        if not evidence_items:
            summaries = [
                observation.summary
                for observation in observations
                if observation.status == "completed" and observation.summary
            ]
            if summaries:
                return "本轮工具已完成，但没有返回可引用的正文证据；可用观察是：\n" + "\n".join(
                    f"- {summary}" for summary in summaries[:5]
                )
            return "本轮没有可用于回答的证据，因此不能可靠生成结论。"

        documents_by_id = {document.id: document for document in documents}
        evidence_by_document: dict[str, list[EvidenceItem]] = {}
        for item in evidence_items:
            document_id = item.document_id or item.source_id or item.title or "unknown"
            evidence_by_document.setdefault(document_id, []).append(item)

        lines = [
            f"针对你的问题“{question.strip()}”，我基于本轮检索到的证据整理如下；没有证据直接支持的部分我不会扩写。",
            "",
        ]
        for index, (document_id, items) in enumerate(evidence_by_document.items(), start=1):
            document = documents_by_id.get(document_id)
            title = (
                document.display_name
                if document and document.display_name
                else document.filename
                if document
                else items[0].title or items[0].citation_label or f"文档 {index}"
            )
            lines.extend([f"## {title}", ""])
            snippets = []
            for item in items[:4]:
                snippet = (item.quote or item.snippet or "").strip()
                if not snippet:
                    continue
                citation = item.citation_label or title
                snippets.append(f"- {snippet}（{citation}）")
            if snippets:
                lines.extend(snippets)
                lines.append("")
                lines.append("基于这些片段，可以先形成上述证据层面的回答；若要更强的创新点/方法/实验结论，需要继续核对论文中对应章节。")
            else:
                lines.append("检索返回了该文档的命中记录，但缺少可展示的正文片段，当前证据不足以形成具体结论。")
            lines.append("")
        lines.append("证据边界：以上内容只来自本轮 evidence，不代表对整篇论文的完整人工审稿。")
        return "\n".join(lines).strip()

    def _get_or_create_category(self, category_name: str):
        existing = next(
            (category for category in self.category_repository.list_categories() if category.name == category_name),
            None,
        )
        if existing is not None:
            return existing, False
        palette = ["#0f5fb8", "#047c71", "#6957d8", "#b76a00", "#b42318"]
        color = palette[len(self.category_repository.list_categories()) % len(palette)]
        category = self.category_repository.create_category(category_name, color)
        return category, True

    @staticmethod
    def _has_completed_library_write(observations: list[_ReactObservation]) -> bool:
        return any(
            observation.status == "completed" and bool(observation.payload.get("library_mutated"))
            for observation in observations
        )

    def _write_goal_fulfilled_by_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> bool:
        """Decide whether the current write request should stop planning and answer from observations."""

        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if self._is_delete_unused_categories_intent(content):
            return "library.operator.delete_unused_categories" in completed_tools
        if self._is_clear_categories_intent(content):
            return "library.operator.clear_categories" in completed_tools
        if self._is_rename_category_intent(content):
            return "library.operator.rename_category" in completed_tools
        rename_pair = self._extract_category_rename_request(content)
        if rename_pair and "library.operator.rename_category" not in completed_tools:
            return False
        if self._is_assignment_intent(content):
            return "library.operator.assign_category" in completed_tools
        if rename_pair:
            return "library.operator.rename_category" in completed_tools
        if self._is_create_category_intent(content):
            return "library.operator.create_category" in completed_tools
        return False

    def _repair_premature_write_final(
        self,
        *,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        observations: list[_ReactObservation],
    ) -> _ReactAction | None:
        """Convert a premature LLM final answer into a validated write action."""

        if not self._is_assignment_intent(content):
            return None
        state = self._read_react_state(session.id)
        category_name = (
            self._extract_category_name_from_request(content)
            or self._category_name_from_observations(observations)
            or state.get("last_category_name")
            or ""
        )
        if not isinstance(category_name, str) or self._category_name_validation_error(category_name):
            return None
        if self._needs_untagged_assignment(content) or self._should_target_untagged_from_context(content, state):
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "scope": "untagged"},
                "模型提前给出完成回答；后端改为执行已校验的未分类论文打标签工具。",
            )
        document_ids = self._document_ids_from_observations(observations) or selected_document_ids
        if document_ids:
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "document_ids": document_ids},
                "模型提前给出完成回答；后端改为执行已校验的指定论文打标签工具。",
            )
        if self._mentions_previous_referent(content) and self._state_document_ids(state):
            return _ReactAction(
                "library.operator.assign_category",
                {"category_name": category_name, "scope": "last_referenced"},
                "模型提前给出完成回答；后端改为执行已校验的上下文指代打标签工具。",
            )
        return None

    @staticmethod
    def _requires_library_write_observation(content: str) -> bool:
        # Guardrail, not semantic routing: report drafting is a read-only generation task even when
        # the user says "write a report" about papers with a tag.
        if KnowledgeAgentRuntime._is_read_only_report_request(content):
            return False
        write_markers = (
            "新建",
            "创建",
            "新增",
            "添加",
            "添加标签",
            "加标签",
            "加个",
            "加一个",
            "加上",
            "打上",
            "打标签",
            "打一个",
            "补上",
            "归类",
            "设置",
            "设为",
            "设成",
            "换成",
            "换为",
            "替换为",
            "替换成",
            "重命名",
            "重命名为",
            "改成",
            "改为",
            "改名为",
            "删除",
            "移除",
            "delete",
            "remove",
            "rename",
            "replace",
            "assign",
                "清除",
                "清空",
                "清楚",
                "去掉",
        )
        target_markers = ("标签", "分类", "论文", "文档", "tag", "category", "paper", "document")
        lowered = content.casefold()
        return any(marker in lowered for marker in write_markers) and any(
            marker in lowered for marker in target_markers
        )

    @staticmethod
    def _is_read_only_report_request(content: str) -> bool:
        lowered = content.casefold()
        has_report_output = any(
            marker in lowered
            for marker in (
                "写一篇报告",
                "写报告",
                "生成报告",
                "总结",
                "综述",
                "概述",
                "对比",
                "比较",
                "report",
                "review",
                "summary",
                "compare",
            )
        )
        has_library_scope = any(marker in lowered for marker in ("论文", "文章", "文档", "paper", "document"))
        destructive_markers = ("清除", "清空", "删除", "移除", "删掉", "去掉", "delete", "remove", "clear")
        write_db_markers = (
            "新建",
            "创建",
            "新增",
            "添加",
            "加标签",
            "打标签",
            "补上",
            "归类",
            "设置分类",
            "重命名",
            "替换成",
            "替换为",
            "改成",
            "改为",
            "rename",
            "replace",
            "assign",
        )
        return has_report_output and has_library_scope and not any(
            marker in lowered for marker in destructive_markers + write_db_markers
        )

    def _requires_tool_observation(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        return self._should_handle_with_react(content, selected_document_ids, attachments)

    def _requires_document_category_observation(
        self,
        content: str,
        selected_document_ids: list[str],
    ) -> bool:
        if self._is_assignment_intent(content) or self._is_clear_categories_intent(content):
            return False
        return self._requests_all_document_categories(content) or (
            bool(selected_document_ids) and self._is_document_category_query(content)
        )

    def _document_category_target_ids(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> list[str]:
        if self._requests_tagged_document_categories(content):
            category_stats = self._latest_payload(observations, "library.explorer.category_stats")
            if isinstance(category_stats, dict):
                return [str(item) for item in category_stats.get("tagged_document_ids") or [] if item]
            return [
                document.id
                for document in self.document_library_service.list_documents()
                if document.categories
            ]
        return [document.id for document in self.document_library_service.list_documents()]

    @staticmethod
    def _requests_all_document_categories(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        has_document_target = any(marker in content for marker in ("每篇", "每一篇", "每个", "全部", "所有", "论文", "文章", "文档"))
        has_mapping = any(marker in content for marker in ("对应", "分别", "各自", "是什么", "哪些", "列出", "查看"))
        return has_category_target and has_document_target and has_mapping

    @staticmethod
    def _requests_tagged_document_categories(content: str) -> bool:
        return any(marker in content for marker in ("带标签", "有标签", "已打标签", "已分类")) and any(
            marker in content for marker in ("分别", "对应", "是什么", "哪些", "列出", "查看")
        )

    @staticmethod
    def _requested_metadata_fields(content: str) -> list[str]:
        fields: list[str] = []

        def add(field: str) -> None:
            if field not in fields:
                fields.append(field)

        lowered = content.casefold()
        if any(marker in content for marker in ("标题", "题名", "名称")) or "title" in lowered:
            add("title")
        if any(marker in content for marker in ("作者", "author")) or "authors" in lowered:
            add("authors")
        if any(marker in content for marker in ("期刊", "会议", "出处", "来源", "发表在哪", "出自")) or any(
            marker in lowered for marker in ("journal", "venue", "conference")
        ):
            add("venue")
        if any(marker in content for marker in ("时间", "日期", "发表时间", "发表日期", "什么时候")) or any(
            marker in lowered for marker in ("published", "publication date", "date")
        ):
            add("published")
        if any(marker in content for marker in ("年份", "哪年", "年")) or "year" in lowered:
            add("year")
        if any(marker in content for marker in ("标签", "分类")) or any(marker in lowered for marker in ("tag", "category")):
            add("tags")
        if "published" in fields and "year" not in fields and any(marker in content for marker in ("时间", "日期")):
            add("year")
        return fields

    @classmethod
    def _is_metadata_query(
        cls,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if not selected_document_ids and not any(attachment.document_id for attachment in attachments):
            return False
        fields = cls._requested_metadata_fields(content)
        if not fields:
            return False
        analysis_markers = ("摘要", "总结", "创新", "贡献", "方法", "实验", "结论", "局限", "对比", "比较", "分析")
        field_only = set(fields).issubset({"title", "authors", "venue", "published", "year", "tags"})
        return field_only and not any(marker in content for marker in analysis_markers)

    @staticmethod
    def _has_document_category_observation(observations: list[_ReactObservation]) -> bool:
        return any(
            observation.tool == "library.explorer.document_categories" and observation.status == "completed"
            for observation in observations
        )

    @staticmethod
    def _needs_library_stats(content: str) -> bool:
        return "论文" in content and any(marker in content for marker in ("几篇", "多少", "数量", "总共", "共有"))

    @staticmethod
    def _needs_category_stats(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类", "无标签", "没有标签")) and any(
            marker in content
            for marker in (
                "几篇",
                "几类",
                "多少",
                "分别",
                "哪些",
                "统计",
                "列出",
                "最多",
                "最少",
                "最大",
                "最小",
                "排名",
                "分布",
                "占比",
            )
        )

    @staticmethod
    def _is_tagged_count_question(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("几篇", "多少", "数量", "有几")
        )

    @staticmethod
    def _is_clear_categories_intent(content: str) -> bool:
        content = KnowledgeAgentRuntime._strip_guardrail_negations(content)
        if KnowledgeAgentRuntime._is_delete_unused_categories_intent(content):
            return False
        return any(KnowledgeAgentRuntime._is_clear_categories_clause(clause) for clause in KnowledgeAgentRuntime._goal_clauses(content))

    @staticmethod
    def _is_delete_unused_categories_intent(content: str) -> bool:
        normalized = content.casefold()
        has_category_target = any(
            marker in normalized
            for marker in (
                "标签",
                "分类",
                "tag",
                "tags",
                "category",
                "categories",
                "label",
                "labels",
                "鏍囩",
                "鍒嗙被",
            )
        )
        if not has_category_target:
            return False
        has_delete = any(
            marker in normalized
            for marker in (
                "删除",
                "删掉",
                "移除",
                "清理",
                "删",
                "delete",
                "remove",
                "cleanup",
                "clean up",
                "鍒犻櫎",
                "鍒犳帀",
                "绉婚櫎",
            )
        )
        if not has_delete:
            return False
        has_unused_scope = any(
            marker in normalized
            for marker in (
                "空",
                "空的",
                "没有任何一篇论文",
                "没有任何论文",
                "没有论文",
                "没有关联",
                "无关联",
                "count=0",
                "0 篇",
                "0篇",
                "unused",
                "empty",
                "zero",
            )
        )
        has_document_relation_target = any(
            marker in normalized
            for marker in (
                "没有标签的论文",
                "无标签论文",
                "未打标签论文",
                "untagged documents",
                "untagged papers",
            )
        )
        return has_unused_scope and not has_document_relation_target

    @staticmethod
    def _is_clear_categories_clause(content: str) -> bool:
        if KnowledgeAgentRuntime._is_delete_unused_categories_intent(content):
            return False
        if KnowledgeAgentRuntime._is_ambiguous_category_delete_request(content):
            return False
        if any(marker in content for marker in KnowledgeAgentRuntime._rename_category_markers()):
            return False
        if any(marker in content for marker in ("加个", "加一个", "添加", "新增", "打上", "补上", "归类", "assign", "add")):
            return False
        lowered = content.casefold()
        has_category_target = any(marker in lowered for marker in ("标签", "分类", "tag", "category"))
        if not has_category_target:
            return False
        if any(marker in lowered for marker in ("清除", "清空", "清楚", "去掉", "clear")):
            return True
        if any(marker in lowered for marker in ("移除", "删掉", "删除", "remove")):
            if any(marker in lowered for marker in ("这篇", "这几篇", "这些", "刚刚", "刚才", "上面", "选中", "selected")):
                return True
            return any(
                marker in lowered
                for marker in (
                    "所有论文",
                    "全部论文",
                    "所有标签",
                    "全部标签",
                    "所有分类",
                    "全部分类",
                    "标签都",
                    "分类都",
                )
            )
        return False

    @staticmethod
    def _is_ambiguous_category_delete_request(content: str) -> bool:
        normalized = content.casefold()
        has_delete = any(marker in normalized for marker in ("删除", "删掉", "移除", "去掉", "清掉", "清理", "delete", "remove"))
        has_category = any(marker in normalized for marker in ("标签", "分类", "tag", "category"))
        if not has_delete or not has_category:
            return False
        if KnowledgeAgentRuntime._is_delete_unused_categories_intent(content):
            return False
        if any(marker in normalized for marker in ("空标签", "空分类", "空的标签", "空的分类", "unused", "empty")):
            return False
        ambiguous_patterns = (
            "标签下的论文",
            "分类下的论文",
            "这个标签下",
            "该标签下",
            "这个分类下",
            "该分类下",
            "无用标签",
            "没用标签",
            "无用分类",
            "没用分类",
            "这些都清掉",
            "这些都删掉",
            "整理一下分类",
            "整理分类",
        )
        return any(pattern in normalized for pattern in ambiguous_patterns)

    @staticmethod
    def _goal_clauses(content: str) -> list[str]:
        """Fallback clause splitter for compound goals; LLM remains the primary planner."""

        separators = (
            "然后",
            "再",
            "之后",
            "随后",
            "同时",
            "并且",
            "另外",
            "接着",
            "；",
            ";",
        )
        pattern = "|".join(re.escape(separator) for separator in separators)
        clauses = [item.strip(" ，,。.\n\t") for item in re.split(pattern, content) if item.strip(" ，,。.\n\t")]
        return clauses or [content]

    @staticmethod
    def _strip_guardrail_negations(content: str) -> str:
        """Guardrail: do not treat safety instructions that forbid deletion as user intent."""

        cleaned_lines: list[str] = []
        destructive_markers = ("删除", "清空", "清除", "移除", "删掉", "delete", "remove", "clear")
        negation_markers = ("不得", "不要", "不能", "禁止", "不允许", "must not", "do not")
        for line in content.splitlines():
            normalized = line.casefold()
            if any(marker in normalized for marker in destructive_markers) and any(
                marker in normalized for marker in negation_markers
            ):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _is_rename_category_intent(content: str) -> bool:
        if KnowledgeAgentRuntime._is_assignment_intent(content):
            return False
        return any(marker in content for marker in KnowledgeAgentRuntime._rename_category_markers()) and any(
            marker in content for marker in ("标签", "分类", "tag", "category")
        )

    @staticmethod
    def _rename_category_markers() -> tuple[str, ...]:
        return ("换成", "换为", "替换为", "替换成", "改成", "改为", "改名为", "重命名为", "重命名", "rename", "replace")

    @staticmethod
    def _is_tag_category_semantics_conflict(content: str) -> bool:
        normalized = content.casefold()
        clear_tag = (
            any(marker in normalized for marker in ("清除标签", "清空标签", "清楚标签", "删除标签", "移除标签", "去掉标签", "clear tags", "remove tags"))
            or ("标签" in normalized and any(marker in normalized for marker in ("清除", "清空", "清楚", "删除", "移除", "去掉")))
        )
        keep_category = any(
            marker in normalized
            for marker in (
                "保留分类",
                "分类保留",
                "保留对应的分类",
                "保留所属的分类",
                "keep categories",
                "preserve categories",
            )
        )
        return clear_tag and keep_category

    @staticmethod
    def _is_create_category_intent(content: str) -> bool:
        return any(marker in content for marker in ("新建", "创建", "新增", "建立", "加一个", "添加一个")) and any(
            marker in content for marker in ("分类", "标签")
        )

    @staticmethod
    def _is_assignment_intent(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "打标签",
                "打上",
                "打一个",
                "加标签",
                "加个",
                "加一个",
                "添加标签",
                "加上",
                "补上",
                "归类",
                "设置分类",
                "添加分类",
                "设置成",
                "设为",
                "设成",
                "改成",
                "标为",
                "新增",
            )
        )

    @staticmethod
    def _needs_untagged_assignment(content: str) -> bool:
        return any(marker in content for marker in ("没有标签", "无标签", "未打标签", "未分类", "没标签")) and any(
            marker in content for marker in ("加上", "补上", "添加", "打上", "都", "所有", "新增")
        )

    def _should_target_untagged_from_context(self, content: str, state: dict[str, Any]) -> bool:
        if not any(marker in content for marker in ("另外", "剩下", "剩余", "其余", "其他")):
            return False
        if not self._is_assignment_intent(content):
            return False
        current_untagged_count = self._category_stats_payload(limit_documents=0).get("untagged_document_count", 0)
        if current_untagged_count <= 0:
            return False
        last_goal = str(state.get("last_user_goal") or "")
        if not last_goal:
            return True
        if not any(marker in last_goal for marker in ("标签", "分类", "有标签", "无标签", "未分类")):
            return False
        if not any(marker in last_goal for marker in ("几篇", "多少", "数量", "统计", "哪些", "列出")):
            return False
        return True

    @staticmethod
    def _mentions_all_categories(content: str) -> bool:
        return any(marker in content for marker in ("所有标签", "全部标签", "所有分类", "全部分类", "所有论文的标签", "所有论文的分类"))

    @staticmethod
    def _is_document_category_query(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("什么", "哪些", "分别", "属于", "查看", "列出")
        )

    def _category_name_from_request_or_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> str | None:
        return (
            self._extract_existing_category_mention(content)
            or self._extract_category_name_from_request(content)
            or self._category_name_from_observations(observations)
        )

    @staticmethod
    def _category_name_from_observations(observations: list[_ReactObservation]) -> str | None:
        for observation in reversed(observations):
            category_name = observation.payload.get("category_name")
            if isinstance(category_name, str) and category_name.strip():
                return category_name.strip()
            category = observation.payload.get("category")
            if isinstance(category, dict) and isinstance(category.get("name"), str):
                return category["name"].strip()
        return None

    @staticmethod
    def _category_names_from_observations(observations: list[_ReactObservation]) -> list[str]:
        names: list[str] = []
        for observation in reversed(observations):
            for item in observation.payload.get("category_names") or []:
                name = str(item).strip()
                if name and name not in names:
                    names.append(name)
            category_name = observation.payload.get("category_name")
            if isinstance(category_name, str) and category_name.strip() and category_name.strip() not in names:
                names.append(category_name.strip())
            if names:
                return names
        return names

    def _extract_category_names_from_request(self, content: str) -> list[str]:
        content = self._category_name_extraction_segment(content)
        names: list[str] = []

        def add(value: str) -> None:
            candidate = self._clean_category_name(value)
            if candidate and not self._category_name_validation_error(candidate) and candidate not in names:
                names.append(candidate)

        for value in re.findall(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content):
            add(value)
        for pattern in (
            r"(?:标签|分类)\s*[：:]\s*([^，。！？；;\n]+)",
            r"(?:名为|叫做|叫|命名为)\s*[\"'“”‘’「」『』《》]?([^\"'“”‘’「」『』《》，。！？；;\n]{1,40})",
        ):
            match = re.search(pattern, content)
            if match:
                add(match.group(1))
        for match in re.finditer(
            r"(?:一个|1个)\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})\s*(?:标签|分类)",
            content,
        ):
            add(match.group(1))
        for match in re.finditer(r"\b([A-Za-z0-9_-]{1,40})\s*(?:标签|分类)", content):
            add(match.group(1))
        tag_sequence = re.search(
            r"(?:加上|添加|新增|新建|创建|打上|补上)\s*(.+)",
            content,
        )
        if tag_sequence:
            raw = re.split(r"(?:然后|之后|接着|再把|再将|，|,|。|；|;|\n)", tag_sequence.group(1), maxsplit=1)[0]
            for value in re.split(r"(?:和|与|及|、|,|，|/|\\|&|\+|以及)", raw):
                cleaned = re.sub(r"^(?:一个|1个|一個|个|個)\s*", "", value.strip())
                cleaned = re.sub(r"^(?:标签|分类)\s*[：:]\s*", "", cleaned)
                cleaned = re.sub(r"(?:标签|分类)$", "", cleaned).strip()
                add(cleaned)
        if not names:
            single = self._extract_category_name_from_request(content)
            if single:
                add(single)
        return names

    def _category_name_extraction_segment(self, content: str) -> str:
        clauses = [
            clause.strip()
            for clause in re.split(r"(?:然后|之后|接着|随后|再把|再将|并且把|同时把)", content)
            if clause.strip()
        ]
        if len(clauses) <= 1:
            return content
        for clause in clauses:
            if self._is_assignment_intent(clause) or self._is_create_category_intent(clause):
                return clause
        return clauses[0]

    def _category_names_from_arguments(self, arguments: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for item in arguments.get("category_names") or []:
            candidate = self._clean_category_name(str(item))
            if candidate and not self._category_name_validation_error(candidate) and candidate not in names:
                names.append(candidate)
        if not names:
            category_name = self._clean_category_name(str(arguments.get("category_name") or ""))
            if category_name and not self._category_name_validation_error(category_name):
                names.append(category_name)
        return names

    def _category_names_validation_error(self, category_names: list[str]) -> str | None:
        if not category_names:
            return "我没有可靠识别出要使用的标签/分类名称，因此没有改动论文库。"
        for category_name in category_names:
            error = self._category_name_validation_error(category_name)
            if error:
                return error
        return None

    def _extract_category_name_from_request(self, content: str) -> str | None:
        quote_match = re.search(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content)
        if quote_match:
            candidate = self._clean_category_name(quote_match.group(1))
            if candidate and not self._category_name_validation_error(candidate):
                return candidate

        patterns = [
            r"(?:标签|分类)\s*[：:]\s*([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
            r"(?:名为|叫做|叫|命名为)\s*[\"'“”‘’「」『』《》]?([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
            r"(?:新增|新建|创建|建立|添加|加一个)\s*(?:一个|1个)?\s*[\"'“”‘’「」『』《》]?([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})[\"'“”‘’「」『』《》]?\s*(?:标签|分类)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            candidate = self._clean_category_name(match.group(1))
            if candidate and not self._category_name_validation_error(candidate):
                return candidate
        existing = self._extract_existing_category_mention(content)
        if existing:
            return existing
        return None

    def _extract_category_rename_request(self, content: str) -> tuple[str, str] | None:
        rename_markers = self._rename_category_markers()
        if not any(marker in content for marker in rename_markers):
            return None
        categories = sorted(self.category_repository.list_categories(), key=lambda item: len(item.name), reverse=True)
        marker_positions = [
            position
            for marker in rename_markers
            if (position := content.casefold().find(marker.casefold())) >= 0
        ]
        replace_position = min(marker_positions) if marker_positions else len(content)
        source_candidates: list[tuple[int, str]] = []
        for category in categories:
            if not category.name:
                continue
            position = content.rfind(category.name, 0, replace_position)
            if position >= 0:
                source_candidates.append((position, category.name))
        source_name = None
        if source_candidates:
            source_candidates.sort(key=lambda item: item[0], reverse=True)
            source_name = source_candidates[0][1]
        else:
            for category in categories:
                if category.name and category.name in content:
                    source_name = category.name
                    break
        if source_name is None:
            return None
        source_index = content.find(source_name)
        trailing = content[source_index + len(source_name) :] if source_index >= 0 else content
        target_patterns = [
            r"(?:换成|换为|替换为|替换成|改成|改为|改名为|重命名为|重命名)\s*[\"'“”‘’「」『』《》]?([^\"'“”‘’「」『』《》，,。；;\s]{1,40})",
            r"(?:rename|replace).{0,20}?(?:to|as)\s*[\"']?([A-Za-z0-9_\-\u4e00-\u9fff ]{1,40})",
        ]
        target_name = None
        for pattern in target_patterns:
            match = re.search(pattern, trailing, flags=re.IGNORECASE)
            if match:
                target_name = self._clean_category_name(match.group(1))
                break
        if not target_name:
            quoted = re.findall(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content)
            for candidate in quoted:
                cleaned = self._clean_category_name(candidate)
                if cleaned and cleaned != source_name:
                    target_name = cleaned
                    break
        if not target_name or self._category_name_validation_error(target_name):
            return None
        return source_name, target_name

    def _extract_existing_category_mention(self, content: str) -> str | None:
        categories = sorted(self.category_repository.list_categories(), key=lambda item: len(item.name), reverse=True)
        for category in categories:
            name = category.name.strip()
            if not name or name not in content:
                continue
            if any(
                marker in content
                for marker in (
                    f"{name}标签",
                    f"{name} 标签",
                    f"{name}分类",
                    f"{name} 分类",
                    f"带着{name}",
                    f"带有{name}",
                    f"属于{name}",
                )
            ):
                return name
            if any(marker in content for marker in ("标签", "分类")):
                return name
        return None

    def _category_entity_names_for_request(self, content: str) -> list[str]:
        """Resolve explicit tag/category/group mentions as structured library entities.

        This is a conservative fallback/grounding helper, not the semantic router:
        it only activates when the user names a tag/category/group-like scope, then
        validates the names against the project's existing CategoryRepository.
        """

        if not self._is_labeled_document_collection_request(content):
            return []

        existing_names: list[str] = []
        categories = sorted(
            self.category_repository.list_categories(),
            key=lambda item: len(item.name.strip()),
            reverse=True,
        )
        for category in categories:
            name = category.name.strip()
            if not name or name in existing_names:
                continue
            if self._category_name_mentioned_with_entity_context(content, name):
                existing_names.append(name)

        if existing_names:
            return existing_names

        candidates: list[str] = []

        def add(value: str) -> None:
            candidate = self._clean_category_name(value)
            if (
                candidate
                and not self._category_name_validation_error(candidate)
                and candidate not in {"没有标签", "无标签", "未分类", "没标签"}
                and candidate not in candidates
            ):
                candidates.append(candidate)

        quoted = re.findall(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{1,40})[\"'“”‘’「」『』《》]", content)
        if quoted and any(marker in content for marker in ("标签", "分类", "分组", "集合")):
            for value in quoted:
                add(value)

        entity_patterns = (
            r"([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})\s*(?:标签|分类|分组|集合)\s*(?:下|下面|里的|中的|内|对应|关联|的)",
            r"(?:标签|分类|分组|集合)\s*[：:]\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})",
            r"(?:属于|带有|带着)\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})\s*(?:标签|分类|分组)?",
            r"(?:标签|分类|分组|集合)\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,40})\s*(?:下|下面|里的|中的|内|对应|关联)",
        )
        for pattern in entity_patterns:
            for match in re.finditer(pattern, content):
                add(match.group(1))

        multi = re.search(
            r"(?:对|把|将|分析|总结|统计|列出|查看)\s*(.+?)\s*(?:标签|分类|分组|集合)(?:下|下面|里的|中的|内|对应|关联|的)",
            content,
        )
        if multi:
            raw = multi.group(1)
            raw = re.sub(r"^(?:我的|这些|这几个|这几类|所有|全部)\s*", "", raw.strip())
            for value in re.split(r"(?:和|与|及|、|,|，|/|\\|&|\+|以及)", raw):
                add(value)

        return candidates

    @staticmethod
    def _category_name_mentioned_with_entity_context(content: str, name: str) -> bool:
        escaped = re.escape(name)
        patterns = (
            rf"{escaped}\s*(?:标签|分类|分组|集合)",
            rf"{escaped}\s*(?:下|下面|里的|中的|内|对应|关联)的?(?:论文|文章|文档|paper|papers|document|documents)",
            rf"(?:标签|分类|分组|集合)\s*[：:]\s*{escaped}(?:\b|[，。！？；;、\s]|$)",
            rf"(?:属于|带有|带着)\s*{escaped}\s*(?:标签|分类|分组|集合)?",
        )
        if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns):
            return True
        if name in content and any(marker in content for marker in ("标签", "分类", "分组", "集合")):
            return True
        return False

    def _category_lookup_payload(self, category_names: list[str]) -> dict[str, Any]:
        requested = []
        for name in category_names:
            cleaned = self._clean_category_name(str(name))
            if cleaned and cleaned not in requested:
                requested.append(cleaned)

        categories = self.category_repository.list_categories()

        def normalize(value: str) -> str:
            return re.sub(r"[\s_\-：:，,。.;；、\"'“”‘’「」『』《》]+", "", value.casefold())

        matched_names: list[str] = []
        missing_names: list[str] = []
        ambiguous_names: list[str] = []
        candidate_names: list[str] = []
        matches: list[dict[str, Any]] = []

        for requested_name in requested:
            exact = [category for category in categories if category.name == requested_name]
            if not exact:
                normalized = normalize(requested_name)
                exact = [category for category in categories if normalize(category.name) == normalized]
            if len(exact) == 1:
                category = exact[0]
                if category.name not in matched_names:
                    matched_names.append(category.name)
                matches.append(
                    {
                        "requested_name": requested_name,
                        "id": category.id,
                        "name": category.name,
                        "match_type": "exact" if category.name == requested_name else "normalized",
                    }
                )
                continue
            if len(exact) > 1:
                ambiguous_names.append(requested_name)
                for category in exact:
                    if category.name not in candidate_names:
                        candidate_names.append(category.name)
                continue

            close = self._similar_category_names(requested_name, categories)
            if close:
                candidate_names.extend(name for name in close if name not in candidate_names)
            missing_names.append(requested_name)

        if not candidate_names:
            candidate_names = [category.name for category in categories[:8]]

        return {
            "requested_names": requested,
            "matched_names": matched_names,
            "missing_names": missing_names,
            "ambiguous_names": ambiguous_names,
            "candidate_names": candidate_names[:8],
            "matches": matches,
        }

    @staticmethod
    def _similar_category_names(name: str, categories: list[Any]) -> list[str]:
        scored: list[tuple[float, str]] = []
        lowered = name.casefold()
        for category in categories:
            candidate = str(getattr(category, "name", "") or "").strip()
            if not candidate:
                continue
            candidate_lower = candidate.casefold()
            score = difflib.SequenceMatcher(None, lowered, candidate_lower).ratio()
            if lowered in candidate_lower or candidate_lower in lowered:
                score = max(score, 0.72)
            if score >= 0.45:
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [name for _score, name in scored[:5]]

    @staticmethod
    def _is_labeled_document_collection_request(content: str) -> bool:
        lowered = content.casefold()
        if any(marker in lowered for marker in ("没有标签", "无标签", "未分类", "untagged")):
            return False
        if any(marker in lowered for marker in ("有标签", "带标签", "已打标签", "已分类", "tagged")) and not re.search(
            r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,40}\s*(?:标签|分类|分组|集合)\s*(?:下|下面|里的|中的|内|对应|关联)",
            content,
        ):
            return False
        has_entity_marker = any(marker in lowered for marker in ("标签", "分类", "分组", "集合", "tag", "category", "group"))
        has_document_scope = any(
            marker in lowered
            for marker in ("论文", "文章", "文档", "paper", "papers", "document", "documents", "下", "下面", "里的", "中的")
        )
        return has_entity_marker and has_document_scope

    @staticmethod
    def _is_labeled_document_analysis_request(content: str) -> bool:
        lowered = content.casefold()
        return any(
            marker in lowered
            for marker in (
                "分析",
                "报告",
                "总结",
                "综述",
                "概述",
                "对比",
                "比较",
                "创新",
                "贡献",
                "方法",
                "实验",
                "结论",
                "不足",
                "report",
                "summary",
                "review",
                "compare",
                "analysis",
            )
        )

    @staticmethod
    def _has_labeled_document_set_observation(observations: list[_ReactObservation]) -> bool:
        return any(
            observation.tool == "library.explorer.find_documents"
            and (
                observation.payload.get("category_lookup")
                or observation.payload.get("category_names")
                or observation.payload.get("category_name")
            )
            for observation in observations
        )

    @staticmethod
    def _clean_category_name(value: str) -> str:
        cleaned = value.strip().strip("：:，,。.;；、 \t\r\n\"'“”‘’「」『』《》")
        cleaned = re.split(r"(?:并且|然后|同时|再|并|且|都|给|把|目前|所有|没有标签的|无标签的|没有标签|无标签)", cleaned, maxsplit=1)[0]
        return cleaned.strip().strip("：:，,。.;；、 \t\r\n\"'“”‘’「」『』《》")

    @staticmethod
    def _category_name_validation_error(category_name: str) -> str | None:
        name = category_name.strip()
        if not name:
            return "我没有可靠识别出要使用的标签/分类名称，因此没有改动论文库。"
        if name in {"标签", "分类", "分组", "集合", "一个", "1个", "不存在", "不存在的"}:
            return "识别到的标签/分类名称不是具体名称，因此没有改动论文库。"
        if len(name) > 40:
            return "识别到的标签/分类名称过长，像是把整句话当成了标签名；本轮没有改动论文库。"
        command_markers = ("帮我", "并且", "然后", "所有", "目前", "没有标签", "无标签", "加上", "补上", "这个标签", "不存在的")
        if any(marker in name for marker in command_markers):
            return "识别到的标签/分类名称包含后续命令，已拦截，避免创建错误标签。"
        return None

    def _category_exists(self, category_name: str) -> bool:
        return any(category.name == category_name for category in self.category_repository.list_categories())

    def _documents_for_category(self, category_name: str) -> list[LibraryDocument]:
        return [
            document
            for document in self.document_library_service.list_documents()
            if document.status == "ready" and any(category.name == category_name for category in document.categories)
        ]

    @staticmethod
    def _category_filter_document_ids_from_observations(observations: list[_ReactObservation]) -> list[str] | None:
        for observation in reversed(observations):
            if observation.tool != "library.explorer.find_documents":
                continue
            payload = observation.payload or {}
            category_lookup = payload.get("category_lookup")
            has_category_filter = bool(
                payload.get("category_name")
                or payload.get("category_names")
                or isinstance(category_lookup, dict)
            )
            if not has_category_filter:
                continue
            if isinstance(category_lookup, dict) and (
                category_lookup.get("missing_names") or category_lookup.get("ambiguous_names")
            ):
                return []
            document_ids: list[str] = []
            for item in payload.get("document_ids") or []:
                if isinstance(item, str) and item not in document_ids:
                    document_ids.append(item)
            for document in payload.get("documents") or []:
                if isinstance(document, dict):
                    document_id = document.get("id")
                    if isinstance(document_id, str) and document_id not in document_ids:
                        document_ids.append(document_id)
            return document_ids
        return None

    @staticmethod
    def _document_ids_from_observations(observations: list[_ReactObservation]) -> list[str]:
        document_ids: list[str] = []
        for observation in observations:
            for item in observation.payload.get("document_ids") or []:
                if isinstance(item, str) and item not in document_ids:
                    document_ids.append(item)
            for document in observation.payload.get("documents") or []:
                if isinstance(document, dict):
                    document_id = document.get("id")
                    if isinstance(document_id, str) and document_id not in document_ids:
                        document_ids.append(document_id)
        return document_ids

    @staticmethod
    def _merge_document_ids(existing: list[str], payload: dict[str, Any]) -> list[str]:
        merged = list(existing)
        for item in payload.get("document_ids") or []:
            if isinstance(item, str) and item not in merged:
                merged.append(item)
        for document in payload.get("documents") or []:
            if isinstance(document, dict):
                document_id = document.get("id")
                if isinstance(document_id, str) and document_id not in merged:
                    merged.append(document_id)
        for group in payload.get("category_groups") or []:
            if not isinstance(group, dict):
                continue
            for document_id in group.get("document_ids") or []:
                if isinstance(document_id, str) and document_id not in merged:
                    merged.append(document_id)
        return merged

    @staticmethod
    def _merge_evidence_items(existing: list[EvidenceItem], payload: dict[str, Any]) -> list[EvidenceItem]:
        merged = list(existing)
        seen = {item.id for item in merged}
        for item in payload.get("evidence_items") or []:
            try:
                evidence = item if isinstance(item, EvidenceItem) else EvidenceItem(**item)
            except Exception:
                continue
            if evidence.id not in seen:
                seen.add(evidence.id)
                merged.append(evidence)
        return merged

    def _evidence_items_from_observations(self, observations: list[_ReactObservation]) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        for observation in observations:
            payload_view = self._react_observation_payload(observation)
            if "evidence_items" not in payload_view:
                payload_view = {**payload_view, "evidence_items": self._react_observation_evidence(observation)}
            evidence_items = self._merge_evidence_items(evidence_items, payload_view)
        return evidence_items

    @staticmethod
    def _category_groups_from_observations(observations: list[_ReactObservation]) -> list[dict[str, Any]]:
        for observation in reversed(observations):
            groups = observation.payload.get("category_groups")
            if isinstance(groups, list) and groups:
                return [group for group in groups if isinstance(group, dict)]
        return []

    def _category_groups_from_library(self, category_names: list[str] | None = None) -> list[dict[str, Any]]:
        requested = {name for name in (category_names or []) if name}
        groups: dict[str, dict[str, Any]] = {}
        for document in self.document_library_service.list_documents():
            if document.status != "ready":
                continue
            for category in document.categories:
                if requested and category.name not in requested:
                    continue
                group = groups.setdefault(
                    category.name,
                    {
                        "category_name": category.name,
                        "document_ids": [],
                        "documents": [],
                        "evidence_items": [],
                    },
                )
                group["document_ids"].append(document.id)
                group["documents"].append(self._document_payload(document))
        return list(groups.values())

    def _compose_read_observation_answer(
        self,
        content: str,
        stats: dict[str, Any] | None,
        category_stats: dict[str, Any] | None,
        document_categories: dict[str, Any] | None,
    ) -> str:
        parts: list[str] = []
        if stats and self._needs_library_stats(content):
            parts.append(
                f"当前本地论文库共有 {stats['total']} 篇论文，其中 {stats['ready']} 篇可用、"
                f"{stats['processing']} 篇处理中、{stats['failed']} 篇处理失败。"
            )
        include_full_category_stats = bool(
            category_stats
            and self._needs_category_stats(content)
            and not document_categories
            and not self._asks_category_extreme(content)
            and self._asks_category_overview(content)
        )
        if category_stats and self._is_tagged_count_question(content) and not include_full_category_stats:
            parts.append(f"当前带分类/标签的论文有 {category_stats['tagged_document_count']} 篇。")
        if document_categories and self._should_include_document_category_detail(content):
            lines = []
            for document in document_categories.get("documents", []):
                categories = document.get("categories") or ["暂无标签/分类"]
                lines.append(f"- {document.get('name')}：{'、'.join(categories)}")
            if lines:
                parts.append("这些论文当前的标签/分类是：\n" + "\n".join(lines))
        if category_stats and self._asks_category_extreme(content):
            extreme_answer, category_name = self._category_extreme_answer(content, category_stats)
            if extreme_answer:
                parts.append(extreme_answer)
                repeated = self._repeat_text_answer(content, category_name)
                if repeated:
                    parts.append(repeated)
        elif category_stats:
            repeated = self._repeat_text_answer(content, self._category_name_from_request_or_observations(content, []))
            if repeated:
                parts.append(repeated)
        if (not parts or include_full_category_stats) and category_stats and self._needs_category_stats(content):
            category_names = [
                f"{item['name']}（{item['document_count']} 篇）"
                for item in category_stats.get("categories", [])
            ]
            labels = "、".join(category_names) if category_names else "暂无标签/分类"
            parts.append(
                f"有标签的论文 {category_stats['tagged_document_count']} 篇，"
                f"没有标签的论文 {category_stats['untagged_document_count']} 篇；"
                f"共有 {category_stats['category_count']} 类标签/分类：{labels}。"
            )
        if not parts and document_categories:
            lines = []
            for document in document_categories.get("documents", []):
                categories = document.get("categories") or ["暂无标签/分类"]
                lines.append(f"- {document.get('name')}：{'、'.join(categories)}")
            if lines:
                parts.append("这些论文当前的标签/分类是：\n" + "\n".join(lines))
        return "\n".join(part for part in parts if part.strip())

    def _compose_labeled_document_set_answer(
        self,
        content: str,
        payload: dict[str, Any] | None,
    ) -> str:
        if not payload:
            return ""
        category_lookup = payload.get("category_lookup")
        category_names = [
            str(item)
            for item in (
                (category_lookup or {}).get("matched_names")
                if isinstance(category_lookup, dict)
                else payload.get("category_names") or ([payload.get("category_name")] if payload.get("category_name") else [])
            )
            if str(item).strip()
        ]
        if not category_names:
            return ""
        if isinstance(category_lookup, dict) and category_lookup.get("missing_names"):
            missing = "、".join(str(item) for item in category_lookup.get("missing_names") or [])
            candidates = "、".join(str(item) for item in category_lookup.get("candidate_names") or [])
            suffix = f"；相近标签/分类：{candidates}" if candidates else ""
            return f"没有找到标签/分类「{missing}」{suffix}。"
        documents = [item for item in payload.get("documents") or [] if isinstance(item, dict)]
        label_text = "、".join(category_names)
        prefix = f"已识别到「{label_text}」是标签/分类，下面共有 {len(documents)} 篇 ready 论文。"
        if not documents:
            return prefix
        if any(marker in content for marker in ("几篇", "多少", "统计", "数量")) and not any(
            marker in content for marker in ("哪些", "列出", "名单", "分别")
        ):
            return prefix
        lines = [prefix, "关联论文如下："]
        for document in documents:
            name = str(document.get("name") or document.get("filename") or document.get("title") or "未命名论文")
            title = str(document.get("title") or "").strip()
            if title and title != name:
                lines.append(f"- {name}（题名：{title}）")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

    def _compose_metadata_observation_answer(
        self,
        content: str,
        document_metadata: dict[str, Any] | None,
    ) -> str:
        if not document_metadata:
            return ""
        documents = [item for item in document_metadata.get("documents") or [] if isinstance(item, dict)]
        if not documents:
            return ""
        requested_fields = document_metadata.get("requested_fields")
        if not isinstance(requested_fields, list) or not requested_fields:
            requested_fields = self._requested_metadata_fields(content)
        labels = {
            "title": "题名",
            "authors": "作者",
            "venue": "期刊/会议",
            "published": "发表时间",
            "year": "年份",
            "tags": "标签",
        }
        field_order = [field for field in ("title", "authors", "venue", "published", "year", "tags") if field in requested_fields]
        if not field_order:
            field_order = ["venue", "published", "year"]
        lines = ["按当前论文库元数据逐篇列出如下："]
        for document in documents:
            name = str(document.get("name") or document.get("filename") or document.get("title") or "未命名论文")
            values = [
                f"{labels.get(field, field)}：{self._metadata_display_value(document.get(field))}"
                for field in field_order
            ]
            lines.append(f"- {name}：" + "；".join(values))
        if any(
            self._is_missing_metadata_value(document.get(field))
            for document in documents
            for field in field_order
        ):
            lines.append("缺失说明：上述标为“缺失”的字段在当前论文库元数据或解析片段中没有可靠记录。")
        return "\n".join(lines)

    @staticmethod
    def _metadata_display_value(value: Any) -> str:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return "、".join(cleaned) if cleaned else "缺失"
        if value is None:
            return "缺失"
        text = str(value).strip()
        return text if text else "缺失"

    @staticmethod
    def _is_missing_metadata_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, list):
            return not any(str(item).strip() for item in value)
        return not str(value).strip()

    @staticmethod
    def _should_include_document_category_detail(content: str) -> bool:
        has_detail_marker = any(marker in content for marker in ("分别", "对应", "各自", "是什么", "哪些", "列出", "查看"))
        has_document_marker = any(marker in content for marker in ("论文", "文章", "文档", "每篇", "每个"))
        has_category_marker = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        return has_detail_marker and has_document_marker and has_category_marker

    @staticmethod
    def _asks_category_overview(content: str) -> bool:
        return any(
            marker in content
            for marker in ("几类", "无标签", "没有标签", "统计", "分布", "占比", "有哪些标签", "哪些标签", "列出标签")
        )

    @staticmethod
    def _asks_category_extreme(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        has_extreme = any(marker in content for marker in ("最多", "最少", "最大", "最小", "top", "most", "least"))
        return has_category_target and has_extreme

    @staticmethod
    def _category_extreme_answer(content: str, category_stats: dict[str, Any]) -> tuple[str, str | None]:
        categories = [
            item
            for item in category_stats.get("categories", [])
            if isinstance(item, dict) and item.get("name") and isinstance(item.get("document_count"), int)
        ]
        if not categories:
            return "当前还没有可统计的标签/分类。", None
        wants_min = any(marker in content for marker in ("最少", "最小", "least"))
        target_count = (min if wants_min else max)(item["document_count"] for item in categories)
        winners = [item for item in categories if item["document_count"] == target_count]
        names = "、".join(str(item["name"]) for item in winners)
        label = "最少" if wants_min else "最多"
        return f"论文数量{label}的标签/分类是：{names}（{target_count} 篇）。", str(winners[0]["name"])

    @staticmethod
    def _repeat_text_answer(content: str, value: str | None) -> str:
        if not value:
            return ""
        match = re.search(r"(?:重复输出|重复|输出).{0,12}?([0-9一二三四五六七八九十两]+)\s*次", content)
        if not match:
            return ""
        count = KnowledgeAgentRuntime._parse_small_count(match.group(1))
        if count <= 0:
            return ""
        count = min(count, 20)
        return "\n".join(str(value) for _ in range(count))

    @staticmethod
    def _parse_small_count(value: str) -> int:
        if value.isdigit():
            return int(value)
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if value == "十":
            return 10
        if "十" in value:
            left, _, right = value.partition("十")
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        return digits.get(value, 0)

    @staticmethod
    def _has_final_literal_output_request(content: str) -> bool:
        return KnowledgeAgentRuntime._final_literal_output(content) is not None

    @classmethod
    def _is_compound_question_request(cls, content: str) -> bool:
        return len(cls._split_questions(content)) >= 2

    def _should_answer_compound_with_evidence(
        self,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if selected_document_ids or any(attachment.document_id for attachment in attachments):
            return True
        return bool(self._extract_document_tokens(content))

    @staticmethod
    def _split_questions(content: str) -> list[str]:
        questions: list[str] = []
        start = 0
        for match in re.finditer(r"[？?]", content):
            question = content[start : match.end()].strip(" \t\r\n；;。")
            if question:
                questions.append(question)
            start = match.end()
        if questions:
            return questions

        question_markers = (
            "谁",
            "什么",
            "哪些",
            "哪个",
            "多少",
            "几",
            "怎么",
            "如何",
            "为什么",
            "为何",
            "是否",
            "who",
            "what",
            "which",
            "when",
            "where",
            "why",
            "how",
        )
        parts = [part.strip() for part in re.split(r"[；;\n。]+", content) if part.strip()]
        return [part for part in parts if any(marker in part.casefold() for marker in question_markers)]

    @staticmethod
    def _final_literal_output(content: str) -> str | None:
        patterns = [
            r"(?:最后|最终|末尾).{0,12}?输出(?:一个|一句|文本)?[「“\"']([^」”\"']{1,80})[」”\"']",
            r"(?:最后|最终|末尾).{0,12}?回复[「“\"']([^」”\"']{1,80})[」”\"']",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return None

    def _synthesize_react_answer(self, content: str, observations: list[_ReactObservation]) -> str:
        if not observations:
            return "我还没有拿到论文库观察结果，无法可靠回答。"
        last = observations[-1]
        if last.status in {"needs_clarification", "validation_failed", "failed", "confirmation_required"}:
            if last.tool == "library.explorer.find_documents" and last.payload.get("candidates"):
                candidates = "、".join(str(item) for item in last.payload["candidates"][:6])
                return f"{last.summary} 可以从这些候选里确认一下：{candidates}"
            if last.status == "confirmation_required":
                completed_summaries = self._completed_write_step_summaries(observations[:-1])
                if completed_summaries:
                    lines = [f"已完成第 {index} 步：{summary}" for index, summary in enumerate(completed_summaries, start=1)]
                    lines.append(f"下一步等待确认：{last.summary}")
                    return "\n".join(lines)
            return last.summary

        stats = self._latest_payload(observations, "library.explorer.stats")
        category_stats = self._latest_payload(observations, "library.explorer.category_stats")
        document_categories = self._latest_payload(observations, "library.explorer.document_categories")
        document_metadata = self._latest_payload(observations, "library.explorer.document_metadata")
        labeled_documents = self._latest_payload(observations, "library.explorer.find_documents")
        created = self._latest_payload(observations, "library.operator.create_category")
        assigned = self._latest_payload(observations, "library.operator.assign_category")
        renamed = self._latest_payload(observations, "library.operator.rename_category")
        deleted_unused = self._latest_payload(observations, "library.operator.delete_unused_categories")
        cleared = self._latest_payload(observations, "library.operator.clear_categories")
        report = self._latest_payload(observations, "report.drafter.write")
        category_report = self._latest_payload(observations, "report.drafter.write_by_category")

        if category_report and category_report.get("answer"):
            return str(category_report["answer"])
        labeled_answer = self._compose_labeled_document_set_answer(content, labeled_documents)
        if labeled_answer:
            return labeled_answer
        metadata_answer = self._compose_metadata_observation_answer(content, document_metadata)
        if metadata_answer:
            return metadata_answer
        if report and report.get("answer"):
            return str(report["answer"])
        if deleted_unused:
            deleted_names = [
                str(item)
                for item in deleted_unused.get("deleted_category_names") or []
                if str(item).strip()
            ]
            if deleted_unused.get("deleted_count", 0):
                return (
                    f"已删除 {deleted_unused.get('deleted_count', 0)} 个 count=0 的空标签/分类实体："
                    f"{'、'.join(deleted_names)}。已二次校验：没有修改任何论文的标签关系，非空标签/分类仍然保留。"
                )
            return "当前没有 count=0 的空标签/分类实体，无需删除；本轮没有修改任何论文标签关系。"
        if cleared:
            verified_state = cleared.get("verified_state")
            operation = cleared.get("operation")
            if operation == "remove_single_category_link":
                return (
                    f"已从 {cleared.get('updated_count', 0)} 篇论文中移除标签/分类"
                    f"「{cleared.get('category_name') or ''}」，其他标签已保留并二次校验。"
                )
            if isinstance(verified_state, dict):
                return (
                    f"已清空 {cleared.get('updated_count', 0)} 篇论文的分类/标签关系。"
                    f"当前带分类/标签的论文有 {verified_state.get('tagged_document_count', 0)} 篇，"
                    f"未分类/无标签的论文有 {verified_state.get('untagged_document_count', 0)} 篇。"
                )
            return f"已清空 {cleared.get('updated_count', 0)} 篇论文的分类/标签关系。"
        if renamed:
            source_name = renamed.get("source_category_name", "源标签")
            target_name = renamed.get("target_category_name", "目标标签")
            updated_count = renamed.get("updated_count", 0)
            rename_text = (
                f"已将标签/分类「{source_name}」"
                f"{'合并到已有标签' if renamed.get('merged') else '重命名为'}「{target_name}」，"
                f"迁移了 {updated_count} 篇论文的关联。"
            )
            if category_stats and target_name:
                for item in category_stats.get("categories") or []:
                    if isinstance(item, dict) and item.get("name") == target_name:
                        rename_text += f" 当前「{target_name}」标签下有 {item.get('document_count', 0)} 篇论文。"
                        break
            if assigned:
                category_name = assigned.get("category_name", "该标签")
                return (
                    f"已为 {assigned.get('updated_count', 0)} 篇论文追加「{category_name}」标签；"
                    + rename_text
                )
            return rename_text
        if assigned:
            category_names = [str(item) for item in assigned.get("category_names") or [] if str(item).strip()]
            category_name = "、".join(category_names) if category_names else assigned.get("category_name", "该标签")
            updated_count = assigned.get("updated_count", 0)
            prefix = ""
            if created:
                created_names = created.get("created_names") or []
                if created_names:
                    prefix = f"已新建标签/分类「{'、'.join(created_names)}」；"
                else:
                    prefix = f"已确认已有标签/分类「{category_name}」；"
            scope = assigned.get("scope")
            if scope == "untagged":
                verified_state = assigned.get("verified_state")
                suffix = ""
                if isinstance(verified_state, dict):
                    if len(category_names) > 1:
                        suffix = f" 当前仍无标签论文 {verified_state.get('untagged_document_count', 0)} 篇。"
                    else:
                        suffix = " 写操作已二次读取验证。"
                return f"{prefix}已为 {updated_count} 篇原本没有标签的论文补上「{category_name}」标签。{suffix}".strip()
            names = "、".join(item.get("name", "") for item in assigned.get("documents", [])[:6] if item.get("name"))
            suffix = f"：{names}" if names else ""
            return f"{prefix}已把标签/分类「{category_name}」追加到 {updated_count} 篇论文{suffix}。"
        if created:
            names = "、".join(created.get("category_names") or [created.get("category_name")])
            verb = "已新建" if created.get("created") else "已存在"
            return f"{verb}标签/分类「{names}」。"
        read_answer = self._compose_read_observation_answer(content, stats, category_stats, document_categories)
        if read_answer:
            return read_answer
        return last.summary

    @staticmethod
    def _completed_write_step_summaries(observations: list[_ReactObservation]) -> list[str]:
        summaries: list[str] = []
        for observation in observations:
            if observation.status != "completed" or not observation.tool.startswith("library.operator."):
                continue
            if observation.tool == "library.operator.rename_category":
                source = observation.payload.get("source_category_name")
                target = observation.payload.get("target_category_name")
                count = observation.payload.get("updated_count", 0)
                summaries.append(f"已将「{source}」迁移/重命名为「{target}」，影响 {count} 篇论文。")
            elif observation.tool == "library.operator.assign_category":
                names = "、".join(str(item) for item in observation.payload.get("category_names") or [observation.payload.get("category_name")] if item)
                summaries.append(f"已为 {observation.payload.get('updated_count', 0)} 篇论文追加标签「{names}」。")
            elif observation.tool == "library.operator.delete_unused_categories":
                names = "、".join(str(item) for item in observation.payload.get("deleted_category_names") or [] if item)
                summaries.append(f"已删除 {observation.payload.get('deleted_count', 0)} 个空标签/分类实体：{names}。")
            elif observation.tool == "library.operator.clear_categories":
                summaries.append(observation.summary)
            elif observation.tool == "library.operator.create_category":
                names = "、".join(str(item) for item in observation.payload.get("category_names") or [observation.payload.get("category_name")] if item)
                summaries.append(f"已确认标签/分类「{names}」。")
        return summaries

    def _user_visible_final_answer(
        self,
        content: str,
        observations: list[_ReactObservation],
        candidate: str,
    ) -> str:
        text = candidate.strip()
        if not text:
            return self._synthesize_react_answer(content, observations)
        if self._must_answer_from_observations(observations) or self._contains_internal_process_wording(text):
            return self._synthesize_react_answer(content, observations)
        return self._strip_internal_process_lines(text) or self._synthesize_react_answer(content, observations)

    @staticmethod
    def _must_answer_from_observations(observations: list[_ReactObservation]) -> bool:
        if not observations:
            return False
        completed_tools = {observation.tool for observation in observations if observation.status == "completed"}
        if any(tool.startswith("library.operator.") for tool in completed_tools):
            return True
        return bool(
            completed_tools
            & {
                "library.explorer.stats",
                "library.explorer.category_stats",
                "library.explorer.find_documents",
                "library.explorer.document_categories",
                "report.drafter.write",
                "report.drafter.write_by_category",
            }
        )

    @classmethod
    def _contains_internal_process_wording(cls, text: str) -> bool:
        return cls._strip_internal_process_lines(text) != text.strip()

    @staticmethod
    def _strip_internal_process_lines(text: str) -> str:
        process_markers = (
            "运行态摘要",
            "根据刚刚",
            "根据上下文",
            "根据对话记录",
            "根据我们的对话",
            "我将使用",
            "我需要确认",
            "首先，我需要",
            "正在执行",
            "工具",
            "tool",
            "library.",
        )
        kept = [
            line
            for line in text.strip().splitlines()
            if not any(marker in line for marker in process_markers)
        ]
        return "\n".join(line for line in kept if line.strip()).strip()

    @staticmethod
    def _latest_payload(observations: list[_ReactObservation], tool: str) -> dict[str, Any] | None:
        for observation in reversed(observations):
            if observation.tool == tool and observation.status == "completed":
                return KnowledgeAgentRuntime._react_observation_payload(observation)
        return None

    def _react_memory_payload(self, session_id: str) -> dict[str, Any]:
        session_summary = self.file_store.read_session_summary(session_id)
        reflection_path = self.file_store.get_session_dir(session_id) / "react_reflections.jsonl"
        reflections: list[dict[str, Any]] = []
        if reflection_path.exists():
            for line in reflection_path.read_text(encoding="utf-8").splitlines()[-5:]:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reflections.append(payload)
        return {
            "project_rules": self.file_store.read_project_rules()[:1200],
            "user_preferences": self.file_store.read_user_preferences()[:8],
            "session_summary": session_summary[:1200],
            "recent_reflections": reflections,
            "fixed_lessons": [
                "标签/分类统计要读取 document_categories 与文档关联，不能只报论文总数。",
                "复合写命令要拆步执行，标签名不能取后半句。",
            ],
        }

    def _react_state_path(self, session_id: str):
        self.file_store.initialize_session(session_id, "新对话")
        return self.file_store.get_session_dir(session_id) / "react_state.json"

    def _read_react_state(self, session_id: str) -> dict[str, Any]:
        path = self._react_state_path(session_id)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_react_state(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._react_state_path(session_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_react_state(
        self,
        *,
        session: ChatSession,
        user_goal: str,
        observations: list[_ReactObservation],
        status: str,
    ) -> None:
        if status not in {"completed", "degraded"}:
            return
        state = self._read_react_state(session.id)
        referent = self._derive_referent_from_observations(user_goal, observations)
        if referent is not None:
            state["last_document_set"] = referent
            ids = [str(item) for item in referent.get("document_ids") or [] if item]
            if len(ids) == 1:
                state["last_single_document"] = {
                    **referent,
                    "document_id": ids[0],
                    "document_ids": ids,
                    "count": 1,
                }
            elif len(ids) > 1:
                state["last_multi_document_set"] = {
                    **referent,
                    "document_ids": ids,
                    "count": len(ids),
                }
        category_name = self._category_name_from_observations(observations)
        if category_name:
            state["last_category_name"] = category_name
        state["last_user_goal"] = user_goal[:240]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_react_state(session.id, state)

    def _derive_referent_from_observations(
        self,
        user_goal: str,
        observations: list[_ReactObservation],
    ) -> dict[str, Any] | None:
        for observation in reversed(observations):
            payload = observation.payload
            if observation.tool == "library.operator.assign_category":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才打标签的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.find_documents":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才定位到的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.document_categories":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才查询标签的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.document_metadata":
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才查询元数据的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool in {"evidence.retriever.search", "report.drafter.write", "report.drafter.write_by_category"}:
                ids = [str(item) for item in payload.get("document_ids") or [] if item]
                if not ids:
                    ids = [
                        str(item.get("document_id") or item.get("source_id"))
                        for item in payload.get("evidence_items") or []
                        if isinstance(item, dict) and (item.get("document_id") or item.get("source_id"))
                    ]
                ids = list(dict.fromkeys(ids))
                if ids:
                    return {
                        "label": f"刚才分析的 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.category_stats":
                if any(marker in user_goal for marker in ("无标签", "没有标签", "未分类", "未打标签")):
                    ids = [str(item) for item in payload.get("untagged_document_ids") or [] if item]
                    label = f"刚才统计到的 {len(ids)} 篇无标签论文"
                elif any(marker in user_goal for marker in ("有标签", "已打标签", "已分类")):
                    ids = [str(item) for item in payload.get("tagged_document_ids") or [] if item]
                    label = f"刚才统计到的 {len(ids)} 篇有标签论文"
                else:
                    ids = []
                    label = ""
                if ids:
                    return {
                        "label": label,
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
            if observation.tool == "library.explorer.stats":
                ids = [str(item) for item in payload.get("referent_document_ids") or [] if item]
                if ids:
                    return {
                        "label": f"刚才统计到的全部 {len(ids)} 篇论文",
                        "document_ids": ids,
                        "source_tool": observation.tool,
                        "count": len(ids),
                    }
        return None

    @staticmethod
    def _state_document_ids(state: dict[str, Any]) -> list[str]:
        document_set = state.get("last_document_set")
        if not isinstance(document_set, dict):
            return []
        ids = document_set.get("document_ids")
        if not isinstance(ids, list):
            return []
        return [str(item) for item in ids if isinstance(item, str) and item]

    def _recent_scope_document_ids(self, session_id: str, *, singular: bool) -> list[str]:
        state = self._read_react_state(session_id)
        key = "last_single_document" if singular else "last_multi_document_set"
        ids = self._document_ids_from_state_entry(state.get(key))
        if ids:
            return self._real_document_ids(ids)
        fallback_ids = self._real_document_ids(self._state_document_ids(state))
        if singular:
            return fallback_ids if len(fallback_ids) == 1 else []
        return fallback_ids if len(fallback_ids) > 1 else []

    @staticmethod
    def _document_ids_from_state_entry(entry: Any) -> list[str]:
        if not isinstance(entry, dict):
            return []
        ids = entry.get("document_ids")
        if isinstance(ids, list):
            return [str(item) for item in ids if isinstance(item, str) and item]
        document_id = entry.get("document_id")
        if isinstance(document_id, str) and document_id:
            return [document_id]
        return []

    @staticmethod
    def _mentions_previous_referent(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "这几篇",
                "这四篇",
                "这些",
                "这几个",
                "上述",
                "上面",
                "刚才",
                "刚刚",
                "另外",
                "剩下",
                "剩余",
                "其余",
                "其他",
                "它们",
                "全部这些",
                "这批",
            )
        )

    def _record_react_reflection(
        self,
        *,
        session: ChatSession,
        user_goal: str,
        observations: list[_ReactObservation],
        status: str,
    ) -> None:
        self.file_store.initialize_session(session.id, session.title)
        path = self.file_store.get_session_dir(session.id) / "react_reflections.jsonl"
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_goal": user_goal[:240],
            "status": status,
            "tools": [observation.tool for observation in observations],
            "summary": self._reflection_summary(user_goal, observations, status),
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _reflection_summary(user_goal: str, observations: list[_ReactObservation], status: str) -> str:
        tools = " -> ".join(observation.tool for observation in observations) or "no tools"
        return f"ReAct task {status}: {user_goal[:80]} | tools: {tools}"

    def _append_react_trace(self, *, run_id: str, status: str, payload: dict[str, Any]) -> None:
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status=status,
            message=status.replace("_", " "),
            payload=payload,
        )

    @staticmethod
    def _safe_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= 4000:
            return payload
        return {"truncated": True, "preview": text[:4000]}

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _begin_run(self, session: ChatSession, topic: str) -> str:
        run_id = f"chat-{uuid4().hex}"
        self.research_repository.create_run(run_id, f"Chat: {topic[:80]}")
        self.research_repository.update_run_status(run_id, ResearchRunStatus.RUNNING_TASK)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="knowledge_agent_started",
            message="Knowledge main agent started.",
            payload={"session_id": session.id, "topic": topic},
        )
        return run_id

    def _finish_run(self, run_id: str, status: ResearchRunStatus = ResearchRunStatus.COMPLETED) -> None:
        self.research_repository.update_run_status(run_id, status)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="knowledge_agent_finished",
            message="Knowledge main agent finished.",
            payload={},
        )

    def _run_subagent(
        self,
        *,
        run_id: str,
        role: str,
        profile: SubagentProfile,
        goal: str,
        tool_policy: ToolPolicy,
        context_bundle: dict[str, Any],
        worker: Callable[[], _TaskOutcome],
    ) -> _TaskOutcome:
        if not self.enable_subagent_execution:
            self.message_bus.append_trace(
                run_id=run_id,
                task_id=None,
                trace_type=TraceEventType.CONTROL,
                status="knowledge_internal_step_started",
                message=f"{role} internal step started without subagent execution.",
                payload={
                    "agent_role": role,
                    "subagent_execution": "disabled",
                    "experimental_feature": "subagent",
                    "trigger_reason": "knowledge_runtime_internal_step",
                    "config_flag": "ENABLE_SUBAGENT_EXECUTION=false",
                    "tool_policy": tool_policy.model_dump(mode="json"),
                },
            )
            try:
                outcome = worker()
            except Exception as exc:
                self.message_bus.append_trace(
                    run_id=run_id,
                    task_id=None,
                    trace_type=TraceEventType.CONTROL,
                    status="knowledge_internal_step_failed",
                    message=f"{role} internal step failed.",
                    payload={
                        "agent_role": role,
                        "subagent_execution": "disabled",
                        "error": str(exc),
                    },
                )
                raise
            self.message_bus.append_trace(
                run_id=run_id,
                task_id=None,
                trace_type=TraceEventType.CONTROL,
                status="knowledge_internal_step_completed",
                message=f"{role} internal step completed without subagent execution.",
                payload={
                    "agent_role": role,
                    "subagent_execution": "disabled",
                    "summary": outcome.summary,
                },
            )
            return outcome

        task = AgentTask(
            id=f"{role}-{uuid4().hex[:8]}",
            run_id=run_id,
            profile=profile,
            goal=goal,
            context_bundle={"agent_role": role, **context_bundle},
            done_criteria="Return a compact result payload for the knowledge main agent.",
            tool_policy=tool_policy,
            artifact_dir=str(self.file_store.runtime_context_path if hasattr(self.file_store, "runtime_context_path") else self.file_store.root),
        )
        self.runtime_repository.create_task(task)
        self.runtime_repository.update_task_status(task.id, AgentTaskStatus.RUNNING)
        self.message_bus.publish_control(
            run_id=run_id,
            task_id=task.id,
            event_type="knowledge_subagent_started",
            message=f"{role} started.",
            payload={"agent_role": role, "tool_policy": tool_policy.model_dump(mode="json")},
        )
        try:
            outcome = worker()
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=profile,
                status=AgentTaskStatus.COMPLETED,
                summary=outcome.summary,
                result_payload={"agent_role": role, **outcome.payload},
                created_at=datetime.now(timezone.utc),
            )
            self.runtime_repository.update_task_status(task.id, AgentTaskStatus.COMPLETED)
            self.message_bus.publish_notification(
                run_id=run_id,
                notification=notification,
                event_type="knowledge_subagent_completed",
            )
            return outcome
        except Exception as exc:
            notification = TaskNotification(
                task_id=task.id,
                agent_profile=profile,
                status=AgentTaskStatus.FAILED,
                summary=f"{role} failed.",
                result_payload={"agent_role": role},
                error=str(exc),
                created_at=datetime.now(timezone.utc),
            )
            self.runtime_repository.update_task_status(task.id, AgentTaskStatus.FAILED)
            self.message_bus.publish_notification(
                run_id=run_id,
                notification=notification,
                event_type="knowledge_subagent_failed",
            )
            raise

    def _answer_library_count(self, session: ChatSession, content: str) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            documents = self.document_library_service.list_documents()
            ready_count = sum(1 for document in documents if document.status == "ready")
            processing_count = sum(1 for document in documents if document.status == "processing")
            failed_count = sum(1 for document in documents if document.status == "failed")
            return _TaskOutcome(
                summary=f"Found {len(documents)} library documents.",
                payload={
                    "total": len(documents),
                    "ready": ready_count,
                    "processing": processing_count,
                    "failed": failed_count,
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Count local library documents.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content},
            worker=worker,
        )
        self._finish_run(run_id)
        payload = outcome.payload
        answer = (
            f"当前本地论文库共有 {payload['total']} 篇论文，其中 {payload['ready']} 篇已可用、"
            f"{payload['processing']} 篇处理中、{payload['failed']} 篇处理失败。"
        )
        return KnowledgeAgentResult(content=answer, action_status="completed", agent_trace_id=run_id)

    def _answer_document_categories(
        self,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, selected_document_ids, allow_all=False)
            if not documents:
                return _TaskOutcome(
                    summary="No document matched the category query.",
                    payload={"documents": [], "candidates": self._candidate_titles()},
                )
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s) for category lookup.",
                payload={
                    "documents": [
                        {
                            "id": document.id,
                            "name": document.display_name or document.filename,
                            "title": document.title,
                            "categories": [category.name for category in document.categories],
                        }
                        for document in documents
                    ]
                },
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Read document categories.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content},
            worker=worker,
        )
        self._finish_run(run_id)
        documents = outcome.payload.get("documents") or []
        if not documents:
            return KnowledgeAgentResult(
                content="我没有在论文库里唯一定位到你说的那篇论文。可以把标题多给几个字，或先在下方选中论文后再问。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )
        lines = []
        for document in documents:
            categories = document["categories"] or ["暂无分类/标签"]
            lines.append(f"- {document['name']}：{'、'.join(categories)}")
        return KnowledgeAgentResult(
            content="这些论文当前的分类/标签是：\n" + "\n".join(lines),
            action_status="completed",
            used_document_ids=[item["id"] for item in documents],
            agent_trace_id=run_id,
        )

    def _create_category(self, session: ChatSession, content: str, category_name: str) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)

        def worker() -> _TaskOutcome:
            existing = next(
                (category for category in self.category_repository.list_categories() if category.name == category_name),
                None,
            )
            if existing is not None:
                return _TaskOutcome(
                    summary=f"Category already exists: {category_name}.",
                    payload={"category": existing.model_dump(mode="json"), "created": False},
                )
            palette = ["#0f5fb8", "#047c71", "#6957d8", "#b76a00", "#b42318"]
            color = palette[len(self.category_repository.list_categories()) % len(palette)]
            category = self.category_repository.create_category(category_name, color)
            return _TaskOutcome(
                summary=f"Created category: {category.name}.",
                payload={"category": category.model_dump(mode="json"), "created": True},
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Create a non-destructive library category.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name},
            worker=worker,
        )
        self._finish_run(run_id)
        category = outcome.payload["category"]
        verb = "已新建" if outcome.payload.get("created") else "已经存在"
        return KnowledgeAgentResult(
            content=f"{verb}分类「{category['name']}」。",
            action_status="completed",
            agent_trace_id=run_id,
            library_mutated=bool(outcome.payload.get("created")),
        )

    def _assign_category(
        self,
        session: ChatSession,
        content: str,
        assignment: dict[str, str],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)
        category_name = assignment["category_name"]

        def worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, [], allow_all=False)
            if not documents:
                return _TaskOutcome(
                    summary="No document matched category assignment.",
                    payload={"documents": [], "category_name": category_name},
                )
            category = next(
                (item for item in self.category_repository.list_categories() if item.name == category_name),
                None,
            )
            if category is None:
                category = self.category_repository.create_category(category_name, "#0f5fb8")
            updated = []
            for document in documents:
                existing_ids = [item.id for item in document.categories]
                next_ids = list(dict.fromkeys([*existing_ids, category.id]))
                categories = self.category_repository.replace_document_categories(document.id, next_ids) or []
                updated.append(
                    {
                        "id": document.id,
                        "name": document.display_name or document.filename,
                        "categories": [item.name for item in categories],
                    }
                )
            return _TaskOutcome(
                summary=f"Assigned category {category_name} to {len(updated)} document(s).",
                payload={"documents": updated, "category": category.model_dump(mode="json")},
            )

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Assign a category to matched library documents.",
            tool_policy=ToolPolicy(read_only=False, db_write=True),
            context_bundle={"category_name": category_name},
            worker=worker,
        )
        self._finish_run(run_id)
        documents = outcome.payload.get("documents") or []
        if not documents:
            return KnowledgeAgentResult(
                content="我没有唯一定位到要打标签的论文。请把论文标题多给几个字，或先在下方选中论文。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )
        names = "、".join(item["name"] for item in documents)
        return KnowledgeAgentResult(
            content=f"已把分类/标签「{category_name}」添加到：{names}。",
            action_status="completed",
            used_document_ids=[item["id"] for item in documents],
            agent_trace_id=run_id,
            library_mutated=True,
        )

    def _draft_report_like_answer(
        self,
        session: ChatSession,
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> KnowledgeAgentResult:
        run_id = self._begin_run(session, content)
        attachment_ids = [item.document_id for item in attachments if item.document_id]
        seed_ids = list(dict.fromkeys([*selected_document_ids, *[item for item in attachment_ids if item]]))

        def resolve_worker() -> _TaskOutcome:
            documents = self._resolve_documents(content, seed_ids, allow_all=self._mentions_all_library(content))
            return _TaskOutcome(
                summary=f"Resolved {len(documents)} document(s) for drafting.",
                payload={
                    "documents": [
                        {
                            "id": document.id,
                            "name": document.display_name or document.filename,
                            "title": document.title,
                            "status": document.status,
                        }
                        for document in documents
                    ]
                },
            )

        resolved = self._run_subagent(
            run_id=run_id,
            role="library-explorer",
            profile=SubagentProfile.EXPLORE,
            goal="Resolve documents for summary/review request.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"question": content, "selected_document_ids": seed_ids},
            worker=resolve_worker,
        )
        document_ids = [item["id"] for item in resolved.payload.get("documents", [])]
        documents = [document for document in self.document_library_service.list_documents() if document.id in document_ids]
        if not documents:
            self._finish_run(run_id)
            return KnowledgeAgentResult(
                content="我还没有定位到可用于总结的论文。你可以先在论文库选择论文，或告诉我更完整的标题。",
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )

        def retrieve_worker() -> _TaskOutcome:
            evidence = self.rag_service.retrieve_evidence(
                question=content,
                documents=documents,
                top_k=min(10, max(4, len(documents) * 3)),
            )
            return _TaskOutcome(
                summary=f"Retrieved {len(evidence)} evidence item(s).",
                payload={"evidence_items": [item.model_dump(mode="json") for item in evidence]},
            )

        retrieved = self._run_subagent(
            run_id=run_id,
            role="evidence-retriever",
            profile=SubagentProfile.EXPLORE,
            goal="Retrieve local evidence for drafting.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"document_ids": document_ids},
            worker=retrieve_worker,
        )
        evidence_items = [EvidenceItem(**item) for item in retrieved.payload.get("evidence_items", [])]
        if not evidence_items:
            self._finish_run(run_id)
            return KnowledgeAgentResult(
                content=(
                    "本轮没有检索到可引用的论文正文证据，因此我不能只根据文件名、标题或元数据生成综述报告。"
                    "请确认所选论文已完成入库、正文 chunk 与向量/关键词索引可用后，再重新发送总结请求。"
                ),
                retrieval_status="skipped",
                warning="没有检索到足够正文片段，本轮未生成基于论文内容的总结。",
                citations=[],
                used_document_ids=document_ids,
                evidence_items=[],
                action_status="needs_clarification",
                agent_trace_id=run_id,
            )

        def draft_worker() -> _TaskOutcome:
            draft = self._draft_with_llm(content, documents, evidence_items)
            return _TaskOutcome(
                summary="Drafted chat-side Markdown answer.",
                payload={
                    "answer": draft.answer,
                    "llm_draft_success": draft.llm_draft_success,
                    "fallback_used": draft.fallback_used,
                    "drafting_error": draft.drafting_error,
                },
            )

        drafted = self._run_subagent(
            run_id=run_id,
            role="report-drafter",
            profile=SubagentProfile.VERIFY,
            goal="Draft the final chat-visible Markdown answer.",
            tool_policy=ToolPolicy(read_only=True),
            context_bundle={"document_ids": document_ids, "evidence_count": len(evidence_items)},
            worker=draft_worker,
        )
        self._finish_run(run_id)
        citations = self._collect_citations(evidence_items)
        return KnowledgeAgentResult(
            content=drafted.payload["answer"],
            retrieval_status="ready" if drafted.payload.get("llm_draft_success") else "degraded",
            warning=None if drafted.payload.get("llm_draft_success") else drafted.payload.get("drafting_error") or "LLM 综述撰写失败，已返回降级草稿。",
            citations=citations,
            used_document_ids=document_ids,
            evidence_items=evidence_items,
            action_status="completed" if drafted.payload.get("llm_draft_success") else "degraded",
            agent_trace_id=run_id,
        )

    def _request_confirmation(
        self,
        session: ChatSession,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult:
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        pending = self._build_destructive_action(content)
        if pending is not None:
            self._write_pending_action(session.id, pending)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="confirmation_required",
            message="Destructive action requires confirmation.",
            payload={"pending_action": pending or {"raw_request": content}},
        )
        if owns_run:
            self._finish_run(run_id)
        target = pending.get("label") if pending else content
        if pending is not None:
            affected_count = pending.get("expected_affected_count")
            expected_scope = pending.get("expected_scope")
            phrase = pending.get("confirmation_phrase") or "确认删除"
            return KnowledgeAgentResult(
                content=(
                    f"需要确认：该操作将作用于「{target}」，"
                    f"预计影响 {affected_count if affected_count is not None else '未知'} 个对象，"
                    f"范围为 {expected_scope or '未解析'}。请回复「{phrase}」后执行。"
                ),
                action_status="confirmation_required",
                agent_trace_id=run_id,
            )
        return KnowledgeAgentResult(
            content=f"这个操作会删除「{target}」，我需要你明确回复“确认删除”后才会执行。",
            action_status="confirmation_required",
            agent_trace_id=run_id,
        )

    def _defer_destructive_part_after_safe_plan(
        self,
        *,
        session: ChatSession,
        content: str,
        observations: list[_ReactObservation],
        current_answer: str,
        run_id: str,
    ) -> str | None:
        if not self._is_destructive_intent(content):
            return None
        if not self._should_plan_before_destructive_confirmation(content, [], []):
            return None
        if self._is_clear_categories_intent(content):
            return None
        if self._is_delete_unused_categories_intent(content) and any(
            observation.tool == "library.operator.delete_unused_categories"
            and observation.status in {"completed", "confirmation_required"}
            for observation in observations
        ):
            return None
        if self._is_ambiguous_category_delete_request(content):
            return (
                f"{current_answer}\n\n"
                "删除部分还不够明确：你是要删除论文实体、删除空标签/分类实体，"
                "还是只移除论文和标签/分类之间的关系？请明确目标后我再继续。"
            ).strip()
        pending = self._build_destructive_action_from_observations(content, observations)
        if pending is None:
            return (
                f"{current_answer}\n\n"
                "删除部分还没有可靠定位到具体论文、标签或分类；请明确要删除的对象后我再继续。"
            ).strip()
        literal = self._final_literal_output(content)
        if literal:
            pending["post_confirm_literal"] = literal
        pending["source_goal"] = content[:500]
        self._write_pending_action(session.id, pending)
        self.message_bus.append_trace(
            run_id=run_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status="confirmation_required",
            message="Destructive subtask requires confirmation after safe subtasks.",
            payload={"pending_action": pending},
        )
        target = pending.get("label") or "该对象"
        suffix = f"确认后我会继续输出「{literal}」。" if literal else "确认后我会执行删除。"
        return (
            f"{current_answer}\n\n"
            f"删除部分会删除「{target}」，需要你明确回复“确认删除”后才会执行。{suffix}"
        ).strip()

    def _build_destructive_action_from_observations(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> dict[str, Any] | None:
        explicit = self._build_destructive_action(content)
        if explicit is not None:
            return explicit
        category_name = self._destructive_category_referent(content, observations)
        if category_name:
            category = next(
                (item for item in self.category_repository.list_categories() if item.name == category_name),
                None,
            )
            if category is not None:
                return {"type": "delete_category", "category_id": category.id, "label": category.name}
        return None

    def _destructive_category_referent(
        self,
        content: str,
        observations: list[_ReactObservation],
    ) -> str | None:
        if not any(marker in content for marker in ("这个标签", "该标签", "此标签", "这个分类", "该分类", "此分类")):
            return None
        category_stats = self._latest_payload(observations, "library.explorer.category_stats")
        if category_stats and self._asks_category_extreme(content):
            _answer, category_name = self._category_extreme_answer(content, category_stats)
            if category_name:
                return category_name
        state_category = self._category_name_from_observations(observations)
        if state_category:
            return state_category
        return None

    def _maybe_execute_pending_action(
        self,
        session: ChatSession,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> KnowledgeAgentResult | None:
        normalized = content.casefold()
        if not (
            any(marker in normalized for marker in self._CONFIRM_MARKERS)
            or ("确认" in normalized and any(marker in normalized for marker in self._DELETE_MARKERS))
        ):
            return None
        pending = self._read_pending_action(session.id)
        if pending is None:
            return None
        run_id = trace_id or self._begin_run(session, content)
        owns_run = trace_id is None
        phrase = str(pending.get("confirmation_phrase") or "").strip()
        if pending.get("risk_level") == "critical" and phrase and phrase not in content:
            if owns_run:
                self._finish_run(run_id)
            return KnowledgeAgentResult(
                content=f"这是 critical 级写操作，需要完整回复「{phrase}」才能执行；本次没有改动论文库。",
                action_status="confirmation_required",
                agent_trace_id=run_id,
            )

        def worker() -> _TaskOutcome:
            if pending.get("type") == "tool_action":
                return self._execute_pending_tool_action(run_id, session, content, pending)
            if pending["type"] == "delete_category":
                category_id = str(pending["category_id"])
                label = str(pending["label"])
                before_snapshot = pending.get("before_snapshot") if isinstance(pending.get("before_snapshot"), dict) else {}
                deleted = self.category_repository.delete_category(pending["category_id"])
                category_still_exists = self.category_repository.get_category(category_id) is not None
                verified_state = self._category_stats_payload()
                linked_document_ids = [
                    document["id"]
                    for document in verified_state.get("tagged_documents", [])
                    if label in (document.get("categories") or [])
                ]
                range_error = self._verify_delete_category_effect(label, before_snapshot)
                if range_error:
                    self._rollback_category_snapshot(before_snapshot)
                    verified_state = self._category_stats_payload()
                verified_deleted = bool(deleted) and not category_still_exists and not linked_document_ids and not range_error
                return _TaskOutcome(
                    summary=(
                        f"Deleted category {label}."
                        if verified_deleted
                        else f"Category deletion was not verified: {label}. {range_error or ''}".strip()
                    ),
                    payload={
                        "deleted": verified_deleted,
                        "delete_attempted": bool(deleted),
                        "label": label,
                        "category_id": category_id,
                        "category_still_exists": category_still_exists,
                        "linked_document_ids": linked_document_ids,
                        "verification_error": range_error,
                        "rollback_attempted": bool(range_error),
                        "verified_state": verified_state,
                        "library_mutated": verified_deleted,
                    },
                )
            if pending["type"] == "delete_document":
                deleted = self.document_library_service.delete_document(pending["document_id"])
                return _TaskOutcome(
                    summary=f"Deleted document {pending['label']}.",
                    payload={"deleted": bool(deleted), "label": pending["label"], "library_mutated": bool(deleted)},
                )
            return _TaskOutcome(summary="No supported pending action.", payload={"deleted": False})

        outcome = self._run_subagent(
            run_id=run_id,
            role="library-operator",
            profile=SubagentProfile.IMPLEMENT,
            goal="Execute confirmed destructive library action.",
            tool_policy=ToolPolicy(read_only=False, db_write=True, workspace_write=True),
            context_bundle={"pending_action": pending},
            worker=worker,
        )
        self._clear_pending_action(session.id)
        if owns_run:
            self._finish_run(run_id)
        if outcome.payload.get("completed"):
            return KnowledgeAgentResult(
                content=str(outcome.summary),
                action_status="completed",
                agent_trace_id=run_id,
                library_mutated=bool(outcome.payload.get("library_mutated")),
            )
        if pending.get("type") == "tool_action" and pending.get("tool_name") == "library.operator.assign_category":
            return KnowledgeAgentResult(
                content=str(outcome.summary),
                action_status="failed",
                agent_trace_id=run_id,
                library_mutated=False,
            )
        if outcome.payload.get("deleted"):
            verified_state = outcome.payload.get("verified_state")
            state_text = ""
            if isinstance(verified_state, dict):
                category_names = [
                    f"{item['name']}（{item['document_count']} 篇）"
                    for item in verified_state.get("categories", [])
                    if isinstance(item, dict)
                ]
                if category_names:
                    state_text = "当前分类：" + "、".join(category_names) + "。"
            post_confirm_literal = str(pending.get("post_confirm_literal") or "").strip()
            content_parts = [f"已删除「{outcome.payload['label']}」。"]
            if state_text:
                content_parts.append(state_text)
            if post_confirm_literal:
                content_parts.append(post_confirm_literal)
            return KnowledgeAgentResult(
                content="\n".join(content_parts),
                action_status="completed",
                agent_trace_id=run_id,
                library_mutated=bool(outcome.payload.get("library_mutated")),
            )
        return KnowledgeAgentResult(
            content=(
                f"我没有能验证「{outcome.payload.get('label', '该对象')}」已从论文库中删除，"
                "因此没有把这次操作标记为成功。"
            ),
            action_status="failed",
            agent_trace_id=run_id,
        )

    def _execute_pending_tool_action(
        self,
        run_id: str,
        session: ChatSession,
        content: str,
        pending: dict[str, Any],
    ) -> _TaskOutcome:
        tool_name = str(pending.get("tool_name") or "")
        tool_args = pending.get("tool_args") if isinstance(pending.get("tool_args"), dict) else {}
        expected_count = int(pending.get("expected_affected_count") or 0)
        before_snapshot = pending.get("before_snapshot") if isinstance(pending.get("before_snapshot"), dict) else {}
        if tool_name == "library.operator.delete_unused_categories":
            preview = self._preview_delete_unused_categories(dict(tool_args))
            if isinstance(preview, str):
                return _TaskOutcome(
                    summary=f"确认前复核失败，未执行：{preview}",
                    payload={"completed": False, "library_mutated": False, "verification_error": preview},
                )
            expected_ids = {str(item) for item in (tool_args.get("category_ids") or []) if item}
            actual_ids = {
                str(item.get("id"))
                for item in preview.affected_entities
                if isinstance(item, dict) and item.get("id")
            }
            if preview.affected_count != expected_count or actual_ids != expected_ids:
                return _TaskOutcome(
                    summary=(
                        "确认前复核发现空标签/分类实体范围已经变化，未执行。"
                        f"预期 {expected_count} 个对象，当前 {preview.affected_count} 个对象。"
                    ),
                    payload={
                        "completed": False,
                        "library_mutated": False,
                        "expected_affected_count": expected_count,
                        "actual_affected_count": preview.affected_count,
                        "expected_category_ids": sorted(expected_ids),
                        "actual_category_ids": sorted(actual_ids),
                    },
                )
            confirmed_args = {
                **tool_args,
                "__confirmed_pending": True,
                "__before_snapshot": before_snapshot,
                "__expected_affected_count": expected_count,
            }
            observation = self._tool_delete_unused_categories(run_id, session, content, confirmed_args, [])
            if observation.status != "completed":
                return _TaskOutcome(
                    summary=observation.summary,
                    payload={**observation.payload, "completed": False, "library_mutated": False},
                )
            self._append_react_trace(
                run_id=run_id,
                status="pending_write_executed",
                payload={
                    "tool": tool_name,
                    "operation": pending.get("operation"),
                    "operation_level": "entity-level",
                    "write_type": "delete",
                    "target_type": "category",
                    "expected_affected_count": expected_count,
                    "library_mutated": bool(observation.payload.get("library_mutated")),
                },
            )
            if observation.payload.get("library_mutated"):
                self._append_react_trace(
                    run_id=run_id,
                    status="library_write_verified",
                    payload={
                        "tool": tool_name,
                        "risk_level": pending.get("risk_level"),
                        "operation_level": "entity-level",
                        "write_type": "delete",
                        "target_type": "category",
                        "affected_category_count": observation.payload.get("deleted_count", 0),
                        "deleted_category_names": observation.payload.get("deleted_category_names") or [],
                        "verified_state": self._safe_trace_payload(observation.payload.get("verified_state") or {}),
                    },
            )
            return _TaskOutcome(summary=observation.summary, payload={**observation.payload, "completed": True})

        if tool_name == "library.operator.assign_category":
            document_ids = [str(item) for item in tool_args.get("document_ids") or [] if item]
            existing_ids = {
                document.id
                for document in self.document_library_service.list_documents()
                if document.id in set(document_ids)
            }
            if len(existing_ids) != expected_count or existing_ids != set(document_ids):
                return _TaskOutcome(
                    summary=(
                        "确认前复核发现目标论文范围已经变化，未执行。"
                        f"预期 {expected_count} 篇论文，当前可确认 {len(existing_ids)} 篇论文。"
                    ),
                    payload={
                        "completed": False,
                        "library_mutated": False,
                        "expected_affected_count": expected_count,
                        "actual_affected_count": len(existing_ids),
                        "expected_document_ids": document_ids,
                        "actual_document_ids": sorted(existing_ids),
                    },
                )
            confirmed_args = {
                **tool_args,
                "__confirmed_pending": True,
                "__before_snapshot": before_snapshot,
                "__expected_affected_count": expected_count,
            }
            observation = self._tool_assign_category(run_id, session, content, confirmed_args, [])
            if observation.status != "completed":
                return _TaskOutcome(
                    summary=observation.summary,
                    payload={**observation.payload, "completed": False, "library_mutated": False},
                )
            actual_count = len(observation.payload.get("document_ids") or [])
            if actual_count != expected_count:
                self._rollback_category_snapshot(before_snapshot, set(document_ids))
                return _TaskOutcome(
                    summary=(
                        "写入后复核发现影响范围与预览不一致，已尝试回滚。"
                        f"预期 {expected_count} 篇论文，实际 {actual_count} 篇论文。"
                    ),
                    payload={
                        **observation.payload,
                        "completed": False,
                        "library_mutated": False,
                        "rollback_attempted": True,
                    },
                )
            self._append_react_trace(
                run_id=run_id,
                status="pending_write_executed",
                payload={
                    "tool": tool_name,
                    "operation": pending.get("operation"),
                    "operation_level": "relation-level",
                    "write_type": "append",
                    "target_type": "paper-category relation",
                    "expected_affected_count": expected_count,
                    "library_mutated": bool(observation.payload.get("library_mutated")),
                },
            )
            if observation.payload.get("library_mutated"):
                self._append_react_trace(
                    run_id=run_id,
                    status="library_write_verified",
                    payload={
                        "tool": tool_name,
                        "risk_level": pending.get("risk_level"),
                        "operation_level": "relation-level",
                        "write_type": "append",
                        "target_type": "paper-category relation",
                        "affected_document_count": actual_count,
                        "category_names": observation.payload.get("category_names") or [],
                        "verified_state": self._safe_trace_payload(observation.payload.get("verified_state") or {}),
                    },
                )
            return _TaskOutcome(summary=observation.summary, payload={**observation.payload, "completed": True})

        if tool_name != "library.operator.clear_categories":
            return _TaskOutcome(
                summary="Unsupported pending write action; no database changes were made.",
                payload={"completed": False, "library_mutated": False, "tool_name": tool_name},
            )
        preview = self._preview_clear_categories(str(pending.get("source_goal") or ""), dict(tool_args), [])
        if isinstance(preview, str):
            return _TaskOutcome(
                summary=f"确认前复核失败，未执行：{preview}",
                payload={"completed": False, "library_mutated": False, "verification_error": preview},
            )
        if preview.affected_count != expected_count:
            return _TaskOutcome(
                summary=(
                    "确认前复核发现影响范围已变化，未执行。"
                    f"预期 {expected_count} 个对象，当前 {preview.affected_count} 个对象。"
                ),
                payload={
                    "completed": False,
                    "library_mutated": False,
                    "expected_affected_count": expected_count,
                    "actual_affected_count": preview.affected_count,
                },
            )
        confirmed_args = {
            **tool_args,
            "__confirmed_pending": True,
            "__before_snapshot": before_snapshot,
            "__expected_affected_count": expected_count,
        }
        observation = self._tool_clear_categories(run_id, session, content, confirmed_args, [])
        if observation.status != "completed":
            return _TaskOutcome(
                summary=observation.summary,
                payload={
                    **observation.payload,
                    "completed": False,
                    "library_mutated": False,
                },
            )
        self._append_react_trace(
            run_id=run_id,
            status="pending_write_executed",
            payload={
                "tool": tool_name,
                "operation": pending.get("operation"),
                "expected_affected_count": expected_count,
                "library_mutated": bool(observation.payload.get("library_mutated")),
            },
        )
        if observation.payload.get("library_mutated"):
            self._append_react_trace(
                run_id=run_id,
                status="library_write_verified",
                payload={
                    "tool": tool_name,
                    "risk_level": pending.get("risk_level"),
                    "affected_document_count": len(observation.payload.get("affected_document_ids") or []),
                    "verified_state": self._safe_trace_payload(observation.payload.get("verified_state") or {}),
                },
            )
        return _TaskOutcome(
            summary=observation.summary,
            payload={
                **observation.payload,
                "completed": True,
            },
        )

    def _build_destructive_action(self, content: str) -> dict[str, Any] | None:
        if self._is_delete_unused_categories_intent(content) or self._is_ambiguous_category_delete_request(content):
            return None
        categories = self.category_repository.list_categories()
        for category in categories:
            if category.name and category.name in content:
                snapshot = self._category_snapshot()
                affected = [
                    {
                        "document_id": document.id,
                        "name": document.display_name or document.filename,
                        "before_categories": [item.name for item in document.categories],
                        "after_categories": [item.name for item in document.categories if item.id != category.id],
                    }
                    for document in self.document_library_service.list_documents()
                    if any(item.id == category.id for item in document.categories)
                ]
                return {
                    "type": "delete_category",
                    "operation": "delete_category_entity",
                    "risk_level": "destructive",
                    "category_id": category.id,
                    "label": category.name,
                    "resolved_target_entity": {"type": "category", "id": category.id, "name": category.name},
                    "expected_affected_count": len(affected),
                    "expected_scope": "single_category_entity",
                    "affected_entities": affected,
                    "before_snapshot": snapshot,
                    "confirmation_phrase": f"确认删除{category.name}标签",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
        documents = self._resolve_documents(content, [], allow_all=False)
        if len(documents) == 1:
            document = documents[0]
            return {
                "type": "delete_document",
                "document_id": document.id,
                "label": document.display_name or document.filename,
            }
        return None

    def _draft_with_llm(
        self,
        question: str,
        documents: list[LibraryDocument],
        evidence_items: list[EvidenceItem],
    ) -> _DraftResult:
        if self._is_field_only_metadata_request(question):
            requested_fields = self._requested_metadata_fields(question)
            payload = {
                "documents": [
                    self._document_metadata_payload(document, requested_fields)
                    for document in documents
                ],
                "requested_fields": requested_fields,
            }
            return _DraftResult(
                answer=self._compose_metadata_observation_answer(question, payload),
                llm_draft_success=False,
                fallback_used=False,
            )
        if self._requests_per_paper_abstract_format(question):
            return _DraftResult(
                answer=self._draft_per_paper_abstract_template(question, documents),
                llm_draft_success=False,
                fallback_used=False,
            )
        document_block = "\n".join(
            (
                f"- {document.display_name or document.filename} | 标题：{document.title or '未知'} "
                f"| 页数：{document.page_count} | 摘要：{self._extract_document_abstract(document)}"
            )
            for document in documents
        )
        evidence_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"来源：{item.citation_label}",
                        f"标题：{item.title}",
                        f"页码：{item.page_number if item.page_number is not None else '未知'}",
                        f"证据：{item.quote or item.snippet}",
                    ]
                )
                for item in evidence_items[:10]
            ]
        )
        category_names = self._category_entity_names_for_request(question)
        fallback = self._draft_template(question, documents, evidence_items, category_names=category_names)
        if not evidence_items:
            return _DraftResult(
                answer=fallback,
                llm_draft_success=False,
                fallback_used=True,
                drafting_error="insufficient_evidence",
            )
        if not self.api_key:
            return _DraftResult(
                answer=fallback,
                llm_draft_success=False,
                fallback_used=True,
                drafting_error="llm_not_configured",
            )
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 report-drafter 子代理。请用中文输出 Markdown，"
                            "只基于给定论文元数据和证据撰写，不要编造引用。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"用户请求：{question}",
                                "论文列表：\n" + document_block,
                                "证据列表：\n" + (evidence_block or "暂无可引用证据。"),
                                "请严格保留用户指定的结构；如果用户要求论文一/论文名称/论文完整摘要/论文页数，"
                                "每篇论文都必须包含这些字段，摘要只能使用给定摘要文本，缺失时写“未提取到真实摘要”。"
                                "如果证据不足，要明确说明边界。",
                            ]
                        ),
                    },
                ],
            )
        except Exception as exc:
            return _DraftResult(
                answer=fallback,
                llm_draft_success=False,
                fallback_used=True,
                drafting_error=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
        answer = self._extract_message_text(response)
        if not answer:
            return _DraftResult(
                answer=fallback,
                llm_draft_success=False,
                fallback_used=True,
                drafting_error="empty_llm_response",
            )
        return _DraftResult(answer=answer, llm_draft_success=True)

    @classmethod
    def _is_field_only_metadata_request(cls, content: str) -> bool:
        fields = cls._requested_metadata_fields(content)
        if not fields:
            return False
        non_field_intent_markers = (
            "\u6458\u8981",
            "\u603b\u7ed3",
            "\u7efc\u8ff0",
            "\u521b\u65b0",
            "\u8d21\u732e",
            "\u65b9\u6cd5",
            "\u5b9e\u9a8c",
            "\u7ed3\u8bba",
            "\u5c40\u9650",
            "\u5bf9\u6bd4",
            "\u6bd4\u8f83",
            "\u5206\u6790",
            "\u62a5\u544a",
            "\u5199\u4e00\u4efd",
            "\u5199\u4e00\u7bc7",
            "\u751f\u6210",
            "abstract",
            "summary",
            "summarize",
            "analysis",
            "analyze",
            "compare",
            "comparison",
            "contribution",
            "novelty",
            "report",
            "write",
        )
        return set(fields).issubset({"title", "authors", "venue", "published", "year", "tags"}) and not any(
            marker in content.casefold() for marker in non_field_intent_markers
        )

    def _draft_category_summaries_with_llm(
        self,
        *,
        question: str,
        category_groups: list[dict[str, Any]],
        target_chars: int,
    ) -> str:
        fallback = self._draft_category_summaries_template(
            question=question,
            category_groups=category_groups,
            target_chars=target_chars,
        )
        if not self.api_key:
            return fallback

        group_blocks = []
        for group in category_groups:
            documents = group.get("documents") or []
            evidence_items = group.get("evidence_items") or []
            document_lines = [
                f"- {item.get('name') or item.get('filename')} | 标题：{item.get('title') or '未知'}"
                for item in documents
                if isinstance(item, dict)
            ]
            evidence_lines = []
            for item in evidence_items[:8]:
                if not isinstance(item, dict):
                    continue
                evidence_lines.append(
                    "\n".join(
                        [
                            f"来源：{item.get('citation_label')}",
                            f"标题：{item.get('title')}",
                            f"页码：{item.get('page_number')}",
                            f"证据：{item.get('quote') or item.get('snippet')}",
                        ]
                    )
                )
            group_blocks.append(
                "\n".join(
                    [
                        f"## 标签：{group.get('category_name')}",
                        "论文：",
                        *document_lines,
                        "证据：",
                        "\n\n".join(evidence_lines) or "暂无可引用证据。",
                    ]
                )
            )

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 PaperDesk 的 report-drafter 子代理。请用中文输出 Markdown。"
                            "任务是按标签分别写总结，每个标签一节，尽量接近用户要求的字数。"
                            "只能基于给定论文元数据和证据，不要编造引用。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"用户目标：{question}",
                                f"每个标签目标长度：约 {target_chars} 字。",
                                "按标签分组的论文与证据：",
                                "\n\n".join(group_blocks),
                            ]
                        ),
                    },
                ],
            )
        except Exception:
            return fallback
        return self._extract_message_text(response) or fallback

    @staticmethod
    def _draft_category_summaries_template(
        *,
        question: str,
        category_groups: list[dict[str, Any]],
        target_chars: int,
    ) -> str:
        lines = [
            f"# {question}",
            "",
            f"以下按标签分别整理，每部分目标约 {target_chars} 字；如果证据不足，会明确说明边界。",
            "",
        ]
        if not category_groups:
            lines.append("当前没有可按标签分组的 ready 论文。")
            return "\n".join(lines)
        for group in category_groups:
            category_name = group.get("category_name") or "未命名标签"
            documents = [item for item in group.get("documents", []) if isinstance(item, dict)]
            evidence_items = [item for item in group.get("evidence_items", []) if isinstance(item, dict)]
            lines.append(f"已识别到「{category_name}」是标签/分类，下面共有 {len(documents)} 篇 ready 论文。")
            lines.append("")
            lines.extend(
                [
                    f"## {category_name}",
                    "",
                    "涉及论文："
                    + ("、".join(item.get("name") or item.get("filename") or "未命名论文" for item in documents) or "暂无论文"),
                    "",
                ]
            )
            if evidence_items:
                snippets = [
                    str(item.get("quote") or item.get("snippet") or "").strip()
                    for item in evidence_items[:4]
                    if str(item.get("quote") or item.get("snippet") or "").strip()
                ]
                lines.append(
                    "该标签下的论文目前检索到的证据主要集中在："
                    + "；".join(snippets)
                    + "。基于这些片段，可以先形成一份初步总结，但仍建议回到原 PDF 核对关键结论和页码。"
                )
            else:
                boundary = str(group.get("evidence_boundary") or "").strip()
                lines.append(boundary or "该标签下论文存在，但正文证据不足。")
                lines.append("因此这里先基于题名和元数据占位总结，强结论需要后续补充 RAG 证据。")
            quality = group.get("evidence_quality")
            if isinstance(quality, dict) and quality.get("warnings"):
                lines.append("证据覆盖边界：" + "、".join(str(item) for item in quality.get("warnings", [])))
            lines.append("")
        return "\n".join(lines).strip()

    @classmethod
    def _requests_per_paper_abstract_format(cls, question: str) -> bool:
        return any(marker in question for marker in ("论文一", "论文名称", "论文完整摘要", "完整摘要")) and any(
            marker in question for marker in ("页数", "多少页", "page")
        )

    def _draft_per_paper_abstract_template(
        self,
        question: str,
        documents: list[LibraryDocument],
    ) -> str:
        lines: list[str] = []
        chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        for index, document in enumerate(documents):
            ordinal = chinese_numbers[index] if index < len(chinese_numbers) else str(index + 1)
            lines.extend(
                [
                    f"论文{ordinal}：",
                    f"论文名称：{document.display_name or document.filename}",
                    f"论文完整摘要：{self._extract_document_abstract(document)}",
                    f"论文页数：{document.page_count or 0}页",
                    "",
                ]
            )
        if documents:
            names = "、".join(document.display_name or document.filename for document in documents)
            lines.append(
                f"这{len(documents)}篇论文都来自用户指定标签下的真实论文集合，分别是{names}；"
                "相同点和不同点需要以已解析正文和摘要为边界进行判断，当前模板优先保证论文名称、真实摘要提取结果和页数不被编造。"
            )
        else:
            lines.append("没有找到符合该标签范围的论文，因此无法生成该格式报告。")
        return "\n".join(lines).strip()

    def _extract_document_abstract(self, document: LibraryDocument) -> str:
        try:
            chunks = self.rag_service.chunk_repository.list_chunks(document_ids=[document.id])
        except Exception:
            chunks = []
        if not chunks:
            return "未提取到真实摘要"
        for chunk in chunks:
            section = f"{chunk.section or ''} {chunk.title or ''}".casefold()
            if "abstract" in section or "摘要" in section:
                candidate = self._clean_abstract_text(chunk.content or chunk.text or "")
                if candidate:
                    return candidate
        first_pages = "\n".join(
            chunk.content or chunk.text or ""
            for chunk in chunks
            if (chunk.page_number or 0) <= 2
        )
        patterns = [
            r"(?is)\babstract\b\s*[:：]?\s*(.{80,2200}?)(?:\b(?:keywords|index terms|introduction)\b|1\s+introduction|\n\s*1[.\s])",
            r"(?s)摘要\s*[:：]?\s*(.{40,1600}?)(?:关键词|关键字|引言|一、|1\s*[\.、]\s*)",
        ]
        for pattern in patterns:
            try:
                match = re.search(pattern, first_pages)
            except re.error:
                continue
            if match:
                candidate = self._clean_abstract_text(match.group(1))
                if candidate:
                    return candidate
        return "未提取到真实摘要"

    @staticmethod
    def _clean_abstract_text(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" ：:;；")
        if not cleaned:
            return ""
        if len(cleaned) < 30:
            return ""
        return cleaned[:2200]

    @staticmethod
    def _draft_template(
        question: str,
        documents: list[LibraryDocument],
        evidence_items: list[EvidenceItem],
        category_names: list[str] | None = None,
    ) -> str:
        names = "、".join(document.display_name or document.filename for document in documents)
        lines = [
            f"# {question}",
            "",
            "## 涉及论文",
            "",
            *[
                f"- {document.display_name or document.filename}：{document.title or '未提取标题'}"
                for document in documents
            ],
            "",
            "## 初步总结",
            "",
        ]
        if category_names:
            lines.extend(
                [
                    f"已识别到「{'、'.join(category_names)}」是标签/分类，下面共有 {len(documents)} 篇 ready 论文。",
                    "",
                ]
            )
            if len(category_names) > 1:
                grouped: dict[str, list[str]] = {name: [] for name in category_names}
                for document in documents:
                    for category in document.categories:
                        if category.name in grouped:
                            grouped[category.name].append(document.display_name or document.filename)
                for category_name, document_names in grouped.items():
                    lines.append(f"- {category_name}：{'、'.join(document_names) if document_names else '暂无 ready 论文'}")
                lines.append("")
        lines.append(f"本次围绕 {names} 进行整理。")
        if evidence_items:
            lines.extend(
                [
                    "当前检索到的证据主要覆盖以下片段：",
                    "",
                    *[
                        f"- {item.quote or item.snippet}（{item.citation_label}）"
                        for item in evidence_items[:5]
                    ],
                ]
            )
        else:
            lines.append("当前没有检索到足够正文片段，因此无法生成基于论文正文的分析结论。")
        lines.extend(
            [
                "",
                "## 结论边界",
                "",
                "强结论应回到原 PDF 与引用页码继续核验；未被证据直接支持的内容不应作为最终学术结论。",
            ]
        )
        return "\n".join(lines)

    def _resolve_documents(
        self,
        content: str,
        selected_document_ids: list[str],
        *,
        allow_all: bool,
    ) -> list[LibraryDocument]:
        documents = self.document_library_service.list_documents()
        ready_documents = [document for document in documents if document.status == "ready"]
        if selected_document_ids:
            selected = [document for document in ready_documents if document.id in set(selected_document_ids)]
            if selected:
                return selected
        if allow_all:
            return ready_documents

        tokens = self._extract_document_tokens(content)
        matches: list[LibraryDocument] = []
        for token in tokens:
            token_matches = self._match_documents(token, ready_documents)
            for document in token_matches:
                if document.id not in {item.id for item in matches}:
                    matches.append(document)
        return matches

    @staticmethod
    def _match_documents(token: str, documents: list[LibraryDocument]) -> list[LibraryDocument]:
        normalized_token = KnowledgeAgentRuntime._normalize_match_text(token)
        if not normalized_token:
            return []
        exact: list[LibraryDocument] = []
        scored: list[tuple[float, LibraryDocument]] = []
        for document in documents:
            haystacks = [
                document.display_name or "",
                document.filename or "",
                document.title or "",
            ]
            normalized_haystacks = [KnowledgeAgentRuntime._normalize_match_text(item) for item in haystacks]
            if any(normalized_token in item for item in normalized_haystacks):
                exact.append(document)
                continue
            score = max(
                (difflib.SequenceMatcher(None, normalized_token, item).ratio() for item in normalized_haystacks),
                default=0.0,
            )
            if score >= 0.55:
                scored.append((score, document))
        if exact:
            return exact
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _score, document in scored[:2]]

    @staticmethod
    def _extract_document_tokens(content: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"《([^》]{2,120})》", content))
        tokens.extend(re.findall(r"[A-Za-z]+\d+[A-Za-z0-9_-]*|\d+[A-Za-z]+[A-Za-z0-9_-]*", content))
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", content):
            if raw.casefold() not in {"pdf", "paper", "review", "summary"}:
                tokens.append(raw)
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            cleaned = token.strip("：:，,。？?！!、 ")
            key = cleaned.casefold()
            if len(cleaned) < 2 or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _candidate_titles(self) -> list[str]:
        return [
            document.display_name or document.filename
            for document in self.document_library_service.list_documents()[:8]
        ]

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).casefold()

    @staticmethod
    def _extract_create_category_name(content: str) -> str | None:
        if not any(marker in content for marker in ("新建", "创建", "新增", "建立")):
            return None
        if not any(marker in content for marker in ("分类", "标签")):
            return None
        patterns = [
            r"(?:分类|标签)[：:\s]+([^，。！？\n]+)",
            r"(?:叫|名为|命名为)[「“\"]?([^」”\"，。！？\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                name = match.group(1).strip(" 「」“”\"'")
                return name[:40] if name else None
        tail = re.split(r"分类|标签", content, maxsplit=1)[-1].strip(" ：:，,。 ")
        return tail[:40] if tail else None

    @staticmethod
    def _extract_category_assignment(content: str) -> dict[str, str] | None:
        if not any(marker in content for marker in ("打标签", "添加标签", "加标签", "归类", "设置分类", "添加分类")):
            return None
        match = re.search(r"(?:标签|分类)[：:\s为到成]*[「“\"]?([^」”\"，。！？\n]+)", content)
        if not match:
            return None
        category_name = match.group(1).strip(" 「」“”\"'")
        if not category_name:
            return None
        return {"category_name": category_name[:40]}

    @staticmethod
    def _is_library_count_question(content: str) -> bool:
        return "论文" in content and any(marker in content for marker in ("几篇", "多少篇", "数量", "count"))

    @staticmethod
    def _is_category_question(content: str) -> bool:
        return any(marker in content for marker in ("标签", "分类")) and any(
            marker in content for marker in ("什么", "哪些", "多少", "列出", "查看")
        )

    @classmethod
    def _is_destructive_intent(cls, content: str) -> bool:
        normalized = cls._strip_quoted_text(cls._strip_guardrail_negations(content)).casefold()
        if any(cls._is_clear_categories_clause(clause) for clause in cls._goal_clauses(normalized)):
            return True
        if cls._is_non_destructive_category_write(normalized):
            return False
        if any(marker in normalized for marker in cls._rename_category_markers()) and any(
            marker in normalized for marker in ("标签", "分类", "tag", "category")
        ):
            return False
        if not any(marker in normalized for marker in cls._DELETE_MARKERS):
            return False
        if any(marker in normalized for marker in ("论文", "文档", "paper", "document")):
            return True
        if any(marker in normalized for marker in ("标签", "分类", "tag", "category")):
            return True
        return bool(re.search(r"(删除|移除|清空|删掉|delete|remove|clear)\s*(?:这个|该|此)?\s*(?:分类|标签)", normalized))

    @staticmethod
    def _strip_quoted_text(content: str) -> str:
        stripped = content
        for pattern in (
            r"“[^”]*”",
            r"‘[^’]*’",
            r"「[^」]*」",
            r"『[^』]*』",
            r"'[^']*'",
            r'"[^"]*"',
        ):
            stripped = re.sub(pattern, "", stripped)
        return stripped

    @staticmethod
    def _is_non_destructive_category_write(content: str) -> bool:
        has_category_target = any(marker in content for marker in ("标签", "分类", "tag", "category"))
        if not has_category_target:
            return False
        return any(
            marker in content
            for marker in (
                "新建",
                "新增",
                "创建",
                "建立",
                "加个",
                "加一个",
                "添加",
                "打上",
                "打个",
                "打一个",
                "补上",
                "归类",
                "换成",
                "换为",
                "替换为",
                "替换成",
                "重命名",
                "重命名为",
                "改成",
                "改为",
                "改名为",
                "assign",
                "create",
                "add",
                "rename",
                "replace",
            )
        )

    @classmethod
    def _is_summary_request(cls, content: str, selected_document_ids: list[str]) -> bool:
        if any(marker in content.casefold() for marker in cls._SUMMARY_MARKERS):
            return True
        return bool(selected_document_ids) and any(marker in content for marker in ("写", "生成", "分析"))

    @staticmethod
    def _is_selected_document_answer_request(
        content: str,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> bool:
        if not selected_document_ids and not any(attachment.document_id for attachment in attachments):
            return False
        if KnowledgeAgentRuntime._is_assignment_intent(content) or KnowledgeAgentRuntime._is_clear_categories_intent(content):
            return False
        if KnowledgeAgentRuntime._is_document_category_query(content):
            return False
        answer_markers = (
            "创新点",
            "贡献",
            "方法",
            "问题",
            "结论",
            "局限",
            "实验",
            "结果",
            "区别",
            "相同点",
            "不同点",
            "分别",
            "各自",
            "是什么",
            "有哪些",
            "说明",
            "解释",
            "概括",
            "概览",
            "亮点",
            "novelty",
            "contribution",
            "method",
            "result",
            "finding",
        )
        return any(marker in content.casefold() for marker in answer_markers)

    @classmethod
    def _is_grouped_category_summary_request(cls, content: str) -> bool:
        return cls._is_summary_request(content, []) and any(
            marker in content for marker in ("按标签", "按分类", "每个标签", "每个分类", "分组")
        )

    @staticmethod
    def _mentions_all_library(content: str) -> bool:
        return any(marker in content for marker in ("所有论文", "全部论文", "论文库", "全库", "所有文献"))

    def _pending_path(self, session_id: str):
        return self.pending_action_store.path_for(session_id)

    def _write_pending_action(self, session_id: str, payload: dict[str, Any]) -> None:
        self.pending_action_store.write(session_id, payload)

    def _read_pending_action(self, session_id: str) -> dict[str, Any] | None:
        return self.pending_action_store.read(session_id)

    def _clear_pending_action(self, session_id: str) -> None:
        self.pending_action_store.clear(session_id)

    @staticmethod
    def _collect_citations(evidence_items: list[EvidenceItem]) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for item in evidence_items:
            if item.citation_label in seen:
                continue
            seen.add(item.citation_label)
            citations.append(item.citation_label)
        return citations

    @staticmethod
    def _extract_message_text(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip() if parts else None
        return None
