"""Read-only Workbench aggregation services."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.config import Settings
from app.models import (
    ChatMessage,
    WorkbenchAgentProfile,
    WorkbenchCapabilitiesResponse,
    WorkbenchCapability,
    WorkbenchCompactTraceStep,
    WorkbenchConfigResponse,
    WorkbenchExperimentalCapability,
    WorkbenchFileContextResponse,
    WorkbenchMessageTraceSummary,
    WorkbenchModelOption,
    WorkbenchSlashCommand,
    WorkbenchTraceArtifactStatus,
    WorkbenchTraceToolStep,
)
from app.repositories import ChatRepository, LibraryRepository, ReportRepository, RuntimeRepository


DEFAULT_AGENT_PROFILES = [
    WorkbenchAgentProfile(
        id="paper_qa",
        label="论文问答",
        description="基于论文内容、元数据和可用证据回答问题。",
    ),
    WorkbenchAgentProfile(
        id="review_writer",
        label="综述/报告草稿",
        description="辅助组织综述、报告草稿和引用材料。",
    ),
    WorkbenchAgentProfile(
        id="library_organizer",
        label="标签与分类整理",
        description="辅助查看、整理和维护论文标签分类。",
    ),
    WorkbenchAgentProfile(
        id="general_chat",
        label="通用科研助手",
        description="处理通用科研对话和非库内问答。",
    ),
]


class WorkbenchService:
    """Build Workbench read models from existing repositories and runtime state."""

    def __init__(
        self,
        *,
        settings: Settings,
        library_repository: LibraryRepository,
        chat_repository: ChatRepository,
        report_repository: ReportRepository,
        runtime_repository: RuntimeRepository,
        knowledge_agent_runtime=None,
    ) -> None:
        self.settings = settings
        self.library_repository = library_repository
        self.chat_repository = chat_repository
        self.report_repository = report_repository
        self.runtime_repository = runtime_repository
        self.knowledge_agent_runtime = knowledge_agent_runtime

    def get_config(self) -> WorkbenchConfigResponse:
        current_model = self.settings.effective_llm_model
        return WorkbenchConfigResponse(
            current_model=current_model,
            masked_base_url=self._mask_base_url(self.settings.effective_llm_base_url),
            available_models=[
                WorkbenchModelOption(
                    id=current_model,
                    label=current_model,
                    is_current=True,
                )
            ],
            agent_profiles=DEFAULT_AGENT_PROFILES,
        )

    def get_capabilities(self) -> WorkbenchCapabilitiesResponse:
        from app.runtime.tool_registry import ToolRegistry

        registry = ToolRegistry(
            enable_experimental_mcp=self.settings.enable_experimental_mcp,
            enable_mcp_in_knowledge=self.settings.enable_mcp_in_knowledge,
        )
        stable_capabilities = [
            self._capability_from_tool(tool)
            for tool in registry.list_default_candidates(scope="knowledge")
            if tool.spec is not None
        ]
        confirmation_required = [
            capability
            for capability in stable_capabilities
            if capability.destructive or capability.requires_confirmation
        ]
        return WorkbenchCapabilitiesResponse(
            stable_capabilities=stable_capabilities,
            confirmation_required=confirmation_required,
            experimental_capabilities=self._experimental_capabilities(),
            slash_commands=self._slash_commands(),
        )

    def get_file_context(self, session_id: str) -> WorkbenchFileContextResponse:
        session = self.chat_repository.get_session(session_id)
        if session is None:
            raise ValueError("Chat session not found")

        messages = self._safe_list_messages(session_id)
        referents = self._safe_conversation_referents(session_id)
        attachment_document_ids = self._collect_attachment_document_ids(messages)
        used_document_ids = self._collect_used_document_ids(messages)
        selected_document_ids = self._latest_user_attachment_document_ids(messages)
        recent_document_ids = self._collect_recent_document_ids(messages, referents)
        report_referenced_document_ids = self._collect_report_referenced_document_ids(messages)

        return WorkbenchFileContextResponse(
            session_id=session.id,
            library_documents=self._safe_list_library_documents(),
            selected_document_ids=selected_document_ids,
            attachment_document_ids=attachment_document_ids,
            recent_document_ids=recent_document_ids,
            used_document_ids=used_document_ids,
            report_referenced_document_ids=report_referenced_document_ids,
            referents=referents,
        )

    def get_message_trace_summary(self, message_id: str) -> WorkbenchMessageTraceSummary:
        message = self.chat_repository.get_message(message_id)
        if message is None:
            raise ValueError("Chat message not found")

        traces = self._safe_list_traces(message.agent_trace_id)
        route = None
        risk_level = "unknown"
        evidence_count = 0
        confirmation_status = self._initial_confirmation_status(message.action_status)
        compact_steps: list[WorkbenchCompactTraceStep] = []
        tool_steps: list[WorkbenchTraceToolStep] = []

        for trace in traces:
            payload = trace.payload if isinstance(trace.payload, dict) else {}
            if trace.status == "agent_mode_selected":
                route = self._string_value(payload.get("route")) or route
                risk_level = self._normalize_risk_level(payload.get("risk_level"), fallback=risk_level)
            if trace.status in {"retrieval_tool_finished", "tool_call_log", "react_observation"}:
                evidence_count = max(evidence_count, self._int_value(payload.get("evidence_count")))
            if trace.status == "agent_orchestrator_finished":
                evidence_count = max(evidence_count, self._int_value(payload.get("evidence_count")))
            if trace.status == "confirmation_required":
                confirmation_status = "required"
            elif trace.status == "pending_write_executed":
                confirmation_status = "executed"
            elif trace.status in {"library_write_verification_failed", "validation_failed", "failed"}:
                if confirmation_status in {"required", "executed"}:
                    confirmation_status = "failed"

            step = self._compact_step_from_trace(trace)
            if step is not None:
                compact_steps.append(step)

            tool_step = self._tool_step_from_trace(trace)
            if tool_step is not None:
                tool_steps.append(tool_step)
                evidence_count = max(evidence_count, tool_step.evidence_count)
                risk_level = self._normalize_risk_level(tool_step.risk_level, fallback=risk_level)

        if message.saved_report_id or message.action_status == "report_saved":
            compact_steps.append(
                WorkbenchCompactTraceStep(
                    kind="report_saved",
                    label="Report saved",
                    status="completed",
                    detail="Assistant answer was saved as a report.",
                    created_at=message.created_at.isoformat(),
                )
            )

        if message.action_status == "confirmation_required":
            confirmation_status = "required"
        elif message.action_status == "failed" and confirmation_status in {"required", "executed"}:
            confirmation_status = "failed"
        elif message.action_status == "report_saved" and confirmation_status == "none":
            confirmation_status = "none"

        report_saved = bool(message.saved_report_id)
        can_save_report = (
            message.role == "assistant"
            and message.status == "completed"
            and bool(message.content.strip())
            and not report_saved
        )

        return WorkbenchMessageTraceSummary(
            message_id=message.id,
            trace_id=message.agent_trace_id if traces else None,
            route=route,
            action_status=message.action_status,
            retrieval_status=message.retrieval_status,
            used_document_ids=list(message.used_document_ids),
            evidence_count=evidence_count,
            tool_steps=tool_steps,
            risk_level=self._normalize_risk_level(risk_level),
            confirmation_status=confirmation_status,
            saved_report_id=message.saved_report_id,
            artifact_status=WorkbenchTraceArtifactStatus(
                report_saved=report_saved,
                can_save_report=can_save_report,
                report_id=message.saved_report_id,
            ),
            compact_steps=compact_steps,
        )

    def _capability_from_tool(self, tool) -> WorkbenchCapability:
        spec = tool.spec
        return WorkbenchCapability(
            id=tool.tool_id,
            name=spec.display_name or tool.name,
            description=spec.description or tool.description,
            group=self._capability_group(tool.tool_id),
            scope=spec.scope,
            maturity=spec.maturity,
            source=spec.source,
            operation_level=spec.operation_level,
            io_type=spec.io_type,
            write_type=spec.write_type,
            destructive=spec.destructive,
            requires_confirmation=spec.requires_confirmation,
            available_by_default=spec.available_by_default,
            current_available=True,
            slash_command=self._slash_command_for_tool(tool.tool_id),
            user_hint=self._capability_hint(tool.tool_id, spec.io_type, spec.requires_confirmation, spec.destructive),
        )

    @staticmethod
    def _capability_group(tool_id: str) -> str:
        if tool_id.startswith("library.explorer."):
            return "library_files"
        if tool_id.startswith("evidence.retriever.") or tool_id.startswith("report.drafter."):
            return "qa_summary_report"
        if tool_id.startswith("library.operator."):
            return "tags_categories"
        if tool_id.startswith("memory."):
            return "memory"
        if tool_id.startswith("tool.registry."):
            return "tool_registry"
        return "knowledge"

    @staticmethod
    def _slash_command_for_tool(tool_id: str) -> str | None:
        if tool_id in {
            "library.explorer.stats",
            "library.explorer.category_stats",
            "library.explorer.find_documents",
            "library.explorer.document_metadata",
            "library.explorer.document_categories",
        }:
            return "library"
        if tool_id.startswith("library.operator."):
            return "tag"
        if tool_id in {"evidence.retriever.search", "report.drafter.write"}:
            return "summary"
        if tool_id in {"evidence.retriever.search_by_category", "report.drafter.write_by_category"}:
            return "compare"
        if tool_id == "tool.registry.list":
            return "help"
        return None

    @staticmethod
    def _capability_hint(tool_id: str, io_type: str, requires_confirmation: bool, destructive: bool) -> str:
        if destructive or requires_confirmation:
            return "Requires preview and explicit confirmation before the runtime can apply this change."
        if tool_id.startswith("library.operator."):
            return "Uses the existing write guardrails and post-write verification path."
        if io_type == "read":
            return "Available for read-only Workbench display and Knowledge chat planning."
        return "Available through the existing guarded Knowledge chat flow."

    @staticmethod
    def _experimental_capabilities() -> list[WorkbenchExperimentalCapability]:
        return [
            WorkbenchExperimentalCapability(
                id="skills",
                name="Skills",
                description="File-backed capability definitions reserved for later Workbench productization.",
                feature_flag=None,
                user_hint="Shown as a future capability only; this API does not manage or execute Skills.",
            ),
            WorkbenchExperimentalCapability(
                id="mcp",
                name="MCP",
                description="Read-only external academic tool declarations remain isolated behind experimental gates.",
                feature_flag="ENABLE_EXPERIMENTAL_MCP",
                user_hint="Shown as an experimental summary only; this API does not edit, install, or execute MCP tools.",
            ),
            WorkbenchExperimentalCapability(
                id="subagent",
                name="Subagent",
                description="Delegated worker execution is reserved for future stages.",
                feature_flag="ENABLE_SUBAGENT_EXECUTION",
                user_hint="Shown as unavailable in this panel; no subagent can be launched here.",
            ),
            WorkbenchExperimentalCapability(
                id="research_runtime",
                name="Research Runtime",
                description="Research loop capabilities are separate from the default Knowledge Workbench path.",
                feature_flag="ENABLE_RESEARCH_TASK_AGENT",
                user_hint="Shown as a future Workbench capability summary; this panel does not start research runs.",
            ),
        ]

    @staticmethod
    def _slash_commands() -> list[WorkbenchSlashCommand]:
        return [
            WorkbenchSlashCommand(id="summary", label="/summary", description="Summarize selected papers."),
            WorkbenchSlashCommand(id="compare", label="/compare", description="Compare selected papers."),
            WorkbenchSlashCommand(id="tag", label="/tag", description="Query or organize tags and categories."),
            WorkbenchSlashCommand(id="library", label="/library", description="Query library counts, status, and tags."),
            WorkbenchSlashCommand(id="help", label="/help", description="Show available slash commands."),
        ]

    def _safe_list_library_documents(self):
        try:
            return self.library_repository.list_documents()
        except Exception:
            return []

    def _safe_list_messages(self, session_id: str) -> list[ChatMessage]:
        try:
            return self.chat_repository.list_messages(session_id)
        except Exception:
            return []

    def _safe_list_traces(self, trace_id: str | None):
        if not trace_id:
            return []
        try:
            return self.runtime_repository.list_traces(trace_id)
        except Exception:
            return []

    def _safe_conversation_referents(self, session_id: str) -> dict[str, Any]:
        if self.knowledge_agent_runtime is None:
            return {}
        try:
            referents = self.knowledge_agent_runtime.conversation_referents(session_id)
        except Exception:
            return {}
        return referents if isinstance(referents, dict) else {}

    @classmethod
    def _collect_attachment_document_ids(cls, messages: list[ChatMessage]) -> list[str]:
        return cls._dedupe(
            attachment.document_id
            for message in messages
            for attachment in message.attachments
            if attachment.document_id
        )

    @classmethod
    def _collect_used_document_ids(cls, messages: list[ChatMessage]) -> list[str]:
        return cls._dedupe(
            document_id
            for message in messages
            for document_id in message.used_document_ids
            if document_id
        )

    @classmethod
    def _latest_user_attachment_document_ids(cls, messages: list[ChatMessage]) -> list[str]:
        for message in reversed(messages):
            if message.role != "user":
                continue
            document_ids = [
                attachment.document_id
                for attachment in message.attachments
                if attachment.document_id
            ]
            if document_ids:
                return cls._dedupe(document_ids)
        return []

    @classmethod
    def _collect_recent_document_ids(cls, messages: list[ChatMessage], referents: dict[str, Any]) -> list[str]:
        referent_ids = cls._document_ids_from_referents(referents)
        if referent_ids:
            return referent_ids

        recent_candidates: list[str] = []
        for message in reversed(messages):
            recent_candidates.extend(reversed(message.used_document_ids))
            recent_candidates.extend(
                attachment.document_id
                for attachment in reversed(message.attachments)
                if attachment.document_id
            )
            if len(recent_candidates) >= 10:
                break
        return cls._dedupe(recent_candidates)[:10]

    def _collect_report_referenced_document_ids(self, messages: list[ChatMessage]) -> list[str]:
        document_ids: list[str] = []
        for report_id in self._dedupe(message.saved_report_id for message in messages if message.saved_report_id):
            try:
                report = self.report_repository.get_report(report_id)
            except Exception:
                continue
            if report is None:
                continue
            document_ids.extend(report.paper_ids)
            document_ids.extend(
                citation.document_id
                for citation in report.citation_items
                if citation.document_id
            )
        return self._dedupe(document_ids)

    @classmethod
    def _document_ids_from_referents(cls, referents: dict[str, Any]) -> list[str]:
        values: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                document_id = node.get("document_id")
                if isinstance(document_id, str):
                    values.append(document_id)
                document_ids = node.get("document_ids")
                if isinstance(document_ids, list):
                    values.extend(str(item) for item in document_ids if item)
                for child in node.values():
                    visit(child)
                return
            if isinstance(node, list):
                for child in node:
                    visit(child)

        visit(referents)
        return cls._dedupe(values)

    @staticmethod
    def _dedupe(values: Iterable[str | None]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            item = str(value)
            if item in seen:
                continue
            seen.add(item)
            results.append(item)
        return results

    @staticmethod
    def _mask_base_url(base_url: str | None) -> str | None:
        if not base_url:
            return None
        try:
            parts = urlsplit(base_url)
        except ValueError:
            return f"{base_url[:8]}***" if len(base_url) > 8 else "***"
        if not parts.netloc:
            return f"{base_url[:8]}***" if len(base_url) > 8 else "***"
        masked_path = "/***"
        return urlunsplit((parts.scheme, parts.netloc, masked_path, "", ""))

    @classmethod
    def _compact_step_from_trace(cls, trace) -> WorkbenchCompactTraceStep | None:
        payload = trace.payload if isinstance(trace.payload, dict) else {}
        mapping = {
            "agent_mode_selected": ("route_selected", "Route selected", cls._route_selected_detail),
            "deterministic_observation": ("deterministic_read", "Deterministic read", cls._deterministic_read_detail),
            "retrieval_tool_started": ("retrieval_started", "Retrieval started", cls._retrieval_started_detail),
            "retrieval_tool_finished": ("retrieval_finished", "Retrieval finished", cls._retrieval_finished_detail),
            "tool_call_log": ("tool_call", "Tool call", cls._tool_call_detail),
            "react_observation": ("tool_call", "Tool observation", cls._tool_call_detail),
            "tool_started": ("tool_call", "Tool started", cls._tool_call_detail),
            "tool_finished": ("tool_call", "Tool finished", cls._tool_call_detail),
            "evidence_reused": ("evidence_reused", "Evidence reused", cls._evidence_reused_detail),
            "write_preview_created": ("write_preview_created", "Write preview created", cls._write_preview_detail),
            "confirmation_required": ("confirmation_required", "Confirmation required", cls._confirmation_required_detail),
            "pending_write_executed": ("pending_write_executed", "Pending write executed", cls._pending_write_detail),
            "library_write_verified": ("write_verified", "Write verified", cls._write_verified_detail),
            "slash_command_hint_recorded": ("slash_command_hint", "Slash command hint", cls._slash_hint_detail),
            "workbench_hint_recorded": ("workbench_hint", "Workbench hint", cls._workbench_hint_detail),
        }
        item = mapping.get(trace.status)
        if item is None:
            return None
        kind, label, detail_factory = item
        return WorkbenchCompactTraceStep(
            kind=kind,
            label=label,
            status=cls._display_status(trace.status, payload),
            detail=detail_factory(payload),
            created_at=cls._created_at_text(trace.created_at),
        )

    @classmethod
    def _tool_step_from_trace(cls, trace) -> WorkbenchTraceToolStep | None:
        if trace.status not in {"tool_call_log", "react_observation", "tool_started", "tool_finished"}:
            return None
        payload = trace.payload if isinstance(trace.payload, dict) else {}
        tool_name = (
            cls._string_value(payload.get("tool_name"))
            or cls._string_value(payload.get("tool"))
            or cls._string_value(payload.get("selected_tool"))
        )
        if not tool_name:
            return None
        status = cls._string_value(payload.get("status")) or cls._display_status(trace.status, payload)
        summary = cls._short_summary(payload.get("summary")) or cls._short_summary(trace.message)
        return WorkbenchTraceToolStep(
            tool_name=tool_name,
            display_name=cls._tool_display_name(tool_name),
            status=status,
            summary=summary,
            evidence_count=cls._int_value(payload.get("evidence_count")),
            risk_level=cls._normalize_risk_level(payload.get("risk_level")),
        )

    @staticmethod
    def _initial_confirmation_status(action_status: str | None) -> str:
        if action_status == "confirmation_required":
            return "required"
        if action_status == "failed":
            return "failed"
        return "none"

    @staticmethod
    def _normalize_risk_level(value: object, *, fallback: str = "unknown") -> str:
        raw = str(value or fallback or "unknown").strip().casefold()
        aliases = {
            "read": "read_only",
            "low": "read_only",
            "safe": "read_only",
            "safe_write": "safe_write",
            "scoped_write": "scoped_write",
            "destructive": "destructive",
            "high": "destructive",
        }
        normalized = aliases.get(raw, raw)
        if normalized in {"read_only", "safe_write", "scoped_write", "destructive", "unknown"}:
            return normalized
        return fallback if fallback in {"read_only", "safe_write", "scoped_write", "destructive"} else "unknown"

    @staticmethod
    def _display_status(status: str, payload: dict[str, Any]) -> str:
        if status.endswith("_started"):
            return "started"
        if status.endswith("_finished") or status in {"library_write_verified", "pending_write_executed"}:
            return "completed"
        if status == "confirmation_required":
            return "required"
        if status == "tool_call_log":
            return "completed" if payload.get("success", True) is not False else "failed"
        if status == "react_observation":
            return str(payload.get("status") or "completed")
        return "recorded"

    @staticmethod
    def _created_at_text(value: object) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value or "")

    @staticmethod
    def _string_value(value: object) -> str:
        if value is None:
            return ""
        cleaned = str(value).strip()
        return WorkbenchService._redact_display_text(cleaned) if cleaned else ""

    @staticmethod
    def _int_value(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _short_summary(value: object, *, limit: int = 180) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = " ".join(value.split())
        cleaned = WorkbenchService._redact_display_text(cleaned)
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 1]}..."

    @staticmethod
    def _redact_display_text(value: str) -> str:
        pattern = re.compile(
            "|".join(
                re.escape(fragment)
                for fragment in (
                    "system prompt",
                    "user_prompt",
                    "prompt",
                    "llm raw json",
                    "completion raw json",
                    "raw_json",
                    "api_key",
                    "base_url",
                    "authorization",
                    "bearer",
                    "pending_action",
                    "input_schema",
                    "output_schema",
                    "chunk_text",
                )
            ),
            flags=re.IGNORECASE,
        )
        return pattern.sub("[redacted]", value)

    @staticmethod
    def _route_selected_detail(payload: dict[str, Any]) -> str:
        route = WorkbenchService._string_value(payload.get("route")) or "unknown"
        runtime = WorkbenchService._string_value(payload.get("target_runtime")) or "selected runtime"
        return f"Selected {route} route with {runtime}."

    @staticmethod
    def _deterministic_read_detail(payload: dict[str, Any]) -> str:
        return WorkbenchService._short_summary(payload.get("summary")) or "Read local Workbench data deterministically."

    @staticmethod
    def _retrieval_started_detail(payload: dict[str, Any]) -> str:
        tool = WorkbenchService._string_value(payload.get("tool")) or WorkbenchService._string_value(payload.get("tool_name"))
        return f"Started retrieval with {tool or 'retrieval tool'}."

    @staticmethod
    def _retrieval_finished_detail(payload: dict[str, Any]) -> str:
        count = WorkbenchService._int_value(payload.get("evidence_count"))
        status = WorkbenchService._string_value(payload.get("status")) or "completed"
        return f"Retrieval {status}; evidence count: {count}."

    @staticmethod
    def _tool_call_detail(payload: dict[str, Any]) -> str:
        tool = (
            WorkbenchService._string_value(payload.get("tool_name"))
            or WorkbenchService._string_value(payload.get("tool"))
            or WorkbenchService._string_value(payload.get("selected_tool"))
            or "tool"
        )
        status = WorkbenchService._string_value(payload.get("status")) or "completed"
        return f"{WorkbenchService._tool_display_name(tool)} {status}."

    @staticmethod
    def _evidence_reused_detail(payload: dict[str, Any]) -> str:
        count = WorkbenchService._int_value(payload.get("evidence_count"))
        return f"Reused prior evidence; evidence count: {count}."

    @staticmethod
    def _write_preview_detail(payload: dict[str, Any]) -> str:
        operation = WorkbenchService._string_value(payload.get("operation")) or "write operation"
        affected_count = WorkbenchService._int_value(payload.get("affected_count"))
        target_type = WorkbenchService._string_value(payload.get("target_type")) or "target"
        return f"Previewed {operation} for {affected_count} {target_type} item(s)."

    @staticmethod
    def _confirmation_required_detail(payload: dict[str, Any]) -> str:
        pending = payload.get("pending_action") if isinstance(payload.get("pending_action"), dict) else {}
        operation = WorkbenchService._string_value(pending.get("operation") if pending else payload.get("operation"))
        target_type = WorkbenchService._string_value(pending.get("target_type") if pending else payload.get("target_type"))
        if operation or target_type:
            return f"User confirmation required for {operation or 'write operation'} on {target_type or 'target'}."
        return "User confirmation required before applying the protected action."

    @staticmethod
    def _pending_write_detail(payload: dict[str, Any]) -> str:
        operation = WorkbenchService._string_value(payload.get("operation")) or "write operation"
        affected_count = WorkbenchService._int_value(payload.get("affected_count") or payload.get("updated_count"))
        return f"Executed confirmed {operation}; affected count: {affected_count}."

    @staticmethod
    def _write_verified_detail(payload: dict[str, Any]) -> str:
        count = WorkbenchService._int_value(
            payload.get("affected_document_count")
            or payload.get("affected_category_count")
            or payload.get("updated_count")
        )
        return f"Verified library write; affected count: {count}."

    @staticmethod
    def _slash_hint_detail(payload: dict[str, Any]) -> str:
        command = WorkbenchService._string_value(payload.get("command")) or "unknown"
        selected_count = WorkbenchService._int_value(payload.get("selected_document_count"))
        return f"Applied /{command} hint with {selected_count} selected document(s)."

    @staticmethod
    def _workbench_hint_detail(payload: dict[str, Any]) -> str:
        profile = WorkbenchService._string_value(payload.get("agent_profile_id")) or "default profile"
        model = WorkbenchService._string_value(payload.get("model_id"))
        return f"Applied Workbench hint for {profile}{' and model hint' if model else ''}."

    @staticmethod
    def _tool_display_name(tool_name: str) -> str:
        mapping = {
            "tool.registry.list": "List PaperDesk chat tools",
            "library.explorer.stats": "Read library statistics",
            "library.explorer.category_stats": "Read tag/category statistics",
            "library.explorer.find_documents": "Resolve documents or tags",
            "library.explorer.document_metadata": "Read document metadata",
            "library.explorer.document_categories": "Read document tags/categories",
            "evidence.retriever.search": "Retrieve document evidence",
            "evidence.retriever.search_by_category": "Retrieve evidence by tag/category",
            "report.drafter.write": "Synthesize grounded answer",
            "report.drafter.write_by_category": "Synthesize grouped answer",
            "memory.read": "Read chat memory",
            "memory.write": "Write chat memory",
            "library.operator.create_category": "Create tag/category",
            "library.operator.assign_category": "Assign tag/category",
            "library.operator.rename_category": "Rename tag/category",
            "library.operator.delete_unused_categories": "Delete unused tags/categories",
            "library.operator.clear_categories": "Clear document tags/categories",
        }
        return mapping.get(tool_name, tool_name)
