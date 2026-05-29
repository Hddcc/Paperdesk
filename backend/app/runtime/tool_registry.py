"""Unified backend tool declarations for PaperDesk agent capabilities.

ToolRegistry is a declaration and metadata catalog. It is intentionally not a
tool execution center: Knowledge execution still lives in KnowledgeAgentRuntime,
Research execution still lives in the research loop, and MCP declarations stay
behind explicit experimental gates.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.models import ResearchActionType, ToolDeclaration, ToolSource, ToolSpec
from .mcp_adapter import (
    ReadOnlyMcpAdapter,
    default_read_only_academic_mcp_declarations,
)


class ToolRegistry:
    """Register builtin and external tool declarations behind one lookup API.

    This registry describes capabilities and metadata; runtime classes still
    execute tools. Scope and feature flags keep knowledge, research, MCP, and
    experimental tools from leaking into the default Knowledge path.

    Boundary vocabulary:
    - scope="knowledge": stable PaperDesk Knowledge Chat tools.
    - scope="research": Research Agent Loop tools, not default Knowledge tools.
    - scope="mcp": external MCP declarations behind feature flags.
    - scope="experimental": opt-in learning or future-facing capabilities.
    - available_by_default means "eligible for default selection" only after
      maturity, scope, source, and feature-flag filters also pass.
    """

    def __init__(
        self,
        declarations: Iterable[ToolDeclaration] | None = None,
        *,
        enable_experimental_mcp: bool = False,
        enable_mcp_in_knowledge: bool = False,
    ) -> None:
        self._tools: dict[str, ToolDeclaration] = {}
        self.enable_experimental_mcp = enable_experimental_mcp
        self.enable_mcp_in_knowledge = enable_mcp_in_knowledge
        for declaration in builtin_tool_declarations():
            self.register(declaration)
        external_declarations = (
            list(declarations)
            if declarations is not None
            else default_mcp_tool_declarations(enabled=enable_experimental_mcp)
        )
        for declaration in external_declarations:
            self.register(declaration)

    def register(self, declaration: ToolDeclaration) -> None:
        self._tools[declaration.tool_id] = declaration

    def get(self, tool_id: str) -> ToolDeclaration | None:
        tool = self._tools.get(tool_id)
        if tool is None or not tool.enabled:
            return None
        return tool

    def list_enabled(self) -> list[ToolDeclaration]:
        return sorted(
            [tool for tool in self._tools.values() if tool.enabled],
            key=lambda tool: tool.tool_id,
        )

    def filter_enabled(self, tool_ids: Iterable[str]) -> list[ToolDeclaration]:
        return [tool for tool_id in tool_ids if (tool := self.get(tool_id)) is not None]

    def list_default_candidates(
        self,
        *,
        scope: str = "knowledge",
        operation_levels: Iterable[str] | None = None,
    ) -> list[ToolDeclaration]:
        """Return default candidates only when structured metadata allows it.

        This is the registry-side boundary for default tool exposure. It does
        not execute tools and does not relax runtime guardrails: destructive or
        write-capable tools still need their runtime preview, confirmation, and
        verification paths when selected.
        """

        allowed_levels = set(operation_levels or [])
        candidates: list[ToolDeclaration] = []
        for tool in self.list_enabled():
            spec = tool.spec
            if spec is None:
                continue
            if not spec.available_by_default:
                continue
            if spec.maturity != "stable":
                continue
            if spec.scope == "experimental":
                continue
            if spec.source == ToolSource.MCP.value and not (
                self.enable_experimental_mcp and (scope != "knowledge" or self.enable_mcp_in_knowledge)
            ):
                continue
            if spec.source == ToolSource.MCP.value and (
                spec.io_type != "read" or spec.operation_level != "query-level" or spec.destructive
            ):
                continue
            visible_scopes = {scope, "shared", "common"}
            if scope == "knowledge":
                visible_scopes.add("workspace")
            if spec.scope not in visible_scopes:
                continue
            if allowed_levels and spec.operation_level not in allowed_levels:
                continue
            candidates.append(tool)
        return candidates

    def label_for(self, tool_id: str, fallback: str | None = None) -> str:
        tool = self.get(tool_id)
        if tool is None:
            return fallback or tool_id
        return tool.name


def builtin_tool_declarations() -> list[ToolDeclaration]:
    """Declare builtin Research Agent Loop tools.

    These declarations are scoped as research capabilities and remain separate
    from the Knowledge Chat default candidate set.
    """

    common_output = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "payload": {"type": "object"},
        },
    }
    tools = [
        _tool(
            "plan/rule_based_initial",
            "规则初始规划",
            "根据研究主题和任务路由生成初始计划项。",
            ResearchActionType.PLAN,
            {"topic": "string", "task_route": "object"},
            common_output,
        ),
        _tool(
            "search_online/openalex_primary",
            "OpenAlex 优先检索",
            "优先使用 OpenAlex 检索在线论文候选。",
            ResearchActionType.SEARCH_ONLINE,
            {"query": "string", "top_k_online": "integer"},
            common_output,
        ),
        _tool(
            "search_online/arxiv_primary",
            "arXiv 优先检索",
            "优先使用 arXiv 检索在线论文候选。",
            ResearchActionType.SEARCH_ONLINE,
            {"query": "string", "top_k_online": "integer"},
            common_output,
        ),
        _tool(
            "search_online/mixed_broad_recall",
            "混合宽召回检索",
            "按当前 provider 配置聚合在线论文候选。",
            ResearchActionType.SEARCH_ONLINE,
            {"query": "string", "search_provider": "string", "top_k_online": "integer"},
            common_output,
        ),
        _tool(
            "search_local/vector_recall_default",
            "默认向量召回",
            "从本地论文库向量索引检索相关证据片段。",
            ResearchActionType.SEARCH_LOCAL,
            {"query": "string", "document_ids": "array", "top_k_local": "integer"},
            common_output,
        ),
        _tool(
            "summarize_evidence/task_level_merge",
            "任务级证据合并",
            "把在线论文候选和本地证据合并为任务级总结。",
            ResearchActionType.SUMMARIZE_EVIDENCE,
            {"paper_records": "array", "evidence_items": "array"},
            common_output,
        ),
        _tool(
            "summarize_evidence/degraded_closeout",
            "证据不足降级收束",
            "在证据不足时生成带边界说明的任务总结。",
            ResearchActionType.SUMMARIZE_EVIDENCE,
            {"paper_records": "array", "evidence_items": "array", "degraded": "boolean"},
            common_output,
        ),
        _tool(
            "revise_plan/rewrite_query",
            "改写检索 query",
            "在证据不足时改写当前计划项 query。",
            ResearchActionType.REVISE_PLAN,
            {"task_id": "string", "query": "string"},
            common_output,
        ),
        _tool(
            "revise_plan/split_task",
            "拆分过宽任务",
            "把过宽计划项拆分为更可执行的子计划。",
            ResearchActionType.REVISE_PLAN,
            {"task_id": "string"},
            common_output,
        ),
        _tool(
            "revise_plan/reorder_priority",
            "重排待办优先级",
            "在连续无增量时调整待办顺序。",
            ResearchActionType.REVISE_PLAN,
            {"task_ids": "array"},
            common_output,
        ),
        _tool(
            "finalize_report/report_writer_default",
            "默认报告生成",
            "基于任务总结生成最终研究报告。",
            ResearchActionType.FINALIZE_REPORT,
            {"topic": "string", "task_summaries": "array"},
            common_output,
            read_only=False,
        ),
        _tool(
            "finalize_report/task_artifact_writer",
            "任务型结果生成",
            "基于任务路由协议生成任务型最终结果。",
            ResearchActionType.FINALIZE_REPORT,
            {"topic": "string", "task_summaries": "array", "task_route": "object"},
            common_output,
            read_only=False,
        ),
        _tool(
            "finish/runtime_complete",
            "运行完成",
            "标记当前研究流程完成。",
            ResearchActionType.FINISH,
            {},
            common_output,
        ),
        _tool(
            "fail/runtime_stop",
            "运行停止",
            "标记当前研究流程失败并停止。",
            ResearchActionType.FAIL,
            {"reason": "string"},
            common_output,
        ),
    ]
    tools.extend(paperdesk_chat_tool_declarations(common_output))
    tools.extend(workspace_file_tool_declarations())
    return tools


def workspace_file_tool_declarations() -> list[ToolDeclaration]:
    """Declare session workspace file tools without wiring Agent execution.

    Execution stays in the workspace tool adapter and existing workspace
    services. These declarations are visible to Knowledge/workspace callers and
    are kept out of research and MCP surfaces by scope/source metadata.
    """

    def workspace_tool(
        tool_id: str,
        name: str,
        description: str,
        properties: dict[str, dict],
        output_properties: dict[str, dict],
        *,
        required: list[str] | None = None,
        read_only: bool = True,
        risk_level: str = "low",
        operation_level: str = "query-level",
        write_type: str = "none",
        requires_confirmation: bool = False,
        available_by_default: bool = True,
        output_summary_policy: str = "Record relative paths and counts only.",
        security_notes: list[str] | None = None,
    ) -> ToolDeclaration:
        spec = ToolSpec(
            name=tool_id,
            display_name=name,
            description=description,
            scope="workspace",
            operation_level=operation_level,
            io_type="read" if read_only else "write",
            write_type=write_type,
            destructive=False,
            requires_confirmation=requires_confirmation,
            input_object_types=["workspace_file"],
            output_observation_type="workspace_file_tool_observation",
            requires_post_read_verification=False,
            verification_tool=None,
            available_by_default=available_by_default,
            maturity="stable",
            source=ToolSource.BUILTIN.value,
        )
        return ToolDeclaration(
            tool_id=tool_id,
            source=ToolSource.BUILTIN,
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "action_type": "workspace_file_tool",
                "risk_level": risk_level,
                "scope_type": "session_workspace",
                "operation_level": operation_level,
                "io_type": spec.io_type,
                "write_type": write_type,
                "requires_confirmation": requires_confirmation,
                "required": required or [],
                "properties": properties,
                "output_summary_policy": output_summary_policy,
                "security_notes": security_notes or [],
            },
            output_schema={
                "type": "object",
                "properties": output_properties,
                "output_summary_policy": output_summary_policy,
            },
            read_only=read_only,
            enabled=True,
            spec=spec,
        )

    safe_path_notes = [
        "Only session workspace relative paths are accepted.",
        "Absolute paths, drive paths, UNC paths, traversal, hidden paths, sensitive names, and symlink escapes are rejected.",
        "Server filesystem locations are never returned.",
    ]
    return [
        workspace_tool(
            "workspace.file.list",
            "List workspace files",
            "List safe files inside the current session workspace, returning relative paths and file metadata.",
            {
                "path": {"type": "string", "default": ""},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {"type": "integer", "default": 100, "minimum": 1, "maximum": 500},
            },
            {
                "items": {"type": "array"},
                "count": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "max_entries": {"type": "integer"},
            },
            risk_level="low",
            security_notes=safe_path_notes,
        ),
        workspace_tool(
            "workspace.file.read",
            "Read workspace file",
            "Read one safe UTF-8 text/code file from the current session workspace with content truncation.",
            {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 12000, "minimum": 1, "maximum": 50000},
            },
            {
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
                "included_chars": {"type": "integer"},
                "char_count": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "mime_type": {"type": ["string", "null"]},
                "size_bytes": {"type": "integer"},
            },
            required=["path"],
            risk_level="low",
            output_summary_policy="Tool result may include content; trace summaries keep only path, size, included_chars, and truncation.",
            security_notes=safe_path_notes
            + ["Binary files, PDFs, unsupported extensions, and sensitive files are rejected."],
        ),
        workspace_tool(
            "workspace.file.write_new",
            "Create workspace file",
            "Create a new safe UTF-8 text/code file in the current session workspace; existing files are rejected and cannot be overwritten.",
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "display_name": {"type": "string"},
            },
            {
                "relative_path": {"type": "string"},
                "display_name": {"type": "string"},
                "file_kind": {"type": "string"},
                "mime_type": {"type": ["string", "null"]},
                "size_bytes": {"type": "integer"},
                "checksum": {"type": "string"},
                "status": {"type": "string"},
            },
            required=["path", "content"],
            read_only=False,
            risk_level="low_to_medium",
            operation_level="content-level",
            write_type="create",
            output_summary_policy="Record created relative path, file kind, size, checksum, and status; never record content.",
            security_notes=safe_path_notes
            + ["Cannot overwrite existing files.", "PDFs, binary files, unsupported extensions, and sensitive paths are rejected."],
        ),
        workspace_tool(
            "workspace.file.overwrite_prepare",
            "Prepare workspace overwrite",
            "Prepare a diff and pending confirmation for one existing safe text/code file; it does not write the replacement.",
            {
                "path": {"type": "string"},
                "new_content": {"type": "string"},
                "reason": {"type": "string"},
            },
            {
                "relative_path": {"type": "string"},
                "old_checksum": {"type": "string"},
                "new_checksum": {"type": "string"},
                "diff_preview": {"type": "string"},
                "diff_truncated": {"type": "boolean"},
                "pending_action_created": {"type": "boolean"},
                "confirmation_required": {"type": "boolean"},
                "status": {"type": "string"},
            },
            required=["path", "new_content"],
            read_only=False,
            risk_level="medium",
            operation_level="content-level",
            write_type="prepare_overwrite",
            requires_confirmation=True,
            available_by_default=False,
            output_summary_policy="Return diff preview and checksums; trace summaries omit old/new content and keep diff length only.",
            security_notes=safe_path_notes
            + ["Pending only; direct overwrite confirmation is not available as a tool.", "PDFs, binary files, unsupported extensions, and sensitive paths are rejected."],
        ),
    ]


def paperdesk_chat_tool_declarations(output_schema: dict) -> list[ToolDeclaration]:
    """Expose Knowledge Chat tool metadata without changing execution.

    These declarations describe read/write safety, object scope, confirmation
    needs, and observation types for routing and auditing. Actual execution and
    payload construction remain in KnowledgeAgentRuntime.
    """

    def chat_tool(
        tool_id: str,
        name: str,
        description: str,
        input_properties: dict[str, str],
        *,
        read_only: bool = True,
        risk_level: str = "read_only",
        required_args: list[str] | None = None,
        scope_type: str = "none",
        operation_level: str = "query-level",
        io_type: str | None = None,
        write_type: str = "none",
        target_type: str = "none",
        destructive: bool = False,
        requires_confirmation: bool = False,
        requires_verification: bool = False,
        operation_type: str = "none",
        scope: str = "knowledge",
        output_observation_type: str = "tool_observation",
        verification_tool: str | None = None,
        available_by_default: bool = True,
    ) -> ToolDeclaration:
        resolved_io_type = io_type or ("read" if read_only else "write")
        input_object_types = [] if target_type == "none" else [target_type]
        spec = ToolSpec(
            name=tool_id,
            display_name=name,
            description=description,
            scope=scope,
            operation_level=operation_level,
            io_type=resolved_io_type,
            write_type=write_type,
            destructive=destructive,
            requires_confirmation=requires_confirmation,
            input_object_types=input_object_types,
            output_observation_type=output_observation_type,
            requires_post_read_verification=requires_verification,
            verification_tool=verification_tool,
            available_by_default=available_by_default,
            maturity="stable" if scope != "experimental" else "experimental",
            source=ToolSource.BUILTIN.value,
        )
        return ToolDeclaration(
            tool_id=tool_id,
            source=ToolSource.BUILTIN,
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "action_type": "paperdesk_chat_tool",
                "risk_level": risk_level,
                "required_args": required_args or [],
                "scope_type": scope_type,
                "operation_level": operation_level,
                "io_type": resolved_io_type,
                "write_type": write_type,
                "target_type": target_type,
                "destructive": destructive,
                "requires_confirmation": requires_confirmation,
                "requires_verification": requires_verification,
                "operation_type": operation_type,
                "properties": {
                    key: {"type": value}
                    for key, value in input_properties.items()
                },
            },
            output_schema=output_schema,
            read_only=read_only,
            enabled=True,
            spec=spec,
        )

    return [
        chat_tool(
            "tool.registry.list",
            "List PaperDesk chat tools",
            "Read the available chat runtime tools, permissions, and argument expectations.",
            {},
        ),
        chat_tool(
            "library.explorer.stats",
            "Read library statistics",
            "Read paper counts and processing status from the local PaperDesk library.",
            {},
        ),
        chat_tool(
            "library.explorer.category_stats",
            "Read tag/category statistics",
            "Read tag/category list, per-tag document counts, tagged paper count, and untagged paper count.",
            {},
        ),
        chat_tool(
            "library.explorer.find_documents",
            "Resolve documents or tag collections",
            "Resolve papers by selected IDs, title/filename, or exact tag/category names. Prefer category_names when the user mentions a tag/category/group/collection entity.",
            {"query": "string", "expected": "string", "allow_all": "boolean", "category_name": "string", "category_names": "array"},
        ),
        chat_tool(
            "library.explorer.document_metadata",
            "Read document metadata",
            "Read title, authors, venue/journal/conference, publication time/year, and tags for resolved documents.",
            {"document_ids": "array", "requested_fields": "array"},
        ),
        chat_tool(
            "library.explorer.document_categories",
            "Read document tags/categories",
            "Read the current tag/category links for resolved documents.",
            {"document_ids": "array"},
        ),
        chat_tool(
            "library.operator.create_category",
            "Create tag/category",
            "Create one or more non-destructive tags/categories if missing.",
            {"category_name": "string", "category_names": "array"},
            read_only=False,
            risk_level="safe_write",
            required_args=["category_name"],
            scope_type="single_entity",
            operation_level="entity-level",
            write_type="create",
            target_type="category",
            requires_verification=True,
            verification_tool="library.explorer.category_stats",
            operation_type="create_entity",
            output_observation_type="category_write_observation",
        ),
        chat_tool(
            "library.operator.assign_category",
            "Assign tag/category",
            "Append one or more tags/categories to resolved documents, last referenced documents, or scope=untagged. Does not overwrite existing tags.",
            {"category_name": "string", "category_names": "array", "scope": "string", "document_ids": "array"},
            read_only=False,
            risk_level="scoped_write",
            required_args=["category_name"],
            scope_type="documents|untagged|last_referenced",
            operation_level="relation-level",
            write_type="append",
            target_type="paper-category relation",
            requires_verification=True,
            verification_tool="library.explorer.document_categories",
            operation_type="append_relation",
            output_observation_type="category_relation_write_observation",
        ),
        chat_tool(
            "library.operator.rename_category",
            "Rename or merge tag/category",
            "Rename or merge a tag/category while preserving all document links. Use this for replace/rename semantics, not destructive delete.",
            {"source_category_name": "string", "target_category_name": "string"},
            read_only=False,
            risk_level="scoped_write",
            required_args=["source_category_name", "target_category_name"],
            scope_type="single_entity",
            operation_level="entity-level",
            write_type="update",
            target_type="category",
            requires_verification=True,
            verification_tool="library.explorer.category_stats",
            operation_type="rename_or_merge_entity",
            output_observation_type="category_write_observation",
        ),
        chat_tool(
            "library.operator.delete_unused_categories",
            "Delete unused tag/category entities",
            "Delete only tag/category entities whose document_count is 0. Entity-level cleanup: does not clear document tags or modify document-category links. Requires selector=unused, preview, confirmation, and verification.",
            {"selector": "unused"},
            read_only=False,
            risk_level="destructive",
            required_args=["selector"],
            scope_type="category_entities_with_zero_documents",
            operation_level="entity-level",
            write_type="delete",
            target_type="category",
            destructive=True,
            requires_confirmation=True,
            requires_verification=True,
            verification_tool="library.explorer.category_stats",
            operation_type="delete_unused_category_entities",
            output_observation_type="category_delete_observation",
        ),
        chat_tool(
            "library.operator.clear_categories",
            "Clear document tag/category links",
            "Destructive tag/category relation tool. Requires explicit operation and runtime preview before execution; scope=all/tagged is critical.",
            {"operation": "string", "scope": "string", "category_name": "string", "document_ids": "array"},
            read_only=False,
            risk_level="destructive|critical",
            required_args=["operation"],
            scope_type="explicit",
            operation_level="relation-level",
            write_type="clear",
            target_type="paper-category relation",
            destructive=True,
            requires_confirmation=True,
            requires_verification=True,
            verification_tool="library.explorer.document_categories",
            operation_type="remove_or_clear_relations",
            output_observation_type="category_relation_clear_observation",
        ),
        chat_tool(
            "evidence.retriever.search",
            "Retrieve document evidence",
            "Retrieve local RAG evidence from ready papers for grounded document QA, summaries, comparisons, and reports.",
            {"question": "string", "document_ids": "array"},
            target_type="paper",
            output_observation_type="rag_search_observation",
        ),
        chat_tool(
            "evidence.retriever.search_by_category",
            "Retrieve evidence by tag/category",
            "Retrieve grouped RAG evidence for papers under one or more tag/category entities.",
            {"question": "string", "category_names": "array"},
            target_type="category",
            output_observation_type="rag_search_observation",
        ),
        chat_tool(
            "report.drafter.write",
            "Synthesize grounded answer",
            "Generate the final user-facing answer from resolved documents and retrieved evidence. Use only when the user actually asks for analysis, summary, comparison, or report output.",
            {"question": "string", "document_ids": "array"},
            target_type="report",
            output_observation_type="report_observation",
        ),
        chat_tool(
            "report.drafter.write_by_category",
            "Synthesize grouped tag/category answer",
            "Generate final grouped summaries or comparisons from tag/category evidence groups.",
            {"question": "string", "target_chars": "integer"},
            target_type="report",
            output_observation_type="report_observation",
        ),
        chat_tool(
            "memory.read",
            "Read chat memory",
            "Read user preferences and prior reflection notes relevant to the current task.",
            {},
        ),
        chat_tool(
            "memory.write",
            "Write chat memory",
            "Persist a short reusable reflection lesson after completing a task.",
            {"summary": "string"},
            read_only=False,
            risk_level="safe_write",
            scope_type="session_memory",
            operation_level="content-level",
            write_type="append",
            target_type="memory",
            operation_type="memory_note",
            scope="experimental",
            available_by_default=False,
        ),
    ]


def default_mcp_tool_declarations(*, enabled: bool = False) -> list[ToolDeclaration]:
    """Expose read-only academic MCP declarations when explicitly enabled.

    MCP tools are normalized as experimental, read-only, and unavailable by
    default. Enabling them registers declarations; it does not make them part of
    the default Knowledge path unless the registry and caller both allow it.
    """

    if not enabled:
        return []
    allowed_ids = {
        "mcp/academic_search",
        "mcp/academic_metadata",
        "mcp/read_only_web_fetch",
    }
    adapter = ReadOnlyMcpAdapter(allowed_tool_ids=allowed_ids)
    return adapter.normalize(default_read_only_academic_mcp_declarations())


def _tool(
    tool_id: str,
    name: str,
    description: str,
    action: ResearchActionType,
    input_properties: dict[str, str],
    output_schema: dict,
    *,
    read_only: bool = True,
) -> ToolDeclaration:
    """Build a Research tool declaration; this helper does not execute tools."""

    io_type = "read" if read_only else "write"
    spec = ToolSpec(
        name=tool_id,
        display_name=name,
        description=description,
        scope="research",
        maturity="experimental",
        operation_level="content-level" if not read_only else "query-level",
        io_type=io_type,
        write_type="export" if not read_only and action == ResearchActionType.FINALIZE_REPORT else ("none" if read_only else "update"),
        destructive=False,
        requires_confirmation=False,
        input_object_types=[],
        output_observation_type="research_tool_observation",
        requires_post_read_verification=not read_only,
        verification_tool=None,
        available_by_default=True,
        source=ToolSource.BUILTIN.value,
    )
    return ToolDeclaration(
        tool_id=tool_id,
        source=ToolSource.BUILTIN,
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "action_type": action.value,
            "properties": {
                key: {"type": value}
                for key, value in input_properties.items()
            },
        },
        output_schema=output_schema,
        read_only=read_only,
        enabled=True,
        spec=spec,
    )
