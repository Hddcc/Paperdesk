"""Unified backend tool declarations for research capabilities."""

from __future__ import annotations

from collections.abc import Iterable

from app.models import ResearchActionType, ToolDeclaration, ToolSource


class ToolRegistry:
    """Register builtin and external tool declarations behind one lookup API."""

    def __init__(self, declarations: Iterable[ToolDeclaration] | None = None) -> None:
        self._tools: dict[str, ToolDeclaration] = {}
        for declaration in builtin_tool_declarations():
            self.register(declaration)
        for declaration in declarations or []:
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

    def label_for(self, tool_id: str, fallback: str | None = None) -> str:
        tool = self.get(tool_id)
        if tool is None:
            return fallback or tool_id
        return tool.name


def builtin_tool_declarations() -> list[ToolDeclaration]:
    """Declare current builtin research tools without changing execution logic."""

    common_output = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "payload": {"type": "object"},
        },
    }
    return [
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
    )
