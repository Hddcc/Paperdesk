# 对比分析

## Body

先明确对比对象和对比维度，再分别检索本地与在线证据。输出必须覆盖各对象表现、共性、差异、适用建议和结论边界。

## Runtime Contract

```json
{
  "available_tools": [
    "plan/rule_based_initial",
    "search_local/vector_recall_default",
    "search_online/mixed_broad_recall",
    "search_online/openalex_primary",
    "search_online/arxiv_primary",
    "mcp/academic_search",
    "summarize_evidence/task_level_merge",
    "summarize_evidence/degraded_closeout",
    "revise_plan/rewrite_query",
    "revise_plan/split_task",
    "revise_plan/reorder_priority",
    "finalize_report/report_writer_default",
    "finish/runtime_complete",
    "fail/runtime_stop"
  ],
  "references": [
    "对比维度应来自用户目标或任务证据，不额外发散。",
    "无证据对象不能强行给出胜负判断。",
    "结论边界沿用阶段 19 的弱证据表达。"
  ],
  "inputs": {
    "required": ["topic"],
    "optional": ["notes", "selected_document_ids", "search_provider"]
  }
}
```
