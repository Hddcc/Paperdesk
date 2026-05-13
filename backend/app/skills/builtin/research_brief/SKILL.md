# 研究路线建议

## Body

围绕主题归纳研究方向和可执行路线，结合在线论文和本地证据给出建议。输出必须说明可用证据、路线建议、后续验证和结论边界。

## Runtime Contract

```json
{
  "available_tools": [
    "plan/rule_based_initial",
    "search_local/vector_recall_default",
    "search_online/mixed_broad_recall",
    "search_online/openalex_primary",
    "search_online/arxiv_primary",
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
    "路线建议应以证据覆盖范围为边界。",
    "不要把启发式建议写成已验证结论。",
    "保留后续验证事项，方便用户继续推进。"
  ],
  "inputs": {
    "required": ["topic"],
    "optional": ["notes", "selected_document_ids", "search_provider"]
  }
}
```
