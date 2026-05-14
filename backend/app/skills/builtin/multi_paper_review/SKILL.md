# 多篇论文综述

## Body

围绕研究主题拆分任务，结合在线论文候选和本地知识库证据，按综述脉络、关键研究方向、趋势归纳、证据来源和局限问题收束。

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
    "继续使用单主 Agent 主控，不新增 worker。",
    "不要重做阶段 19 的综述型 Markdown builder。",
    "报告阶段不新增事实，只整合任务总结和已收集证据。"
  ],
  "inputs": {
    "required": ["topic"],
    "optional": ["notes", "selected_document_ids", "search_provider", "top_k_online", "top_k_local"]
  }
}
```
