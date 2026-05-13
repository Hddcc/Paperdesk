# 方法解释

## Body

围绕用户提出的方法或概念组织解释，优先使用本地材料，必要时补充在线证据。输出必须覆盖概念定义、方法流程、适用场景、证据来源和局限注意事项。

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
    "finalize_report/report_writer_default",
    "finalize_report/task_artifact_writer",
    "finish/runtime_complete",
    "fail/runtime_stop"
  ],
  "references": [
    "解释要服务用户当前问题，不扩展成教材章节。",
    "方法优缺点必须回到证据或明确标为初步判断。",
    "不向前端暴露 skill 或 MCP 信息。"
  ],
  "inputs": {
    "required": ["topic"],
    "optional": ["notes", "selected_document_ids", "search_provider"]
  }
}
```
