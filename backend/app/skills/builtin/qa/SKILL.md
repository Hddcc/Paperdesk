# 知识问答

## Body

优先围绕用户问题组织直接答案，先读取可用本地证据，再在路由要求时补充在线论文候选。输出必须区分直接答案、关键证据、必要引用和结论边界。

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
    "回答不得把弱证据写成强结论。",
    "引用表达沿用阶段 19 的证据来源与引用映射规则。",
    "前端不展示 skill 名称或工具来源。"
  ],
  "inputs": {
    "required": ["topic"],
    "optional": ["notes", "selected_document_ids", "search_provider"]
  }
}
```
