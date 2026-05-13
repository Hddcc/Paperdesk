# 单篇论文总结

## Body

优先使用用户指定的本地论文证据，按论文主题、研究问题、核心方法、主要贡献、结果结论、局限性和适用场景收束。证据不足时必须说明总结范围受限。

## Runtime Contract

```json
{
  "available_tools": [
    "plan/rule_based_initial",
    "search_local/vector_recall_default",
    "summarize_evidence/task_level_merge",
    "summarize_evidence/degraded_closeout",
    "finalize_report/task_artifact_writer",
    "finish/runtime_complete",
    "fail/runtime_stop"
  ],
  "references": [
    "只基于当前可检索片段总结，不伪造全文实验细节。",
    "需要保留本地页码或引用线索。",
    "不默认补在线检索，除非路由显式要求。"
  ],
  "inputs": {
    "required": ["topic", "selected_document_ids"],
    "optional": ["notes"]
  }
}
```
