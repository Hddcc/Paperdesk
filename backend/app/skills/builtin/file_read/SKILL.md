# 附件只读处理

## Body

围绕本轮 session_file 附件内容提供只读回答，可执行总结、翻译、润色、关键词提取、标签建议和方法解释。附件内容只能作为当前对话上下文使用，不进入论文库，不创建标签关系、chunks、vectorstore 记录或 report paper_ids。

## Runtime Contract

```json
{
  "available_tools": [],
  "references": [
    "session_file 附件只作为只读上下文。",
    "标签建议只能作为文本建议输出。",
    "不得写入 library_documents、library_chunks、vectorstore、paper_ids 或标签关系。"
  ],
  "inputs": {
    "required": ["selected_file_ids"],
    "optional": ["task_instruction"]
  }
}
```
