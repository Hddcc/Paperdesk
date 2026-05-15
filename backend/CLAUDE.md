# Backend Rules

## 主要服务

- `DocumentLibraryService` 负责 PDF 上传、去重、后台索引。
- `RagService` 负责在指定文档集合中检索证据。
- `ResearchOrchestrator` 负责研究任务运行、checkpoint、报告生成和 SSE 事件。
- `ChatService` 负责聊天页会话、附件、记忆、主模型回答。

## 聊天页原则

- 聊天主流程默认可用；Milvus 失败时，只影响知识库增强状态，不应该直接让聊天接口返回 500。
- 聊天消息、会话、附件、记忆记录和刷新日志走 SQLite 仓储层。
- 长上下文摘要和偏好缓存可写入运行时上下文目录；该目录必须保持在 git ignore 范围内。
- 记忆只存摘要、偏好、引用索引；真正回答前必须重新读取当前文档或记录校验。

## 数据与存储

- SQLite 是事实来源，schema 由 `app/repositories/base.py` 初始化。
- 聊天会话、聊天消息、附件、记忆记录、记忆刷新日志都走仓储层，不要在服务里直接写 SQL。
- 库内论文仍使用 Milvus 作为正式向量检索后端。
- Chroma 和早期 stub vectorstore 不属于当前正式后端。

## Agent 运行时

- `MainAgentRuntime`、`MessageBus`、`ToolRegistry`、`SkillRegistry` 是研究任务运行时的长期结构。
- `SubagentRunner` 和 MCP adapter 属于可扩展骨架；除非实际接入主流程，否则文档中不要描述为完整多 Agent 并发或真实外部 MCP 执行。
- 任何新增工具调用都必须有失败收束路径，避免同一工具无进展重复调用。

## 错误处理

- 对前端返回业务可读的错误，不要把长栈直接返回给用户。
- 图片能力如果模型不支持，要明确告诉前端“当前模型未开启图片理解或图片请求失败”。
