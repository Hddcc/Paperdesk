# PaperDesk

PaperDesk 是一个面向科研论文阅读、知识库问答、研究任务编排和报告沉淀的本地优先项目。当前代码以 FastAPI、SQLite、Milvus、Vue 3 和 Pinia 组成一套可运行的论文助手，而不是只展示单点 Agent demo。

本仓库的文档按“当前真实能力”描述。已经保留但尚未完全接入主流程的 Agent 扩展点，会明确标注为运行时骨架或后续扩展点。

## 当前产品入口

- `/knowledge`：聊天式主页，支持会话、附件、选中库内论文和知识库增强回答。
- `/library`：本地论文库，负责 PDF 上传、分类、解析、索引和 PDF 打开入口。
- `/research`：PDF 阅读区。这里不再是旧的研究任务工作台，也不恢复三栏任务面板。
- `/reports`：历史报告列表、报告预览和 Markdown 导出。

建议体验顺序：

1. 在 `Library` 上传 PDF。
2. 等文档状态进入可用状态。
3. 在 `Knowledge` 选择论文并提问。
4. 在 `Reports` 查看已生成的历史报告。
5. 从 `Library` 点击论文名称，在 `Research` 阅读 PDF。

## 启动方式

后端：

```bash
cd backend
uv sync
uv run uvicorn app.api.main:app --reload --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

常用校验：

```bash
cd backend
.\.venv\Scripts\pytest.exe -q
```

```bash
cd frontend
npm run build
```

配置示例见 `backend/.env.example` 和 `frontend/.env.example`。

## 后端结构

- `backend/app/api`：FastAPI 入口、依赖装配和路由注册。
- `backend/app/repositories`：SQLite 仓储层，是业务事实来源。
- `backend/app/services`：PDF 解析、建库、RAG、聊天上下文、研究任务编排和报告导出。
- `backend/app/agents`：Planner、Search、Summarizer、ReportWriter 等任务能力封装。
- `backend/app/runtime`：主 Agent loop、工具注册、skill registry、MCP adapter、消息总线和 subagent runner。
- `backend/app/vectorstores`：当前正式向量检索后端是 Milvus。

当前 PDF 建库链路是：

1. 文档元数据写入 SQLite。
2. `PdfParser` 解析 PDF 正文。
3. `TextChunker` 切分 chunk。
4. `EmbeddingService` 生成向量。
5. `KnowledgeIngestionService` 写入 Milvus。
6. chunk 元数据同步写入 SQLite。
7. 文档状态更新为可用。

## Agent 能力边界

当前已接入主流程的能力：

- `ResearchTaskRouter` 会根据主题、输入模式和选中文档判断任务类型与证据策略。
- `MainAgentRuntime` 维护主循环状态，并带有步数、重复工具、无进展和重规划次数限制。
- `ResearchOrchestrator` 负责 research run、checkpoint、SSE 事件、任务总结和报告生成。
- `SkillRegistry` 会装载后端内置 skill manifest，用作任务类型与工具边界说明。

当前属于运行时骨架、尚未等同于完整外部系统的能力：

- `SubagentRunner` 已实现并发 worker 的注册、进度、结果和失败记录，但主研究流程仍以 orchestrator 顺序编排为主。
- MCP adapter 当前负责只读工具声明过滤和注册，并不代表已经接入完整外部 MCP server 执行链路。
- `MessageBus` 和 runtime repository 可记录 trace、notification、artifact；实际事件丰富度取决于 orchestrator 调用点。

文档和代码维护时，不要把“已有骨架”写成“已完整接入主流程”。

## 记忆与上下文

聊天消息、会话、附件、记忆记录和刷新日志存入 SQLite 仓储层。

长上下文摘要、用户偏好缓存和 compact 文件由 `ContextFileStore` 管理，默认写入：

```text
backend/runtime/context
```

这个目录属于运行时文件，应保持在 git ignore 范围内。旧的 `CLAUDE_DIR` 环境变量仍可作为兼容入口使用，但 `CLAUDE.md` 文件本身只应记录开发协作规则，不作为业务记忆文档。

上下文预算由 `ContextBudgetService` 管理；超预算时优先压缩证据，再压缩历史消息，最后保留最近若干轮原始对话。

## 维护规则

- 删除代码前先确认全仓无业务引用、路由无依赖、测试或构建不依赖。
- 当前不要恢复旧 Research Workbench。`/research` 的定位是 PDF 阅读。
- Milvus 是正式向量库；不要重新引入 Chroma 或早期 stub vectorstore，除非先完成架构决策。
- 前端整理应避免视觉重排；只删除不可达组件或修正文案。
- Agent、MCP、skills、memory 的描述必须和当前实现状态一致。
