# PaperDesk 论文阅读 Agent

PaperDesk 是一个面向论文阅读、论文库管理和研究写作的本地 AI Agent 应用。项目围绕论文业务闭环构建，同时把 Agent 执行框架抽象为可复用的 `Agent Core`，让后续接入 draw.io、token 消耗统计、外部知识源、MCP 工具等能力时有清晰的扩展位置。

项目核心结构采用 `Agent Core + Domain Pack + Infrastructure Adapter`：

- `Agent Core`：负责路由决策、上下文装配、记忆管理、运行时分发、工具治理、Skills 选择、写操作安全和运行追踪。
- `Domain Pack`：承载论文、工作区、报告等业务能力，避免把业务逻辑堆在单一 ChatService 中。
- `Infrastructure Adapter`：隔离模型、Embedding、向量库、文件系统、第三方 API 等外部依赖。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3, TypeScript, Vite |
| 后端 | FastAPI, Pydantic, SQLite |
| Agent | Lifecycle, Capability Registry, Runtime Dispatcher, Tool Registry, Skills, Safety, Trace |
| RAG | PDF parsing, chunking, embedding, Milvus, metadata filter, evidence assembly |
| 工程能力 | SSE streaming, write confirmation, route trace, runtime metrics, modular domain packs |

## 项目截图

| 对话与 RAG | 论文库与标签 |
| --- | --- |
| ![对话与 RAG](docs/images/readme/chat-rag.png) | ![论文库与标签](docs/images/readme/library-tags.png) |

| 报告保存 | Skills 与会话文件 |
| --- | --- |
| ![报告保存](docs/images/readme/report-save.png) | ![Skills 与会话文件](docs/images/readme/skills-files.png) |

## 核心能力

- 论文上传与解析：上传 PDF 后解析文本、切分 chunk、写入论文库并建立向量索引。
- 论文库问答：支持全库检索、选中文档范围、metadata 过滤、证据拼接和引用信息。
- 选中文章综述：根据当前选择的论文生成总结、对比、方法解释和综述草稿。
- 标签与分类管理：支持查看、创建、更新和分配分类，写操作需要明确 scope。
- 报告保存与导出：可将聊天中的研究结果保存为报告，并导出 Markdown。
- 普通聊天：无论文意图时走轻量 direct chat，不强行进入 RAG 或工具链。
- 工作区文件：支持会话文件上传、工作区文件读取、生成文件和覆盖确认。
- 自定义 Skills：支持内置和用户自定义 skill，允许绑定工具，并受 route、capability、tool policy 约束。
- Trace 与调试：记录 route、capability、skill、context、tool policy、RAG evidence、runtime、错误原因等信息。

## 架构设计

```text
Frontend Vue
  -> FastAPI Routes
  -> Application Use Cases
  -> Agent Core
       Ingress
       Route Decision
       Capability Resolution
       Skill Selection
       Context Assembly
       Memory Snapshot
       Tool Policy
       Runtime Dispatch
       Response Recorder
       Trace
  -> Domain Packs
       Paper Domain
       Workspace Domain
       Artifact Domain
  -> Infrastructure Adapters
       LLM / Embedding / VectorStore / Files / External APIs
```

PaperDesk 的请求不会直接散落到各个业务 service 中。所有聊天类请求先进入 Agent Core，由 Agent Core 判断请求类型、当前上下文、可用能力和工具边界，再分发到对应 runtime。论文业务、工作区业务和报告业务通过 domain pack 暴露稳定能力，底层模型和存储细节由 infrastructure adapter 管理。

## 执行链路

```text
User Request
  -> Agent Ingress
  -> RouteDecisionPacket
  -> CapabilityDeclaration
  -> Skill Selection
  -> AgentContext
  -> MemorySnapshot
  -> ToolPolicyDecision
  -> Runtime Executor
  -> RuntimeMetricsEnvelope
  -> Stream / Response
```

关键对象：

- `RouteDecisionPacket`：记录 route、runtime、编排策略、scope、RAG/tool/confirmation 需求。
- `CapabilityDeclaration`：声明能力 id、routes、tools、domain package、infrastructure 依赖和文档摘要。
- `AgentContext`：聚合最近对话、选中文档、检索证据、会话文件、当前操作范围和 token 预算。
- `MemorySnapshot`：保存会话摘要、用户偏好、最近任务状态，作为轻量记忆输入。
- `ToolPolicyDecision`：根据 route、capability、skill、scope、risk、确认状态过滤工具。
- `RuntimeMetricsEnvelope`：记录 route、runtime、capability、证据数、工具数、token 可用状态和错误信息。

## 上下文管理

PaperDesk 的上下文管理目标是让模型拿到当前任务需要的信息，同时避免普通聊天被论文库和工具链拖重。

```text
Recent Messages
  + Active Paper Scope
  + Retrieved Evidence
  + Session Files
  + Workspace Scope
  + Pending Action
  + Token Budget
  -> AgentContext
```

上下文由 Agent Core 统一装配：

- 最近对话：保留当前会话的短期上下文，用于延续普通聊天和多轮论文问答。
- 选中文档：记录用户当前选择的论文范围，RAG 召回和写操作默认只在明确 scope 内进行。
- 检索证据：只在 `paper_rag` 或相关 route 中注入 evidence，普通聊天不默认拼接论文内容。
- 会话文件：用户上传到当前会话的文件独立进入上下文，和论文库文档保持边界。
- 工作区范围：文件读写操作需要明确路径和操作意图，防止模糊指令扩大影响面。
- Token 控制：上下文装配阶段会控制消息、证据和文件片段数量，优先保留当前任务相关信息。

## 记忆管理

PaperDesk 使用轻量记忆，不引入复杂长期人格记忆。当前记忆重点服务于论文阅读和研究任务连续性。

```text
Session Summary
  + User Preference
  + Recent Task State
  + Active Paper Scope
  -> MemorySnapshot
```

记忆分为三类：

- 会话摘要：当对话变长时，将历史交流压缩为摘要，用于维持研究任务的连续性。
- 用户偏好：记录稳定偏好，例如回答语言、引用格式、综述风格、输出结构。
- 最近任务状态：记录刚执行过的查询、报告保存、标签操作、待确认写操作和当前论文范围。

写操作相关状态会通过 pending action 管理。需要修改标签、分类、删除、覆盖、保存报告时，系统先生成预览并等待用户确认，确认后再执行。

## 编排策略

每次请求选择一个主 runtime 和一个主编排策略，保持执行链路可解释、可调试、可扩展。

| 用户意图 | Route | 主策略 | Runtime |
| --- | --- | --- | --- |
| 普通聊天 | `direct_chat` | single-turn | `DirectChatRuntime` |
| 论文问答、总结、对比 | `paper_rag` | retrieve-then-synthesize | `PaperRagRuntime` |
| 论文库只读查询 | `library_read` / `tool_action` | bounded-react | `ToolActionRuntime` |
| 标签、分类、删除、覆盖等写操作 | `write_pending` / `write_confirmed` | preview-confirm-execute-verify | `ToolActionRuntime` / `ConfirmedWriteRuntime` |
| 报告保存和导出 | `report_action` | service-workflow | `ReportActionRuntime` |
| 工作区文件 | `workspace_read` / `workspace_write` | service-workflow / preview-confirm | `WorkspaceActionRuntime` |
| 实验研究能力 | `experimental_research` | plan-execute-replan | `ExperimentalRuntime` |

Planner、reflection、MCP、subagent、research-task runtime 放在 experimental runtime 中，作为可选研究能力保留。

## RAG 设计

```text
PDF Upload
  -> Text Parse
  -> Chunk Split
  -> Embedding
  -> Vector Index
  -> Metadata Filter
  -> Evidence Recall
  -> Answer Synthesis
  -> Citation / Trace
```

论文 RAG 保留论文场景最关键的能力：

- PDF 解析：提取页面文本和基础 metadata。
- Chunk 切分：按页面和段落生成可召回片段，保留页码、标题、文档 id、版本等 metadata。
- Embedding：通过 infrastructure LLM/embedding 边界生成向量。
- 向量库：使用 Milvus 执行向量召回。
- 范围控制：支持选中文档、文档 id、分类、标签等过滤。
- 证据拼接：将文本片段、页码、标题和文档信息拼入模型上下文。
- 输出边界：证据不足时明确说明，不把无证据内容包装成确定结论。

## Skills

Skills 支持内置和用户自定义。一个 skill 可以声明触发方式、适用 route、需要的 capability、允许绑定的工具和输出协议。

```json
{
  "skill_id": "custom_review",
  "name": "自定义论文评审",
  "enabled": true,
  "capability_ids": ["paper"],
  "allowed_tool_ids": ["search_local/vector_recall_default"],
  "trigger": {
    "keywords": ["custom-review"],
    "routes": ["paper_rag"],
    "capability_ids": ["paper"]
  }
}
```

Skill 负责表达“用户想让 Agent 用什么方式工作”。真正暴露给 runtime 的工具仍会经过 Tool Registry、route、capability、scope、risk、feature flag 和确认状态过滤。

## Tool Registry 与写安全

所有工具通过统一 Registry 声明 metadata：

- tool id、描述、schema；
- capability id、scope、integration source；
- read/write 类型；
- operation level；
- risk level；
- destructive 标记；
- confirmation requirement；
- verification 和 observation 规则。

写操作统一走：

```text
preview -> pending action -> explicit confirmation -> execute -> optional verification
```

安全规则：

- 模糊指代不会默认执行全库写操作。
- 删除、覆盖、清空、批量改标签必须有明确 scope。
- 未确认前不暴露危险写工具。
- 工具结果统一回灌为结构化 observation，便于 trace 和调试。

## 可观测性

Agent 运行过程记录轻量 trace 和 metrics：

- route、runtime、orchestration pattern；
- active capability；
- active skill；
- context scope；
- allowed/filtered tools；
- RAG evidence count 与 metadata filter；
- pending action 与写操作 scope；
- response status、error reason；
- token usage availability。

当模型供应商没有返回 token 信息时，系统记录 `token_usage_available=false`，请求仍可正常完成。

## 扩展能力

新增能力优先以 capability 接入：

```text
Capability Declaration
  -> Domain Pack
  -> Tool Metadata
  -> Runtime Binding
  -> API / UI Entry
  -> README / Docs
```

示例方向：

- `drawio`：新增图形产物 domain 或复用 artifact domain，在 integration adapter 中接入 draw.io 能力，在 Tool Registry 声明创建、编辑、导出工具。
- `token_usage`：在 observability 中扩展 usage 聚合，在 Tool Registry 声明只读查询工具，通过 API 暴露会话维度的 token、latency、cost 信息。

## 快速启动

后端：

```bash
cd backend
uv sync
.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

默认访问：

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:8000
```

## API 概览

| 能力 | API |
| --- | --- |
| 会话列表 | `GET /api/chat/sessions` |
| 创建会话 | `POST /api/chat/sessions` |
| 发送消息 | `POST /api/chat/sessions/{session_id}/messages` |
| SSE 流式消息 | `POST /api/chat/sessions/{session_id}/messages/stream` |
| 上传论文 | `POST /api/documents/upload` |
| 文档列表 | `GET /api/documents` |
| RAG 问答 | `POST /api/rag/ask` |
| 报告列表 | `GET /api/reports` |
| 保存报告 | `POST /api/reports/from-message` |
| Workbench trace | `GET /api/workbench/messages/{message_id}/trace` |

## 验证

```bash
backend\.venv\Scripts\python.exe -m compileall -q backend\app backend\tests
backend\.venv\Scripts\pytest.exe -q backend\tests
cd frontend
npm run build
```

## 联系

项目维护者：Hddcc

GitHub：<https://github.com/Hddcc/Paperdesk>
