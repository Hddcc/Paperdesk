# PaperDesk 论文阅读 Agent

> 面向论文阅读、资料整理和研究报告生成的轻量级 AI Agent 应用。

`FastAPI` `Vue 3` `TypeScript` `RAG` `Tool Registry` `Skills` `Write Safety` `Agent Runtime`

PaperDesk 以论文业务闭环为核心：上传 PDF、解析入库、切分向量化、论文库管理、选中文章问答、综述对比、标签分类、报告保存和普通聊天。项目在此基础上加入轻量 Agent 工程能力，包括路由决策、运行时编排、工具治理、用户自定义 Skills 和安全写操作。

## 项目截图

| 聊天与论文问答 | 论文库与标签管理 |
| --- | --- |
| ![聊天与论文问答](docs/images/readme/chat-rag.png) | ![论文库与标签管理](docs/images/readme/library-tags.png) |

| 报告保存 | 本地文件与 Skills |
| --- | --- |
| ![报告保存](docs/images/readme/report-save.png) | ![本地文件与 Skills](docs/images/readme/skills-files.png) |

## 核心能力

- 论文上传与解析：PDF 文本解析、chunk 切分、embedding 生成、向量库写入。
- 论文库管理：论文状态、元数据、标签、分类、选中文档范围。
- RAG 问答：向量召回、metadata 过滤、选中文档范围、证据拼接、引用生成。
- 论文综述：支持围绕选中文章做总结、对比、方法分析和证据归纳。
- 普通聊天：无论文意图时走轻量 single-turn 聊天路径。
- 报告保存：将回答或综述保存为报告，支持后续导出。
- Skills 扩展：支持内置 Skills 和用户自定义 Skills，可声明触发条件与绑定工具。
- Tool Registry：统一声明工具 schema、读写类型、风险等级、确认要求和验证策略。
- 写操作安全：标签、分类、删除、覆盖、报告保存等写操作必须有明确 scope 和确认流程。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Vue 3、TypeScript、Vite |
| 后端 | Python、FastAPI、Pydantic、Uvicorn |
| Agent 核心 | Lifecycle Service、Route Decision、Runtime Dispatcher、Tool Policy、Skills |
| RAG | PDF Parser、Text Chunker、Embedding Service、Vector Store、Evidence Assembly |
| 数据与文件 | 本地仓储、论文文件、报告文件、向量索引 |

## 架构总览

```text
Chat API
  -> ChatService
  -> AgentLifecycleService
       -> Ingress
       -> Route Decision
       -> Skill Selection
       -> Context Assembly
       -> Tool Policy
       -> Runtime Dispatch
            -> DirectChatRuntime
            -> PaperRagRuntime
            -> ToolActionRuntime
            -> ConfirmedWriteRuntime
            -> ReportActionRuntime
            -> WorkspaceActionRuntime
            -> ExperimentalRuntime
  -> Chat Response
```

Agent 入口位于 `backend/app/agent`：

```text
backend/app/agent/lifecycle      生命周期、路由、上下文、工具策略
backend/app/agent/runtimes       路由运行时与 executor
backend/app/agent/tools          Tool Registry 与 Tool Policy
backend/app/agent/skills         Skills 注册、选择、工具绑定
```

## Agent 编排策略

PaperDesk 支持多种 Agent 编排模式，每个请求选择一个主模式，保证普通问题保持轻量，论文任务保留必要的业务能力。

| 请求类型 | 主编排模式 | Runtime | 说明 |
| --- | --- | --- | --- |
| 普通聊天 | `single-turn` | `DirectChatRuntime` | 一轮模型回答，不进入论文库和工具循环 |
| 论文问答 | `retrieve-then-synthesize` | `PaperRagRuntime` | 检索证据后生成答案和引用 |
| 论文库查询 | `bounded-react` | `ToolActionRuntime` | 有最大步数、结构化 observation、明确停止原因 |
| 写操作 | `preview-confirm-execute-verify` | `ToolActionRuntime` / `ConfirmedWriteRuntime` | 预览、挂起、确认、执行、验证 |
| 报告/工作区 | `service-workflow` | `ReportActionRuntime` / `WorkspaceActionRuntime` | 确定性服务编排 |
| 实验研究任务 | `plan-execute-replan` | `ExperimentalRuntime` | planner、reflection、MCP、subagent 等显式实验能力 |

## RAG 设计

```text
PDF
  -> parse text
  -> chunk
  -> embed
  -> vector index
  -> selected scope / metadata filter
  -> retrieve evidence
  -> synthesize answer
  -> citations
```

默认 RAG 保留论文场景最需要的能力：选中文档范围、metadata 过滤、证据去重、引用拼接。复杂 rerank、多跳检索、外部 Web RAG 等能力可以放在明确工作流中扩展。

## Skills 与工具绑定

用户自定义 Skill 采用本地目录格式：

```text
<custom-skills-root>/<skill_id>/
  manifest.json
  SKILL.md
```

`manifest.json` 可声明：

- 支持任务类型；
- 触发关键词、命令、意图提示；
- 输出协议；
- `allowed_tool_ids` 工具绑定；
- 是否默认启用；
- maturity、scope、source 等元数据。

Skill 可以表达希望使用哪些工具。实际工具暴露仍由 route、scope、风险等级、feature flag、确认状态和 Tool Registry 共同决定。

## Tool Registry 与写安全

Tool Registry 统一声明：

- tool id、描述、输入输出 schema；
- read/write 类型；
- operation level；
- risk level；
- 是否 destructive；
- 是否需要确认；
- 验证工具或验证策略。

写操作遵循：

```text
intent
  -> scoped preview
  -> pending action
  -> user confirmation
  -> execute
  -> verify
```

模糊指代会进入澄清或非执行预览，系统不会把“删除/清空/覆盖/保存”这类指令默认扩展到全库。

## 快速启动

### 后端

```bash
cd backend
uv sync
copy .env.example .env
uv run uvicorn app.api.main:app --reload
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

## API 概览

| 模块 | 说明 |
| --- | --- |
| Chat API | 会话、消息、普通聊天、论文问答 |
| Library API | 论文上传、论文库、标签分类 |
| Report API | 报告保存、列表、导出 |
| Workspace API | 普通文件上下文、工作区读写 |

## 项目结构

```text
backend/app/agent          Agent 生命周期、运行时、工具、Skills
backend/app/api            FastAPI 路由
backend/app/models         API、论文、Agent lifecycle 数据模型
backend/app/services       业务服务与兼容服务
backend/app/runtime        Agent runtime、registry、planner、实验能力
backend/app/skills         内置 Skills
backend/app/repositories   会话、文件、论文库仓储
frontend                   Vue 前端
docs/images/readme         README 展示截图
```

## 简历亮点写法

- 基于运行时优先架构设计论文阅读 Agent，将请求统一收敛到 Ingress、Route、Context、Tool Policy、Runtime 的执行链路。
- 设计多编排策略映射：普通聊天使用 single-turn，论文 RAG 使用 retrieve-then-synthesize，工具查询使用 bounded ReAct，写操作使用 preview-confirm-execute-verify。
- 构建 Tool Registry 与 Skill Registry，支持用户自定义 Skills 绑定工具，并通过 route、scope、risk、confirmation 进行二次权限过滤。
- 针对论文场景实现轻量 RAG 链路，支持 chunk 切分、向量召回、metadata 过滤、选中文档范围、证据拼接和引用生成。
- 设计写操作安全边界，对标签、分类、删除、覆盖、报告保存等操作进行 scope 校验、pending confirmation 和执行验证。

## 联系方式

- 邮箱：31210118@qq.com
