# PaperDesk

PaperDesk 是一个面向科研场景的论文工作台项目，围绕“上传论文、整理材料、发起研究、沉淀结果”这一条完整链路来设计。它把本地论文库、RAG 问答、研究任务编排、报告生成和前端工作台放进同一个系统里，适合作为一个论文助手项目来使用，也适合作为一个 Agent 工程案例来阅读和继续扩展。

如果你正在学习 `helloagent` 一类项目，想继续往前走到“如何把记忆、上下文管理、RAG、Skill、MCP、主循环决策和前端工作台串成一个完整应用”，PaperDesk 就是在回答这件事。

你可以把它理解成三个连续问题：

- 本地 PDF 如何建库，才能支持页码级引用、连续追问和后续研究任务
- 聊天式知识库如何做记忆管理和上下文压缩，避免对话越长越乱
- 研究任务如何从一句自然语言开始，被路由、拆分、补证据、汇总并最终生成报告

---

## 1. 这个项目解决什么问题

在论文阅读场景里，用户的需求通常不是一次性的。

最开始可能只是上传几篇 PDF，问一个问题；接着会想做总结、对比、综述；最后又希望把过程沉淀成可回看的报告。PaperDesk 的目标，就是把这些原本分散的动作放进一个连续工作流里，让“问答”和“研究”不再是两套割裂的系统。

围绕这个目标，PaperDesk 提供四条主链路：

### 1.1 本地论文库

- 上传 PDF
- 解析标题、页数和正文
- 切分 chunk
- 生成向量
- 写入 Milvus
- 保存文档、chunk 和分类信息

### 1.2 知识库问答

- 在本地论文库上直接提问
- 支持选择指定论文范围
- 支持聊天态连续追问
- 回答中保留来源、页码和引用标签

### 1.3 研究工作台

- 从一句主题发起研究
- 自动路由成总结、综述、对比、问答、方法解释或研究 brief
- 在本地知识库与在线论文之间补充证据
- 逐步推进任务并生成研究报告

### 1.4 结果沉淀

- 保存研究运行态
- 保存历史报告
- 支持 Markdown 导出
- 支持围绕已有材料继续发起下一轮任务

---

## 2. 项目怎么启动

### 2.1 启动后端

进入后端目录：

```bash
cd backend
```

安装依赖：

```bash
uv sync
```

如需配置模型、向量库和运行目录，可以先参考：

- `backend/.env.example`

启动后端：

```bash
uv run uvicorn app.api.main:app --reload --port 8000
```

默认地址：

- `http://localhost:8000`

### 2.2 启动前端

进入前端目录：

```bash
cd frontend
```

安装依赖：

```bash
npm install
```

如需单独指定后端地址，可以参考：

- `frontend/.env.example`

启动前端：

```bash
npm run dev
```

默认地址：

- `http://localhost:5173`

### 2.3 第一次建议怎么体验

建议按下面的顺序走一遍：

1. 在 `Library` 页面上传 PDF。
2. 等文档进入可用状态。
3. 去 `Knowledge` 页面选择论文并直接提问。
4. 用快捷入口发起总结、综述或对比任务。
5. 去 `Research` 页面观察任务推进、证据补充和报告生成。
6. 在 `Reports` 页面查看历史报告并导出 Markdown。

---

## 3. 整体架构怎么理解

PaperDesk 虽然是一个 FastAPI + Vue 项目，但真正重要的是它把几条能力链路接到了同一个运行时里。比较好理解的方式，是把它拆成下面六层：

### 3.1 接口层

位置：

- `backend/app/api`

职责：

- 接收前端请求
- 管理依赖注入
- 提供聊天、论文库、RAG、研究流和报告导出接口
- 通过 SSE 持续把研究状态推送给前端

### 3.2 仓储层

位置：

- `backend/app/repositories`

职责：

- 基于 SQLite 保存业务事实
- 管理文档、chunk、会话、记忆、研究运行态、报告、分类、通知和 trace

### 3.3 服务层

位置：

- `backend/app/services`

职责：

- 文档解析与建库
- RAG 检索与回答
- 记忆读写与上下文装配
- 研究任务路由
- 研究运行编排
- 报告导出

### 3.4 Agent 与运行时层

位置：

- `backend/app/agents`
- `backend/app/runtime`

职责：

- Planner、Search、Summarizer、ReportWriter 等能力封装
- 主 Agent 循环决策
- Skill 装载
- MCP 工具接入
- 子任务通信与并发 worker 调度

### 3.5 向量层

位置：

- `backend/app/vectorstores`

职责：

- 封装 Milvus 检索接口
- 为 RAG 和研究链路提供统一的本地证据入口

### 3.6 前端状态层

位置：

- `frontend/src/stores`

职责：

- 管理聊天状态
- 管理论文库文档和分类
- 消费研究 SSE 事件
- 缓存任务、运行态和报告列表

---

## 4. 本地论文库和 RAG 是怎么做的

### 4.1 PDF 建库链路

文档进入系统后，会经过一条离线建库流水线：

1. 文档元数据先写入 SQLite。
2. `PdfParser` 解析 PDF，提取标题、页数和正文。
3. `TextChunker` 把正文切成 chunk。
4. `EmbeddingService` 用 `BAAI/bge-m3` 生成向量。
5. `KnowledgeIngestionService` 把文档和 chunk 写入 Milvus。
6. chunk 元数据同步写入 SQLite。
7. 文档状态更新为 `ready`。

核心入口：

- `backend/app/services/knowledge_ingestion_service.py`

### 4.2 RAG 检索链路

RAG 的入口是 `RagService`。

它的基本策略是：

- 先确定可用文档范围
- 如配置了翻译服务，会补充英文 query
- 对每个 query 去向量库检索证据
- 合并重复命中的 chunk
- 按 score 排序
- 取前 `top_k`

PaperDesk 里的本地证据不会只保留纯文本，而会尽量保留这些结构化字段：

- `citation_label`
- `title`
- `page_number`
- `filename`
- `score`
- `source_id`

这样后面的聊天回答、任务总结和最终报告都可以继续依赖这些字段生成引用与边界说明。

核心入口：

- `backend/app/services/rag_service.py`

---

## 5. 记忆管理是怎么做的

### 5.1 记忆不是全堆在数据库里

PaperDesk 的聊天记忆采用“文件优先”的思路。

系统不会只把信息存在数据库里，而是把会长期影响对话质量的上下文写进 `.claude` 目录，形成一个可见、可恢复、可人工检查的文件体系。

主要包括：

- `.claude/CLAUDE.md`
- `.claude/runtime/user.md`
- `.claude/sessions/<session>/session.md`
- `.claude/sessions/<session>/compact/compact-xxx.md`
- `.claude/sessions/<session>/context_state.json`

其中：

- `CLAUDE.md` 负责项目级规则
- `user.md` 负责长期用户偏好
- `session.md` 负责当前会话摘要
- `compact` 目录负责历史压缩摘要
- `context_state.json` 负责记录当前上下文阶段和压缩状态

核心入口：

- `backend/app/services/context_file_store.py`

### 5.2 会记录什么记忆

`ChatMemoryService` 会把这些内容转成可复用记忆：

- 用户偏好
- 当前会话主题
- 曾引用过的论文
- 当前选中的论文
- 文件型上下文摘要

这类记忆一部分存文件，一部分通过仓储层索引出来，用于后续对话和会话恢复。

核心入口：

- `backend/app/services/chat_memory_service.py`

---

## 6. 上下文管理是怎么做的

### 6.1 token 预算怎么计算

上下文预算由 `ContextBudgetService` 统一管理。

核心参数在 `backend/app/config.py`：

- `response_reserve_tokens = 12000`
- `compact_warn_ratio = 0.82`
- `compact_force_ratio = 0.92`
- `recent_turns_min = 3`
- `max_evidence_items = 4`
- `max_evidence_chars_per_item = 280`

默认最大上下文窗口会按模型名推断：

- `gpt-4.1 / gpt-4o / o3 / o4`：按 `128000`
- `qwen / deepseek / glm`：按 `64000`
- 其他模型：按 `64000`

真正可用于 prompt 的预算是：

```text
budget_tokens = max_context_tokens - response_reserve_tokens
```

### 6.2 超预算后怎么处理

聊天态上下文的处理顺序是固定的：

1. 先压缩证据。
2. 如果还超预算，就把旧消息写成 compact 摘要文件。
3. 如果还超预算，就只保留最近若干轮原始消息。

这意味着系统优先保住“最近几轮对话”和“高相关证据”，而不是机械地截断前面的历史。

相关实现：

- `backend/app/services/context_budget_service.py`
- `backend/app/services/context_compaction_service.py`
- `backend/app/services/context_assembler.py`

### 6.3 研究态上下文和聊天态有什么不同

研究态不是按对话轮次压缩，而是按任务决策需要来组织。

系统会显式维护这些状态：

- `working_summary`
- `active_task`
- `tool_history`
- `evidence_buffer`
- `compacted_evidence`
- `context_state`

`ResearchContextAssembler` 默认只展示最近 `8` 步工具历史，同时会对证据做：

- 相关性判断
- 覆盖度判断
- 冲突检测
- 可见性筛选

这样主循环在做下一步决策时，看到的是“为当前任务整理过的上下文”，而不是一份不断膨胀的原始日志。

相关实现：

- `backend/app/services/research_context_assembler.py`

---

## 7. 研究任务怎么被路由

研究请求进入系统后，第一步不是立刻搜索，而是先经过 `ResearchTaskRouter`。

它会根据：

- `topic`
- `notes`
- `input_modes`
- `selected_document_ids`
- 当前准备好的本地文档数量

把请求识别成这些任务类型之一：

- `qa`
- `paper_summary`
- `multi_paper_review`
- `comparison`
- `method_explainer`
- `research_brief`

同时，它还会决定：

- 是否需要本地知识
- 是否需要在线检索
- 证据策略是 `local_only / local_first / online_first / online_supplement`
- 是否允许 lightweight single-pass
- 是否进入主 Agent loop

轻量任务在本地材料足够时可以直接 single-pass 完成，复杂任务则会进入主循环。

核心入口：

- `backend/app/services/research_task_router.py`

---

## 8. Skill、MCP、RAG 是怎么接到一起的

### 8.1 Skill 怎么做

PaperDesk 里的 Skill 主要是后端运行时约束，不是前端显式给用户选择的模板。

每个 Skill 都会描述：

- 对应任务类型
- 可用工具列表
- 产物协议
- 默认执行模式

当前内置 Skill 包括：

- `paper_summary`
- `multi_paper_review`
- `comparison`
- `qa`
- `method_explainer`
- `research_brief`

相关目录：

- `backend/app/skills`
- `backend/app/runtime/skill_registry.py`

### 8.2 MCP 怎么做

MCP 在 PaperDesk 里承担的是“把外部只读能力统一接进工具体系”的职责。

当前默认声明的只读工具包括：

- `mcp/academic_search`
- `mcp/academic_metadata`
- `mcp/read_only_web_fetch`

这些声明先经过 `ReadOnlyMcpAdapter` 过滤，只保留白名单、只读、启用、schema 合法的工具，然后再交给 `ToolRegistry`。

相关实现：

- `backend/app/runtime/mcp_adapter.py`
- `backend/app/runtime/tool_registry.py`

### 8.3 为什么要统一成工具注册表

因为一旦进入研究主循环，系统关心的核心问题就变成了：

- 当前动作是什么
- 应该选哪个工具策略
- 执行之后拿到了什么证据
- 下一步该补证据、改计划，还是收束报告

所以 builtin tool、RAG 本地检索能力和 MCP 外部只读能力，最终都要进入统一的工具声明体系。

---

## 9. 主 Agent loop 决策是怎么做的

### 9.1 编排入口

研究编排的中心在：

- `backend/app/services/research_orchestrator.py`

它负责：

- 创建 run
- 保存运行态
- 发出 SSE 事件
- 调用 Planner、Search、Summarizer、ReportWriter
- 维护 `runtime_state`
- 处理恢复运行

### 9.2 主循环在看什么

`MainAgentRuntime.next_action()` 每一步不是简单地“再问一次模型下一步怎么办”，而是根据显式运行态做判断：

- 当前计划是否存在
- 当前任务是否缺本地证据
- 当前任务是否缺在线论文候选
- 当前证据是否已经足够
- 同一工具是否重复调用过多
- 是否已经长时间没有增量
- 是否需要 revise plan
- 是否该降级收束
- 是否该生成最终报告

相关实现：

- `backend/app/runtime/main_agent_runtime.py`

### 9.3 一个典型决策顺序

可以把它理解成这样一条状态机：

1. 没有计划，先 `PLAN`。
2. 如果当前任务要求本地证据且本地优先，先 `SEARCH_LOCAL`。
3. 在线证据还缺时，执行 `SEARCH_ONLINE`。
4. 证据达到阈值后，执行 `SUMMARIZE_EVIDENCE`。
5. 证据不足但还有调整空间时，执行 `REVISE_PLAN`。
6. 所有任务完成后，执行 `FINALIZE_REPORT`。
7. 报告生成完成后，进入 `FINISH`。

### 9.4 loop 怎么避免空转

主循环内置了几层刹车：

- `max_step_budget = 64`
- `max_no_progress_count = 3`
- `max_same_tool_streak = 2`
- `max_replan_count = 2`

如果出现连续失败、连续无增量，或者同一工具反复调用却没有新证据，系统会触发：

- 改写 query
- 拆任务
- 重排优先级
- 降级总结
- 直接结束本轮研究

---

## 10. 多 Agent 怎么通信，怎么处理并发

### 10.1 当前结构

PaperDesk 当前采用的是“单主 Agent + 可并发 worker”的结构：

- 主 Agent 负责决定下一步干什么
- worker 负责执行具体任务
- 运行态和消息总线负责同步结果

这种结构的好处，是恢复运行、前端回放和状态解释都会更稳定。

### 10.2 Agent 之间怎么通信

多 Agent 之间不会直接彼此改状态，而是通过 `MessageBus` 和 `RuntimeRepository` 传递进度、通知和结果。

`MessageBus` 负责三件事：

- 记录 trace
- 记录 notification
- 向前端 SSE sink 发事件

相关实现：

- `backend/app/runtime/message_bus.py`

### 10.3 并发怎么做

`SubagentRunner` 内部使用 `ThreadPoolExecutor` 执行并发 worker。

它提供两个入口：

- `spawn()`：单个子任务
- `run_parallel()`：多个子任务并发执行

执行时会：

- 先注册任务
- 标记状态为 `running`
- 通过 `progress` 回调发进度事件
- 完成后写 notification 和 artifact
- 失败后写 failed notification
- 最后按任务原始顺序返回结果

相关实现：

- `backend/app/runtime/subagent_runner.py`

### 10.4 并发结果怎么合并

PaperDesk 不把“谁先跑完谁先覆盖状态”当成默认策略，而是把结果先归档到运行态，再由 orchestrator 决定何时 merge。

这样做有两个直接好处：

- 防止并发 worker 互相踩状态
- 让恢复运行和回放历史更稳定

---

## 11. 前端工作台怎么和这些能力接起来

前端不是单纯的接口展示层，它承担的是“把研究状态组织成一个连续工作台”的职责。

主要页面包括：

- `Knowledge`：聊天、选论文、快捷发起任务
- `Library`：上传、分类、筛选、排序
- `Research`：查看任务进度、恢复当前运行、继续围绕当前材料发起下一轮任务
- `Reports`：查看历史报告和导出 Markdown

关键代码主要在：

- `frontend/src/views`
- `frontend/src/stores`

如果你想理解 PaperDesk 的前后端协作方式，建议从这里看：

- `frontend/src/views/KnowledgeView.vue`
- `frontend/src/views/LibraryView.vue`
- `frontend/src/views/ResearchView.vue`
- `frontend/src/views/ReportView.vue`

---

## 12. 建议按什么顺序读代码

如果你是从教程视角来读这个项目，比较推荐按下面顺序往下看：

### 12.1 先看产品入口

- `frontend/src/views/KnowledgeView.vue`
- `frontend/src/views/LibraryView.vue`
- `frontend/src/views/ResearchView.vue`
- `frontend/src/views/ReportView.vue`

### 12.2 再看后端主入口

- `backend/app/api/main.py`

### 12.3 想理解本地建库

- `backend/app/services/document_library_service.py`
- `backend/app/services/knowledge_ingestion_service.py`
- `backend/app/vectorstores/milvus_store.py`

### 12.4 想理解记忆和上下文

- `backend/app/services/context_file_store.py`
- `backend/app/services/chat_memory_service.py`
- `backend/app/services/context_assembler.py`
- `backend/app/services/context_budget_service.py`
- `backend/app/services/context_compaction_service.py`

### 12.5 想理解研究主循环

- `backend/app/services/research_orchestrator.py`
- `backend/app/runtime/main_agent_runtime.py`
- `backend/app/services/research_context_assembler.py`
- `backend/app/services/research_task_router.py`

### 12.6 想理解 Skill、MCP 和工具层

- `backend/app/runtime/skill_registry.py`
- `backend/app/runtime/tool_registry.py`
- `backend/app/runtime/mcp_adapter.py`
- `backend/app/skills`

---

## 13. 适合继续往哪里扩展

如果你打算基于这个项目继续做自己的版本，比较自然的扩展方向包括：

- 接入更多在线学术数据源
- 扩展新的 Skill 和任务类型
- 增加更细粒度的报告模板
- 把并发 worker 用在更复杂的证据收集任务上
- 增强会话恢复、任务重试和人工介入节点
- 在前端补充更清晰的材料组织和研究回看能力

PaperDesk 的价值，不在某一个单点功能，而在它把“本地论文库、记忆、上下文、RAG、任务路由、工具调用、研究循环、结果沉淀”放进了同一个工程里。你可以把它当成一个可运行的论文助手，也可以把它当成一个完整的 Agent 系统样板来拆解。
