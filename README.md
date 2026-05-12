# PaperDesk

PaperDesk 是一个面向科研场景的论文工作台。当前版本已经完成教程 `09.5` 之后的升级版收口，系统不再只是固定研究流程演示，而是同时具备三条可独立验收的主链路：

- 研究工作流：围绕研究主题拆分 TODO、串联在线论文检索与本地证据、生成任务总结和最终综述
- 本地知识库索引：上传 PDF、解析文本、切片、向量化、写入 Milvus、保存 chunk 元数据
- RAG 知识问答：对本地知识库自由提问，或做单篇分析、多篇比较、在线候选论文筛选

阶段 `10-测试、排错与升级版RAG验收` 的重点不是继续扩功能，而是把现有能力整理成可验证、可排错、可持续扩展的工程形态。

## 当前能力

### 前端页面

- `/`：总览与研究入口
- `/library`：本地论文库，负责上传、查看和删除 PDF
- `/knowledge`：知识问答、本地论文分析、多篇比较、在线候选筛选
- `/research`：研究工作台，展示 SSE 任务推进、单任务总结与最终综述
- `/reports`：历史报告预览与 Markdown 导出

### 后端接口

- `GET /healthz`
- `GET /api/documents`
- `POST /api/documents/upload`
- `DELETE /api/documents/{document_id}`
- `POST /api/papers/search`
- `POST /api/papers/analyze`
- `POST /api/papers/curate`
- `POST /api/rag/ask`
- `POST /api/research/stream`
- `GET /api/research/{run_id}`
- `GET /api/reports`
- `GET /api/reports/{report_id}`
- `DELETE /api/reports/{report_id}`
- `GET /api/export/{report_id}`

### 当前已支持

- 在线论文检索：`all / auto / openalex / arxiv`
- 本地 PDF 上传、去重、重传同名重建、索引状态追踪
- 递归切片、本地 embedding、Milvus 向量检索、chunk 元数据落库
- 知识问答、单篇分析、多篇比较、候选论文筛选
- SSE 研究工作流、结构化综述、统一引用、Markdown 导出
- 失败退化：文档索引失败、知识库证据不足、研究中途任务失败

### 当前未支持

- 真实外部 Milvus 联调已完成的正式验收口径
- Google Scholar 或更多在线 provider
- 目录轮询、Webhook、消息队列等自动化入库更新
- 更重的生产化运维、监控、权限或多租户能力

## 目录结构

- `backend/`：FastAPI 后端、数据层、服务层、Agent、测试
- `frontend/`：Vue 3 + Pinia 工作台界面

## 本地启动

### 后端

```bash
cd backend
uv sync
uv run uvicorn app.api.main:app --reload --port 8000
```

后端默认地址：`http://localhost:8000`

### 前端

```bash
cd frontend
npm install
npm run dev
```

前端默认地址：`http://localhost:5173`

## 环境变量

### backend/.env

关键变量如下：

- `LLM_API_KEY`：可选；为空时，报告润色和 RAG 回答会走稳定模板回退
- `LLM_BASE_URL`：可选；兼容代理或自定义网关
- `LLM_MODEL`：默认 `gpt-4o-mini`
- `EMBEDDING_MODEL`：默认 `BAAI/bge-m3`
- `SQLITE_PATH`：SQLite 数据库路径
- `MILVUS_URI`：Milvus 或 Milvus Lite 地址
- `MILVUS_TOKEN`：可选
- `MILVUS_DATABASE`：Milvus 数据库名
- `MILVUS_COLLECTION`：本地知识库 collection 名
- `WORKSPACE_DIR`：运行时工作目录
- `UPLOAD_DIR`：PDF 上传目录
- `REPORT_DIR`：Markdown 报告输出目录

示例文件见 [backend/.env.example](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/backend/.env.example)。

### frontend/.env

- `VITE_API_BASE_URL`：前端请求后端的基础地址，默认 `http://localhost:8000`

示例文件见 [frontend/.env.example](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/frontend/.env.example)。

## 推荐测试分层

升级版 PaperDesk 至少按 5 层看：

1. 数据层测试：SQLite 文档、chunk、研究任务、报告与引用是否落库正确
2. 索引层测试：PDF 解析、切片、embedding、Milvus 写入、chunk 元数据是否完整
3. 检索与生成服务层测试：`KnowledgeIngestionService`、`RagService`、`PaperSearchService`、`ResearchOrchestrator`
4. 接口层测试：上传、检索、问答、研究流、报告导出接口是否稳定
5. 手工验收：从上传 PDF 到研究与导出的完整体验是否闭环

当前自动化测试主要位于：

- [backend/tests/test_documents.py](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/backend/tests/test_documents.py)
- [backend/tests/test_knowledge.py](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/backend/tests/test_knowledge.py)
- [backend/tests/test_research.py](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/backend/tests/test_research.py)
- [backend/tests/test_papers.py](/D:/个人资料/aicoding/LangChain/helloagent/hello-agents-main/科研Agent/paperdesk/backend/tests/test_papers.py)

## 验收命令

### 后端测试

```bash
cd backend
.\.venv\Scripts\pytest.exe -q
```

### 前端构建

```bash
cd frontend
npm.cmd run build
```

## 升级版手工验收清单

### 工程启动

- 后端能启动
- 前端能启动
- `GET /healthz` 返回 `{"status":"ok"}`

### 离线索引

- 上传 PDF 成功
- 文档列表出现新记录
- 文档状态从 `processing` 进入 `ready`
- `parser_status` 进入 `indexed`
- 文档有页数、`version`、`indexed_at`

### 在线检索

- 主题检索能返回标准化论文结果
- 不同来源结果可以合并显示

### RAG 问答

- 问题能召回本地证据
- 回答中带来源和页码
- `sources`、`pages`、`retrieval_count` 字段合理
- 证据不足时返回明确提示，而不是编造答案

### 研究工作流

- 能拆出 3 到 5 个 TODO
- SSE 按顺序返回 `status`、`todo_list`、`task_status`、`task_summary`、`report`
- 每个 TODO 都有总结
- 最终综述生成成功

### 导出

- `/reports` 可查看历史报告
- `/api/export/{report_id}` 可导出 Markdown
- 引用清单存在且可读

## 常见故障定位表

### 上传成功，但问答始终检索不到

优先排查：

- 文档是否已经到 `ready`
- `parser_status` 是否已经是 `indexed`
- embedding 是否成功
- Milvus 是否写入成功
- chunk 元数据里的 `document_id / filename / page_number` 是否完整

### 检索有结果，但回答明显跑偏

优先排查：

- Top-K 召回是否相关
- prompt 是否只拼入了当前证据
- LLM 指令是否要求“仅基于证据回答”
- 当前问题是否过宽，导致检索命中噪声 chunk

### 本地 PDF 总被错引

优先排查：

- chunk 是否切得过碎
- query 是否太泛
- 检索后是否缺少相关性过滤
- 页码、标题、文件名映射是否正确

### 研究一直转圈

优先排查：

- `/api/research/stream` 是否持续输出 SSE
- `ResearchOrchestrator` 是否卡在某个任务阶段
- 在线检索、知识检索、报告生成是否超时

### 前端状态混乱

优先排查：

- SSE 事件类型是否符合既有约定
- store 是否按 `run_id / task_id` 更新
- 是否出现重复消费或中途异常导致的半状态

## 失败时的退化策略

当前版本推荐并已部分落地的退化口径：

- 在线检索失败：仍可继续使用本地知识库
- 本地证据为空：RAG 问答明确返回“当前证据不足”，不伪造答案
- 文档索引失败：文档状态标记为 `failed`，方便重试和排查
- 单个任务失败：研究运行标记失败，并把错误通过 SSE 返回
- LLM 不可用：报告生成与知识回答退回稳定模板，不阻断主流程

## Milvus 抽象边界

当前实现不是让业务层直接依赖 Milvus 原始返回，而是通过 `vectorstores/` 抽象层统一输出到上层。

上层应只依赖这些稳定概念：

- `ChunkRecord`
- `EvidenceItem`
- `document_id`
- `source`
- `page`
- `chunk_id`
- `score`

这样无论底层是 `Milvus Lite`、完整 `Milvus`，还是未来切回其他向量库，业务层改动都能保持最小。

## 已验证口径

当前阶段 10 收口时，建议至少确认以下结果：

- 后端测试通过
- 前端构建通过
- 空知识库或无效 `document_ids` 时，`/api/rag/ask` 能诚实返回证据不足
- 已上传并索引完成的 PDF 可以支撑知识问答、研究总结和报告引用

如果以上都稳定成立，就可以认为当前版本已经从“教程功能堆叠”进入“可测试、可排错、可继续演进”的升级版项目形态。
