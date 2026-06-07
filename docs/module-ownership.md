# PaperDesk 模块归属说明

PaperDesk 当前采用 `Agent Core + Domain Pack + Infrastructure Adapter` 的代码组织方式。新代码优先从 `app.agent`、`app.domains`、`app.infrastructure` 导入；`app.services`、`app.runtime`、`app.agents` 中保留的文件主要承担兼容导出。

## Agent Core

| 归属 | 主要路径 | 职责 |
| --- | --- | --- |
| 生命周期 | `backend/app/agent/lifecycle` | Ingress、route decision、lifecycle service |
| 运行时 | `backend/app/agent/runtimes` | Runtime dispatcher、route runtime、runtime executor |
| 实验运行时 | `backend/app/agent/runtimes/experimental` | planner、reflection、subagent、research task |
| Capability | `backend/app/agent/capabilities` | capability 声明、默认注册表、扩展入口 |
| Tools | `backend/app/agent/tools` | Tool Registry、tool policy、tool observation、MCP 声明 |
| Skills | `backend/app/agent/skills` | Skill Registry、selector、context、lifecycle binding |
| Safety | `backend/app/agent/safety` | pending action、写安全、workspace path guard |
| Memory | `backend/app/agent/memory` | lifecycle context、轻量 memory 边界 |
| Observability | `backend/app/agent/observability` | trace recorder、RAG trace、response recorder |

## Domain Packs

| 归属 | 主要路径 | 职责 |
| --- | --- | --- |
| Paper | `backend/app/domains/paper` | 论文上传、解析、切分、RAG、论文搜索、报告生命周期、research agents |
| Workspace | `backend/app/domains/workspace` | 会话文件、工作区文件、路径解析、覆盖确认、Workbench read model |
| Artifact | `backend/app/domains/artifact` | 报告导出和未来图形/产物输出 |

## Infrastructure

| 归属 | 主要路径 | 职责 |
| --- | --- | --- |
| LLM | `backend/app/infrastructure/llm` | embedding provider |
| Files | `backend/app/infrastructure/files` | context file store、file asset、text extraction |
| Integrations | `backend/app/infrastructure/integrations` | arXiv、OpenAlex、未来 draw.io 等外部系统 |
| Vectorstore | `backend/app/infrastructure/vectorstore` | Milvus bootstrap 和向量库边界 |
| Persistence | `backend/app/infrastructure/persistence` | repository 导出边界 |

## Compatibility Surfaces

| 旧路径 | 当前用途 |
| --- | --- |
| `backend/app/services/agent_*` | 兼容导出到 `app.agent.*` |
| `backend/app/runtime/tool_registry.py` | 兼容导出到 `app.agent.tools.registry` |
| `backend/app/runtime/skill_registry.py` | 兼容导出到 `app.agent.skills.registry` |
| `backend/app/runtime/main_agent_runtime.py` 等 | 兼容导出到 `app.agent.runtimes.experimental` |
| `backend/app/agents/*` | 兼容导出到 `app.domains.paper.research_agents` |
| `backend/app/services/document_library_service.py` 等 | 兼容导出到 `app.domains.paper` |
| `backend/app/services/workspace_*` | 兼容导出到 `app.domains.workspace` |
| `backend/app/services/file_*`、`embedding_service.py` | 兼容导出到 `app.infrastructure` |

## 默认与实验边界

默认产品路径覆盖普通聊天、论文 RAG、只读工具、确认写操作、报告、workspace 文件。Planner、reflection、MCP、subagent、research-task runtime 位于 `backend/app/agent/runtimes/experimental`，需要显式 route 或 feature flag 才进入。
