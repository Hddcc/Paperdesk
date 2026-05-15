# PaperDesk

## 项目概览

- 这是一个面向论文阅读、知识库问答、研究任务编排与报告沉淀的本地优先项目。
- 当前产品入口分为：`/knowledge` 聊天与知识库入口、`/library` 本地论文库、`/research` PDF 阅读区、`/reports` 历史报告。
- 本文件只记录长期有效的项目规则，不记录阶段性目标、待办事项或运行时用户记忆。

## 常用命令

- 后端启动：`.\.venv\Scripts\uvicorn.exe app.api.main:app --reload --port 8000`
- 后端测试：`.\.venv\Scripts\pytest.exe -q`
- 前端启动：`npm run dev`
- 前端构建：`npm run build`

## 架构约束

- 聊天链路必须在知识库检索降级时仍能返回可读结果；Milvus 失败只应影响 RAG 增强状态。
- 本地 PDF 通过文档上传、解析、chunk、embedding、Milvus 入库进入知识库。
- `/research` 是 PDF 阅读区，不恢复旧的三栏研究工作台；研究任务入口以 `/knowledge` 和后端 research API 为准。
- Agent 相关能力要区分“已接入主流程”和“已有运行时骨架但未接入”，文档不能把后者写成已完成能力。

## 记忆规则

- 记忆是索引，不是真相。
- 任何来自记忆的文档、配置、项目状态、引用结论，在再次使用前都要先读取当前真实文件或数据库记录确认。
- 分层 `CLAUDE.md` 用于帮助维护者和 AI 理解项目，不作为终端用户聊天记忆的业务数据。
- 运行时上下文文件由后端配置管理，默认写入 `backend/runtime/context`；旧的 `CLAUDE_DIR` 环境变量仅作为兼容入口保留。
