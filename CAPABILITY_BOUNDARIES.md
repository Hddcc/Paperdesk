# PaperDesk 当前能力边界

PaperDesk 当前稳定主线是 Library、Knowledge 和 Reports：PDF 上传入库、解析切分、索引、选择论文或标签后的 RAG 问答、报告保存、预览和 Markdown 导出。

Research Task Agent Loop 属于 experimental 能力。它保留后端实验入口，但默认不会接管 Knowledge 页面里的普通“总结论文”“对比论文”请求。Knowledge 中即使用户明确说“分步骤研究”或“按研究任务执行”，在 `ENABLE_RESEARCH_FROM_KNOWLEDGE=false` 时也只会提示使用研究任务入口，不会自动启动研究循环。

MCP、Subagent、Skills 深度编排、长期记忆和自动 Reflection 属于 experimental/future 能力。默认配置下，Knowledge 只使用 stable 的 knowledge/shared 工具；MCP 不进入 Knowledge 候选；Subagent 不作为 Knowledge 默认执行器；Skills 不在 Knowledge 每轮路由中全量加载；Reflection 只在用户明确纠错或配置允许自动审查时触发。

相关开关集中在后端配置中：

- `ENABLE_RESEARCH_TASK_AGENT`
- `ENABLE_RESEARCH_FROM_KNOWLEDGE`
- `ENABLE_EXPERIMENTAL_MCP`
- `ENABLE_MCP_IN_KNOWLEDGE`
- `ENABLE_SUBAGENT_EXECUTION`
- `ENABLE_AUTO_REFLECTION`

默认值都偏向保护当前稳定主流程。
