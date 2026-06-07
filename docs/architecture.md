# PaperDesk Runtime-First Agent Architecture

PaperDesk is a lightweight paper-reading Agent application with platform-grade extension points. The default product path stays focused on paper upload, parsing, library management, selected-paper RAG, safe library writes, reports, and ordinary chat.

## Lifecycle

```text
request
  -> ingress
  -> route
  -> skill
  -> context
  -> runtime
  -> RAG/tools/write safety/service workflow
  -> trace
  -> response finalization
```

The lifecycle entrypoint is `AgentLifecycleService`. Route-specific runtimes identify the primary orchestration pattern, execution policy, target scope, and compact metrics for each request. `AgentRuntimeResponseRecorder` finalizes response metadata so trace records both the selected runtime and the emitted response status.

New Agent-facing imports are exposed under `backend/app/agent`. Paper, workspace, and artifact business code is exposed under `backend/app/domains`, while external adapters live under `backend/app/infrastructure`. Legacy `app.services`, `app.runtime`, and `app.agents` paths remain compatibility surfaces for older imports and tests.

## Route Vocabulary

| Route | Runtime target | Purpose |
| --- | --- | --- |
| `direct_chat` | `DirectChatRuntime` | General conversation without paper or tool requirements. |
| `paper_rag` | `PaperRagRuntime` | Paper-grounded answers using selected documents, metadata filters, evidence, and citations. |
| `library_read` | `ToolActionRuntime` | Read-only paper library, tag, category, and metadata queries. |
| `tool_action` | `ToolActionRuntime` | Bounded non-destructive tool execution. |
| `write_pending` | `ToolActionRuntime` | Preview and pending confirmation for risky writes. |
| `write_confirmed` | `ConfirmedWriteRuntime` | Execution of a confirmed pending write. |
| `report_action` | `ReportActionRuntime` | Report save, list, and export actions. |
| `workspace_read` | `WorkspaceActionRuntime` | Workspace file reads and inspection. |
| `workspace_write` | `WorkspaceActionRuntime` | Workspace writes with path and overwrite safety. |
| `experimental_research` | `ExperimentalRuntime` | Planner, reflection, MCP, subagent, or research-task surfaces. |

## Module Ownership

See `docs/module-ownership.md` for the current ownership map. The short version is:

| Lifecycle stage | Primary module direction |
| --- | --- |
| Ingress | Normalize chat sessions, attachments, selected documents, commands, and pending action references. |
| Route | Classify user intent into product routes with auditable reasons. |
| Skill | Select built-in or custom skills and render prompt-safe skill context. |
| Context | Assemble recent messages, selected scope, evidence, pending actions, workspace scope, and preferences under token budget. |
| Runtime | Dispatch through `app.agent.runtimes` to direct chat, paper RAG, tool action, confirmed write, report, workspace, or experimental runtime. |
| Tool policy | Intersect route, skill, scope, risk, feature flags, and confirmation state before exposing tools. |
| RAG | Preserve paper parsing, chunking, vector recall, keyword fallback, metadata filters, evidence quality, and citations. |
| Write safety | Use explicit scope, preview, pending confirmation, execution, and verification. |
| Trace | Record route, skill, context scope, evidence, tool filtering, calls, pending actions, verification, and errors. |
| Response | Finalize runtime response metadata and return chat-compatible responses. |

## Orchestration Patterns

| Route family | Primary pattern | Boundary |
| --- | --- | --- |
| Direct chat | `single-turn` | One model answer, no default RAG, tools, planner, reflection, MCP, or subagent execution. |
| Paper RAG | `retrieve-then-synthesize` | Selected document or query-scoped retrieval, evidence assembly, citation-aware synthesis. |
| Library/tool read | `bounded-react` | Tool steps are bounded and carry stop reasons plus structured observations. |
| Write routes | `preview-confirm-execute-verify` | Scope must be explicit or marked as needing clarification; broad default writes are blocked. |
| Report/workspace | `service-workflow` | Deterministic service orchestration with route/runtime trace. |
| Experimental research | `plan-execute-replan` | Planner, reflection, MCP, and subagent behavior stay behind explicit experimental selection or feature flags. |

## Migration Phases

1. Establish regression coverage and baseline ownership.
2. Add lifecycle models, dispatcher, and runtime boundaries.
3. Route existing chat and streaming through ingress, route, dispatcher, and trace.
4. Move context, skills, tools, write safety, and RAG into lifecycle-compatible boundaries.
5. Migrate runtimes by route.
6. Rewrite README and supporting docs with the implemented architecture.
7. Remove or archive duplicated legacy paths after tests prove coverage.

## Runtime Migration Status

Route-specific lifecycle runtimes now exist for direct chat, paper RAG, tool actions, confirmed writes, reports, workspace actions, and experimental research. They are connected through `RuntimeDispatcher` and exposed under `app.agent.runtimes`.

`ChatService` delegates lifecycle preparation to `AgentLifecycleService`, which owns ingress, route decision, context assembly, tool policy resolution, dispatch, and trace handoff for each chat turn. The response recorder now attaches final response metadata to the selected runtime result. Runtime executors own the route-specific call boundary for direct chat, paper RAG, tool action, write, report, workspace, and experimental execution while existing business services provide the underlying operations.

This keeps the main product behavior stable during migration. Compatibility aliases remain for older imports, while docs and tests use the runtime-first lifecycle names.

`KnowledgeAgentRuntime` is now accessed from `ChatService` through `KnowledgeAgentCapabilityProvider`. This keeps the large legacy runtime behind a narrow provider surface for pending actions, ReAct execution, context lines, conversation referents, and final-answer repair. The next internal split can move provider methods into paper, tool, write, report, and experimental modules without changing the chat API.

## Default and Experimental Surfaces

Default PaperDesk routes cover normal chat, paper RAG, library reads, safe writes, reports, and workspace operations. Planner, reflection, MCP, subagent, and research-task workflows remain explicit experimental capabilities unless a route or feature flag enables them.

## Context and Memory

Lifecycle context is represented by a `ContextPacket`. It keeps the request-scoped material that a runtime needs:

- recent messages,
- selected library document ids,
- selected session file ids,
- retrieved evidence,
- pending action state,
- workspace scope,
- lightweight preferences,
- token budget.

Direct chat keeps recent conversation, session files, pending action state, workspace scope, and preferences. It drops paper document scope and RAG evidence unless the route decision asks for paper grounding. Paper RAG and tool routes can include selected documents, evidence, and target scope. Longer-term memory, reflection-derived lessons, and automatic preference extraction stay outside the default lifecycle until an explicit route or feature flag enables them.
