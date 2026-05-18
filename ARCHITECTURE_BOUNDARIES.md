# PaperDesk Architecture Boundaries

This document records the current PaperDesk architecture boundaries after the Knowledge main path was simplified. It is intentionally descriptive: it does not define new behavior, new APIs, or a migration plan.

## Main Product Line

PaperDesk is a local-first research-paper Agent platform. The stable product line is:

PDF upload -> parse -> chunk -> index -> paper/tag selection -> RAG retrieval -> evidence-grounded Agent answer -> save report / export Markdown.

The Knowledge page should stay optimized for this path. Experimental Agent capabilities are useful for learning, but they should not silently take over this default flow.

## Knowledge Main Path

The current Knowledge request flow is:

Frontend Knowledge page
-> knowledge store
-> chat route
-> ChatService
-> AgentOrchestrator
-> KnowledgeRoute
-> AgentRunMode compatibility layer
-> Direct answer or KnowledgeAgentRuntime
-> Tool execution / Observation
-> Final answer

`ChatService` owns chat-session orchestration, attachment normalization, memory/context assembly, route selection, and runtime dispatch. Deeper Agent behavior should remain in the orchestrator or runtimes, not spread back into API handlers or frontend state.

## KnowledgeRoute And AgentRunMode

`KnowledgeRoute` is the product-semantic route layer for Knowledge Chat. It names what the user-facing request means:

- `DirectAnswer`: ordinary chat or explanation that does not need library state or tools.
- `ToolAction`: read-only or evidence-grounded Knowledge work, including RAG over selected papers.
- `ConfirmedWrite`: write operations that need guardrails, preview, confirmation, execution, and verification.
- `OptionalPlanner`: explicit long or multi-step Knowledge tasks.
- `OptionalReflection`: explicit correction, recheck, or answer-quality repair.

`AgentRunMode` is the execution compatibility layer kept for the existing runtime dispatch:

- `DirectAnswer` usually maps to `AgentRunMode.DIRECT`.
- `ToolAction` usually maps to `AgentRunMode.REACT`.
- `ConfirmedWrite` usually maps to `AgentRunMode.REACT`, but semantically means the write path must be protected.
- `OptionalPlanner` maps to `AgentRunMode.PLANNER`.
- `OptionalReflection` maps to `AgentRunMode.REFLECTION`.

New routing work should prefer the `KnowledgeRoute` vocabulary when describing product behavior. Do not add new logic by casually stacking old `DIRECT`, `REACT`, `PLANNER`, and `REFLECTION` meanings without making the product route clear first.

## Three Core Request Types

Ordinary Q&A, such as "What is RAG?", should route to `DirectAnswer`. The orchestrator may inspect tools and context, but the final dispatch should stay on the direct chat path when no selected paper, library state, RAG evidence, or write operation is needed.

Paper Q&A, such as "Summarize the selected paper", should route to `ToolAction`. Selected document IDs and library-document attachments enter through the Knowledge store and chat request, are normalized by `ChatService`, and force a grounded Knowledge runtime path. `KnowledgeAgentRuntime` then resolves documents, calls evidence retrieval, receives observations, and synthesizes the final answer from retrieved evidence.

Write requests, such as deleting empty tag categories, clearing paper tags, or adding a category to papers, should route to `ConfirmedWrite` when they are destructive or scope-sensitive. The runtime distinguishes entity-level work, such as deleting unused category entities, from relation-level work, such as clearing paper-category links. Protected writes follow preview -> pending action -> user confirmation -> execute -> verify. The successful answer should be based on verified state, not on model intent alone.

## Experimental Capability Boundaries

The following capabilities are retained for learning and future extension, but they are not the default Knowledge main path:

- Research Agent Loop: an experimental, independent research-task workflow.
- MCP: experimental external tool declaration and read-only academic extension.
- Subagent: experimental child-task execution and tracing capability.
- Skills: research-task capability descriptions and extension metadata.
- Planner: optional enhancement for explicit long or multi-step tasks.
- Reflection: optional enhancement for explicit correction or configured quality review.

These capabilities should be activated through configuration flags, explicit entrypoints, or clear user intent. They should not silently pollute ordinary Knowledge Q&A or selected-paper RAG flows.

The current capability flags include:

- `ENABLE_RESEARCH_FROM_KNOWLEDGE`
- `ENABLE_EXPERIMENTAL_MCP`
- `ENABLE_MCP_IN_KNOWLEDGE`
- `ENABLE_SUBAGENT_EXECUTION`
- `ENABLE_AUTO_REFLECTION`

`CAPABILITY_BOUNDARIES.md` remains the concise reference for those experimental capability switches.

## KnowledgeAgentRuntime Boundary

`KnowledgeAgentRuntime` is currently the main Knowledge Agent runtime. It owns the ReAct loop, tool execution, write protection, pending actions, Observation wrapping, evidence merging, answer synthesis, report drafting, category operations, and lightweight memory/reflection helpers.

The file is large by design history, but it should not be split aggressively in one pass. If it is slimmed down later, prefer small extractions with clear behavior-preserving boundaries, such as:

- pending action storage and confirmation handling;
- write guardrails, preview, and verification;
- evidence retrieval tools;
- final answer synthesis and report drafting.

Each extraction should be independently testable and should not change the Knowledge main path.

## ToolRegistry Boundary

`ToolRegistry` currently declares tools and their metadata. It is not the unified execution engine. Actual Knowledge tool execution still lives in the runtime.

Tool declarations must preserve scope boundaries:

- `knowledge`: stable Knowledge Chat tools;
- `research`: research-task tools;
- `mcp`: external MCP declarations;
- `experimental`: opt-in or future-facing tools.

Default Knowledge candidates should remain stable, non-experimental, and scoped for Knowledge unless a configuration flag and explicit route allow otherwise. `available_by_default` is only one eligibility signal; it must still pass maturity, scope, source, and feature-flag filtering before a tool appears in a default candidate list.

Tool metadata is descriptive:

- `io_type`, `write_type`, `destructive`, and `requires_confirmation` describe safety posture for routing, traces, and audits.
- Write and destructive tools still require runtime guardrails such as preview, pending action, confirmation, execution, and verification.
- MCP declarations stay read-only, experimental, and disabled by default unless explicitly enabled.
- Research tools remain Research Agent Loop capabilities and should not appear in the default Knowledge tool set.
- Experimental tools, such as `memory.write`, should remain opt-in and should not silently enter the Knowledge main path.

Do not turn `ToolRegistry` into a cross-runtime execution center before the runtime boundaries are clearer.

## Observation And Payload Boundary

`payload` is still the main data channel inside Knowledge runtime observations. Many retrieval, write verification, answer synthesis, and trace paths read from `_ReactObservation.payload`.

`ToolObservation` is the normalized envelope used for structured trace data and future migration. It is the desired shape for clearer observations, but it does not replace `payload` yet.

Do not delete `payload` directly. A safer future step is to add compatibility helpers that read the current payload while gradually exposing stable `ToolObservation.data`, `ToolObservation.evidence`, and `ToolObservation.verification` fields.
