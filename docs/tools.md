# PaperDesk Tools

Tools are declared centrally and exposed through lifecycle policy. Tool declarations describe capability and safety metadata; runtime code still owns execution.

Agent-facing imports are exposed through `backend/app/agent/tools`. The compatibility implementation still lives under `backend/app/runtime/tool_registry.py` and `backend/app/services/agent_tool_policy_service.py`.

## Tool Registry

`backend/app/runtime/tool_registry.py` is the declaration authority for:

- tool id and description,
- input and output schema,
- source,
- scope,
- maturity,
- operation level,
- read/write type,
- destructive flag,
- confirmation requirement,
- feature flag,
- post-read verification.

Stable default tools can be listed by scope. Experimental, MCP, research, workspace, and write-capable tools remain filtered unless the route and feature flags allow them.

## Tool Policy Resolution

Tool exposure is resolved by `AgentToolPolicyResolver`:

```text
route decision
  + active skill allowed_tool_ids
  + selected scope
  + pending confirmation state
  + feature flags
  + Tool Registry metadata
  -> ToolPolicyDecision
```

`ToolPolicyDecision` contains:

- `allowed_tools`: declarations available to the runtime,
- `filtered_tools`: tool id to policy reason,
- `confirmation_required`: whether a pending confirmation is still needed,
- `reason`: a traceable summary.

## Read and Write Classes

Read tools can inspect papers, metadata, categories, tags, reports, evidence, or workspace files. Write tools mutate library relations, category/tag entities, report content, or workspace files.

Write tools must declare:

- `io_type = "write"`,
- operation level,
- write type,
- destructive status when applicable,
- confirmation requirement,
- verification tool when available.

## Skill Tool Binding

Skills can list tool ids in `allowed_tool_ids`. This expresses what the skill wants to use. It does not bypass route policy, feature flags, confirmation, or write safety.

Example:

```json
{
  "skill_id": "custom_review",
  "allowed_tool_ids": [
    "search_local/vector_recall_default"
  ]
}
```

## Observations

Runtime tool results should be wrapped as `ToolObservation` objects. The lifecycle helper `AgentToolObservationFactory` creates success and error observations with:

- tool name,
- success status,
- operation level,
- IO type,
- write type,
- affected objects,
- counts,
- data,
- evidence,
- verification,
- structured errors.

These observations feed trace, model feedback, and post-write verification.

Runtime dispatch also records compact execution policy metadata such as max steps, stop reason, allowed tool count, filtered tool count, and current target scope.
