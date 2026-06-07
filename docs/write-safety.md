# PaperDesk Write Safety

PaperDesk write operations must be scoped, previewed, confirmed, and traceable.

Write policy is represented in Agent lifecycle route metadata and Tool Registry declarations. The current compatibility implementation is documented under `backend/app/agent/tools`, `backend/app/agent/lifecycle`, and workspace/report services.

## Operation Levels

| Level | Meaning | Examples |
| --- | --- | --- |
| `entity-level` | Mutates an entity record. | Create, rename, or delete category/tag entities. |
| `relation-level` | Mutates relationships between records. | Assign categories/tags to selected papers. |
| `content-level` | Writes report or workspace content. | Save report, overwrite workspace file. |
| `query-level` | Applies a broad filter or query. | Batch operations over a filtered paper set. |

## Scope Requirements

Writes require explicit target scope. Accepted scope keys include:

- `document_ids`,
- `category_id`,
- `tag_id`,
- `report_id`,
- `workspace_path`,
- `query_filter`.

Vague references can resolve only to current selections, recent citations, the current report, or the current workspace file. Broad library writes require preview and confirmation.

## Pending Confirmation

The write lifecycle is:

```text
write intent
  -> scoped preview
  -> pending action
  -> user confirmation
  -> execute
  -> verification
  -> trace
```

`PendingWriteAction` stores:

- action id,
- route,
- operation level,
- target scope,
- affected objects,
- confirmation text,
- creation time,
- expiration time.

Execution is allowed only when the confirmation text and target scope match the pending action.

## Verification

Tools that declare verification behavior should run a post-write read check. The verification result is recorded in the normalized tool observation and lifecycle trace.

## Ambiguous Writes

When the user asks to clear, delete, rename, assign, overwrite, or save without a resolvable target, PaperDesk creates a non-executable plan or asks for clarification. It does not expand vague instructions to the entire paper library or workspace.

Lifecycle route decisions record `target_scope.scope_status`. Ambiguous library writes are marked as `needs_explicit_scope`; workspace writes require explicit path and confirmation before execution.
