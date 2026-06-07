# PaperDesk Skills

Skills are PaperDesk Agent capabilities that describe task style, output protocol, trigger signals, and tool bindings. They can guide how a request is handled, while Tool Registry and runtime guardrails still control actual permissions.

Agent-facing imports are exposed through `backend/app/agent/skills`. The registry implementation now lives there, while `backend/app/runtime/skill_registry.py` remains a compatibility wrapper for older imports.

## Locations

PaperDesk loads built-in skills from:

```text
backend/app/skills/builtin/<skill_id>/
  manifest.json
  SKILL.md
```

Custom skills can be loaded from configured directories passed to `SkillRegistry(custom_skill_dirs=[...])`:

```text
<custom-skills-root>/<skill_id>/
  manifest.json
  SKILL.md
```

The custom directory style follows the same local skill pattern used by Claude Code-like skills: a manifest indexes the capability, and `SKILL.md` contains the human-readable task instructions.

## Manifest Fields

Required or commonly used fields:

```json
{
  "skill_id": "custom_review",
  "name": "Custom Review",
  "enabled": true,
  "supported_task_types": ["qa"],
  "default_execution_mode": "main_agent",
  "description": "Review selected papers with a custom rubric.",
  "artifact_protocol": {
    "protocol_type": "qa",
    "title": "Custom Review",
    "required_sections": ["Summary", "Evidence", "Limitations"],
    "citation_required": true
  },
  "skill_file": "SKILL.md",
  "scope": "shared",
  "source": "custom",
  "maturity": "stable",
  "available_by_default": true,
  "priority": 50,
  "trigger": {
    "keywords": ["custom-review"],
    "commands": [],
    "intent_hints": [],
    "routes": ["paper_rag"],
    "task_types": ["qa"],
    "attachment_kinds": ["library_document"],
    "fallback": false,
    "confidence": 0.9
  },
  "allowed_tool_ids": [
    "search_local/vector_recall_default"
  ]
}
```

## Tool Binding

`allowed_tool_ids` declares which tools the skill can request. It does not grant execution permission by itself.

Runtime exposure is resolved by:

```text
route + active skill + selected scope + confirmation state + feature flags + Tool Registry metadata
```

This means a custom skill can bind read-only evidence tools, while write tools still require explicit scope, preview, pending confirmation, and verification when the Tool Registry declares those requirements.

## Prompt Safety

PaperDesk renders a compact skill context into the model prompt:

- skill id and name,
- short description,
- artifact protocol,
- allowed tool ids,
- output expectations,
- safety constraints,
- trigger reason.

Raw `SKILL.md` content, hidden contracts, local paths, secrets, executable details, and raw tool schemas are kept out of the prompt-safe context.

## Validation

Custom skill manifests must:

- use a relative `skill_file` path inside the skill directory,
- pass Pydantic manifest validation,
- use supported source, maturity, scope, and trigger fields,
- keep tool access in `allowed_tool_ids`,
- provide a readable skill body.

Invalid custom skills are excluded from selection and recorded in `SkillRegistry.validation_errors`.

Custom skills are designed as user-extensible Agent capabilities. A user can add a local skill directory, declare trigger metadata, bind allowed tools, and let PaperDesk apply route/runtime/tool policy before execution.
