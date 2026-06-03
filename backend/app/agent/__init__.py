"""Agent engineering boundary for PaperDesk.

Top-level exports are resolved lazily so service compatibility imports do not
create circular imports during the runtime-first migration.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentLifecycleResult",
    "AgentLifecycleService",
    "AgentToolPolicyResolver",
    "RuntimeDispatcher",
    "SkillRegistry",
    "SkillSelector",
    "ToolRegistry",
    "default_runtime_dispatcher",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentLifecycleResult", "AgentLifecycleService"}:
        from .lifecycle import AgentLifecycleResult, AgentLifecycleService

        return {
            "AgentLifecycleResult": AgentLifecycleResult,
            "AgentLifecycleService": AgentLifecycleService,
        }[name]
    if name in {"RuntimeDispatcher", "default_runtime_dispatcher"}:
        from .runtimes import RuntimeDispatcher, default_runtime_dispatcher

        return {
            "RuntimeDispatcher": RuntimeDispatcher,
            "default_runtime_dispatcher": default_runtime_dispatcher,
        }[name]
    if name in {"SkillRegistry", "SkillSelector"}:
        from .skills import SkillRegistry, SkillSelector

        return {
            "SkillRegistry": SkillRegistry,
            "SkillSelector": SkillSelector,
        }[name]
    if name in {"AgentToolPolicyResolver", "ToolRegistry"}:
        from .tools import AgentToolPolicyResolver, ToolRegistry

        return {
            "AgentToolPolicyResolver": AgentToolPolicyResolver,
            "ToolRegistry": ToolRegistry,
        }[name]
    raise AttributeError(name)
