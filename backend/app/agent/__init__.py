"""Agent engineering boundary for PaperDesk.

Top-level exports are resolved lazily so service compatibility imports do not
create circular imports during the runtime-first migration.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentLifecycleResult",
    "AgentLifecycleService",
    "AgentRagTraceService",
    "AgentRuntimeResponseRecorder",
    "AgentTraceRecorder",
    "AgentToolPolicyResolver",
    "CapabilityRegistry",
    "RuntimeDispatcher",
    "SkillRegistry",
    "SkillSelector",
    "ToolRegistry",
    "default_capability_registry",
    "default_runtime_dispatcher",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentLifecycleResult", "AgentLifecycleService"}:
        from .lifecycle import AgentLifecycleResult, AgentLifecycleService

        return {
            "AgentLifecycleResult": AgentLifecycleResult,
            "AgentLifecycleService": AgentLifecycleService,
        }[name]
    if name in {"AgentRagTraceService", "AgentRuntimeResponseRecorder", "AgentTraceRecorder"}:
        from .observability import AgentRagTraceService, AgentRuntimeResponseRecorder, AgentTraceRecorder

        return {
            "AgentRagTraceService": AgentRagTraceService,
            "AgentRuntimeResponseRecorder": AgentRuntimeResponseRecorder,
            "AgentTraceRecorder": AgentTraceRecorder,
        }[name]
    if name in {"RuntimeDispatcher", "default_runtime_dispatcher"}:
        from .runtimes import RuntimeDispatcher, default_runtime_dispatcher

        return {
            "RuntimeDispatcher": RuntimeDispatcher,
            "default_runtime_dispatcher": default_runtime_dispatcher,
        }[name]
    if name in {"CapabilityRegistry", "default_capability_registry"}:
        from .capabilities import CapabilityRegistry, default_capability_registry

        return {
            "CapabilityRegistry": CapabilityRegistry,
            "default_capability_registry": default_capability_registry,
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
