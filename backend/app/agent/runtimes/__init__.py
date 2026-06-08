"""Route runtime implementations and dispatchers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentModeCompatibilityAdapter",
    "AgentModeExecutionAdapter",
    "ConfirmedWriteRuntime",
    "ChatAnswerRuntime",
    "DEFAULT_ROUTE_RUNTIME_MAP",
    "DirectChatRuntime",
    "DirectChatRuntimeExecutor",
    "ExperimentalRuntime",
    "LifecycleRuntime",
    "KnowledgeAgentCapabilityProvider",
    "PaperRagRuntime",
    "PaperRagRuntimeExecutor",
    "ReportRuntimeExecutor",
    "ReportActionRuntime",
    "RuntimeDispatcher",
    "RuntimeHandler",
    "RuntimeLLMClient",
    "RuntimeAnswerResult",
    "RuntimeLLMDiagnostic",
    "ExperimentalRuntimeExecutor",
    "ToolActionRuntime",
    "ToolActionRuntimeExecutor",
    "WorkspaceActionRuntime",
    "WorkspaceRuntimeExecutor",
    "WriteRuntimeExecutor",
    "default_runtime_dispatcher",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DirectChatRuntimeExecutor",
        "ChatAnswerRuntime",
        "ExperimentalRuntimeExecutor",
        "PaperRagRuntimeExecutor",
        "ReportRuntimeExecutor",
        "RuntimeAnswerResult",
        "RuntimeLLMClient",
        "RuntimeLLMDiagnostic",
        "ToolActionRuntimeExecutor",
        "WorkspaceRuntimeExecutor",
        "WriteRuntimeExecutor",
    }:
        from .chat_execution import (
            ChatAnswerRuntime,
            DirectChatRuntimeExecutor,
            ExperimentalRuntimeExecutor,
            PaperRagRuntimeExecutor,
            ReportRuntimeExecutor,
            RuntimeAnswerResult,
            RuntimeLLMClient,
            RuntimeLLMDiagnostic,
            ToolActionRuntimeExecutor,
            WorkspaceRuntimeExecutor,
            WriteRuntimeExecutor,
        )

        return {
            "ChatAnswerRuntime": ChatAnswerRuntime,
            "DirectChatRuntimeExecutor": DirectChatRuntimeExecutor,
            "ExperimentalRuntimeExecutor": ExperimentalRuntimeExecutor,
            "PaperRagRuntimeExecutor": PaperRagRuntimeExecutor,
            "ReportRuntimeExecutor": ReportRuntimeExecutor,
            "RuntimeAnswerResult": RuntimeAnswerResult,
            "RuntimeLLMClient": RuntimeLLMClient,
            "RuntimeLLMDiagnostic": RuntimeLLMDiagnostic,
            "ToolActionRuntimeExecutor": ToolActionRuntimeExecutor,
            "WorkspaceRuntimeExecutor": WorkspaceRuntimeExecutor,
            "WriteRuntimeExecutor": WriteRuntimeExecutor,
        }[name]
    if name == "KnowledgeAgentCapabilityProvider":
        from .knowledge_provider import KnowledgeAgentCapabilityProvider

        return KnowledgeAgentCapabilityProvider
    if name in {"AgentModeCompatibilityAdapter", "AgentModeExecutionAdapter"}:
        from .compatibility import AgentModeCompatibilityAdapter, AgentModeExecutionAdapter

        return {
            "AgentModeCompatibilityAdapter": AgentModeCompatibilityAdapter,
            "AgentModeExecutionAdapter": AgentModeExecutionAdapter,
        }[name]
    if name in {
        "DEFAULT_ROUTE_RUNTIME_MAP",
        "RuntimeDispatcher",
        "RuntimeHandler",
        "default_runtime_dispatcher",
    }:
        from .dispatcher import DEFAULT_ROUTE_RUNTIME_MAP, RuntimeDispatcher, RuntimeHandler, default_runtime_dispatcher

        return {
            "DEFAULT_ROUTE_RUNTIME_MAP": DEFAULT_ROUTE_RUNTIME_MAP,
            "RuntimeDispatcher": RuntimeDispatcher,
            "RuntimeHandler": RuntimeHandler,
            "default_runtime_dispatcher": default_runtime_dispatcher,
        }[name]
    if name in {
        "ConfirmedWriteRuntime",
        "DirectChatRuntime",
        "ExperimentalRuntime",
        "LifecycleRuntime",
        "PaperRagRuntime",
        "ReportActionRuntime",
        "ToolActionRuntime",
        "WorkspaceActionRuntime",
    }:
        from .lifecycle import (
            ConfirmedWriteRuntime,
            DirectChatRuntime,
            ExperimentalRuntime,
            LifecycleRuntime,
            PaperRagRuntime,
            ReportActionRuntime,
            ToolActionRuntime,
            WorkspaceActionRuntime,
        )

        return {
            "ConfirmedWriteRuntime": ConfirmedWriteRuntime,
            "DirectChatRuntime": DirectChatRuntime,
            "ExperimentalRuntime": ExperimentalRuntime,
            "LifecycleRuntime": LifecycleRuntime,
            "PaperRagRuntime": PaperRagRuntime,
            "ReportActionRuntime": ReportActionRuntime,
            "ToolActionRuntime": ToolActionRuntime,
            "WorkspaceActionRuntime": WorkspaceActionRuntime,
        }[name]
    raise AttributeError(name)
