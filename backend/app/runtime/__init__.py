"""Claude Code-style runtime helpers for PaperDesk."""

from .agent_orchestrator import AgentOrchestrator
from .agent_lifecycle import (
    DEFAULT_ROUTE_RUNTIME_MAP,
    RuntimeDispatcher,
    RuntimeHandler,
    default_runtime_dispatcher,
)
from .lifecycle_runtimes import (
    ConfirmedWriteRuntime,
    DirectChatRuntime,
    ExperimentalRuntime,
    LifecycleRuntime,
    PaperRagRuntime,
    ReportActionRuntime,
    ToolActionRuntime,
    WorkspaceActionRuntime,
)
from .main_agent_runtime import MainAgentRuntime
from .message_bus import MessageBus
from .knowledge_agent_runtime import KnowledgeAgentRuntime, KnowledgeAgentResult
from .knowledge_planner_runtime import KnowledgePlannerRuntime
from .planner_candidate_provider import RuleBasedPlannerCandidateProvider
from .reflection_runtime import ReflectionRuntime
from .research_tool_executor import ResearchToolExecutor
from .scratchpad_store import ScratchpadStore
from .skill_registry import SkillRegistry
from .mcp_adapter import ReadOnlyMcpAdapter, default_read_only_academic_mcp_declarations
from .subagent_runner import SubagentRunner, WorkerResult
from .task_registry import TaskRegistry
from .tool_registry import ToolRegistry

__all__ = [
    "AgentOrchestrator",
    "ConfirmedWriteRuntime",
    "DEFAULT_ROUTE_RUNTIME_MAP",
    "DirectChatRuntime",
    "ExperimentalRuntime",
    "LifecycleRuntime",
    "MainAgentRuntime",
    "MessageBus",
    "KnowledgeAgentRuntime",
    "KnowledgeAgentResult",
    "KnowledgePlannerRuntime",
    "PaperRagRuntime",
    "RuleBasedPlannerCandidateProvider",
    "ReflectionRuntime",
    "ReportActionRuntime",
    "ResearchToolExecutor",
    "RuntimeDispatcher",
    "RuntimeHandler",
    "ReadOnlyMcpAdapter",
    "default_read_only_academic_mcp_declarations",
    "ScratchpadStore",
    "SkillRegistry",
    "SubagentRunner",
    "TaskRegistry",
    "ToolRegistry",
    "ToolActionRuntime",
    "WorkspaceActionRuntime",
    "WorkerResult",
    "default_runtime_dispatcher",
]
