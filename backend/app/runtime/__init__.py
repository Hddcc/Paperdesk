"""Claude Code-style runtime helpers for PaperDesk."""

from .main_agent_runtime import MainAgentRuntime
from .message_bus import MessageBus
from .planner_candidate_provider import RuleBasedPlannerCandidateProvider
from .research_tool_executor import ResearchToolExecutor
from .scratchpad_store import ScratchpadStore
from .skill_registry import SkillRegistry
from .mcp_adapter import ReadOnlyMcpAdapter, default_read_only_academic_mcp_declarations
from .subagent_runner import SubagentRunner, WorkerResult
from .task_registry import TaskRegistry
from .tool_registry import ToolRegistry

__all__ = [
    "MainAgentRuntime",
    "MessageBus",
    "RuleBasedPlannerCandidateProvider",
    "ResearchToolExecutor",
    "ReadOnlyMcpAdapter",
    "default_read_only_academic_mcp_declarations",
    "ScratchpadStore",
    "SkillRegistry",
    "SubagentRunner",
    "TaskRegistry",
    "ToolRegistry",
    "WorkerResult",
]
