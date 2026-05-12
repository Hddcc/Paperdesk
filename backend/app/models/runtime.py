"""Runtime coordination models for Claude Code-style subagents."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SubagentProfile(str, Enum):
    """Supported subagent profiles."""

    EXPLORE = "explore"
    IMPLEMENT = "implement"
    VERIFY = "verify"


class AgentTaskStatus(str, Enum):
    """Lifecycle state for a runtime-managed agent task."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    TIMED_OUT = "timed_out"


class ControlMessageType(str, Enum):
    """Control-plane messages exchanged by the runtime."""

    SPAWN = "spawn"
    CONTINUE = "continue"
    STOP = "stop"
    HEARTBEAT = "heartbeat"
    TIMEOUT = "timeout"


class TraceEventType(str, Enum):
    """Execution trace categories stored for coordinator visibility."""

    CONTROL = "control"
    STATUS = "status"
    NOTIFICATION = "notification"
    ARTIFACT = "artifact"
    MERGE = "merge"


class ToolPolicy(BaseModel):
    """Explicit tool capability envelope for a subagent task."""

    read_only: bool = True
    network_allowed: bool = False
    workspace_write: bool = False
    db_write: bool = False


class TaskArtifactRef(BaseModel):
    """Reference to a scratchpad or workspace artifact."""

    name: str
    path: str
    kind: str = "file"
    description: str | None = None


class AgentTask(BaseModel):
    """Structured task packet passed from the main agent to a subagent."""

    id: str
    run_id: str
    parent_task_id: str | None = None
    profile: SubagentProfile
    goal: str
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    done_criteria: str
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    artifact_dir: str


class TaskNotification(BaseModel):
    """Structured result payload returned by a subagent."""

    task_id: str
    agent_profile: SubagentProfile
    status: AgentTaskStatus
    summary: str
    result_payload: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    artifact_refs: list[TaskArtifactRef] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime

    def to_xml_block(self) -> str:
        """Render a compact Claude Code-like notification block."""

        lines = [
            "<task-notification>",
            f"<task_id>{self.task_id}</task_id>",
            f"<agent_profile>{self.agent_profile.value}</agent_profile>",
            f"<status>{self.status.value}</status>",
            f"<summary>{self.summary}</summary>",
        ]
        if self.error:
            lines.append(f"<error>{self.error}</error>")
        for artifact in self.artifact_refs:
            lines.append(
                f'<artifact name="{artifact.name}" kind="{artifact.kind}" path="{artifact.path}">'
                f"{artifact.description or ''}</artifact>"
            )
        lines.append("</task-notification>")
        return "\n".join(lines)


class TaskExecutionTrace(BaseModel):
    """Persisted trace entry for coordinator and subagent activity."""

    run_id: str
    task_id: str | None = None
    trace_type: TraceEventType
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CoordinatorDecision(BaseModel):
    """Main-agent decision for how to process a research request."""

    action: str
    reason: str
    spawn_subagents: bool = False
    profile: SubagentProfile | None = None


class SpawnTask(BaseModel):
    """Control-plane command for creating a fresh subagent task."""

    type: ControlMessageType = ControlMessageType.SPAWN
    task: AgentTask


class ContinueTask(BaseModel):
    """Control-plane command for reusing an existing subagent context."""

    type: ControlMessageType = ControlMessageType.CONTINUE
    task_id: str
    goal: str | None = None
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    done_criteria: str | None = None


class StopTask(BaseModel):
    """Control-plane command for stopping a subagent task."""

    type: ControlMessageType = ControlMessageType.STOP
    task_id: str
    reason: str | None = None


class Heartbeat(BaseModel):
    """Control-plane liveness message."""

    type: ControlMessageType = ControlMessageType.HEARTBEAT
    task_id: str
    message: str = "alive"


class TaskTimeout(BaseModel):
    """Control-plane timeout signal."""

    type: ControlMessageType = ControlMessageType.TIMEOUT
    task_id: str
    timeout_seconds: float


class StoredAgentTask(BaseModel):
    """Persisted task metadata with runtime status fields."""

    id: str
    run_id: str
    parent_task_id: str | None = None
    profile: SubagentProfile
    goal: str
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    done_criteria: str
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    artifact_dir: str
    status: AgentTaskStatus = AgentTaskStatus.CREATED
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: AgentTaskStatus | str) -> AgentTaskStatus:
        if isinstance(value, AgentTaskStatus):
            return value
        return AgentTaskStatus(value)
