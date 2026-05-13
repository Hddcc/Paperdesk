"""Backend skill and tool capability models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .task_routing import ResearchArtifactProtocol, ResearchTaskType


class SkillExecutionMode(str, Enum):
    """Preferred execution path for a backend skill."""

    LIGHTWEIGHT = "lightweight"
    MAIN_AGENT = "main_agent"


class ToolSource(str, Enum):
    """Origin of a normalized tool declaration."""

    BUILTIN = "builtin"
    MCP = "mcp"


class ToolDeclaration(BaseModel):
    """Stable declaration for a tool that can be selected by the main agent."""

    tool_id: str
    source: ToolSource = ToolSource.BUILTIN
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True
    enabled: bool = True


class SkillDefinition(BaseModel):
    """File-backed backend capability definition."""

    skill_id: str
    name: str
    enabled: bool = True
    supported_task_types: list[ResearchTaskType] = Field(default_factory=list)
    default_execution_mode: SkillExecutionMode = SkillExecutionMode.MAIN_AGENT
    description: str
    body: str
    available_tools: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifact_protocol: ResearchArtifactProtocol
    version: str = "1.0.0"
    priority: int = 100


class SkillManifest(BaseModel):
    """Lightweight skill index loaded during discovery."""

    skill_id: str
    name: str
    enabled: bool = True
    supported_task_types: list[ResearchTaskType] = Field(default_factory=list)
    default_execution_mode: SkillExecutionMode = SkillExecutionMode.MAIN_AGENT
    description: str
    artifact_protocol: ResearchArtifactProtocol
    version: str = "1.0.0"
    priority: int = 100
    skill_file: str = "SKILL.md"
