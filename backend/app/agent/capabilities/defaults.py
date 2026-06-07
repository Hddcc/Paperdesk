"""Default capability declarations for PaperDesk."""

from __future__ import annotations

from app.models import (
    AgentOrchestrationPattern,
    CapabilityDeclaration,
    CapabilityMaturity,
    CapabilityRouteBinding,
    CapabilityToolBinding,
    PaperDeskRoute,
    PaperDeskRuntimeKind,
)

from .registry import CapabilityRegistry


def paper_capability_declaration() -> CapabilityDeclaration:
    """PaperDesk's main paper-reading domain capability."""

    return CapabilityDeclaration(
        capability_id="paper",
        display_name="Paper Reading",
        description="论文上传、解析、RAG、标签分类、论文库读写和报告生成能力。",
        domain_package="app.domains.paper",
        routes=[
            _route(PaperDeskRoute.PAPER_RAG, PaperDeskRuntimeKind.PAPER_RAG, AgentOrchestrationPattern.RETRIEVE_THEN_SYNTHESIZE),
            _route(PaperDeskRoute.LIBRARY_READ, PaperDeskRuntimeKind.TOOL_ACTION, AgentOrchestrationPattern.BOUNDED_REACT),
            _route(PaperDeskRoute.WRITE_PENDING, PaperDeskRuntimeKind.TOOL_ACTION, AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY),
            _route(PaperDeskRoute.WRITE_CONFIRMED, PaperDeskRuntimeKind.CONFIRMED_WRITE, AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY),
            _route(PaperDeskRoute.REPORT_ACTION, PaperDeskRuntimeKind.REPORT_ACTION, AgentOrchestrationPattern.SERVICE_WORKFLOW),
        ],
        tools=[
            CapabilityToolBinding(tool_id="library.explorer.stats"),
            CapabilityToolBinding(tool_id="library.explorer.find_documents"),
            CapabilityToolBinding(tool_id="evidence.retriever.search"),
            CapabilityToolBinding(tool_id="report.drafter.write"),
            CapabilityToolBinding(tool_id="library.operator.assign_category", read_only=False, risk_level="scoped_write", requires_confirmation=True),
            CapabilityToolBinding(tool_id="library.operator.clear_categories", read_only=False, risk_level="destructive", requires_confirmation=True),
        ],
        infrastructure_dependencies=["llm", "vectorstore", "persistence", "files"],
        skill_scopes=["paper", "knowledge", "shared"],
        documentation_summary="PaperDesk 的默认领域能力包，承载论文阅读业务闭环。",
    )


def chat_capability_declaration() -> CapabilityDeclaration:
    """General conversation capability."""

    return CapabilityDeclaration(
        capability_id="chat",
        display_name="Direct Chat",
        description="普通对话和无需论文库的单轮回答能力。",
        domain_package="app.agent.runtimes",
        routes=[
            _route(PaperDeskRoute.DIRECT_CHAT, PaperDeskRuntimeKind.DIRECT_CHAT, AgentOrchestrationPattern.SINGLE_TURN),
        ],
        skill_scopes=["chat", "shared"],
        documentation_summary="轻量直接对话入口，不默认进入 RAG 或工具循环。",
    )


def workspace_capability_declaration() -> CapabilityDeclaration:
    """Session workspace file capability."""

    return CapabilityDeclaration(
        capability_id="workspace",
        display_name="Workspace Files",
        description="会话工作区文件读取、创建和安全写入准备能力。",
        domain_package="app.domains.workspace",
        routes=[
            _route(PaperDeskRoute.WORKSPACE_READ, PaperDeskRuntimeKind.WORKSPACE_ACTION, AgentOrchestrationPattern.SERVICE_WORKFLOW),
            _route(PaperDeskRoute.WORKSPACE_WRITE, PaperDeskRuntimeKind.WORKSPACE_ACTION, AgentOrchestrationPattern.PREVIEW_CONFIRM_EXECUTE_VERIFY),
        ],
        tools=[
            CapabilityToolBinding(tool_id="workspace.file.list"),
            CapabilityToolBinding(tool_id="workspace.file.read"),
            CapabilityToolBinding(tool_id="workspace.file.write_new", read_only=False, risk_level="low_to_medium"),
            CapabilityToolBinding(tool_id="workspace.file.overwrite_prepare", read_only=False, risk_level="medium", requires_confirmation=True),
        ],
        infrastructure_dependencies=["files", "persistence"],
        skill_scopes=["workspace", "shared"],
        documentation_summary="会话文件和导出产物的安全工作区能力。",
    )


def research_capability_declaration() -> CapabilityDeclaration:
    """Experimental research orchestration capability."""

    return CapabilityDeclaration(
        capability_id="research",
        display_name="Experimental Research",
        description="显式实验路由中的 planner、reflection、MCP、subagent 等研究能力。",
        domain_package="app.runtime",
        routes=[
            _route(PaperDeskRoute.EXPERIMENTAL_RESEARCH, PaperDeskRuntimeKind.EXPERIMENTAL, AgentOrchestrationPattern.PLAN_EXECUTE_REPLAN),
        ],
        infrastructure_dependencies=["llm", "integrations"],
        skill_scopes=["research"],
        maturity=CapabilityMaturity.EXPERIMENTAL,
        documentation_summary="保留在显式实验入口下的研究型 Agent 能力。",
    )


def drawio_capability_declaration() -> CapabilityDeclaration:
    """Architecture-ready declaration for future draw.io support."""

    return CapabilityDeclaration(
        capability_id="drawio",
        display_name="draw.io Diagram",
        description="未来接入流程图、架构图、思维导图等 draw.io 产物操作。",
        domain_package="app.domains.artifact",
        infrastructure_dependencies=["integrations.drawio", "files"],
        skill_scopes=["artifact", "diagram"],
        maturity=CapabilityMaturity.EXPERIMENTAL,
        enabled=False,
        documentation_summary="扩展示例：通过 capability、tool、artifact、integration 四个落点接入 draw.io。",
        metadata={"example_only": True},
    )


def token_usage_capability_declaration() -> CapabilityDeclaration:
    """Architecture-ready declaration for future token usage views."""

    return CapabilityDeclaration(
        capability_id="token_usage",
        display_name="Token Usage",
        description="未来查看会话 token、latency、cost 等运行指标。",
        domain_package="app.agent.observability",
        infrastructure_dependencies=["llm"],
        skill_scopes=["observability"],
        maturity=CapabilityMaturity.EXPERIMENTAL,
        enabled=False,
        documentation_summary="扩展示例：通过 observability metrics 暴露只读 token 消耗能力。",
        metadata={"example_only": True},
    )


def default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for declaration in [
        chat_capability_declaration(),
        paper_capability_declaration(),
        workspace_capability_declaration(),
        research_capability_declaration(),
        drawio_capability_declaration(),
        token_usage_capability_declaration(),
    ]:
        registry.register(declaration)
    return registry


def _route(
    route: PaperDeskRoute,
    runtime: PaperDeskRuntimeKind,
    pattern: AgentOrchestrationPattern,
) -> CapabilityRouteBinding:
    return CapabilityRouteBinding(
        route=route.value,
        runtime=runtime.value,
        orchestration_pattern=pattern.value,
    )
