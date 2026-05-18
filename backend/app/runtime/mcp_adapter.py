"""First-pass read-only MCP tool declaration adapter."""

from __future__ import annotations

from collections.abc import Iterable

from app.models import ToolDeclaration, ToolSource, ToolSpec


class ReadOnlyMcpAdapter:
    """Normalize whitelisted read-only MCP tool declarations."""

    def __init__(self, allowed_tool_ids: Iterable[str] | None = None) -> None:
        self.allowed_tool_ids = set(allowed_tool_ids or [])

    def normalize(self, declarations: Iterable[dict | ToolDeclaration]) -> list[ToolDeclaration]:
        tools: list[ToolDeclaration] = []
        for declaration in declarations:
            tool = self._normalize_one(declaration)
            if tool is not None:
                tools.append(tool)
        return tools

    def _normalize_one(self, declaration: dict | ToolDeclaration) -> ToolDeclaration | None:
        tool = (
            declaration
            if isinstance(declaration, ToolDeclaration)
            else ToolDeclaration.model_validate(declaration)
        )
        if self.allowed_tool_ids and tool.tool_id not in self.allowed_tool_ids:
            return None
        if not tool.enabled or not tool.read_only:
            return None
        if not tool.name or not tool.description:
            return None
        if not isinstance(tool.input_schema, dict) or not isinstance(tool.output_schema, dict):
            return None
        spec = tool.spec or ToolSpec(
            name=tool.tool_id,
            display_name=tool.name,
            description=tool.description,
            scope="experimental",
            maturity="experimental",
            operation_level="query-level",
            io_type="read",
            write_type="none",
            destructive=False,
            requires_confirmation=False,
            input_object_types=[],
            output_observation_type="mcp_read_observation",
            requires_post_read_verification=False,
            verification_tool=None,
            available_by_default=False,
            source=ToolSource.MCP.value,
        )
        safe_spec = spec.model_copy(
            update={
                "scope": spec.scope if spec.scope in {"research", "mcp", "experimental"} else "experimental",
                "maturity": "experimental",
                "operation_level": "query-level",
                "io_type": "read",
                "write_type": "none",
                "destructive": False,
                "requires_confirmation": False,
                "available_by_default": False,
                "feature_flag": "ENABLE_EXPERIMENTAL_MCP",
                "source": ToolSource.MCP.value,
            }
        )
        return tool.model_copy(update={"source": ToolSource.MCP, "read_only": True, "spec": safe_spec})


def default_read_only_academic_mcp_declarations() -> list[ToolDeclaration]:
    """First-party read-only academic MCP declarations used by the research runtime."""

    common_output = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "payload": {"type": "object"},
        },
    }
    return [
        ToolDeclaration(
            tool_id="mcp/academic_search",
            name="External academic search",
            description=(
                "Read-only academic search capability normalized through the MCP adapter; "
                "implemented by the current online paper search service."
            ),
            input_schema={
                "type": "object",
                "action_type": "search_online",
                "properties": {
                    "query": {"type": "string"},
                    "search_provider": {"type": "string"},
                    "top_k_online": {"type": "integer"},
                },
            },
            output_schema=common_output,
            read_only=True,
            enabled=True,
            spec=ToolSpec(
                name="mcp/academic_search",
                display_name="External academic search",
                description="Read-only external academic search declaration.",
                scope="mcp",
                maturity="experimental",
                operation_level="query-level",
                io_type="read",
                write_type="none",
                destructive=False,
                requires_confirmation=False,
                input_object_types=["paper"],
                output_observation_type="mcp_read_observation",
                available_by_default=False,
                feature_flag="ENABLE_EXPERIMENTAL_MCP",
                source=ToolSource.MCP.value,
            ),
        ),
        ToolDeclaration(
            tool_id="mcp/academic_metadata",
            name="External paper metadata lookup",
            description=(
                "Read-only paper metadata lookup declaration for DOI, title and source enrichment."
            ),
            input_schema={
                "type": "object",
                "action_type": "search_online",
                "properties": {
                    "title": {"type": "string"},
                    "doi": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
            output_schema=common_output,
            read_only=True,
            enabled=True,
            spec=ToolSpec(
                name="mcp/academic_metadata",
                display_name="External paper metadata lookup",
                description="Read-only external paper metadata lookup declaration.",
                scope="mcp",
                maturity="experimental",
                operation_level="query-level",
                io_type="read",
                write_type="none",
                destructive=False,
                requires_confirmation=False,
                input_object_types=["paper"],
                output_observation_type="mcp_read_observation",
                available_by_default=False,
                feature_flag="ENABLE_EXPERIMENTAL_MCP",
                source=ToolSource.MCP.value,
            ),
        ),
        ToolDeclaration(
            tool_id="mcp/read_only_web_fetch",
            name="Read-only source page fetch",
            description=(
                "Read-only web content fetch declaration reserved for future source page expansion."
            ),
            input_schema={
                "type": "object",
                "action_type": "search_online",
                "properties": {
                    "url": {"type": "string"},
                },
            },
            output_schema=common_output,
            read_only=True,
            enabled=True,
            spec=ToolSpec(
                name="mcp/read_only_web_fetch",
                display_name="Read-only source page fetch",
                description="Read-only external source page fetch declaration.",
                scope="mcp",
                maturity="experimental",
                operation_level="query-level",
                io_type="read",
                write_type="none",
                destructive=False,
                requires_confirmation=False,
                input_object_types=["paper"],
                output_observation_type="mcp_read_observation",
                available_by_default=False,
                feature_flag="ENABLE_EXPERIMENTAL_MCP",
                source=ToolSource.MCP.value,
            ),
        ),
    ]
