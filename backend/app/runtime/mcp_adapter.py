"""First-pass read-only MCP tool declaration adapter."""

from __future__ import annotations

from collections.abc import Iterable

from app.models import ToolDeclaration, ToolSource


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
        return tool.model_copy(update={"source": ToolSource.MCP})
