from __future__ import annotations

import json

from app.models import ResearchTaskType, SkillDefinition, SkillManifest, ToolDeclaration, ToolSource
from app.runtime import ReadOnlyMcpAdapter, SkillRegistry, ToolRegistry


def test_builtin_skill_registry_loads_layered_research_skills():
    registry = SkillRegistry()

    manifests = registry.list_enabled()

    assert {skill.skill_id for skill in manifests} == {
        "qa",
        "paper_summary",
        "multi_paper_review",
        "comparison",
        "method_explainer",
        "research_brief",
    }
    for skill in manifests:
        assert skill.description
        assert skill.artifact_protocol.title
        assert not hasattr(skill, "body")
        assert not hasattr(skill, "available_tools")


def test_skill_definition_is_loaded_only_when_requested():
    registry = SkillRegistry()

    manifest = registry.default_for(ResearchTaskType.QA)
    assert manifest is not None
    definition = registry.load_definition(manifest.skill_id)

    assert definition is not None
    assert definition.skill_id == manifest.skill_id
    assert definition.body.startswith("# 知识问答")
    assert definition.available_tools
    assert definition.references


def test_skill_registry_selects_default_by_task_type():
    registry = SkillRegistry()

    skill = registry.default_for(ResearchTaskType.MULTI_PAPER_REVIEW)

    assert skill is not None
    assert skill.skill_id == "multi_paper_review"
    assert skill.artifact_protocol.protocol_type.value == "review"


def test_disabled_skill_is_not_a_candidate(sandbox_dir):
    skill_dir = sandbox_dir / "skills"
    skill_dir.mkdir()
    manifest = SkillManifest(
        skill_id="disabled_qa",
        name="Disabled QA",
        enabled=False,
        supported_task_types=[ResearchTaskType.QA],
        description="Disabled test skill.",
        artifact_protocol=SkillRegistry().default_for(ResearchTaskType.QA).artifact_protocol,
    )
    disabled_dir = skill_dir / "disabled_qa"
    disabled_dir.mkdir()
    (disabled_dir / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    (disabled_dir / "SKILL.md").write_text("# Disabled\n", encoding="utf-8")

    registry = SkillRegistry(skill_dir)

    assert registry.candidates_for(ResearchTaskType.QA) == []


def test_skill_available_tools_resolve_to_tool_declarations():
    skill_registry = SkillRegistry()
    tool_registry = ToolRegistry(enable_experimental_mcp=True)
    manifest = skill_registry.default_for(ResearchTaskType.RESEARCH_BRIEF_TASK)
    assert manifest is not None
    skill = skill_registry.load_definition(manifest.skill_id)

    assert skill is not None
    declarations = tool_registry.filter_enabled(skill.available_tools)

    assert {tool.tool_id for tool in declarations} == set(skill.available_tools)
    assert any(tool.tool_id == "mcp/academic_search" and tool.source == ToolSource.MCP for tool in declarations)
    assert any(tool.source == ToolSource.BUILTIN for tool in declarations)
    assert all(tool.input_schema for tool in declarations)
    assert all(tool.output_schema for tool in declarations)


def test_tool_registry_declarations_are_serializable():
    registry = ToolRegistry(enable_experimental_mcp=True)

    tools = registry.list_enabled()

    assert tools
    assert any(tool.tool_id == "search_online/mixed_broad_recall" for tool in tools)
    assert any(tool.tool_id == "mcp/academic_search" and tool.source == ToolSource.MCP for tool in tools)
    for tool in tools:
        payload = tool.model_dump(mode="json")
        assert payload["tool_id"]
        assert payload["source"] in {"builtin", "mcp"}
        assert isinstance(payload["read_only"], bool)
        assert payload["enabled"] is True
        assert isinstance(payload["input_schema"], dict)
        assert isinstance(payload["output_schema"], dict)
        assert payload["spec"] is not None
        assert payload["spec"]["name"] == payload["tool_id"]
        assert payload["spec"]["operation_level"] in {"query-level", "entity-level", "relation-level", "content-level"}
        assert payload["spec"]["io_type"] in {"read", "write"}


def test_knowledge_default_tool_candidates_are_metadata_filtered():
    registry = ToolRegistry()

    candidates = registry.list_default_candidates(scope="knowledge")
    candidate_ids = {tool.tool_id for tool in candidates}

    assert "library.operator.delete_unused_categories" in candidate_ids
    assert "library.operator.clear_categories" in candidate_ids
    assert "memory.write" not in candidate_ids
    assert all(not tool.tool_id.startswith("mcp/") for tool in candidates)
    assert all(tool.spec is not None for tool in candidates)
    assert all(tool.spec.available_by_default for tool in candidates if tool.spec is not None)
    assert all(tool.spec.scope != "experimental" for tool in candidates if tool.spec is not None)
    assert all(tool.spec.maturity == "stable" for tool in candidates if tool.spec is not None)


def test_knowledge_default_tool_candidates_do_not_include_non_knowledge_scopes():
    registry = ToolRegistry(enable_experimental_mcp=True)

    candidates = registry.list_default_candidates(scope="knowledge")
    candidate_scopes = {tool.spec.scope for tool in candidates if tool.spec is not None}

    assert "research" not in candidate_scopes
    assert "mcp" not in candidate_scopes
    assert "experimental" not in candidate_scopes


def test_knowledge_write_tool_specs_distinguish_entity_and_relation_operations():
    registry = ToolRegistry()

    delete_unused = registry.get("library.operator.delete_unused_categories")
    clear_categories = registry.get("library.operator.clear_categories")
    list_by_category = registry.get("library.explorer.find_documents")

    assert delete_unused is not None and delete_unused.spec is not None
    assert delete_unused.spec.operation_level == "entity-level"
    assert delete_unused.spec.io_type == "write"
    assert delete_unused.spec.write_type == "delete"
    assert delete_unused.spec.destructive is True
    assert delete_unused.spec.requires_confirmation is True
    assert delete_unused.spec.requires_post_read_verification is True

    assert clear_categories is not None and clear_categories.spec is not None
    assert clear_categories.spec.operation_level == "relation-level"
    assert clear_categories.spec.write_type == "clear"
    assert clear_categories.spec.input_object_types == ["paper-category relation"]

    assert list_by_category is not None and list_by_category.spec is not None
    assert list_by_category.spec.operation_level == "query-level"
    assert list_by_category.spec.io_type == "read"
    assert list_by_category.spec.destructive is False
    assert list_by_category.spec.requires_confirmation is False


def test_read_only_mcp_adapter_accepts_only_whitelisted_read_only_tools():
    adapter = ReadOnlyMcpAdapter(allowed_tool_ids={"mcp/search"})
    declarations = [
        {
            "tool_id": "mcp/search",
            "name": "External Search",
            "description": "Read-only external search.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "read_only": True,
            "enabled": True,
        },
        {
            "tool_id": "mcp/write",
            "name": "External Writer",
            "description": "Should be rejected because it writes.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "read_only": False,
            "enabled": True,
        },
        ToolDeclaration(
            tool_id="mcp/disabled",
            name="Disabled",
            description="Should be rejected because disabled.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            read_only=True,
            enabled=False,
        ),
    ]

    tools = adapter.normalize(declarations)

    assert len(tools) == 1
    assert tools[0].tool_id == "mcp/search"
    assert tools[0].source == ToolSource.MCP
    assert tools[0].read_only is True


def test_mcp_absence_keeps_builtin_tools_available():
    adapter = ReadOnlyMcpAdapter()
    registry = ToolRegistry(adapter.normalize([]))

    assert registry.get("plan/rule_based_initial") is not None
    assert registry.get("search_local/vector_recall_default") is not None


def test_default_tool_registry_keeps_mcp_disabled_by_default():
    registry = ToolRegistry()

    assert registry.get("mcp/academic_search") is None


def test_enabled_tool_registry_exposes_read_only_academic_mcp_tools():
    registry = ToolRegistry(enable_experimental_mcp=True)

    academic_search = registry.get("mcp/academic_search")

    assert academic_search is not None
    assert academic_search.source == ToolSource.MCP
    assert academic_search.read_only is True
    assert academic_search.input_schema["action_type"] == "search_online"
    assert academic_search.spec is not None
    assert academic_search.spec.io_type == "read"
    assert academic_search.spec.operation_level == "query-level"
    assert academic_search.spec.maturity == "experimental"
    assert academic_search.spec.available_by_default is False
