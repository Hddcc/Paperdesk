from app.models import AgentOrchestratorInput, AgentRunMode, KnowledgeRoute, MemoryHit, MemorySnapshot
from app.repositories import SQLiteRepository
from app.runtime import AgentOrchestrator, SkillRegistry, ToolRegistry


def _orchestrator(sandbox_dir):
    repository = SQLiteRepository(sandbox_dir / "orchestrator.db")
    return AgentOrchestrator(
        research_repository=repository.research,
        runtime_repository=repository.runtime,
        model="test-model",
        api_key=None,
        base_url=None,
    )


def _payload(prompt: str, *, selected_document_ids: list[str] | None = None, has_pending_action: bool = False):
    return AgentOrchestratorInput(
        session_id="session-1",
        message_id="message-1",
        user_prompt=prompt,
        selected_document_ids=selected_document_ids or [],
        memory_snapshot=MemorySnapshot(),
        available_tools=ToolRegistry().list_enabled(),
        available_skills=SkillRegistry().list_enabled(),
        runtime_context={"has_pending_action": has_pending_action},
    )


def test_route_to_run_mode_compatibility_contract():
    assert AgentOrchestrator._run_mode_for_route(KnowledgeRoute.DIRECT_ANSWER) == AgentRunMode.DIRECT
    assert AgentOrchestrator._run_mode_for_route(KnowledgeRoute.TOOL_ACTION) == AgentRunMode.REACT
    assert AgentOrchestrator._run_mode_for_route(KnowledgeRoute.CONFIRMED_WRITE) == AgentRunMode.REACT
    assert AgentOrchestrator._run_mode_for_route(KnowledgeRoute.OPTIONAL_PLANNER) == AgentRunMode.PLANNER
    assert AgentOrchestrator._run_mode_for_route(KnowledgeRoute.OPTIONAL_REFLECTION) == AgentRunMode.REFLECTION


def test_run_mode_to_default_route_compatibility_contract():
    assert AgentOrchestrator._default_route_for_run_mode(AgentRunMode.DIRECT) == KnowledgeRoute.DIRECT_ANSWER
    assert AgentOrchestrator._default_route_for_run_mode(AgentRunMode.REACT) == KnowledgeRoute.TOOL_ACTION
    assert AgentOrchestrator._default_route_for_run_mode(AgentRunMode.PLANNER) == KnowledgeRoute.OPTIONAL_PLANNER
    assert AgentOrchestrator._default_route_for_run_mode(AgentRunMode.REFLECTION) == KnowledgeRoute.OPTIONAL_REFLECTION


def test_confirmed_write_keeps_react_runtime_compatibility():
    mode = AgentOrchestrator._run_mode_for_route(KnowledgeRoute.CONFIRMED_WRITE)

    assert mode == AgentRunMode.REACT
    assert AgentOrchestrator._target_runtime_for(mode) == "KnowledgeAgentRuntime"


def test_orchestrator_routes_plain_chat_to_direct(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你好，帮我解释一下 RAG 是什么"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.route == KnowledgeRoute.DIRECT_ANSWER
    assert decision.intent == "chat"
    assert decision.requires_tools is False
    assert decision.trace_id.startswith("chat-orch-")
    assert decision.target_runtime == "DirectChatRuntime"


def test_orchestrator_routes_library_question_to_react(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.TOOL_ACTION
    assert decision.requires_tools is True
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_keeps_library_read_questions_in_tool_runtime_when_llm_routes_direct(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated the tag mapping question as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("每篇文章对应的标签是什么？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "library state questions must enter" in decision.reason


def test_orchestrator_routes_multistage_task_to_planner(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(
        _payload("先查哪些论文没有标签，再补上历史标签，然后按标签分别写总结")
    )

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.CONFIRMED_WRITE
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_routes_explicit_long_task_to_optional_planner(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(
        _payload("帮我基于这 20 篇论文写一个综述大纲，先分主题再总结")
    )

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.route == KnowledgeRoute.OPTIONAL_PLANNER
    assert decision.target_runtime == "KnowledgePlannerRuntime"


def test_orchestrator_routes_user_correction_to_reflection(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你刚才回答不对，重新检查一下"))

    assert decision.mode == AgentRunMode.REFLECTION
    assert decision.route == KnowledgeRoute.OPTIONAL_REFLECTION
    assert decision.target_runtime == "ReflectionRuntime"


def test_orchestrator_does_not_route_general_go_explanation_to_reflection(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("不对这个 Go 代码继续解释一下"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.target_runtime == "DirectChatRuntime"


def test_orchestrator_keeps_destructive_actions_on_confirmation_path(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("帮我删除分类法律"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.CONFIRMED_WRITE
    assert decision.requires_confirmation is True
    assert decision.risk_level == "high"
    assert decision.initial_context["permission_policy"] == "confirmation_required"


def test_orchestrator_treats_tag_replacement_as_verified_library_write(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated tag replacement as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("把里面带有hx23标签的论文，标签都换成“更新”"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.CONFIRMED_WRITE
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "library write requests must enter" in decision.reason


def test_orchestrator_treats_replacement_synonyms_as_verified_library_write(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated tag replacement synonym as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("把旧主题标签替换成新主题"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.CONFIRMED_WRITE
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_falls_back_to_rule_candidate_when_llm_candidate_is_absent(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.fallback_used is False


def test_orchestrator_blocks_planner_for_non_explicit_knowledge_question(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.PLANNER,
            reason="用户要先筛选标签再写对比。",
            confidence=0.92,
            target_runtime="KnowledgePlannerRuntime",
            required_capabilities=["knowledge_tools", "structured_plan"],
        ),
    )

    decision = orchestrator.select_mode(_payload("为我所有带着中文标签的论文写一篇对比，对比他们的区别"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.route == KnowledgeRoute.TOOL_ACTION
    assert decision.fallback_used is True
    assert decision.initial_context.get("action_plan") in (None, [])


def test_orchestrator_uses_low_confidence_llm_fallback(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.PLANNER,
            reason="Low confidence semantic guess.",
            confidence=0.2,
            target_runtime="KnowledgePlannerRuntime",
            source="llm",
        ),
    )

    decision = orchestrator.select_mode(_payload("你好，帮我解释一下 RAG 是什么"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.fallback_used is True


def test_orchestrator_keeps_selected_documents_grounded_when_llm_routes_direct(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM guessed this was plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
            source="llm",
        ),
    )

    decision = orchestrator.select_mode(_payload("总结这篇论文", selected_document_ids=["doc-1"]))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "document observations" in decision.reason


def test_orchestrator_preserves_llm_task_plan_for_explicit_long_task(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: self._llm_candidate_from_payload(
                {
                    "mode": "PLANNER",
                    "reason": "User wants a multi-step review outline.",
                    "confidence": 0.93,
                    "risk_level": "read_only",
                    "task_type": "collection_report",
                    "user_intent": "Create a staged literature review outline from selected papers.",
                    "entities": [
                        {"text": "20 篇论文", "type": "collection", "role": "object", "confidence": 0.9},
                    ],
                    "needs_tool": True,
                    "needs_plan": True,
                    "needs_verification": False,
                    "requested_output": "analysis_report",
                    "required_capabilities": ["structured_plan", "knowledge_tools"],
                    "action_plan": [
                        {
                            "tool": "library.explorer.find_documents",
                            "purpose": "Resolve the target paper collection.",
                            "arguments": {"query": "20 篇论文", "expected": "collection"},
                            "requires_verification_after": False,
                        }
                    ],
                }
        ),
    )

    decision = orchestrator.select_mode(_payload("帮我基于这 20 篇论文制定研究计划，先筛选，再分类，然后生成报告"))

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.route == KnowledgeRoute.OPTIONAL_PLANNER
    assert decision.initial_context["user_intent"].startswith("Create a staged")
    assert decision.initial_context["needs_verification"] is False
    assert decision.initial_context["entities"][0]["text"] == "20 篇论文"
    assert decision.initial_context["action_plan"][0]["tool"] == "library.explorer.find_documents"


def test_orchestrator_llm_prompt_includes_handlers_and_memory(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"
    captured: dict = {}

    class FakeMessage:
        content = '{"route":"DirectAnswer","reason":"普通解释问题。","confidence":0.91,"risk_level":"none","intent":"chat"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            _ = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("app.runtime.agent_orchestrator.OpenAI", FakeOpenAI)

    payload = _payload("帮我解释一下 RAG")
    payload.memory_snapshot = MemorySnapshot(
        items=[
            MemoryHit(
                id="memory-1",
                memory_type="user",
                summary="默认使用中文回答。",
                source_kind="runtime_user_file",
            )
        ]
    )

    decision = orchestrator.select_mode(payload)

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.route == KnowledgeRoute.DIRECT_ANSWER
    user_payload = __import__("json").loads(captured["messages"][1]["content"])
    assert [handler["name"] for handler in user_payload["available_handlers"]] == [
        "DirectAnswer",
        "ToolAction",
        "ConfirmedWrite",
    ]
    assert user_payload["memory_items"][0]["summary"] == "默认使用中文回答。"
    assert "available_skill_ids" not in user_payload


def test_orchestrator_filters_unknown_llm_tools_from_action_plan(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: self._validate_llm_candidate(
            payload,
            self._llm_candidate_from_payload(
                {
                    "route": "ToolAction",
                    "reason": "Need a library tool.",
                    "confidence": 0.9,
                    "needs_tool": True,
                    "action_plan": [
                        {"tool": "library.explorer.find_documents", "purpose": "Find documents."},
                        {"tool": "missing.tool", "purpose": "This tool is unavailable."},
                    ],
                }
            ),
        ),
    )

    decision = orchestrator.select_mode(_payload("查一下论文库里有哪些论文"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.initial_context["action_plan"] == [
        {"tool": "library.explorer.find_documents", "purpose": "Find documents."}
    ]
