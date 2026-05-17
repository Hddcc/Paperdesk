from app.models import AgentOrchestratorInput, AgentRunMode, MemorySnapshot
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


def test_orchestrator_routes_plain_chat_to_direct(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你好，帮我解释一下 RAG 是什么"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.trace_id.startswith("chat-orch-")
    assert decision.target_runtime == "DirectChatRuntime"


def test_orchestrator_routes_library_question_to_react(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
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

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.target_runtime == "KnowledgePlannerRuntime"


def test_orchestrator_routes_user_correction_to_reflection(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你刚才回答不对，重新检查一下"))

    assert decision.mode == AgentRunMode.REFLECTION
    assert decision.target_runtime == "ReflectionRuntime"


def test_orchestrator_keeps_destructive_actions_on_confirmation_path(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("帮我删除分类法律"))

    assert decision.mode == AgentRunMode.REACT
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
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_falls_back_to_rule_candidate_when_llm_candidate_is_absent(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.fallback_used is False


def test_orchestrator_uses_high_confidence_llm_intention_over_rule_candidate(sandbox_dir, monkeypatch):
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

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.target_runtime == "KnowledgePlannerRuntime"
    assert decision.reason.startswith("LLM intention decision")
