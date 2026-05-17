from app.models import ReflectionImprovementAction
from app.runtime.knowledge_agent_runtime import KnowledgeAgentResult
from app.runtime.reflection_runtime import ReflectionRuntime


def test_reflection_payload_is_clamped_and_extended_schema_is_recorded():
    runtime = object.__new__(ReflectionRuntime)

    reflection = runtime._reflection_result_from_payload(
        {
            "overall_score": 42,
            "intent_score": -5,
            "tool_score": 8,
            "evidence_score": 7,
            "answer_score": 6,
            "completion_score": 5,
            "risk_level": "invalid",
            "needs_tool_recheck": "true",
            "needed_tool_type": "rag",
            "issues": ["  missing evidence  ", "missing evidence"],
            "improvement_actions": [
                {
                    "type": "call_tool",
                    "tool": "evidence.retriever.search",
                    "args": {},
                    "reason": "recheck",
                }
            ],
            "should_retry": False,
            "memory_lessons": [" use tools "],
        },
        source="llm",
    )

    assert reflection.overall_score == 10
    assert reflection.intent_score == 1
    assert reflection.should_retry is False
    assert reflection.issues == ["missing evidence"]
    assert runtime._reflection_meta(reflection)["risk_level"] == "safe"
    assert runtime._reflection_meta(reflection)["needed_tool_type"] == "rag"
    assert runtime._reflection_meta(reflection)["needs_tool_recheck"] is True


def test_reflection_tool_recheck_can_trigger_improvement_without_low_score():
    runtime = object.__new__(ReflectionRuntime)
    reflection = runtime._reflection_result_from_payload(
        {
            "overall_score": 8,
            "intent_score": 8,
            "tool_score": 8,
            "evidence_score": 8,
            "answer_score": 8,
            "completion_score": 8,
            "risk_level": "read_only",
            "needs_tool_recheck": True,
            "needed_tool_type": "category",
            "issues": [],
            "improvement_actions": [],
            "should_retry": False,
            "memory_lessons": [],
        },
        source="llm",
    )

    assert runtime._should_run_improvement(
        user_goal="重新检查这些论文标签",
        result=KnowledgeAgentResult(content="done"),
        reflection=reflection,
    )


def test_reflection_guardrails_block_destructive_or_mutated_retry():
    runtime = object.__new__(ReflectionRuntime)
    reflection = runtime._with_reflection_meta(
        runtime._normalize_reflection_result(
            runtime._fallback_evaluate(
                user_goal="解释 RAG",
                final_answer="",
                traces=[],
                result=KnowledgeAgentResult(content=""),
                mode="REACT",
                user_feedback=None,
            )
        ),
        {
            "source": "llm",
            "risk_level": "destructive",
            "needs_tool_recheck": True,
            "needed_tool_type": "operator_verify",
            "schema_valid": True,
        },
    )

    assert not runtime._should_run_improvement(
        user_goal="删除这篇论文",
        result=KnowledgeAgentResult(content="old"),
        reflection=reflection,
    )
    assert not runtime._should_run_improvement(
        user_goal="给论文打标签",
        result=KnowledgeAgentResult(content="old", library_mutated=True),
        reflection=reflection,
    )


def test_reflection_fallback_marks_tool_recheck_for_missing_observation():
    runtime = object.__new__(ReflectionRuntime)
    reflection = runtime._fallback_evaluate(
        user_goal="论文库里有几篇论文？",
        final_answer="我猜有三篇",
        traces=[],
        result=KnowledgeAgentResult(content="我猜有三篇"),
        mode="REACT",
        user_feedback=None,
    )

    assert reflection.should_retry is True
    assert runtime._reflection_meta(reflection)["source"] == "fallback_rule"
    assert runtime._reflection_meta(reflection)["needs_tool_recheck"] is True
    assert any(isinstance(action, ReflectionImprovementAction) for action in reflection.improvement_actions)
