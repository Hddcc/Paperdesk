"""One-off Knowledge Chat evaluation for PaperDesk.

The script keeps all evaluation artifacts under the judge directory and does
not modify backend source code. It uses the real FastAPI route through
TestClient and wraps the OpenAI-compatible client symbols imported by the
backend so LLM calls can be measured without touching production modules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import traceback
from typing import Any


JUDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = JUDGE_DIR.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
DEFAULT_CASES_PATH = JUDGE_DIR / "golden_cases.jsonl"


@dataclass
class LlmCall:
    index: int
    started_at: str
    module: str
    model: str | None
    stream: bool
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    usage_missing: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "started_at": self.started_at,
            "module": self.module,
            "model": self.model,
            "stream": self.stream,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_missing": self.usage_missing,
            "error": self.error,
        }


class LlmRecorder:
    def __init__(self) -> None:
        self.calls: list[LlmCall] = []

    def snapshot_len(self) -> int:
        return len(self.calls)

    def calls_since(self, offset: int) -> list[dict[str, Any]]:
        return [call.as_dict() for call in self.calls[offset:]]

    def record(
        self,
        *,
        module: str,
        model: str | None,
        stream: bool,
        started_at: float,
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        usage = getattr(response, "usage", None) if response is not None else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        self.calls.append(
            LlmCall(
                index=len(self.calls) + 1,
                started_at=datetime.now().isoformat(timespec="seconds"),
                module=module,
                model=model,
                stream=stream,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                usage_missing=usage is None,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )


class _CompletionProxy:
    def __init__(self, inner: Any, recorder: LlmRecorder, module_name: str) -> None:
        self._inner = inner
        self._recorder = recorder
        self._module_name = module_name

    def create(self, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        model = kwargs.get("model")
        stream = bool(kwargs.get("stream"))
        try:
            response = self._inner.create(*args, **kwargs)
        except BaseException as exc:
            self._recorder.record(
                module=self._module_name,
                model=model,
                stream=stream,
                started_at=started_at,
                error=exc,
            )
            raise
        if stream:
            return _StreamingResponseProxy(
                response=response,
                recorder=self._recorder,
                module_name=self._module_name,
                model=model,
                started_at=started_at,
            )
        self._recorder.record(
            module=self._module_name,
            model=model,
            stream=False,
            started_at=started_at,
            response=response,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StreamingResponseProxy:
    def __init__(
        self,
        *,
        response: Any,
        recorder: LlmRecorder,
        module_name: str,
        model: str | None,
        started_at: float,
    ) -> None:
        self._response = response
        self._recorder = recorder
        self._module_name = module_name
        self._model = model
        self._started_at = started_at
        self._recorded = False

    def __iter__(self):
        try:
            for item in self._response:
                yield item
        except BaseException as exc:
            self._record_once(error=exc)
            raise
        self._record_once()

    def _record_once(self, error: BaseException | None = None) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._recorder.record(
            module=self._module_name,
            model=self._model,
            stream=True,
            started_at=self._started_at,
            error=error,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class _ChatProxy:
    def __init__(self, inner: Any, recorder: LlmRecorder, module_name: str) -> None:
        self._inner = inner
        self.completions = _CompletionProxy(inner.completions, recorder, module_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_openai_wrapper(real_openai: Any, recorder: LlmRecorder, module_name: str):
    class RecordingOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._client = real_openai(*args, **kwargs)
            self.chat = _ChatProxy(self._client.chat, recorder, module_name)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._client, name)

    RecordingOpenAI.__name__ = f"RecordingOpenAI_{module_name.replace('.', '_')}"
    return RecordingOpenAI


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def preflight(cases: list[dict[str, Any]]) -> dict[str, Any]:
    db_path = BACKEND_ROOT / "data" / "paperdesk.db"
    if not db_path.exists():
        raise RuntimeError(f"SQLite database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        doc_count = conn.execute("SELECT count(*) FROM library_documents").fetchone()[0]
        chunk_count = conn.execute("SELECT count(*) FROM library_chunks").fetchone()[0]
        documents = [
            dict(row)
            for row in conn.execute(
                "SELECT id, title, page_count, status FROM library_documents ORDER BY uploaded_at ASC"
            )
        ]
        categories = [
            dict(row)
            for row in conn.execute("SELECT id, name FROM document_categories ORDER BY name ASC")
        ]
    if doc_count < 9:
        raise RuntimeError(f"Expected at least 9 library documents, got {doc_count}")
    if chunk_count <= 0:
        raise RuntimeError("Expected library_chunks to contain retrievable text")

    selected_ids = {
        document_id
        for case in cases
        for document_id in case.get("selected_document_ids", [])
    }
    known_ids = {item["id"] for item in documents}
    missing_ids = sorted(selected_ids - known_ids)
    if missing_ids:
        raise RuntimeError(f"Golden cases reference missing document ids: {missing_ids}")

    sys.path.insert(0, str(BACKEND_ROOT))
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    if not settings.effective_llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured")
    return {
        "db_path": str(db_path),
        "document_count": doc_count,
        "chunk_count": chunk_count,
        "documents": documents,
        "categories": categories,
        "llm_model": settings.effective_llm_model,
        "llm_base_url_configured": bool(settings.effective_llm_base_url),
        "llm_api_key_configured": bool(settings.effective_llm_api_key),
    }


def install_llm_recorders(recorder: LlmRecorder) -> None:
    import openai
    import app.runtime.agent_orchestrator as agent_orchestrator
    import app.runtime.knowledge_agent_runtime as knowledge_agent_runtime
    import app.services.chat_service as chat_service
    import app.services.query_translation_service as query_translation_service
    import app.services.rag_service as rag_service

    real_openai = openai.OpenAI
    targets = [
        (agent_orchestrator, "app.runtime.agent_orchestrator"),
        (knowledge_agent_runtime, "app.runtime.knowledge_agent_runtime"),
        (chat_service, "app.services.chat_service"),
        (query_translation_service, "app.services.query_translation_service"),
        (rag_service, "app.services.rag_service"),
    ]
    for module, module_name in targets:
        setattr(module, "OpenAI", build_openai_wrapper(real_openai, recorder, module_name))


def create_client():
    from fastapi.testclient import TestClient

    from app.api.main import (
        create_app,
        get_agent_orchestrator,
        get_chat_memory_service,
        get_chat_service,
        get_context_assembler,
        get_context_budget_service,
        get_context_compaction_service,
        get_context_file_store,
        get_document_library_service,
        get_embedding_service,
        get_export_service,
        get_knowledge_agent_runtime,
        get_knowledge_ingestion_service,
        get_knowledge_planner_runtime,
        get_milvus_bootstrap_service,
        get_paper_analysis_agent,
        get_paper_search_service,
        get_paper_selection_agent,
        get_query_translation_service,
        get_rag_service,
        get_reflection_runtime,
        get_report_lifecycle_service,
        get_report_writer,
        get_repository,
        get_research_context_assembler,
        get_research_orchestrator,
        get_research_workspace_service,
        get_vectorstore,
    )
    from app.config import get_settings

    for cached in [
        get_settings,
        get_repository,
        get_embedding_service,
        get_milvus_bootstrap_service,
        get_vectorstore,
        get_paper_search_service,
        get_paper_selection_agent,
        get_paper_analysis_agent,
        get_query_translation_service,
        get_context_file_store,
        get_context_budget_service,
        get_context_compaction_service,
        get_context_assembler,
        get_research_context_assembler,
        get_chat_memory_service,
        get_chat_service,
        get_agent_orchestrator,
        get_document_library_service,
        get_knowledge_ingestion_service,
        get_knowledge_agent_runtime,
        get_knowledge_planner_runtime,
        get_export_service,
        get_research_workspace_service,
        get_report_writer,
        get_report_lifecycle_service,
        get_rag_service,
        get_research_orchestrator,
        get_reflection_runtime,
    ]:
        cached.cache_clear()
    return TestClient(create_app())


def trace_rows(trace_id: str | None) -> list[dict[str, Any]]:
    if not trace_id:
        return []
    from app.api.main import get_runtime_repository

    return [
        trace.model_dump(mode="json")
        for trace in get_runtime_repository().list_traces(trace_id)
    ]


def tools_from_traces(traces: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    for trace in traces:
        payload = trace.get("payload") or {}
        if trace.get("status") == "tool_call_log":
            tool = payload.get("tool_name")
        elif trace.get("status") in {"react_observation", "tool_finished", "retrieval_tool_finished"}:
            tool = payload.get("tool")
        else:
            tool = None
        if isinstance(tool, str) and tool and tool not in tools:
            tools.append(tool)
    return tools


def reasoning_steps(traces: list[dict[str, Any]]) -> int:
    tool_logs = sum(1 for trace in traces if trace.get("status") == "tool_call_log")
    observations = sum(1 for trace in traces if trace.get("status") == "react_observation")
    return max(tool_logs, observations)


def case_token_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    totals: dict[str, Any] = {}
    for field in fields:
        values = [call.get(field) for call in calls if isinstance(call.get(field), int)]
        totals[field] = sum(values) if values else None
    totals["missing_usage_calls"] = sum(1 for call in calls if call.get("usage_missing"))
    return totals


def contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lowered = text.casefold()
    missing = [needle for needle in needles if needle.casefold() not in lowered]
    return not missing, missing


def contains_all_groups(text: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    lowered = text.casefold()
    missing: list[str] = []
    for group in groups:
        if not group:
            continue
        if not any(str(needle).casefold() in lowered for needle in group):
            missing.append(" OR ".join(str(needle) for needle in group))
    return not missing, missing


def contains_any(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lowered = text.casefold()
    hits = [needle for needle in needles if needle.casefold() in lowered]
    return bool(hits), hits


def score_case(
    case: dict[str, Any],
    response_payload: dict[str, Any] | None,
    traces: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    assistant = (response_payload or {}).get("assistant_message") or {}
    content = str(assistant.get("content") or "")
    action_status = assistant.get("action_status")
    retrieval_status = assistant.get("retrieval_status")
    used_document_ids = assistant.get("used_document_ids") or []
    actual_tools = tools_from_traces(traces)

    required_ok, missing_keywords = contains_all(content, case.get("required_keywords", []))
    group_ok, missing_groups = contains_all_groups(content, case.get("required_keyword_groups", []))
    required_ok = required_ok and group_ok
    missing_keywords.extend(missing_groups)
    hallucinated, hallucination_hits = contains_any(content, case.get("forbidden_claims", []))
    expected_tools = case.get("expected_tools", [])
    forbidden_tools = case.get("forbidden_tools", [])
    missing_tools = [tool for tool in expected_tools if tool not in actual_tools]
    forbidden_tool_hits = [tool for tool in forbidden_tools if tool in actual_tools]
    expected_doc_ids = case.get("expected_document_ids", [])
    missing_doc_ids = [doc_id for doc_id in expected_doc_ids if doc_id not in used_document_ids]

    http_ok = error is None and response_payload is not None
    action_ok = action_status not in {"failed", "validation_failed", "retrieval_failed"}
    completion_ok = bool(http_ok and action_ok and required_ok and not missing_doc_ids)
    tool_ok = not missing_tools and not forbidden_tool_hits
    hallucination_ok = not hallucinated

    return {
        "case_id": case["id"],
        "completion_ok": completion_ok,
        "tool_ok": tool_ok,
        "hallucination_ok": hallucination_ok,
        "required_keywords_ok": required_ok,
        "missing_keywords": missing_keywords,
        "actual_tools": actual_tools,
        "missing_tools": missing_tools,
        "forbidden_tool_hits": forbidden_tool_hits,
        "hallucination_hits": hallucination_hits,
        "used_document_ids": used_document_ids,
        "missing_expected_document_ids": missing_doc_ids,
        "action_status": action_status,
        "retrieval_status": retrieval_status,
        "error": error,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def aggregate(case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [item["latency_ms"] for item in case_scores if isinstance(item.get("latency_ms"), int)]
    total_tokens = [item["tokens"]["total_tokens"] for item in case_scores if isinstance(item["tokens"].get("total_tokens"), int)]
    prompt_tokens = [item["tokens"]["prompt_tokens"] for item in case_scores if isinstance(item["tokens"].get("prompt_tokens"), int)]
    completion_tokens = [
        item["tokens"]["completion_tokens"]
        for item in case_scores
        if isinstance(item["tokens"].get("completion_tokens"), int)
    ]
    n = len(case_scores)
    return {
        "case_count": n,
        "e2e_latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 2) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "avg_llm_calls": round(sum(item["llm_call_count"] for item in case_scores) / n, 2) if n else None,
        "token_usage": {
            "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "completion_tokens": sum(completion_tokens) if completion_tokens else None,
            "total_tokens": sum(total_tokens) if total_tokens else None,
            "missing_usage_calls": sum(item["tokens"]["missing_usage_calls"] for item in case_scores),
        },
        "avg_reasoning_steps": round(sum(item["reasoning_steps"] for item in case_scores) / n, 2) if n else None,
        "task_completion_rate": round(sum(1 for item in case_scores if item["completion_ok"]) / n, 4) if n else None,
        "tool_selection_accuracy": round(sum(1 for item in case_scores if item["tool_ok"]) / n, 4) if n else None,
        "hallucination_rate": round(sum(1 for item in case_scores if not item["hallucination_ok"]) / n, 4) if n else None,
    }


def render_summary(run_dir: Path, preflight_info: dict[str, Any], case_scores: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# PaperDesk Knowledge Chat Evaluation",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Cases: {summary['case_count']}",
        f"- Documents: {preflight_info['document_count']}",
        f"- Library chunks: {preflight_info['chunk_count']}",
        f"- LLM model: `{preflight_info['llm_model']}`",
        f"- LLM base URL configured: {preflight_info['llm_base_url_configured']}",
        f"- LLM API key configured: {preflight_info['llm_api_key_configured']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| E2E latency avg | {summary['e2e_latency_ms']['avg']} ms |",
        f"| E2E latency P50 | {summary['e2e_latency_ms']['p50']} ms |",
        f"| E2E latency P95 | {summary['e2e_latency_ms']['p95']} ms |",
        f"| Avg LLM calls | {summary['avg_llm_calls']} |",
        f"| Total tokens | {summary['token_usage']['total_tokens']} |",
        f"| Prompt tokens | {summary['token_usage']['prompt_tokens']} |",
        f"| Completion tokens | {summary['token_usage']['completion_tokens']} |",
        f"| Missing-usage calls | {summary['token_usage']['missing_usage_calls']} |",
        f"| Avg reasoning steps | {summary['avg_reasoning_steps']} |",
        f"| Task completion rate | {summary['task_completion_rate']} |",
        f"| Tool selection accuracy | {summary['tool_selection_accuracy']} |",
        f"| Hallucination rate | {summary['hallucination_rate']} |",
        "",
        "## Case Details",
        "",
        "| Case | Done | Tool OK | Hallucination OK | Latency ms | LLM calls | Tokens | Steps | Tools | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in case_scores:
        notes = []
        if item.get("missing_keywords"):
            notes.append("missing keywords: " + ", ".join(item["missing_keywords"]))
        if item.get("missing_tools"):
            notes.append("missing tools: " + ", ".join(item["missing_tools"]))
        if item.get("forbidden_tool_hits"):
            notes.append("forbidden tools: " + ", ".join(item["forbidden_tool_hits"]))
        if item.get("hallucination_hits"):
            notes.append("claims: " + ", ".join(item["hallucination_hits"]))
        if item.get("error"):
            notes.append(str(item["error"])[:120])
        lines.append(
            "| {case_id} | {done} | {tool_ok} | {hall_ok} | {latency} | {calls} | {tokens} | {steps} | {tools} | {notes} |".format(
                case_id=item["case_id"],
                done="Y" if item["completion_ok"] else "N",
                tool_ok="Y" if item["tool_ok"] else "N",
                hall_ok="Y" if item["hallucination_ok"] else "N",
                latency=item["latency_ms"],
                calls=item["llm_call_count"],
                tokens=item["tokens"].get("total_tokens"),
                steps=item["reasoning_steps"],
                tools="<br>".join(item.get("actual_tools") or []),
                notes="<br>".join(notes),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(cases_path: Path) -> tuple[Path, dict[str, Any]]:
    cases = load_jsonl(cases_path)
    preflight_info = preflight(cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = JUDGE_DIR / f"eval_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    recorder = LlmRecorder()
    install_llm_recorders(recorder)
    client = create_client()

    write_json(run_dir / "preflight.json", preflight_info)
    raw_path = run_dir / "raw_results.jsonl"
    llm_path = run_dir / "llm_calls.jsonl"
    trace_path = run_dir / "trace_events.jsonl"
    case_scores: list[dict[str, Any]] = []

    session_response = client.post("/api/chat/sessions", json={"title": f"judge-{timestamp}"})
    session_response.raise_for_status()
    session_id = session_response.json()["id"]

    for index, case in enumerate(cases, 1):
        llm_offset = recorder.snapshot_len()
        started_at = time.perf_counter()
        response_payload: dict[str, Any] | None = None
        error: str | None = None
        try:
            response = client.post(
                f"/api/chat/sessions/{session_id}/messages",
                json={
                    "content": case["prompt"],
                    "attachments": [],
                    "selected_document_ids": case.get("selected_document_ids", []),
                },
            )
            response_payload = response.json()
            response.raise_for_status()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if response_payload is None:
                response_payload = {"exception": traceback.format_exc()}
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        assistant = (response_payload or {}).get("assistant_message") or {}
        trace_id = assistant.get("agent_trace_id")
        traces = trace_rows(trace_id)
        llm_calls = recorder.calls_since(llm_offset)

        raw_record = {
            "case_index": index,
            "case": case,
            "latency_ms": latency_ms,
            "trace_id": trace_id,
            "response": response_payload,
            "error": error,
        }
        append_jsonl(raw_path, raw_record)
        for call in llm_calls:
            append_jsonl(llm_path, {"case_id": case["id"], **call})
        for trace in traces:
            append_jsonl(trace_path, {"case_id": case["id"], **trace})

        score = score_case(case, response_payload, traces, error)
        score.update(
            {
                "case_index": index,
                "prompt": case["prompt"],
                "latency_ms": latency_ms,
                "trace_id": trace_id,
                "llm_call_count": len(llm_calls),
                "tokens": case_token_totals(llm_calls),
                "reasoning_steps": reasoning_steps(traces),
            }
        )
        case_scores.append(score)

    summary = aggregate(case_scores)
    result = {
        "run_dir": str(run_dir),
        "preflight": preflight_info,
        "summary": summary,
        "cases": case_scores,
    }
    write_json(run_dir / "case_scores.json", result)
    (run_dir / "summary.md").write_text(render_summary(run_dir, preflight_info, case_scores, summary), encoding="utf-8")
    return run_dir, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PaperDesk Knowledge Chat metrics.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()
    run_dir, result = run_evaluation(args.cases)
    print(json.dumps({"run_dir": str(run_dir), "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
