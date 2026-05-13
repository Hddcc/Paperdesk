"""Rule-based product task router for unified research inputs."""

from __future__ import annotations

from app.models import (
    LibraryDocument,
    ResearchArtifactProtocol,
    ResearchArtifactProtocolType,
    ResearchEvidencePolicy,
    ResearchExecutionRoute,
    ResearchInputMode,
    ResearchRequest,
    ResearchTaskRoute,
    ResearchTaskType,
)


class ResearchTaskRouter:
    """Map prompt, file and knowledge-base inputs to a product-level route."""

    def route(
        self,
        request: ResearchRequest,
        *,
        ready_documents: list[LibraryDocument],
    ) -> ResearchTaskRoute:
        input_modes = self._input_modes(request)
        task_type = self._detect_task_type(request, ready_documents)
        protocol = self._artifact_protocol(task_type)
        needs_local = (
            bool(ready_documents)
            or ResearchInputMode.KNOWLEDGE_BASE in input_modes
            or ResearchInputMode.UPLOADED_FILE in input_modes
        )
        needs_online = self._needs_online_search(request, task_type, needs_local)
        execution_route = self._execution_route(task_type)
        return ResearchTaskRoute(
            task_type=task_type,
            input_modes=input_modes,
            evidence_policy=self._evidence_policy(needs_local=needs_local, needs_online=needs_online, task_type=task_type),
            execution_route=execution_route,
            artifact_protocol=protocol,
            selected_document_ids=request.selected_document_ids,
            needs_local_knowledge=needs_local,
            needs_online_search=needs_online,
            use_main_agent_loop=not self._allows_lightweight_route(task_type, ready_documents),
            allow_single_pass=self._allows_lightweight_route(task_type, ready_documents),
            rationale=self._rationale(task_type, needs_local=needs_local, needs_online=needs_online),
        )

    @staticmethod
    def _input_modes(request: ResearchRequest) -> list[ResearchInputMode]:
        modes = list(dict.fromkeys(request.input_modes or [ResearchInputMode.PROMPT]))
        if request.selected_document_ids and ResearchInputMode.KNOWLEDGE_BASE not in modes:
            modes.append(ResearchInputMode.KNOWLEDGE_BASE)
        if (
            ResearchInputMode.UPLOADED_FILE in modes
            and ResearchInputMode.KNOWLEDGE_BASE not in modes
        ):
            modes.append(ResearchInputMode.KNOWLEDGE_BASE)
        return modes

    @staticmethod
    def _detect_task_type(request: ResearchRequest, ready_documents: list[LibraryDocument]) -> ResearchTaskType:
        text = f"{request.topic} {request.notes or ''}".casefold()
        document_count = len(ready_documents) or len(request.selected_document_ids)

        if any(marker in text for marker in ("对比", "比较", "区别", "差异", "compare", "comparison")):
            return ResearchTaskType.COMPARISON
        if any(marker in text for marker in ("综述", "survey", "review", "overview", "文献回顾")):
            return ResearchTaskType.MULTI_PAPER_REVIEW
        if any(marker in text for marker in ("总结", "概括", "summary", "summarize")) and document_count <= 1:
            return ResearchTaskType.PAPER_SUMMARY
        if any(marker in text for marker in ("路线", "路线图", "研究方向", "选题", "roadmap", "brief")):
            return ResearchTaskType.RESEARCH_BRIEF_TASK
        if any(marker in text for marker in ("解释", "讲解", "原理", "方法", "概念", "explain", "method")):
            return ResearchTaskType.METHOD_EXPLAINER
        if document_count >= 2:
            return ResearchTaskType.MULTI_PAPER_REVIEW
        return ResearchTaskType.QA

    @staticmethod
    def _needs_online_search(request: ResearchRequest, task_type: ResearchTaskType, needs_local: bool) -> bool:
        explicit_provider = request.search_provider not in {None, "", "auto"}
        if explicit_provider:
            return True
        if not needs_local:
            return True
        if task_type == ResearchTaskType.PAPER_SUMMARY:
            return False
        return task_type in {
            ResearchTaskType.MULTI_PAPER_REVIEW,
            ResearchTaskType.RESEARCH_BRIEF_TASK,
        }

    @staticmethod
    def _evidence_policy(
        *,
        needs_local: bool,
        needs_online: bool,
        task_type: ResearchTaskType,
    ) -> ResearchEvidencePolicy:
        if needs_local and not needs_online:
            return ResearchEvidencePolicy.LOCAL_ONLY
        if needs_local:
            return ResearchEvidencePolicy.LOCAL_FIRST
        if task_type in {ResearchTaskType.QA, ResearchTaskType.METHOD_EXPLAINER}:
            return ResearchEvidencePolicy.ONLINE_SUPPLEMENT
        return ResearchEvidencePolicy.ONLINE_FIRST

    @staticmethod
    def _execution_route(task_type: ResearchTaskType) -> ResearchExecutionRoute:
        mapping = {
            ResearchTaskType.QA: ResearchExecutionRoute.KNOWLEDGE_QA,
            ResearchTaskType.PAPER_SUMMARY: ResearchExecutionRoute.SINGLE_PAPER_SUMMARY,
            ResearchTaskType.MULTI_PAPER_REVIEW: ResearchExecutionRoute.MAIN_AGENT_REVIEW,
            ResearchTaskType.COMPARISON: ResearchExecutionRoute.COMPARISON_ANALYSIS,
            ResearchTaskType.METHOD_EXPLAINER: ResearchExecutionRoute.METHOD_EXPLANATION,
            ResearchTaskType.RESEARCH_BRIEF_TASK: ResearchExecutionRoute.RESEARCH_BRIEF,
        }
        return mapping[task_type]

    @staticmethod
    def _artifact_protocol(task_type: ResearchTaskType) -> ResearchArtifactProtocol:
        protocols = {
            ResearchTaskType.QA: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.QA,
                title="问答结果",
                required_sections=["直接答案", "关键证据", "必要引用", "不确定性说明"],
            ),
            ResearchTaskType.PAPER_SUMMARY: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.PAPER_SUMMARY,
                title="单篇论文总结",
                required_sections=["论文主题", "研究问题", "核心方法", "主要贡献", "结果结论", "局限性", "适用场景"],
            ),
            ResearchTaskType.MULTI_PAPER_REVIEW: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.REVIEW,
                title="多篇论文综述",
                required_sections=["研究主题", "子方向或章节结构", "代表论文", "核心方法脉络", "共性与差异", "趋势与不足", "引用来源"],
            ),
            ResearchTaskType.COMPARISON: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.COMPARISON,
                title="对比分析",
                required_sections=["对比对象", "对比维度", "各对象表现", "共性", "差异", "适用建议"],
            ),
            ResearchTaskType.METHOD_EXPLAINER: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.METHOD_EXPLAINER,
                title="方法解释",
                required_sections=["概念定义", "方法流程", "适用场景", "证据来源", "局限与注意事项"],
            ),
            ResearchTaskType.RESEARCH_BRIEF_TASK: ResearchArtifactProtocol(
                protocol_type=ResearchArtifactProtocolType.RESEARCH_BRIEF,
                title="研究路线建议",
                required_sections=["方向概览", "关键问题", "可用证据", "路线建议", "后续验证"],
            ),
        }
        return protocols[task_type]

    @staticmethod
    def _allows_lightweight_route(task_type: ResearchTaskType, ready_documents: list[LibraryDocument]) -> bool:
        return task_type in {
            ResearchTaskType.QA,
            ResearchTaskType.PAPER_SUMMARY,
            ResearchTaskType.METHOD_EXPLAINER,
        } and bool(ready_documents)

    @staticmethod
    def _rationale(task_type: ResearchTaskType, *, needs_local: bool, needs_online: bool) -> str:
        source = "优先使用本地知识库" if needs_local else "当前没有指定本地材料"
        supplement = "，必要时补在线检索" if needs_online else "，不默认补在线检索"
        return f"识别为 {task_type.value} 任务；{source}{supplement}。"
