"""Unified research tool wrappers for the phase-13 main-agent loop."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL
from uuid import uuid4

from app.agents import (
    LibraryRetrieverAgent,
    PaperSearchAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)
from app.models import (
    EvidenceItem,
    PaperRecord,
    ResearchArtifactProtocolType,
    ResearchPlanItem,
    ResearchRequest,
    ResearchReport,
    ResearchTaskRoute,
    ResearchTaskType,
    ResearchToolResult,
    ResearchToolResultClassification,
    ResearchToolResultStatus,
    SkillDefinition,
    TaskArtifactRef,
    TodoTask,
    TodoTaskStatus,
)
from app.services.research_workspace_service import ResearchWorkspaceService


class ResearchToolExecutor:
    """Wrap existing research capabilities behind a unified tool contract."""

    def __init__(
        self,
        *,
        topic_planner: TopicPlannerAgent,
        paper_search_agent: PaperSearchAgent,
        library_retriever: LibraryRetrieverAgent,
        reading_summarizer: ReadingSummarizerAgent,
        report_writer: ReportWriterAgent,
        workspace_service: ResearchWorkspaceService,
    ) -> None:
        self.topic_planner = topic_planner
        self.paper_search_agent = paper_search_agent
        self.library_retriever = library_retriever
        self.reading_summarizer = reading_summarizer
        self.report_writer = report_writer
        self.workspace_service = workspace_service

    def plan(
        self,
        run_id: str,
        request: ResearchRequest,
        *,
        direct_task: bool,
        task_route: ResearchTaskRoute | None = None,
        active_skill: SkillDefinition | None = None,
    ) -> ResearchToolResult:
        if direct_task:
            tasks = [
                TodoTask(
                    id=run_id,
                    title=request.topic,
                    intent="Direct research pass for a compact topic",
                    query=request.topic,
                    status=TodoTaskStatus.PENDING,
                )
            ]
        elif task_route is not None and task_route.task_type != ResearchTaskType.MULTI_PAPER_REVIEW:
            tasks = self._routed_tasks(request, task_route)
        else:
            tasks = self.topic_planner.plan(request.topic)
        if task_route is not None:
            for task in tasks:
                task.intent = f"{task.intent}；产物协议：{task_route.artifact_protocol.title}"
        plan_items = [
            ResearchPlanItem(
                task_id=task.id,
                title=task.title,
                intent=task.intent,
                query=task.query,
                objective=task.intent or task.title,
                done_criteria=self._done_criteria(task_route),
                priority=index + 1,
                suggested_tools=self._suggested_tools(task_route, active_skill),
                required_evidence=self._required_evidence(task_route),
                query_history=[task.query],
                status=task.status,
            )
            for index, task in enumerate(tasks)
        ]
        artifact = self._write_json_artifact(
            run_id,
            "__plan__",
            "plan_items.json",
            [item.model_dump(mode="json") for item in plan_items],
            description="Main-agent plan items",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary=f"Planned {len(plan_items)} research tasks.",
            payload={"plan_items": [item.model_dump(mode="json") for item in plan_items]},
            artifacts=[artifact],
        )

    def search_online(
        self,
        run_id: str,
        task: ResearchPlanItem,
        request: ResearchRequest,
    ) -> ResearchToolResult:
        todo_task = self._to_todo_task(task)
        paper_records = self.paper_search_agent.search(
            todo_task,
            top_k=request.top_k_online,
            search_provider=request.search_provider,
        )
        payload = [record.model_dump(mode="json") for record in paper_records]
        artifacts = [
            self._write_json_artifact(
                run_id,
                task.task_id,
                "papers.json",
                payload,
                description="Normalized online paper candidates",
            ),
            self._write_markdown_artifact(
                run_id,
                task.task_id,
                "online-analysis.md",
                "\n".join(
                    [
                        f"# Online Search: {task.title}",
                        "",
                        f"Collected {len(paper_records)} paper candidates.",
                        "",
                        *[f"- {record.title}" for record in paper_records],
                    ]
                ),
                description="Compact online paper summary",
            ),
        ]
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_INSUFFICIENT,
            summary=f"Collected {len(paper_records)} online paper candidates.",
            payload={"paper_records": payload},
            artifacts=artifacts,
            retryable=True,
        )

    def search_local(
        self,
        run_id: str,
        task: ResearchPlanItem,
        documents,
        *,
        top_k_local: int,
    ) -> ResearchToolResult:
        todo_task = self._to_todo_task(task)
        evidence_items = self.library_retriever.retrieve(
            todo_task,
            documents,
            top_k=top_k_local,
        )
        payload = [item.model_dump(mode="json") for item in evidence_items]
        artifacts = [
            self._write_json_artifact(
                run_id,
                task.task_id,
                "evidence.json",
                payload,
                description="Retrieved local evidence items",
            ),
            self._write_markdown_artifact(
                run_id,
                task.task_id,
                "local-analysis.md",
                "\n".join(
                    [
                        f"# Local Search: {task.title}",
                        "",
                        f"Collected {len(evidence_items)} local evidence items.",
                        "",
                        *[f"- {item.citation_label}" for item in evidence_items],
                    ]
                ),
                description="Compact local evidence summary",
            ),
        ]
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_INSUFFICIENT,
            summary=f"Collected {len(evidence_items)} local evidence items.",
            payload={"evidence_items": payload},
            artifacts=artifacts,
            retryable=True,
        )

    def summarize_evidence(
        self,
        run_id: str,
        task: ResearchPlanItem,
        paper_records: list[PaperRecord],
        evidence_items: list[EvidenceItem],
        *,
        degraded: bool = False,
    ) -> ResearchToolResult:
        summary = self.reading_summarizer.summarize(
            self._to_todo_task(task),
            paper_records,
            evidence_items,
        )
        if degraded:
            degraded_note = "证据不足：本任务未检索到足够可用材料，以下总结按当前材料降级收口。"
            summary.summary = f"{degraded_note}\n\n{summary.summary}".strip()
            summary.summary_markdown = f"{degraded_note}\n\n{summary.summary_markdown}".strip()

        artifact = self._write_markdown_artifact(
            run_id,
            task.task_id,
            "summary.md",
            summary.summary_markdown or summary.summary,
            description="Task-level merged summary",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary="Task summary completed.",
            payload={"task_summary": summary.model_dump(mode="json")},
            artifacts=[artifact],
        )

    def finalize_report(
        self,
        run_id: str,
        topic: str,
        task_summaries,
        *,
        task_route: ResearchTaskRoute | None = None,
    ) -> ResearchToolResult:
        report = self._write_routed_report(topic, task_summaries, task_route)
        if task_route is not None:
            report.markdown = self._apply_artifact_protocol(report.markdown, task_route)
        artifact = TaskArtifactRef(
            name="final_report.md",
            path=str(self.workspace_service.write_final_report(run_id, report)),
            kind="markdown",
            description="Final report markdown",
        )
        return ResearchToolResult(
            status=ResearchToolResultStatus.COMPLETED,
            classification=ResearchToolResultClassification.SUCCESS_SUFFICIENT,
            summary="Final report generated.",
            payload={"report": report.model_dump(mode="json")},
            artifacts=[artifact],
            retryable=False,
        )

    def _write_routed_report(
        self,
        topic: str,
        task_summaries,
        task_route: ResearchTaskRoute | None,
    ) -> ResearchReport:
        base_report = self.report_writer.write(topic, task_summaries)
        if task_route is None:
            return base_report

        protocol_type = task_route.artifact_protocol.protocol_type
        if protocol_type == ResearchArtifactProtocolType.QA:
            markdown = self._build_qa_markdown(topic, task_summaries, base_report.citations)
        elif protocol_type == ResearchArtifactProtocolType.PAPER_SUMMARY:
            markdown = self._build_paper_summary_markdown(topic, task_summaries, base_report.citations)
        elif protocol_type == ResearchArtifactProtocolType.REVIEW:
            markdown = self._build_review_markdown(topic, task_summaries, base_report.citations)
        elif protocol_type == ResearchArtifactProtocolType.COMPARISON:
            markdown = self._build_comparison_markdown(topic, task_summaries, base_report.citations)
        elif protocol_type == ResearchArtifactProtocolType.METHOD_EXPLAINER:
            markdown = self._build_protocol_markdown(
                topic,
                "方法解释",
                task_summaries,
                base_report.citations,
                ["概念定义", "方法流程", "适用场景", "证据来源", "局限与注意事项"],
            )
        elif protocol_type == ResearchArtifactProtocolType.RESEARCH_BRIEF:
            markdown = self._build_protocol_markdown(
                topic,
                "研究路线建议",
                task_summaries,
                base_report.citations,
                ["方向概览", "关键问题", "可用证据", "路线建议", "后续验证"],
            )
        else:
            markdown = base_report.markdown

        return ResearchReport(
            id=str(uuid4()),
            topic=topic,
            markdown=markdown,
            task_summaries=task_summaries,
            citations=base_report.citations,
            citation_items=base_report.citation_items,
            created_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _to_todo_task(task: ResearchPlanItem) -> TodoTask:
        return TodoTask(
            id=task.task_id,
            title=task.title,
            intent=task.intent,
            query=task.query,
            status=task.status,
            summary=task.summary,
            summary_markdown=task.summary_markdown,
        )

    @staticmethod
    def _routed_tasks(request: ResearchRequest, task_route: ResearchTaskRoute) -> list[TodoTask]:
        protocol = task_route.artifact_protocol
        sections = "、".join(protocol.required_sections[:4])
        return [
            TodoTask(
                id=f"{task_route.task_type.value}-{run_safe_id(request.topic)}",
                title=f"{protocol.title}：{request.topic}",
                intent=f"按“{protocol.title}”协议生成结果，重点覆盖：{sections}",
                query=request.topic,
                status=TodoTaskStatus.PENDING,
            )
        ]

    @staticmethod
    def _suggested_tools(task_route: ResearchTaskRoute | None, active_skill: SkillDefinition | None = None) -> list[str]:
        if active_skill is not None and active_skill.available_tools:
            return list(active_skill.available_tools)
        if task_route is None:
            return ["search_online", "search_local", "summarize_evidence"]
        tools: list[str] = []
        if task_route.needs_online_search:
            tools.append("search_online")
        if task_route.needs_local_knowledge:
            tools.append("search_local")
        if not tools:
            tools.append("search_local")
        tools.append("summarize_evidence")
        return tools

    @staticmethod
    def _required_evidence(task_route: ResearchTaskRoute | None) -> list[str]:
        if task_route is None:
            return ["online_paper", "local_document"]
        evidence: list[str] = []
        if task_route.needs_online_search:
            evidence.append("online_paper")
        if task_route.needs_local_knowledge:
            evidence.append("local_document")
        return evidence or ["online_paper", "local_document"]

    @staticmethod
    def _done_criteria(task_route: ResearchTaskRoute | None) -> str:
        if task_route is None:
            return "形成可引用的任务级研究总结；若证据不足，说明降级边界。"
        sections = "、".join(task_route.artifact_protocol.required_sections)
        return f"按“{task_route.artifact_protocol.title}”产物协议收束，至少覆盖：{sections}。"

    @staticmethod
    def _apply_artifact_protocol(markdown: str, task_route: ResearchTaskRoute) -> str:
        protocol = task_route.artifact_protocol
        protocol_lines = [
            "> 产物协议",
            f"> - 任务类型：{task_route.task_type.value}",
            f"> - 输出形态：{protocol.title}",
            f"> - 证据策略：{task_route.evidence_policy.value}",
            f"> - 默认路径：{task_route.execution_route.value}",
            f"> - 必含要点：{'、'.join(protocol.required_sections)}",
        ]
        return "\n".join([markdown.strip(), "", *protocol_lines]).strip()

    @staticmethod
    def _build_qa_markdown(topic: str, task_summaries, citations: list[str]) -> str:
        body = ResearchToolExecutor._combined_summary_body(task_summaries)
        citation_hint = ResearchToolExecutor._citation_hint(citations)
        return "\n".join(
            [
                f"# {topic}",
                "",
                "## 直接答案",
                "",
                body,
                "",
                "## 关键证据",
                "",
                ResearchToolExecutor._evidence_lines(task_summaries),
                "",
                "## 证据来源与引用映射",
                "",
                *ResearchToolExecutor._evidence_source_lines(task_summaries, citations),
                "",
                "## 必要引用",
                "",
                citation_hint,
                "",
                "## 结论边界",
                "",
                *ResearchToolExecutor._conclusion_boundary_lines(task_summaries),
                "",
                "## 参考来源",
                "",
                *ResearchToolExecutor._reference_lines(citations),
            ]
        ).strip()

    @staticmethod
    def _build_paper_summary_markdown(topic: str, task_summaries, citations: list[str]) -> str:
        body = ResearchToolExecutor._combined_summary_body(task_summaries)
        return "\n".join(
            [
                f"# {topic}",
                "",
                "## 论文主题",
                "",
                body,
                "",
                "## 研究问题",
                "",
                ResearchToolExecutor._section_from_body(body, "本文围绕用户指定论文提取研究问题。"),
                "",
                "## 核心方法",
                "",
                ResearchToolExecutor._section_from_body(body, "核心方法依据本轮本地证据整理。"),
                "",
                "## 主要贡献",
                "",
                ResearchToolExecutor._section_from_body(body, "主要贡献按当前论文摘要和局部证据归纳。"),
                "",
                "## 结果结论",
                "",
                ResearchToolExecutor._section_from_body(body, "结果结论需要结合原文完整阅读进一步核验。"),
                "",
                "## 局限性",
                "",
                "本摘要只基于当前可检索片段生成，可能未覆盖全文所有实验、消融和附录细节。",
                "",
                "## 适用场景",
                "",
                "适合作为快速阅读入口，后续应结合原 PDF 进行精读和引用核验。",
                "",
                "## 证据来源与引用映射",
                "",
                *ResearchToolExecutor._evidence_source_lines(task_summaries, citations),
                "",
                "## 结论边界",
                "",
                *ResearchToolExecutor._conclusion_boundary_lines(task_summaries),
                "",
                "## 参考来源",
                "",
                *ResearchToolExecutor._reference_lines(citations),
            ]
        ).strip()

    @staticmethod
    def _build_review_markdown(topic: str, task_summaries, citations: list[str]) -> str:
        body = ResearchToolExecutor._combined_summary_body(task_summaries)
        return "\n".join(
            [
                f"# {topic}",
                "",
                "## 综述脉络与主题分组",
                "",
                body,
                "",
                "## 关键研究方向",
                "",
                ResearchToolExecutor._task_bullets(task_summaries),
                "",
                "## 趋势归纳",
                "",
                "当前综述只基于本轮任务总结、在线论文候选和本地 PDF 证据归纳主题趋势，不在报告阶段新增事实。",
                "",
                "## 证据来源与引用映射",
                "",
                *ResearchToolExecutor._evidence_source_lines(task_summaries, citations),
                "",
                "## 局限与待补充问题",
                "",
                *ResearchToolExecutor._conclusion_boundary_lines(task_summaries),
                "",
                "## 参考来源",
                "",
                *ResearchToolExecutor._reference_lines(citations),
            ]
        ).strip()

    @staticmethod
    def _build_comparison_markdown(topic: str, task_summaries, citations: list[str]) -> str:
        body = ResearchToolExecutor._combined_summary_body(task_summaries)
        return "\n".join(
            [
                f"# {topic}",
                "",
                "## 对比对象",
                "",
                body,
                "",
                "## 对比维度",
                "",
                "- 研究目标",
                "- 方法路径",
                "- 证据强度",
                "- 适用边界",
                "",
                "## 各对象表现",
                "",
                ResearchToolExecutor._task_bullets(task_summaries),
                "",
                "## 共性",
                "",
                "当前材料显示这些对象均围绕用户问题提供了可比较的证据线索。",
                "",
                "## 差异",
                "",
                "差异需要结合各任务证据密度、方法假设和适用场景继续核验。",
                "",
                "## 适用建议",
                "",
                "优先采用证据更直接、引用更清晰且与目标场景更接近的方案。",
                "",
                "## 证据来源与引用映射",
                "",
                *ResearchToolExecutor._evidence_source_lines(task_summaries, citations),
                "",
                "## 结论边界",
                "",
                *ResearchToolExecutor._conclusion_boundary_lines(task_summaries),
                "",
                "## 参考来源",
                "",
                *ResearchToolExecutor._reference_lines(citations),
            ]
        ).strip()

    @staticmethod
    def _build_protocol_markdown(
        topic: str,
        title: str,
        task_summaries,
        citations: list[str],
        sections: list[str],
    ) -> str:
        body = ResearchToolExecutor._combined_summary_body(task_summaries)
        lines = [f"# {topic}", "", f"## {title}", "", body]
        for section in sections:
            lines.extend(["", f"## {section}", "", ResearchToolExecutor._section_from_body(body, f"{section}基于当前证据整理。")])
        lines.extend(["", "## 证据来源与引用映射", "", *ResearchToolExecutor._evidence_source_lines(task_summaries, citations)])
        lines.extend(["", "## 结论边界", "", *ResearchToolExecutor._conclusion_boundary_lines(task_summaries)])
        lines.extend(["", "## 参考来源", "", *ResearchToolExecutor._reference_lines(citations)])
        return "\n".join(lines).strip()

    @staticmethod
    def _combined_summary_body(task_summaries) -> str:
        chunks = []
        for summary in task_summaries:
            content = (summary.summary_markdown or summary.summary or "").strip()
            content = ResearchToolExecutor._strip_summary_reference_sections(content)
            if not content:
                continue
            chunks.append(content)
        return "\n\n".join(chunks) or "当前没有可直接整合的任务总结。"

    @staticmethod
    def _strip_summary_reference_sections(content: str) -> str:
        marker = "\n在线论文参考："
        if marker in content:
            return content.split(marker, 1)[0].strip()
        if content.startswith("在线论文参考："):
            return ""
        return content

    @staticmethod
    def _section_from_body(body: str, fallback: str) -> str:
        first_paragraph = next((part.strip() for part in body.split("\n\n") if part.strip()), "")
        return first_paragraph or fallback

    @staticmethod
    def _task_bullets(task_summaries) -> str:
        lines = []
        for summary in task_summaries:
            content = (summary.summary_markdown or summary.summary or "").strip()
            preview = content.splitlines()[0] if content else "暂无总结"
            lines.append(f"- {summary.title}：{preview}")
        return "\n".join(lines) or "- 当前没有可比较的任务总结。"

    @staticmethod
    def _evidence_lines(task_summaries) -> str:
        local_count = sum(len(summary.evidence_items) for summary in task_summaries)
        paper_count = sum(len(summary.paper_records) for summary in task_summaries)
        if local_count or paper_count:
            return f"- 本轮整合本地证据 {local_count} 条，在线论文 {paper_count} 条。"
        return "- 本轮没有检索到可直接引用的证据。"

    @staticmethod
    def _evidence_source_lines(task_summaries, citations: list[str]) -> list[str]:
        local_count = sum(len(summary.evidence_items) for summary in task_summaries)
        paper_count = sum(len(summary.paper_records) for summary in task_summaries)
        local_labels, online_labels = ResearchToolExecutor._citation_groups(citations)

        lines = [
            f"- 本地知识库证据：{local_count} 条；对应引用：{ResearchToolExecutor._label_text(local_labels)}。",
            f"- 在线检索证据：{paper_count} 条；对应引用：{ResearchToolExecutor._label_text(online_labels)}。",
        ]
        if local_count and paper_count:
            lines.append("- 混合结论：同时参考本地 PDF 片段与在线论文候选，正文中的综合判断需要结合两类来源一起核验。")
        elif local_count:
            lines.append("- 当前结论主要来自本地 PDF 证据；在线检索没有参与本轮核心收束。")
        elif paper_count:
            lines.append("- 当前结论主要来自在线论文候选；本地知识库没有提供可直接引用的支撑。")
        else:
            lines.append("- 当前没有可直接定位的证据来源，因此只应视为低置信度收束。")
        return lines

    @staticmethod
    def _conclusion_boundary_lines(task_summaries) -> list[str]:
        local_count = sum(len(summary.evidence_items) for summary in task_summaries)
        paper_count = sum(len(summary.paper_records) for summary in task_summaries)
        has_degraded_summary = any(
            "证据不足" in (summary.summary_markdown or summary.summary or "")
            for summary in task_summaries
        )
        lines = [
            "- 强结论只应来自有明确引用编号或本地页码支撑的内容。",
        ]
        if has_degraded_summary or not (local_count or paper_count):
            lines.append("- 本轮存在证据不足或无直接证据的任务，相关回答只能作为有限材料下的初步判断。")
        elif local_count and paper_count:
            lines.append("- 本轮同时包含本地与在线来源，交叉出现的观点可信度更高；单一来源观点仍需回到原文复核。")
        else:
            lines.append("- 本轮证据来自单一来源类型，结论边界受该来源覆盖范围限制。")
        lines.append("- 未被引用直接支撑的表述不应作为最终学术结论使用。")
        return lines

    @staticmethod
    def _citation_groups(citations: list[str]) -> tuple[list[str], list[str]]:
        local_labels: list[str] = []
        online_labels: list[str] = []
        for line in citations:
            label = ResearchToolExecutor._citation_label(line)
            if not label:
                continue
            if "Local PDF:" in line:
                local_labels.append(label)
            else:
                online_labels.append(label)
        return local_labels, online_labels

    @staticmethod
    def _citation_label(reference_line: str) -> str:
        return reference_line.split(" ", 1)[0].strip()

    @staticmethod
    def _label_text(labels: list[str]) -> str:
        return " ".join(labels) if labels else "暂无"

    @staticmethod
    def _citation_hint(citations: list[str]) -> str:
        labels = []
        for line in citations[:3]:
            label = ResearchToolExecutor._citation_label(line)
            if label:
                labels.append(label)
        return " ".join(labels) if labels else "暂无可用引用。"

    @staticmethod
    def _reference_lines(citations: list[str]) -> list[str]:
        if not citations:
            return ["- 暂无可导出的参考来源。"]
        return [f"- {line}" for line in citations]

    def _write_json_artifact(
        self,
        run_id: str,
        task_id: str,
        filename: str,
        payload: object,
        *,
        description: str,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_json(run_id, task_id, filename, payload)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="json",
            description=description,
        )

    def _write_markdown_artifact(
        self,
        run_id: str,
        task_id: str,
        filename: str,
        content: str,
        *,
        description: str,
    ) -> TaskArtifactRef:
        path = self.workspace_service.write_scratch_markdown(run_id, task_id, filename, content)
        return TaskArtifactRef(
            name=filename,
            path=str(path),
            kind="markdown",
            description=description,
        )


def run_safe_id(value: str) -> str:
    return uuid5(NAMESPACE_URL, value).hex[:8]
