"""Final report writer with stable citations and optional LLM polishing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any
from uuid import uuid4

from openai import OpenAI

from app.models import CitationRecord, EvidenceItem, PaperRecord, ResearchReport, TaskSummary

_REQUIRED_SECTION_HEADINGS = [
    "## 1. 研究概览",
    "## 2. 分任务总结整合",
    "## 3. 关键观点归纳",
    "## 4. 局限与待补充问题",
    "## 参考来源",
]


@dataclass(slots=True)
class _CitationEntry:
    key: str
    citation_record: CitationRecord
    reference_line: str


class ReportWriterAgent:
    """Create a structured Markdown report from task summaries."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def write(self, topic: str, task_summaries: list[TaskSummary]) -> ResearchReport:
        citation_entries = self._build_citation_entries(task_summaries)
        citation_lookup = {entry.key: entry for entry in citation_entries}
        reference_lines = [entry.reference_line for entry in citation_entries]
        rendered_reference_lines = self._render_reference_lines(reference_lines)

        draft_markdown = self._build_template_report(
            topic=topic,
            task_summaries=task_summaries,
            citation_lookup=citation_lookup,
            reference_lines=rendered_reference_lines,
        )
        polished_markdown = self._polish_with_llm(
            topic=topic,
            task_summaries=task_summaries,
            reference_lines=reference_lines,
            draft_markdown=draft_markdown,
        )
        markdown = self._normalize_final_markdown(
            topic=topic,
            markdown=polished_markdown or draft_markdown,
            reference_lines=rendered_reference_lines,
        )
        return ResearchReport(
            id=str(uuid4()),
            topic=topic,
            markdown=markdown,
            task_summaries=task_summaries,
            citations=reference_lines,
            citation_items=[entry.citation_record for entry in citation_entries],
            created_at=datetime.now(timezone.utc),
        )

    def _build_template_report(
        self,
        *,
        topic: str,
        task_summaries: list[TaskSummary],
        citation_lookup: dict[str, _CitationEntry],
        reference_lines: list[str],
    ) -> str:
        task_sections = [
            self._build_task_section(
                index=index,
                task_summary=task_summary,
                citation_lookup=citation_lookup,
            )
            for index, task_summary in enumerate(task_summaries, start=1)
        ]
        key_points = self._build_key_points(task_summaries, citation_lookup)
        limitation_lines = self._build_limitations(task_summaries)
        overview = self._build_overview(topic, task_summaries, reference_lines)

        lines: list[str] = [
            f"# {topic}",
            "",
            "## 1. 研究概览",
            "",
            overview,
            "",
            "## 2. 分任务总结整合",
            "",
            *task_sections,
            "## 3. 关键观点归纳",
            "",
            *key_points,
            "",
            "## 4. 局限与待补充问题",
            "",
            *limitation_lines,
            "",
            "## 参考来源",
            "",
            *(reference_lines or ["暂无可导出的参考来源。"]),
        ]
        return "\n".join(lines).strip()

    def _build_overview(
        self,
        topic: str,
        task_summaries: list[TaskSummary],
        reference_lines: list[str],
    ) -> str:
        unique_paper_keys = {
            self._paper_key(paper)
            for task_summary in task_summaries
            for paper in task_summary.paper_records
        }
        unique_local_keys = {
            self._evidence_key(evidence)
            for task_summary in task_summaries
            for evidence in task_summary.evidence_items
        }
        leading_citations = " ".join(
            self._extract_citation_label(line) for line in reference_lines[:3]
        )
        overview = (
            f"本次研究围绕“{topic}”拆解为 {len(task_summaries)} 个子任务，"
            f"整合了 {len(unique_paper_keys)} 个在线论文来源与 {len(unique_local_keys)} 个本地 PDF 证据来源。"
            "最终报告只基于既有任务总结进行重组与归纳，不在报告阶段新增检索或扩展结论。"
        )
        if leading_citations:
            overview += f" 代表性来源包括 {leading_citations}。"
        return overview

    def _build_task_section(
        self,
        *,
        index: int,
        task_summary: TaskSummary,
        citation_lookup: dict[str, _CitationEntry],
    ) -> str:
        summary_body = self._strip_reference_sections(
            task_summary.summary_markdown or task_summary.summary
        )
        summary_body = self._strip_intro_line(summary_body).strip()
        if not summary_body:
            summary_body = "该任务已完成阶段性总结，当前报告仅保留其核心研究意图与证据链接。"
        labels = self._collect_task_citation_labels(task_summary, citation_lookup)
        label_text = " ".join(labels) if labels else "暂无直接引用"
        lines = [
            f"### 2.{index} {task_summary.title}",
            "",
            f"任务意图：{task_summary.intent}",
            "",
            summary_body,
            "",
            f"关联引用：{label_text}",
            "",
        ]
        return "\n".join(lines)

    def _build_key_points(
        self,
        task_summaries: list[TaskSummary],
        citation_lookup: dict[str, _CitationEntry],
    ) -> list[str]:
        points: list[str] = []
        for task_summary in task_summaries:
            labels = self._collect_task_citation_labels(task_summary, citation_lookup)
            label_text = " ".join(labels) if labels else "暂无直接引用"
            points.append(
                f"- 围绕“{task_summary.title}”的总结聚焦于 {task_summary.intent}，"
                f"相关依据见 {label_text}。"
            )
        if not points:
            points.append("- 当前没有可整合的任务总结。")
        return points

    def _build_limitations(self, task_summaries: list[TaskSummary]) -> list[str]:
        has_local_evidence = any(
            task_summary.evidence_items for task_summary in task_summaries
        )
        has_uncited_task = any(
            not task_summary.evidence_items and not task_summary.paper_records
            for task_summary in task_summaries
        )
        lines = [
            "- 本报告只整合已完成任务的阶段性总结，未在报告阶段新增搜索、检索或实验验证。",
        ]
        if has_local_evidence:
            lines.append(
                "- 并非所有子任务都能获得同等充分的本地 PDF 证据，局部结论仍建议回到原文继续核验。"
            )
        else:
            lines.append(
                "- 本轮未检索到可直接采用的本地 PDF 证据，当前判断主要依赖在线论文来源。"
            )
        if has_uncited_task:
            lines.append("- 个别任务缺少可直接定位的引用项，后续需要补充更明确的来源支撑。")
        else:
            lines.append("- 虽然已统一引用格式，但不同任务间的证据密度仍可能存在不均衡。")
        return lines

    def _build_citation_entries(self, task_summaries: list[TaskSummary]) -> list[_CitationEntry]:
        entries: dict[str, _CitationEntry] = {}

        def add_entry(key: str, record: CitationRecord, reference_line: str) -> None:
            if key in entries:
                return
            label = f"[{len(entries) + 1}]"
            record.citation_label = label
            entries[key] = _CitationEntry(
                key=key,
                citation_record=record,
                reference_line=f"{label} {reference_line}",
            )

        for task_summary in task_summaries:
            for paper in task_summary.paper_records:
                key = self._paper_key(paper)
                add_entry(
                    key,
                    CitationRecord(
                        citation_label="",
                        source_type=paper.source_type.value,
                        title=paper.title,
                        url=paper.url,
                        doi=paper.doi,
                    ),
                    self._format_online_reference(paper),
                )
            for evidence in task_summary.evidence_items:
                key = self._evidence_key(evidence)
                title = self._local_title(evidence)
                add_entry(
                    key,
                    CitationRecord(
                        citation_label="",
                        source_type=evidence.source_type.value,
                        title=title,
                        url=evidence.url,
                        document_id=evidence.document_id or evidence.source_id,
                        page_number=evidence.page_number,
                    ),
                    self._format_local_reference(title, evidence.page_number),
                )
        return list(entries.values())

    def _polish_with_llm(
        self,
        *,
        topic: str,
        task_summaries: list[TaskSummary],
        reference_lines: list[str],
        draft_markdown: str,
    ) -> str | None:
        if not self.api_key:
            return None

        task_payload = "\n\n".join(
            [
                "\n".join(
                    [
                        f"任务标题：{task_summary.title}",
                        f"任务意图：{task_summary.intent}",
                        "任务总结：",
                        self._strip_reference_sections(
                            task_summary.summary_markdown or task_summary.summary
                        ).strip()
                        or "暂无总结",
                    ]
                )
                for task_summary in task_summaries
            ]
        )
        prompt = "\n\n".join(
            [
                f"研究主题：{topic}",
                "请基于以下任务总结润色最终报告，要求：",
                "1. 只能基于现有任务总结与引用信息写作。",
                "2. 不得新增事实，不得虚构引用。",
                "3. 必须完整保留以下标题结构：",
                "# 研究主题",
                "## 1. 研究概览",
                "## 2. 分任务总结整合",
                "## 3. 关键观点归纳",
                "## 4. 局限与待补充问题",
                "## 参考来源",
                "4. 正文观点尽量引用已有编号。",
                "5. 返回纯 Markdown，不要额外解释。",
                "可用引用：",
                "\n".join(reference_lines) or "暂无可用引用。",
                "当前模板稿：",
                draft_markdown,
                "任务总结原文：",
                task_payload,
            ]
        )
        response = self._call_llm(
            system_prompt=(
                "You polish Chinese academic survey markdown. Preserve structure, keep claims grounded, "
                "and never invent citations or facts."
            ),
            user_prompt=prompt,
        )
        if not response:
            return None
        if not self._is_valid_polished_markdown(response, reference_lines):
            return None
        return response

    def _call_llm(self, *, system_prompt: str, user_prompt: str) -> str | None:
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None
        return self._extract_message_text(response)

    def _is_valid_polished_markdown(
        self,
        markdown: str,
        reference_lines: list[str],
    ) -> bool:
        positions: list[int] = []
        for heading in _REQUIRED_SECTION_HEADINGS:
            position = markdown.find(heading)
            if position == -1:
                return False
            positions.append(position)
        if positions != sorted(positions):
            return False

        allowed_labels = {
            self._extract_citation_label(line)
            for line in reference_lines
        }
        for label in re.findall(r"\[\d+\]", markdown):
            if label not in allowed_labels:
                return False
        return True

    def _normalize_final_markdown(
        self,
        *,
        topic: str,
        markdown: str,
        reference_lines: list[str],
    ) -> str:
        body_without_references = re.split(
            r"(?m)^##\s*参考来源\s*$",
            markdown.strip(),
            maxsplit=1,
        )[0].strip()
        body_without_title = re.sub(
            r"(?ms)^#\s+.*?(?:\n{2,}|$)",
            "",
            body_without_references,
            count=1,
        ).strip()
        final_lines = [
            f"# {topic}",
            "",
            body_without_title,
            "",
            "## 参考来源",
            "",
            *(reference_lines or ["暂无可导出的参考来源。"]),
        ]
        return "\n".join(final_lines).strip()

    @staticmethod
    def _render_reference_lines(reference_lines: list[str]) -> list[str]:
        if not reference_lines:
            return ["- 暂无可导出的参考来源。"]
        return [f"- {line}" for line in reference_lines]

    @staticmethod
    def _extract_message_text(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None

        message = getattr(choices[0], "message", None)
        if message is None:
            return None

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip() if parts else None
        return None

    @staticmethod
    def _collect_task_citation_labels(
        task_summary: TaskSummary,
        citation_lookup: dict[str, _CitationEntry],
    ) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for paper in task_summary.paper_records:
            entry = citation_lookup.get(ReportWriterAgent._paper_key(paper))
            if entry is None or entry.citation_record.citation_label in seen:
                continue
            labels.append(entry.citation_record.citation_label)
            seen.add(entry.citation_record.citation_label)
        for evidence in task_summary.evidence_items:
            entry = citation_lookup.get(ReportWriterAgent._evidence_key(evidence))
            if entry is None or entry.citation_record.citation_label in seen:
                continue
            labels.append(entry.citation_record.citation_label)
            seen.add(entry.citation_record.citation_label)
        return labels

    @staticmethod
    def _strip_reference_sections(summary: str) -> str:
        cleaned = re.sub(
            r"(?s)\n*在线论文参考：.*$",
            "",
            summary.strip(),
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_intro_line(summary: str) -> str:
        return re.sub(r"(?m)^任务聚焦：.*\n?", "", summary).strip()

    @staticmethod
    def _paper_key(paper: PaperRecord) -> str:
        if paper.doi:
            return f"online:doi:{paper.doi}"
        if paper.url:
            return f"online:url:{paper.url.strip().lower()}"
        year = paper.year if paper.year is not None else "unknown"
        return f"online:title:{paper.title.strip().lower()}:{year}"

    @staticmethod
    def _evidence_key(evidence: EvidenceItem) -> str:
        page = evidence.page_number if evidence.page_number is not None else "unknown"
        if evidence.document_id:
            return f"local:document:{evidence.document_id}:{page}"
        return f"local:title:{ReportWriterAgent._local_title(evidence).strip().lower()}:{page}"

    @staticmethod
    def _format_online_reference(paper: PaperRecord) -> str:
        author_text = ReportWriterAgent._format_authors(paper.authors)
        year_text = str(paper.year) if paper.year is not None else "年份未知"
        locator = paper.doi or paper.url or "source unavailable"
        return f"{paper.title}. {author_text}. {year_text}. {locator}."

    @staticmethod
    def _format_local_reference(title: str, page_number: int | None) -> str:
        if page_number is None:
            return f"Local PDF: {title}."
        return f"Local PDF: {title}, p.{page_number}."

    @staticmethod
    def _format_authors(authors: list[str]) -> str:
        if not authors:
            return "作者未知"
        if len(authors) <= 3:
            return ", ".join(authors)
        return f"{', '.join(authors[:3])} 等"

    @staticmethod
    def _local_title(evidence: EvidenceItem) -> str:
        filename = evidence.metadata.get("filename")
        if isinstance(filename, str) and filename.strip():
            return filename.strip()
        if evidence.title.strip():
            return evidence.title.strip()
        return evidence.source_id

    @staticmethod
    def _extract_citation_label(reference_line: str) -> str:
        match = re.match(r"(\[\d+\])", reference_line.strip())
        return match.group(1) if match else ""
