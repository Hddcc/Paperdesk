"""Topic planner agent stub."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from app.models import TodoTask
from app.models.enums import TodoTaskStatus


class TopicPlannerAgent:
    """Split a research topic into deterministic TODO items."""

    def plan(self, topic: str, *, current_date: date | None = None) -> list[TodoTask]:
        reference_date = current_date or date.today()
        aspects = [
            ("研究背景与问题定义", "了解主题的核心概念、边界和研究价值"),
            ("代表性方法与论文脉络", "梳理该主题下常见方法、模型和论文方向"),
            ("应用场景与证据线索", "关注落地场景、实验结果和可验证证据"),
            ("挑战、局限与后续方向", "总结不足、风险和未来值得继续研究的方向"),
        ]
        tasks: list[TodoTask] = []
        for index, (title_suffix, intent) in enumerate(aspects, start=1):
            tasks.append(
                TodoTask(
                    id=str(uuid4()),
                    title=f"{topic}：{title_suffix}",
                    intent=intent,
                    query=f"{topic} {title_suffix} {reference_date.year}",
                    status=TodoTaskStatus.PENDING,
                )
            )
        return tasks
