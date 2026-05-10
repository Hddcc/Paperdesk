"""Helpers for building provider-friendly online search queries."""

from __future__ import annotations

import re

_TASK_SUFFIXES = [
    "研究背景与问题定义",
    "代表性方法与论文脉络",
    "应用场景与证据线索",
    "挑战、局限与后续方向",
]


def contains_cjk(text: str) -> bool:
    """Return whether text contains Chinese/Japanese/Korean ideographs."""

    return bool(re.search(r"[\u3400-\u9fff]", text))


def extract_ascii_keywords(text: str) -> list[str]:
    """Extract stable ASCII keywords already present in a query."""

    seen: set[str] = set()
    keywords: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9-]*", text.casefold()):
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def extract_primary_topic(text: str) -> str:
    """Reduce a task-level query back to its core research topic."""

    candidate = text.strip()
    if not candidate:
        return ""

    for separator in ("：", ":"):
        if separator in candidate:
            candidate = candidate.split(separator, 1)[0].strip()
            break

    candidate = re.sub(r"\b(19|20)\d{2}\b", "", candidate).strip()
    for suffix in _TASK_SUFFIXES:
        candidate = candidate.replace(suffix, " ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate or text.strip()
