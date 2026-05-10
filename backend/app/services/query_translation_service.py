"""Optional translation fallback for provider-facing search queries."""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from app.utils.query_utils import contains_cjk

_GLOSSARY = [
    ("检索增强生成", "retrieval augmented generation"),
    ("人脸超分辨率", "face super-resolution"),
    ("人脸超分", "face super-resolution"),
    ("图像超分辨率", "image super-resolution"),
    ("图像超分", "image super-resolution"),
    ("目标检测", "object detection"),
    ("语义分割", "semantic segmentation"),
    ("机器翻译", "machine translation"),
    ("知识蒸馏", "knowledge distillation"),
    ("多模态", "multimodal"),
    ("遥感", "remote sensing"),
    ("行人重识别", "person re-identification"),
    ("视频生成", "video generation"),
    ("图像生成", "image generation"),
    ("问答系统", "question answering"),
    ("推荐系统", "recommender systems"),
    ("联邦学习", "federated learning"),
    ("异常检测", "anomaly detection"),
    ("医学影像", "medical imaging"),
    ("时间序列", "time series"),
    ("强化学习", "reinforcement learning"),
    ("自监督学习", "self-supervised learning"),
    ("对比学习", "contrastive learning"),
    ("小样本学习", "few-shot learning"),
    ("大语言模型", "large language models"),
    ("视觉语言模型", "vision language models"),
    ("人脸", "face"),
    ("超分辨率", "super-resolution"),
    ("超分", "super-resolution"),
]


class QueryTranslationService:
    """Translate Chinese research topics into concise English search phrases."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._cache: dict[str, str | None] = {}

    def translate_to_english(self, query: str) -> str | None:
        """Return a concise English search phrase, or None on failure."""

        cleaned = query.strip()
        if not cleaned:
            return None
        if cleaned in self._cache:
            return self._cache[cleaned]

        glossary_translation = _translate_with_glossary(cleaned)
        if glossary_translation is not None:
            self._cache[cleaned] = glossary_translation
            return glossary_translation
        if not self.api_key:
            self._cache[cleaned] = None
            return None

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You convert Chinese academic topics into concise English literature "
                            "search phrases. Return only the English query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": cleaned,
                    },
                ],
            )
        except Exception:
            self._cache[cleaned] = None
            return None

        translated = _extract_message_text(response)
        if translated:
            translated = translated.strip().strip("\"'`")
        self._cache[cleaned] = translated or None
        return self._cache[cleaned]


def _extract_message_text(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return " ".join(parts) if parts else None
    return None


def _translate_with_glossary(query: str) -> str | None:
    translated = query
    matched = False
    for source, target in _GLOSSARY:
        if source in translated:
            translated = translated.replace(source, f" {target} ")
            matched = True

    translated = re.sub(r"[：:，。；、（）()\[\]【】]+", " ", translated)
    translated = re.sub(r"\s+", " ", translated).strip()
    if not matched:
        return None
    if contains_cjk(translated):
        return None
    return translated or None
