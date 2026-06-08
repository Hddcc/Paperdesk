"""Small SSE formatting helpers shared by route handlers and tests."""

from __future__ import annotations

import json


def sse_event(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def chunk_text(text: str, size: int = 18):
    if not text:
        return
    index = 0
    while index < len(text):
        next_index = min(len(text), index + size)
        newline_index = text.find("\n", index + 1, next_index + 1)
        if newline_index != -1:
            next_index = newline_index + 1
        yield text[index:next_index]
        index = next_index
