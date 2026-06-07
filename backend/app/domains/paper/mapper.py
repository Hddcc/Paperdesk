"""Provider-specific payload mapping into normalized paper records."""

from __future__ import annotations

import re
from uuid import NAMESPACE_URL, uuid5

from app.models import PaperRecord
from app.models.paper import normalize_doi_value


def normalize_title(value: str) -> str:
    """Normalize titles for deterministic deduplication."""

    return " ".join(value.casefold().split())


def restore_openalex_abstract(abstract_index: dict[str, list[int]] | None) -> str | None:
    """Convert OpenAlex inverted abstract index back into plain text."""

    if not abstract_index:
        return None

    max_position = -1
    for positions in abstract_index.values():
        for position in positions:
            if position > max_position:
                max_position = position

    if max_position < 0:
        return None

    tokens = [""] * (max_position + 1)
    for token, positions in abstract_index.items():
        for position in positions:
            if 0 <= position < len(tokens):
                tokens[position] = token

    abstract = " ".join(token for token in tokens if token).strip()
    return abstract or None


def build_stable_paper_id(source: str, *, provider_id: str | None = None, url: str | None = None, title: str) -> str:
    """Build a stable paper identifier with a provider prefix."""

    if provider_id:
        cleaned = provider_id.strip()
        if cleaned.startswith("http"):
            cleaned = cleaned.rstrip("/").rsplit("/", 1)[-1]
        if cleaned:
            return f"{source}:{cleaned}"

    seed = url or title
    return f"{source}:{uuid5(NAMESPACE_URL, f'{source}:{seed}')}"


def map_openalex_work(work: dict) -> PaperRecord:
    """Map an OpenAlex work payload to PaperRecord."""

    title = str(work.get("display_name") or "").strip()
    authors = [
        str(author_name).strip()
        for authorship in work.get("authorships", [])
        for author_name in [
            (authorship.get("author") or {}).get("display_name") or authorship.get("raw_author_name")
        ]
        if author_name and str(author_name).strip()
    ]

    primary_location = work.get("primary_location") or {}
    source_metadata = primary_location.get("source") or {}
    url = (
        primary_location.get("landing_page_url")
        or primary_location.get("pdf_url")
        or work.get("id")
    )
    venue = (
        source_metadata.get("display_name")
        or ((work.get("host_venue") or {}).get("display_name"))
        or None
    )

    return PaperRecord(
        paper_id=build_stable_paper_id(
            "openalex",
            provider_id=work.get("id"),
            url=url,
            title=title,
        ),
        title=title,
        authors=authors,
        abstract=restore_openalex_abstract(work.get("abstract_inverted_index")),
        year=work.get("publication_year"),
        venue=venue,
        doi=normalize_doi_value(work.get("doi")),
        url=url,
        source="openalex",
    )


def _extract_arxiv_identifier(entry_id: str | None) -> str | None:
    if not entry_id:
        return None

    cleaned = entry_id.rstrip("/").rsplit("/", 1)[-1]
    return cleaned or None


def map_arxiv_entry(entry: dict) -> PaperRecord:
    """Map an arXiv entry payload to PaperRecord."""

    title = re.sub(r"\s+", " ", str(entry.get("title") or "")).strip()
    summary = re.sub(r"\s+", " ", str(entry.get("summary") or "")).strip() or None
    published = str(entry.get("published") or "").strip()
    year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
    entry_id = str(entry.get("id") or "").strip() or None
    url = str(entry.get("url") or entry_id or "").strip() or None

    return PaperRecord(
        paper_id=build_stable_paper_id(
            "arxiv",
            provider_id=_extract_arxiv_identifier(entry_id),
            url=url,
            title=title,
        ),
        title=title,
        authors=[str(author).strip() for author in entry.get("authors", []) if str(author).strip()],
        abstract=summary,
        year=year,
        venue="arXiv",
        doi=normalize_doi_value(entry.get("doi")),
        url=url,
        source="arxiv",
    )
