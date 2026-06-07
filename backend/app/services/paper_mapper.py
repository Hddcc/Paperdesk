"""Compatibility wrapper for paper domain mapping helpers."""

from app.domains.paper.mapper import map_arxiv_entry, map_openalex_work, normalize_title

__all__ = ["map_arxiv_entry", "map_openalex_work", "normalize_title"]
