"""arXiv API client for online paper search."""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

import httpx

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivClient:
    """Fetch paper metadata from the arXiv Atom API."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.http_client = http_client

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        """Search arXiv entries and return lightweight parsed payloads."""

        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        xml_text = self._request(params=params)
        return self._parse_entries(xml_text)

    def _request(self, *, params: dict[str, Any]) -> str:
        if self.http_client is not None:
            response = self.http_client.get(self.base_url, params=params, timeout=self.timeout)
        else:
            with httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": "PaperDesk/0.1"},
            ) as client:
                response = client.get(self.base_url, params=params)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_entries(xml_text: str) -> list[dict[str, Any]]:
        root = ElementTree.fromstring(xml_text)
        entries: list[dict[str, Any]] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            authors = [
                name.text.strip()
                for name in entry.findall("atom:author/atom:name", ATOM_NS)
                if name.text and name.text.strip()
            ]
            doi_element = entry.find("arxiv:doi", ATOM_NS)
            alternate_url = None
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
                    alternate_url = link.attrib["href"].strip()
                    break

            entries.append(
                {
                    "id": _safe_text(entry.find("atom:id", ATOM_NS)),
                    "title": _safe_text(entry.find("atom:title", ATOM_NS)),
                    "summary": _safe_text(entry.find("atom:summary", ATOM_NS)),
                    "published": _safe_text(entry.find("atom:published", ATOM_NS)),
                    "updated": _safe_text(entry.find("atom:updated", ATOM_NS)),
                    "authors": authors,
                    "doi": _safe_text(doi_element),
                    "url": alternate_url,
                }
            )

        return entries


def _safe_text(node: ElementTree.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None
