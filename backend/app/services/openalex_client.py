"""OpenAlex API client for online paper search."""

from __future__ import annotations

from typing import Any

import httpx


class OpenAlexClient:
    """Fetch paper metadata from the OpenAlex Works API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.http_client = http_client

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        """Search works and return provider payloads."""

        params: dict[str, Any] = {
            "search": query,
            "per-page": limit,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = self._request("/works", params=params)
        results = response.get("results")
        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]

    def _request(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        if self.http_client is not None:
            response = self.http_client.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        else:
            with httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": "PaperDesk/0.1"},
            ) as client:
                response = client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
