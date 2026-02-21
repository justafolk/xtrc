from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from xtrc.core.errors import AinavError


class HttpAinavClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _normalize_repo_path(repo_path: str) -> str:
        return str(Path(repo_path).expanduser().resolve())

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(method=method, url=url, **kwargs)
        except httpx.RequestError as exc:
            raise AinavError(
                code="SERVER_UNREACHABLE",
                message=f"Could not reach xtrc server at {self.base_url}: {exc}",
                status_code=503,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise AinavError(
                code="INVALID_RESPONSE",
                message=f"Server returned non-JSON response ({response.status_code})",
                status_code=502,
            ) from exc

        if response.status_code >= 400:
            if isinstance(payload, dict) and payload.get("error"):
                error = payload["error"]
                message = str(error.get("message", "Server error"))
                code = str(error.get("code", "SERVER_ERROR"))
                details = error.get("details")
                raise AinavError(code=code, message=message, status_code=response.status_code, details=details)
            raise AinavError(
                code="SERVER_ERROR",
                message=f"Server error {response.status_code}",
                status_code=response.status_code,
            )

        if not isinstance(payload, dict):
            raise AinavError(
                code="INVALID_RESPONSE",
                message="Server returned malformed payload",
                status_code=502,
            )

        return payload

    def index(self, repo_path: str, rebuild: bool = False) -> dict[str, Any]:
        normalized_repo = self._normalize_repo_path(repo_path)
        return self._request("POST", "/index", json={"repo_path": normalized_repo, "rebuild": rebuild})

    def query(self, repo_path: str, query: str, top_k: int = 8) -> dict[str, Any]:
        normalized_repo = self._normalize_repo_path(repo_path)
        return self._request(
            "POST",
            "/query",
            json={"repo_path": normalized_repo, "query": query, "top_k": top_k},
        )

    def status(self, repo_path: str) -> dict[str, Any]:
        normalized_repo = self._normalize_repo_path(repo_path)
        return self._request("GET", "/status", params={"repo_path": normalized_repo})
