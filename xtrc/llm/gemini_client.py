from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Any

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - exercised in integration environments
    genai = None

_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class GeminiClientError(RuntimeError):
    pass


class GeminiTimeoutError(GeminiClientError):
    pass


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        timeout_seconds: float = 3.0,
        cache_size: int = 128,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Add it to dependencies to use Gemini."
            )

        genai.configure(api_key=api_key)
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xtrc-gemini")
        self._cached_completion = lru_cache(maxsize=cache_size)(self._generate_uncached)

    def complete_json(self, prompt: str, *, model_name: str | None = None) -> tuple[dict[str, Any], int]:
        started = time.perf_counter()
        model = model_name or self.default_model
        raw_text = self._cached_completion(model, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._parse_json_object(raw_text), latency_ms

    def complete_text(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        started = time.perf_counter()
        model = model_name or self.default_model
        raw_text = self._cached_completion(model, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = self._parse_rewrite_text(raw_text).strip()
        if not text:
            raise GeminiClientError("Gemini response was empty")
        return text, latency_ms

    def rewrite_query(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        started = time.perf_counter()
        model = model_name or self.default_model
        raw_text = self._cached_completion(model, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        rewritten = self._parse_rewrite_text(raw_text)
        if not rewritten:
            raise GeminiClientError("Gemini rewrite response was empty")
        return rewritten, latency_ms

    def _generate_uncached(self, model_name: str, prompt: str) -> str:
        future = self._executor.submit(self._call_model, model_name, prompt)
        try:
            raw = future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise GeminiTimeoutError(f"Gemini request timed out after {self.timeout_seconds:.1f}s") from exc
        except Exception as exc:
            raise GeminiClientError(f"Gemini request failed: {exc}") from exc

        if not raw.strip():
            raise GeminiClientError("Gemini returned an empty response")
        return raw

    @staticmethod
    def _call_model(model_name: str, prompt: str) -> str:
        if genai is None:
            raise GeminiClientError("google-generativeai is unavailable")

        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 512,
            },
        )

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", None)
                if parts is None:
                    continue
                joined: list[str] = []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        joined.append(part_text)
                if joined:
                    return "\n".join(joined)

        raise GeminiClientError("Gemini response did not include text")

    @staticmethod
    def _parse_json_object(raw_text: str) -> dict[str, Any]:
        candidates = [raw_text]

        for match in _JSON_CODE_BLOCK_RE.finditer(raw_text):
            candidates.append(match.group(1))

        object_match = _JSON_OBJECT_RE.search(raw_text)
        if object_match:
            candidates.append(object_match.group(0))

        for candidate in candidates:
            snippet = candidate.strip()
            if not snippet:
                continue
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        raise GeminiClientError("Gemini did not return a valid JSON object")

    @staticmethod
    def _parse_rewrite_text(raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                query = parsed.get("query")
                if isinstance(query, str):
                    return query.strip()
        except json.JSONDecodeError:
            pass

        first_line = text.splitlines()[0].strip() if text else ""
        if (first_line.startswith('"') and first_line.endswith('"')) or (
            first_line.startswith("'") and first_line.endswith("'")
        ):
            first_line = first_line[1:-1].strip()
        return first_line
