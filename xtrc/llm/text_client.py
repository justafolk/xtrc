from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache


class LLMClientError(RuntimeError):
    pass


class LLMClientTimeoutError(LLMClientError):
    pass


class LLMTextClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        timeout_seconds: float = 2.0,
        cache_size: int = 256,
    ) -> None:
        self.provider = provider.strip().lower()
        self.model = model
        self.timeout_seconds = max(0.1, timeout_seconds)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xtrc-llm")
        self._cached_completion = lru_cache(maxsize=max(1, cache_size))(self._generate_uncached)

        if self.provider not in {"gemini", "openai"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self._gemini_configured = False
        self._openai_client = None

    def complete_text(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        started = time.perf_counter()
        model = model_name or self.model
        output = self._cached_completion(model, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = self._normalize_text(output)
        if not text:
            raise LLMClientError("LLM response was empty")
        return text, latency_ms

    def _generate_uncached(self, model_name: str, prompt: str) -> str:
        future = self._executor.submit(self._call_model, model_name, prompt)
        try:
            output = future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise LLMClientTimeoutError(
                f"LLM request timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except Exception as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc
        return output

    def _call_model(self, model_name: str, prompt: str) -> str:
        if self.provider == "gemini":
            return self._call_gemini(model_name, prompt)
        return self._call_openai(model_name, prompt)

    def _call_gemini(self, model_name: str, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover
            raise LLMClientError("google-generativeai is not installed") from exc

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMClientError("GEMINI_API_KEY is not set")

        if not self._gemini_configured:
            genai.configure(api_key=api_key)
            self._gemini_configured = True

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

        raise LLMClientError("Gemini response did not include text")

    def _call_openai(self, model_name: str, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise LLMClientError("openai package is not installed") from exc

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=api_key)

        response = self._openai_client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0.1,
            max_output_tokens=512,
        )

        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        output = getattr(response, "output", None)
        if output:
            parts: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if content is None:
                    continue
                for piece in content:
                    piece_text = getattr(piece, "text", None)
                    if isinstance(piece_text, str) and piece_text.strip():
                        parts.append(piece_text)
            if parts:
                return "\n".join(parts)

        raise LLMClientError("OpenAI response did not include text")

    @staticmethod
    def _normalize_text(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()
