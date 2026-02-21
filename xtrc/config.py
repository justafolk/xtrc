from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    model_name: str = "BAAI/bge-base-en-v1.5"
    qdrant_dirname: str = "qdrant"
    sqlite_name: str = "metadata.db"
    embedding_cache_name: str = "embeddings.db"
    max_batch_size: int = 256
    use_gemini: bool = False
    gemini_model: str = "gemini-2.5-flash"
    gemini_threshold: float = 0.85
    gemini_timeout_seconds: float = 2.0
    gemini_enable_rewrite: bool = False
    gemini_cache_size: int = 128
    gemini_summarize_on_index: bool = False
    gemini_summary_model: str = ""
    gemini_summary_max_chars: int = 320
    llm_provider: str = "gemini"
    llm_timeout_seconds: float = 2.0
    llm_cache_size: int = 256
    query_rewrite_enabled: bool = False
    query_rewrite_model: str = ""
    local_reranker_enabled: bool = False
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_reranker_top_k: int = 10
    heuristic_route_boost: float = 1.3
    heuristic_noise_penalty: float = 0.7
    heuristic_intent_boost: float = 1.2

    @staticmethod
    def from_env() -> "Settings":
        host = os.getenv("AINAV_HOST", "127.0.0.1")
        port = _env_int("AINAV_PORT", 8765)
        model_name = os.getenv("AINAV_MODEL", "BAAI/bge-base-en-v1.5")
        use_gemini = _env_bool("USE_GEMINI", False)
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_threshold = _env_float("GEMINI_THRESHOLD", 0.85)
        gemini_timeout_seconds = _env_float("GEMINI_TIMEOUT_SECONDS", 2.0)
        gemini_enable_rewrite = _env_bool("GEMINI_ENABLE_REWRITE", False)
        gemini_cache_size = _env_int("GEMINI_CACHE_SIZE", 128)
        gemini_summarize_on_index = _env_bool("GEMINI_SUMMARIZE_ON_INDEX", False)
        gemini_summary_model = os.getenv("GEMINI_SUMMARY_MODEL", "").strip()
        gemini_summary_max_chars = _env_int("GEMINI_SUMMARY_MAX_CHARS", 320)
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
        llm_timeout_seconds = _env_float("LLM_TIMEOUT_SECONDS", 2.0)
        llm_cache_size = _env_int("LLM_CACHE_SIZE", 256)
        query_rewrite_enabled = _env_bool("QUERY_REWRITE_ENABLED", False)
        query_rewrite_model = os.getenv("QUERY_REWRITE_MODEL", "").strip()
        local_reranker_enabled = _env_bool("LOCAL_RERANKER_ENABLED", False)
        local_reranker_model = (
            os.getenv("LOCAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        local_reranker_top_k = _env_int("LOCAL_RERANKER_TOP_K", 10)
        heuristic_route_boost = _env_float("HEURISTIC_ROUTE_BOOST", 1.3)
        heuristic_noise_penalty = _env_float("HEURISTIC_NOISE_PENALTY", 0.7)
        heuristic_intent_boost = _env_float("HEURISTIC_INTENT_BOOST", 1.2)
        return Settings(
            host=host,
            port=port,
            model_name=model_name,
            use_gemini=use_gemini,
            gemini_model=gemini_model,
            gemini_threshold=max(0.0, min(1.0, gemini_threshold)),
            gemini_timeout_seconds=max(0.1, gemini_timeout_seconds),
            gemini_enable_rewrite=gemini_enable_rewrite,
            gemini_cache_size=max(1, gemini_cache_size),
            gemini_summarize_on_index=gemini_summarize_on_index,
            gemini_summary_model=gemini_summary_model,
            gemini_summary_max_chars=max(64, gemini_summary_max_chars),
            llm_provider=llm_provider if llm_provider in {"gemini", "openai"} else "gemini",
            llm_timeout_seconds=max(0.1, llm_timeout_seconds),
            llm_cache_size=max(1, llm_cache_size),
            query_rewrite_enabled=query_rewrite_enabled,
            query_rewrite_model=query_rewrite_model,
            local_reranker_enabled=local_reranker_enabled,
            local_reranker_model=local_reranker_model,
            local_reranker_top_k=max(1, local_reranker_top_k),
            heuristic_route_boost=max(0.1, heuristic_route_boost),
            heuristic_noise_penalty=max(0.1, heuristic_noise_penalty),
            heuristic_intent_boost=max(0.1, heuristic_intent_boost),
        )


def resolve_data_root(repo_path: Path) -> Path:
    env_data_root = os.getenv("AINAV_DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root).expanduser().resolve()
    else:
        data_root = repo_path / ".xtrc"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root
