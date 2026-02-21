from __future__ import annotations

from dataclasses import dataclass

from xtrc.core.route_signals import extract_route_signal
from xtrc.core.tokenizer import normalize_terms

_NOISE_PATH_HINTS = {
    "seed",
    "seeds",
    "migration",
    "migrations",
    "fixture",
    "fixtures",
    "dummy",
    "mock",
    "test",
    "tests",
    "spec",
    "script",
    "scripts",
}

_LOGGING_HINTS = {"log", "logger", "logging", "audit", "trace"}
_ANALYTICS_HINTS = {"analytics", "metric", "metrics", "telemetry", "tracking", "event"}

_CREATE_HINTS = {"create", "insert", "add", "register", "new", "post"}
_UPDATE_HINTS = {"update", "modify", "edit", "patch", "put", "upsert"}
_DELETE_HINTS = {"delete", "remove", "destroy", "drop"}
_READ_HINTS = {"get", "fetch", "read", "list", "find", "retrieve", "query"}


@dataclass(frozen=True)
class IntentMetadata:
    intent_tags: list[str]
    route_method: str | None
    route_path: str | None
    route_intent: str | None
    route_resource: str | None
    structural_terms: list[str]
    is_route_handler: bool


def extract_intent_metadata(
    *,
    file_path: str,
    symbol_kind: str | None,
    symbol: str | None,
    text: str,
) -> IntentMetadata:
    terms = set(normalize_terms(f"{file_path}\n{symbol or ''}\n{text[:8000]}"))

    route_signal = extract_route_signal(text, symbol_name=symbol)
    route_method = route_signal.method if route_signal is not None else None
    route_path = route_signal.path if route_signal is not None else None
    route_intent = route_signal.intent if route_signal is not None else None
    route_resource = route_signal.resource if route_signal is not None else None

    tags: set[str] = set()
    if route_intent:
        tags.add(f"{route_intent}_resource")

    if _has_any(file_path, _NOISE_PATH_HINTS) or _has_any_set(terms, {"fixture", "fixtures", "mock"}):
        if "seed" in file_path or "seeds" in file_path:
            tags.add("seed_data")
        if "migration" in file_path or "migrations" in file_path:
            tags.add("migration_script")
        if _has_any(file_path, {"test", "tests", "spec"}):
            tags.add("test_script")
        if _has_any(file_path, {"script", "scripts"}):
            tags.add("script")

    if _has_any_set(terms, _LOGGING_HINTS):
        tags.add("logging")
    if _has_any_set(terms, _ANALYTICS_HINTS):
        tags.add("analytics")

    if _has_any_set(terms, _CREATE_HINTS):
        tags.add("create_resource")
    if _has_any_set(terms, _UPDATE_HINTS):
        tags.add("update_resource")
    if _has_any_set(terms, _DELETE_HINTS):
        tags.add("delete_resource")
    if _has_any_set(terms, _READ_HINTS):
        tags.add("read_resource")

    if route_signal is not None:
        tags.add("route_handler")

    structural_terms = set(terms)
    if route_signal is not None:
        structural_terms.update(route_signal.structural_terms)
        structural_terms.add(route_signal.method.lower())
        structural_terms.add(route_signal.intent.lower())
        if route_signal.resource:
            structural_terms.update(normalize_terms(route_signal.resource))

    is_route_handler = route_signal is not None or (symbol_kind == "route")

    return IntentMetadata(
        intent_tags=sorted(tags),
        route_method=route_method,
        route_path=route_path,
        route_intent=route_intent,
        route_resource=route_resource,
        structural_terms=sorted(structural_terms),
        is_route_handler=is_route_handler,
    )


def _has_any(value: str, candidates: set[str]) -> bool:
    low = value.lower()
    return any(candidate in low for candidate in candidates)


def _has_any_set(values: set[str], candidates: set[str]) -> bool:
    return bool(values.intersection(candidates))
