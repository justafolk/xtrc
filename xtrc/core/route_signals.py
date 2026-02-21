from __future__ import annotations

import re
from dataclasses import dataclass

from xtrc.core.tokenizer import normalize_terms

HTTP_INTENT_MAP: dict[str, str] = {
    "post": "create",
    "put": "update",
    "patch": "update",
    "delete": "delete",
    "get": "read",
}

_INTENT_ALIASES: dict[str, set[str]] = {
    "create": {"create", "add", "new", "insert", "post", "register", "submit"},
    "update": {"update", "edit", "modify", "put", "patch", "change"},
    "delete": {"delete", "remove", "destroy", "drop"},
    "read": {"read", "get", "fetch", "find", "list", "show", "retrieve"},
}

_STOP_TERMS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "where",
    "when",
    "which",
    "what",
    "does",
    "should",
    "route",
    "endpoint",
    "http",
    "api",
    "resource",
}

_JS_ROUTE_RE = re.compile(r"\.\s*(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_PY_DECORATOR_ROUTE_RE = re.compile(
    r"@[A-Za-z_][A-Za-z0-9_\.]*(?:router|app)?\.?\s*(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_GENERIC_METHOD_RE = re.compile(r"\b(get|post|put|delete|patch)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RouteSignal:
    method: str
    intent: str
    resource: str | None
    path: str | None
    structural_terms: list[str]


@dataclass(frozen=True)
class QuerySignal:
    intents: list[str]
    methods: list[str]
    structural_terms: list[str]


def extract_route_signal(text: str, symbol_name: str | None = None) -> RouteSignal | None:
    method: str | None = None
    path: str | None = None

    js_match = _JS_ROUTE_RE.search(text)
    if js_match:
        method = js_match.group(1).lower()
        path = js_match.group(2)
    else:
        py_match = _PY_DECORATOR_ROUTE_RE.search(text)
        if py_match:
            method = py_match.group(1).lower()
            path = py_match.group(2)

    if method is None:
        generic = _GENERIC_METHOD_RE.search(text)
        if generic:
            method = generic.group(1).lower()

    if method is None:
        return None

    intent = HTTP_INTENT_MAP.get(method)
    if intent is None:
        return None

    resource = _extract_resource(path) if path else _resource_from_symbol(symbol_name)

    terms = set()
    terms.add(method)
    terms.add(intent)

    if path:
        for segment in _path_segments(path):
            for token in normalize_terms(segment):
                terms.add(token)
    if resource:
        terms.add(resource)
    if symbol_name:
        for token in normalize_terms(symbol_name):
            terms.add(token)

    return RouteSignal(
        method=method.upper(),
        intent=intent,
        resource=resource,
        path=path,
        structural_terms=sorted(terms),
    )


def infer_query_signal(query: str) -> QuerySignal:
    terms = normalize_terms(query)
    methods = {term for term in terms if term in HTTP_INTENT_MAP}

    intents: set[str] = set()
    for intent, aliases in _INTENT_ALIASES.items():
        if any(term in aliases for term in terms):
            intents.add(intent)

    for method in methods:
        mapped = HTTP_INTENT_MAP.get(method)
        if mapped:
            intents.add(mapped)

    structural = {term for term in terms if term not in _STOP_TERMS}
    structural.update(methods)
    structural.update(intents)

    return QuerySignal(
        intents=sorted(intents),
        methods=sorted(methods),
        structural_terms=sorted(structural),
    )


def _path_segments(path: str) -> list[str]:
    normalized = path.strip()
    if not normalized:
        return []
    if normalized.startswith("http://") or normalized.startswith("https://"):
        normalized = normalized.split("//", maxsplit=1)[-1]
        normalized = normalized[normalized.find("/") :] if "/" in normalized else ""

    segments: list[str] = []
    for segment in normalized.strip("/").split("/"):
        if not segment:
            continue
        if segment.startswith(":"):
            continue
        if segment.startswith("{") and segment.endswith("}"):
            continue
        segments.append(segment)
    return segments


def _extract_resource(path: str | None) -> str | None:
    if not path:
        return None
    segments = _path_segments(path)
    if not segments:
        return None
    candidate = segments[0]
    tokens = normalize_terms(candidate)
    if not tokens:
        return None
    return _singularize(tokens[0])


def _resource_from_symbol(symbol_name: str | None) -> str | None:
    if not symbol_name:
        return None
    tokens = normalize_terms(symbol_name)
    if not tokens:
        return None
    for token in tokens:
        if token in {"create", "update", "delete", "get", "post", "put", "patch"}:
            continue
        return _singularize(token)
    return None


def _singularize(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return f"{value[:-3]}y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value
