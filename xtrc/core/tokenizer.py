from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")


def estimate_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def normalize_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    return [token for token in tokens if len(token) > 1]
