from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xtrc.core.models import CodeChunk, SymbolBlock
from xtrc.indexer.intent import IntentMetadata, extract_intent_metadata
from xtrc.core.tokenizer import estimate_tokens, normalize_terms


@dataclass(frozen=True)
class _ChunkDraft:
    start_line: int
    end_line: int
    symbol: str | None
    symbol_kind: str | None
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


class ChunkBuilder:
    def __init__(self, min_tokens: int = 200, max_tokens: int = 800, target_tokens: int = 500) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens

    def build_chunks(
        self,
        *,
        repo_path: Path,
        file_path: Path,
        language: str,
        file_hash: str,
        content: str,
        symbols: list[SymbolBlock],
    ) -> list[CodeChunk]:
        relative_path = str(file_path.relative_to(repo_path))
        drafts = self._initial_drafts(content, symbols)
        drafts = self._split_large_drafts(drafts)
        drafts = self._merge_small_drafts(drafts)

        chunks: list[CodeChunk] = []
        for draft in drafts:
            intent_meta = extract_intent_metadata(
                file_path=relative_path,
                symbol_kind=draft.symbol_kind,
                symbol=draft.symbol,
                text=draft.text,
            )
            description = self._describe(relative_path, draft, intent_meta)
            symbol_terms = normalize_terms(draft.symbol or "")
            structural_terms = list(intent_meta.structural_terms)
            structural_terms.extend(normalize_terms(relative_path))
            structural_terms = sorted(set(structural_terms))
            if structural_terms:
                symbol_terms.extend(structural_terms)
            symbol_terms = sorted(set(symbol_terms))

            keyword_source = [description, draft.text[:4000]]
            if intent_meta.route_method:
                route_context = [
                    f"Intent: {intent_meta.route_intent or 'unknown'} resource",
                    f"HTTP method: {intent_meta.route_method}",
                ]
                if intent_meta.route_resource:
                    route_context.append(f"Resource: {intent_meta.route_resource}")
                if intent_meta.route_path:
                    route_context.append(f"Path: {intent_meta.route_path}")
                keyword_source.append("\n".join(route_context))
            keyword_terms = normalize_terms("\n".join(keyword_source))
            chunk_hash = hashlib.sha256(
                f"{relative_path}|{draft.start_line}|{draft.end_line}|{draft.text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_hash,
                    repo_path=str(repo_path),
                    file_path=relative_path,
                    language=language,
                    start_line=draft.start_line,
                    end_line=draft.end_line,
                    symbol=draft.symbol,
                    symbol_kind=draft.symbol_kind,
                    description=description,
                    text=draft.text,
                    content_hash=file_hash,
                    tokens=estimate_tokens(draft.text),
                    keywords=keyword_terms,
                    symbol_terms=symbol_terms,
                    route_method=intent_meta.route_method,
                    route_path=intent_meta.route_path,
                    route_intent=intent_meta.route_intent,
                    route_resource=intent_meta.route_resource,
                    intent_tags=intent_meta.intent_tags,
                    structural_terms=structural_terms,
                )
            )

        return chunks

    def _initial_drafts(self, content: str, symbols: list[SymbolBlock]) -> list[_ChunkDraft]:
        lines = content.splitlines()
        if not symbols:
            return self._slice_file_fallback(lines)

        ordered = sorted(symbols, key=lambda s: (s.start_line, s.end_line))
        drafts: list[_ChunkDraft] = []
        for symbol in ordered:
            start = max(symbol.start_line, 1)
            end = max(symbol.end_line, start)
            text = "\n".join(lines[start - 1 : end]).strip()
            if not text:
                continue
            drafts.append(
                _ChunkDraft(
                    start_line=start,
                    end_line=end,
                    symbol=symbol.name,
                    symbol_kind=symbol.kind,
                    text=text,
                )
            )

        return drafts or self._slice_file_fallback(lines)

    def _slice_file_fallback(self, lines: list[str]) -> list[_ChunkDraft]:
        if not lines:
            return []
        content = "\n".join(lines)
        tokens = estimate_tokens(content)
        if tokens <= self.max_tokens:
            return [
                _ChunkDraft(
                    start_line=1,
                    end_line=len(lines),
                    symbol=None,
                    symbol_kind="major_block",
                    text=content,
                )
            ]

        drafts: list[_ChunkDraft] = []
        offset = 1
        for text, start, end in self._split_text_by_lines(content, offset):
            drafts.append(
                _ChunkDraft(
                    start_line=start,
                    end_line=end,
                    symbol=None,
                    symbol_kind="major_block",
                    text=text,
                )
            )
        return drafts

    def _split_large_drafts(self, drafts: list[_ChunkDraft]) -> list[_ChunkDraft]:
        out: list[_ChunkDraft] = []
        for draft in drafts:
            if draft.tokens <= self.max_tokens:
                out.append(draft)
                continue
            for part_text, start, end in self._split_text_by_lines(draft.text, draft.start_line):
                out.append(
                    _ChunkDraft(
                        start_line=start,
                        end_line=end,
                        symbol=draft.symbol,
                        symbol_kind=draft.symbol_kind,
                        text=part_text,
                    )
                )
        return out

    def _split_text_by_lines(self, text: str, start_line: int) -> list[tuple[str, int, int]]:
        lines = text.splitlines()
        chunks: list[tuple[str, int, int]] = []
        current: list[str] = []
        current_tokens = 0
        block_start = start_line

        for idx, line in enumerate(lines):
            line_tokens = estimate_tokens(line)
            projected = current_tokens + line_tokens
            if current and projected > self.target_tokens and current_tokens >= self.min_tokens:
                end_line = block_start + len(current) - 1
                chunks.append(("\n".join(current), block_start, end_line))
                block_start = start_line + idx
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

            if current_tokens >= self.max_tokens:
                end_line = block_start + len(current) - 1
                chunks.append(("\n".join(current), block_start, end_line))
                block_start = end_line + 1
                current = []
                current_tokens = 0

        if current:
            end_line = block_start + len(current) - 1
            chunks.append(("\n".join(current), block_start, end_line))

        return chunks

    def _merge_small_drafts(self, drafts: list[_ChunkDraft]) -> list[_ChunkDraft]:
        if not drafts:
            return []

        sorted_drafts = sorted(drafts, key=lambda d: (d.start_line, d.end_line))
        merged: list[_ChunkDraft] = []
        buffer: list[_ChunkDraft] = []

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            if len(buffer) == 1:
                merged.append(buffer[0])
            else:
                text = "\n\n".join(part.text for part in buffer)
                merged.append(
                    _ChunkDraft(
                        start_line=buffer[0].start_line,
                        end_line=buffer[-1].end_line,
                        symbol=buffer[0].symbol if len(buffer) == 1 else None,
                        symbol_kind=buffer[0].symbol_kind if len(buffer) == 1 else "major_block",
                        text=text,
                    )
                )
            buffer = []

        for draft in sorted_drafts:
            if draft.tokens >= self.min_tokens:
                flush_buffer()
                merged.append(draft)
                continue

            if not buffer:
                buffer.append(draft)
                continue

            current_tokens = estimate_tokens("\n\n".join(part.text for part in buffer))
            gap = draft.start_line - buffer[-1].end_line
            if current_tokens + draft.tokens <= self.max_tokens and gap <= 40:
                buffer.append(draft)
            else:
                flush_buffer()
                buffer.append(draft)

        flush_buffer()

        # If tail chunks are still very small, merge them backward where possible.
        if len(merged) >= 2 and merged[-1].tokens < self.min_tokens:
            tail = merged.pop()
            prev = merged.pop()
            combined_text = f"{prev.text}\n\n{tail.text}"
            if estimate_tokens(combined_text) <= self.max_tokens:
                merged.append(
                    _ChunkDraft(
                        start_line=prev.start_line,
                        end_line=tail.end_line,
                        symbol=prev.symbol,
                        symbol_kind=prev.symbol_kind,
                        text=combined_text,
                    )
                )
            else:
                merged.extend([prev, tail])

        return merged

    @staticmethod
    def _describe(file_path: str, draft: _ChunkDraft, intent_meta: IntentMetadata) -> str:
        first_line = draft.text.strip().splitlines()[0].strip() if draft.text.strip() else ""
        preview = first_line[:120]
        route_suffix = ""
        if intent_meta.route_method:
            route_suffix = (
                f" Intent: {intent_meta.route_intent or 'unknown'} resource."
                f" HTTP method: {intent_meta.route_method}."
            )
            if intent_meta.route_resource:
                route_suffix += f" Resource: {intent_meta.route_resource}."
            if intent_meta.route_path:
                route_suffix += f" Path: {intent_meta.route_path}."
        if intent_meta.intent_tags:
            route_suffix += f" Tags: {', '.join(intent_meta.intent_tags)}."
        if draft.symbol_kind == "class" and draft.symbol:
            return f"Class {draft.symbol} in {file_path}. Starts with: {preview}{route_suffix}".strip()
        if draft.symbol_kind == "route":
            route_name = draft.symbol or "unnamed route"
            return (
                f"Route handler {route_name} in {file_path}. Starts with: {preview}{route_suffix}"
            ).strip()
        if draft.symbol_kind == "handler":
            handler = draft.symbol or "anonymous handler"
            return f"Handler {handler} in {file_path}. Starts with: {preview}{route_suffix}".strip()
        if draft.symbol and draft.symbol_kind == "function":
            return f"Function {draft.symbol} in {file_path}. Starts with: {preview}{route_suffix}".strip()
        return (
            f"Major code block in {file_path} (lines {draft.start_line}-{draft.end_line}). "
            f"Starts with: {preview}{route_suffix}"
        ).strip()
