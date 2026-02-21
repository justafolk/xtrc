from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser

import tree_sitter_javascript as ts_javascript
import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_typescript

from xtrc.core.models import SymbolBlock

_ROUTE_PATTERN = re.compile(r"\.(get|post|put|delete|patch|route|use)\s*\(", re.IGNORECASE)
_PATH_ARG_PATTERN = re.compile(r"\(\s*['\"]/[^'\"\)]*['\"]")
_HANDLER_NAME_PATTERN = re.compile(r"handler|callback", re.IGNORECASE)


@dataclass(frozen=True)
class _SymbolDraft:
    kind: str
    name: str | None
    start_line: int
    end_line: int
    text: str


def _resolve_language(obj: object) -> Language:
    if isinstance(obj, Language):
        return obj
    try:
        return Language(obj)  # type: ignore[arg-type]
    except TypeError:
        return obj  # type: ignore[return-value]


def _set_parser_language(parser: Parser, language: Language) -> None:
    if hasattr(parser, "language"):
        parser.language = language
    else:
        parser.set_language(language)


def _node_text(source: bytes, start_byte: int, end_byte: int) -> str:
    return source[start_byte:end_byte].decode("utf-8", errors="ignore")


def _line_range(node: object) -> tuple[int, int]:
    start_row = int(node.start_point[0]) + 1
    end_row = int(node.end_point[0]) + 1
    return start_row, end_row


class TreeSitterCodeParser:
    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        parser_py = Parser()
        _set_parser_language(parser_py, _resolve_language(ts_python.language()))

        parser_js = Parser()
        _set_parser_language(parser_js, _resolve_language(ts_javascript.language()))

        parser_ts = Parser()
        _set_parser_language(parser_ts, _resolve_language(ts_typescript.language_typescript()))

        parser_tsx = Parser()
        _set_parser_language(parser_tsx, _resolve_language(ts_typescript.language_tsx()))

        self._parsers = {
            "python": parser_py,
            "javascript": parser_js,
            "typescript": parser_ts,
            "tsx": parser_tsx,
        }

    def parse_symbols(self, file_path: Path, language: str, content: str) -> list[SymbolBlock]:
        parser = self._parsers.get(language)
        if parser is None:
            return []

        source = content.encode("utf-8")
        tree = parser.parse(source)
        root = tree.root_node

        drafts: list[_SymbolDraft] = []
        stack = [root]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.children))

            if language == "python":
                self._collect_python(node, source, drafts)
            else:
                self._collect_js_ts(node, source, drafts)

        self._add_major_blocks(root, source, drafts)

        unique: dict[tuple[str, str | None, int, int], SymbolBlock] = {}
        for draft in drafts:
            key = (draft.kind, draft.name, draft.start_line, draft.end_line)
            if key in unique:
                continue
            unique[key] = SymbolBlock(
                kind=draft.kind,
                name=draft.name,
                start_line=draft.start_line,
                end_line=draft.end_line,
                text=draft.text,
            )

        return sorted(unique.values(), key=lambda s: (s.start_line, s.end_line, s.kind))

    def _collect_python(self, node: object, source: bytes, drafts: list[_SymbolDraft]) -> None:
        node_type = node.type

        if node_type in {"function_definition", "async_function_definition"}:
            name_node = node.child_by_field_name("name")
            name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
            kind = "handler" if name and _HANDLER_NAME_PATTERN.search(name) else "function"
            start_line, end_line = _line_range(node)
            drafts.append(
                _SymbolDraft(
                    kind=kind,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, node.start_byte, node.end_byte),
                )
            )
            return

        if node_type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
            start_line, end_line = _line_range(node)
            drafts.append(
                _SymbolDraft(
                    kind="class",
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, node.start_byte, node.end_byte),
                )
            )
            return

        if node_type == "decorated_definition":
            text = _node_text(source, node.start_byte, node.end_byte)
            if _ROUTE_PATTERN.search(text) or "@app" in text:
                start_line, end_line = _line_range(node)
                name_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", text)
                drafts.append(
                    _SymbolDraft(
                        kind="route",
                        name=name_match.group(1) if name_match else None,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                    )
                )
            return

        if node_type == "call":
            text = _node_text(source, node.start_byte, node.end_byte)
            if _ROUTE_PATTERN.search(text) and _PATH_ARG_PATTERN.search(text):
                start_line, end_line = _line_range(node)
                drafts.append(
                    _SymbolDraft(
                        kind="route",
                        name=None,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                    )
                )

    def _collect_js_ts(self, node: object, source: bytes, drafts: list[_SymbolDraft]) -> None:
        node_type = node.type

        if node_type in {"function_declaration", "generator_function_declaration"}:
            name_node = node.child_by_field_name("name")
            name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
            kind = "handler" if name and _HANDLER_NAME_PATTERN.search(name) else "function"
            start_line, end_line = _line_range(node)
            drafts.append(
                _SymbolDraft(
                    kind=kind,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, node.start_byte, node.end_byte),
                )
            )
            return

        if node_type == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
            start_line, end_line = _line_range(node)
            drafts.append(
                _SymbolDraft(
                    kind="class",
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, node.start_byte, node.end_byte),
                )
            )
            return

        if node_type == "method_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
            start_line, end_line = _line_range(node)
            drafts.append(
                _SymbolDraft(
                    kind="handler" if name and _HANDLER_NAME_PATTERN.search(name) else "function",
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, node.start_byte, node.end_byte),
                )
            )
            return

        if node_type == "variable_declarator":
            value_node = node.child_by_field_name("value")
            if value_node and value_node.type in {
                "arrow_function",
                "function",
                "function_expression",
            }:
                name_node = node.child_by_field_name("name")
                name = _node_text(source, name_node.start_byte, name_node.end_byte) if name_node else None
                start_line, end_line = _line_range(node)
                drafts.append(
                    _SymbolDraft(
                        kind="handler" if name and _HANDLER_NAME_PATTERN.search(name) else "function",
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        text=_node_text(source, node.start_byte, node.end_byte),
                    )
                )
            return

        if node_type == "call_expression":
            text = _node_text(source, node.start_byte, node.end_byte)
            if _ROUTE_PATTERN.search(text) and _PATH_ARG_PATTERN.search(text):
                start_line, end_line = _line_range(node)
                route_name = self._extract_route_name(text)
                drafts.append(
                    _SymbolDraft(
                        kind="route",
                        name=route_name,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                    )
                )

    def _add_major_blocks(self, root: object, source: bytes, drafts: list[_SymbolDraft]) -> None:
        occupied: list[tuple[int, int]] = [(draft.start_line, draft.end_line) for draft in drafts]
        for child in root.children:
            if not child.is_named:
                continue
            if child.type in {
                "import_statement",
                "import_from_statement",
                "lexical_declaration",
                "variable_declaration",
                "comment",
                "expression_statement",
            }:
                continue
            start_line, end_line = _line_range(child)
            span = end_line - start_line + 1
            if span < 15:
                continue
            if any(start_line >= s and end_line <= e for s, e in occupied):
                continue
            drafts.append(
                _SymbolDraft(
                    kind="major_block",
                    name=None,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(source, child.start_byte, child.end_byte),
                )
            )
            occupied.append((start_line, end_line))

    @staticmethod
    def _extract_route_name(text: str) -> str | None:
        method_match = re.search(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        path_match = re.search(r"\(\s*['\"](/[^'\"]*)['\"]", text)
        if not method_match:
            return None
        method = method_match.group(1)
        path = path_match.group(1) if path_match else ""
        return f"{method.upper()} {path}".strip()
