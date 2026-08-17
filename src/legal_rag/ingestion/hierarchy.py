"""Executable hierarchy-regex.v1 parser in canonical NFC offset space."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes

HierarchyKind = Literal["part", "chapter", "section", "subsection", "article", "clause", "point"]

_H = r"[^\S\r\n]"
_NUM = r"[0-9]+[A-Za-zĐđ]?"
_TAIL = rf"(?:[.:\-–—]{_H}*(?P<title>.*))?{_H}*$"
_FLAGS = re.IGNORECASE | re.UNICODE


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    kind: HierarchyKind
    level: int
    pattern: str
    label: str
    implicit: bool = False


_EXPLICIT_RULES = (
    _Rule(
        "HIER_PART",
        "part",
        0,
        rf"^{_H}*Phần{_H}+(?P<ordinal>[IVXLCDM]+|[0-9]+){_H}*{_TAIL}",
        "Phần",
    ),
    _Rule(
        "HIER_CHAPTER",
        "chapter",
        1,
        rf"^{_H}*Chương{_H}+(?P<ordinal>[IVXLCDM]+|[0-9]+){_H}*{_TAIL}",
        "Chương",
    ),
    _Rule(
        "HIER_SUBSECTION",
        "subsection",
        3,
        rf"^{_H}*Tiểu{_H}+mục{_H}+(?P<ordinal>{_NUM}){_H}*{_TAIL}",
        "Tiểu mục",
    ),
    _Rule(
        "HIER_SECTION",
        "section",
        2,
        rf"^{_H}*Mục{_H}+(?P<ordinal>{_NUM}){_H}*{_TAIL}",
        "Mục",
    ),
    _Rule(
        "HIER_ARTICLE",
        "article",
        4,
        rf"^{_H}*Điều{_H}+(?P<ordinal>{_NUM}){_H}*{_TAIL}",
        "Điều",
    ),
    _Rule(
        "HIER_CLAUSE",
        "clause",
        5,
        rf"^{_H}*Khoản{_H}+(?P<ordinal>{_NUM}){_H}*{_TAIL}",
        "Khoản",
    ),
    _Rule(
        "HIER_POINT",
        "point",
        6,
        rf"^{_H}*Điểm{_H}+(?P<ordinal>[A-Za-zĐđ]){_H}*{_TAIL}",
        "Điểm",
    ),
)
_IMPLICIT_CLAUSE = _Rule(
    "IMPLICIT_CLAUSE",
    "clause",
    5,
    rf"^{_H}*(?P<ordinal>[0-9]+)[.)]{_H}+(?P<body>.+){_H}*$",
    "Khoản",
    implicit=True,
)
_IMPLICIT_POINT = _Rule(
    "IMPLICIT_POINT",
    "point",
    6,
    rf"^{_H}*(?P<ordinal>[A-Za-zĐđ])[.)]{_H}+(?P<body>.+){_H}*$",
    "Điểm",
    implicit=True,
)
_ALL_RULES = (*_EXPLICIT_RULES, _IMPLICIT_CLAUSE, _IMPLICIT_POINT)
_COMPILED = {rule.rule_id: re.compile(rule.pattern, _FLAGS) for rule in _ALL_RULES}


@dataclass(frozen=True, slots=True)
class HierarchyNode:
    rule_id: str
    kind: HierarchyKind
    ordinal: str
    title: str | None
    label: str
    hierarchy_path: tuple[str, ...]
    canonical_start: int
    canonical_end: int
    line_text: str
    heading_only: bool
    level: int


@dataclass(frozen=True, slots=True)
class HierarchyWarning:
    code: str
    canonical_start: int
    message: str


@dataclass(frozen=True, slots=True)
class HierarchyParseResult:
    nodes: tuple[HierarchyNode, ...]
    warnings: tuple[HierarchyWarning, ...]


def hierarchy_regex_manifest() -> dict[str, Any]:
    rules = [
        {
            "rule_id": rule.rule_id,
            "kind": rule.kind,
            "pattern": rule.pattern,
            "precedence": precedence,
        }
        for precedence, rule in enumerate(_ALL_RULES)
    ]
    return {
        "schema_version": "hierarchy-regex.v1",
        "unicode_version": unicodedata.unidata_version,
        "flags": ["IGNORECASE", "UNICODE"],
        "rules": rules,
        "patterns_checksum": checksum_bytes(canonical_json_bytes({"rules": rules})),
    }


def _line_spans(passage: str) -> tuple[tuple[int, int, str], ...]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for raw_line in passage.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        spans.append((offset, offset + len(content), content))
        offset += len(raw_line)
    if offset < len(passage) or not passage:
        spans.append((offset, len(passage), passage[offset:]))
    return tuple(spans)


def _matching_rule(
    line: str,
    active: dict[int, HierarchyNode],
) -> tuple[_Rule, re.Match[str]] | None:
    for rule in _EXPLICIT_RULES:
        match = _COMPILED[rule.rule_id].match(line)
        if match is not None:
            return rule, match
    if 4 in active:
        match = _COMPILED[_IMPLICIT_CLAUSE.rule_id].match(line)
        if match is not None:
            return _IMPLICIT_CLAUSE, match
    if 5 in active:
        match = _COMPILED[_IMPLICIT_POINT.rule_id].match(line)
        if match is not None:
            return _IMPLICIT_POINT, match
    return None


def parse_hierarchy(passage: str) -> HierarchyParseResult:
    """Parse hierarchy markers without altering canonical passage coordinates."""

    canonical = unicodedata.normalize("NFC", passage)
    active: dict[int, HierarchyNode] = {}
    nodes: list[HierarchyNode] = []
    warnings: list[HierarchyWarning] = []
    previous_numeric: dict[tuple[HierarchyKind, tuple[str, ...]], int] = {}

    for canonical_start, canonical_end, line in _line_spans(canonical):
        selected = _matching_rule(line, active)
        if selected is None:
            continue
        rule, match = selected
        for level in tuple(active):
            if level >= rule.level:
                del active[level]
        display_ordinal = match.group("ordinal")
        ordinal = display_ordinal.casefold()
        label = f"{rule.label} {display_ordinal}"
        ancestors = tuple(active[level].label for level in sorted(active))
        hierarchy_path = (*ancestors, label)
        raw_title = match.groupdict().get("body" if rule.implicit else "title")
        title = raw_title.strip() if raw_title and raw_title.strip() else None
        node = HierarchyNode(
            rule_id=rule.rule_id,
            kind=rule.kind,
            ordinal=ordinal,
            title=title,
            label=label,
            hierarchy_path=hierarchy_path,
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            line_text=line,
            heading_only=not rule.implicit and title is None,
            level=rule.level,
        )
        parent_key = (rule.kind, ancestors)
        if ordinal.isdigit():
            numeric = int(ordinal)
            previous = previous_numeric.get(parent_key)
            if previous is not None and numeric != previous + 1:
                warnings.append(
                    HierarchyWarning(
                        code="HIER_NUMBERING_DISCONTINUOUS",
                        canonical_start=canonical_start,
                        message=f"{rule.kind} numbering changed from {previous} to {numeric}",
                    )
                )
            previous_numeric[parent_key] = numeric
        nodes.append(node)
        active[rule.level] = node
    return HierarchyParseResult(nodes=tuple(nodes), warnings=tuple(warnings))
