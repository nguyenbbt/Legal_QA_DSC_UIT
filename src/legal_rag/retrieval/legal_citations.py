"""Deterministic Vietnamese legal-citation parsing for retrieval supervision v2."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

CoordinateKind = Literal["part", "chapter", "section", "subsection"]

_SPACE = r"[^\S\r\n]"
_FLAGS = re.IGNORECASE | re.UNICODE | re.VERBOSE
_DOCUMENT_NUMBER = re.compile(
    rf"""
    (?<!\w)(?:số {_SPACE}*)?
    (?P<number>
      [0-9]+/
      (?:[0-9]{{4}}/)?
      [0-9a-zđ]+(?:-[0-9a-zđ]+)*
    )
    (?!\w)
    """,
    _FLAGS,
)
_FULL_COORDINATE = re.compile(
    rf"""
    (?<!\w)
    (?:điểm {_SPACE}+ (?P<point>[a-zđ]) {_SPACE}* (?:[,;:] {_SPACE}*)?)?
    (?:khoản {_SPACE}+ (?P<clause>[0-9]+[a-zđ]?) {_SPACE}* (?:[,;:] {_SPACE}*)?)?
    điều {_SPACE}+ (?P<article>[0-9]+[a-zđ]?)
    (?!\w)
    """,
    _FLAGS,
)
_CLAUSE = re.compile(r"(?<!\w)khoản\s+(?P<ordinal>[0-9]+[a-zđ]?)(?!\w)", re.I)
_OTHER_COORDINATE = re.compile(
    r"(?<!\w)(?P<label>tiểu\s+mục|phần|chương|mục)\s+"
    r"(?P<ordinal>[0-9]+(?:\.[0-9]+)*|[ivxlcdm]+)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_LAW_IDENTITY = re.compile(
    r"(?<!\w)(?P<kind>bộ\s+luật|luật)\s+"
    r"(?P<title>[^\n.;:()]{1,100}?)\s+(?P<year>(?:19|20)[0-9]{2})(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_BOUNDARY = re.compile(r"[.;:\n]")


@dataclass(frozen=True, slots=True)
class ParsedLegalCitation:
    """One normalized citation anchored to canonical NFC code-point offsets."""

    document_number: str | None
    law_identity: str | None
    article: str | None
    clause: str | None
    point: str | None
    other_coordinates: tuple[tuple[CoordinateKind, str], ...]
    canonical_start: int
    canonical_end: int

    def as_dict(self) -> dict[str, object]:
        return {
            "document_number": self.document_number,
            "law_identity": self.law_identity,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
            "other_coordinates": [
                {"kind": kind, "ordinal": ordinal} for kind, ordinal in self.other_coordinates
            ],
            "canonical_start": self.canonical_start,
            "canonical_end": self.canonical_end,
            "offset_space": "canonical_nfc",
        }


@dataclass(frozen=True, slots=True)
class _DocumentMention:
    document_number: str | None
    law_identity: str | None
    start: int
    end: int


def _fold_ascii(value: str) -> str:
    folded = unicodedata.normalize("NFD", unicodedata.normalize("NFC", value).casefold())
    unmarked = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return " ".join(unmarked.replace("đ", "d").replace("-", " ").split())


def normalize_legal_ordinal(kind: str, value: str) -> str | None:
    """Normalize only grammar-valid ordinals for a known hierarchy kind."""

    folded = unicodedata.normalize("NFC", value).casefold().strip()
    if kind in {"article", "clause"}:
        match = re.fullmatch(r"0*([0-9]+)([a-zđ]?)", folded)
        if match is None:
            return None
        return f"{int(match.group(1))}{match.group(2)}"
    if kind == "point":
        return folded if re.fullmatch(r"[a-zđ]", folded) else None
    if kind in {"part", "chapter"}:
        if re.fullmatch(r"[ivxlcdm]+", folded):
            return folded
        return str(int(folded)) if folded.isdigit() else None
    if kind in {"section", "subsection"}:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", folded):
            return None
        return ".".join(str(int(part)) for part in folded.split("."))
    return None


def _document_mentions(text: str) -> tuple[_DocumentMention, ...]:
    mentions = [
        _DocumentMention(match.group("number"), None, match.start(), match.end())
        for match in _DOCUMENT_NUMBER.finditer(text)
    ]
    for match in _LAW_IDENTITY.finditer(text):
        identity = _fold_ascii(
            f"{match.group('kind')} {match.group('title')} {match.group('year')}"
        )
        mentions.append(_DocumentMention(None, identity, match.start(), match.end()))
    return tuple(sorted(mentions, key=lambda item: (item.start, item.end)))


def _nearest_document(
    start: int, end: int, mentions: tuple[_DocumentMention, ...]
) -> _DocumentMention | None:
    following = [item for item in mentions if 0 <= item.start - end <= 240]
    if following:
        return min(following, key=lambda item: (item.start - end, item.start))
    preceding = [item for item in mentions if 0 <= start - item.end <= 240]
    if preceding:
        return min(preceding, key=lambda item: (start - item.end, -item.start))
    return None


def _nearby_other_coordinates(
    text: str, start: int, occupied: set[tuple[int, int]]
) -> tuple[tuple[CoordinateKind, str], ...]:
    boundary = max((match.end() for match in _BOUNDARY.finditer(text, 0, start)), default=0)
    result: list[tuple[CoordinateKind, str]] = []
    for match in _OTHER_COORDINATE.finditer(text, max(boundary, start - 80), start):
        item = _other_coordinate(match)
        if item is not None:
            result.append(item)
            occupied.add((match.start(), match.end()))
    return tuple(result)


def _other_coordinate(match: re.Match[str]) -> tuple[CoordinateKind, str] | None:
    label = " ".join(match.group("label").casefold().split())
    kind_by_label: dict[str, CoordinateKind] = {
        "phần": "part",
        "chương": "chapter",
        "mục": "section",
        "tiểu mục": "subsection",
    }
    kind = kind_by_label[label]
    ordinal = normalize_legal_ordinal(kind, match.group("ordinal"))
    return (kind, ordinal) if ordinal is not None else None


def _order_other_coordinates(
    values: tuple[tuple[CoordinateKind, str], ...],
) -> tuple[tuple[CoordinateKind, str], ...]:
    levels = {"part": 0, "chapter": 1, "section": 2, "subsection": 3}
    return tuple(sorted(dict.fromkeys(values), key=lambda item: (levels[item[0]], item[1])))


def _preceding_clauses(text: str, start: int) -> tuple[str, ...]:
    boundary = max((match.end() for match in _BOUNDARY.finditer(text, 0, start)), default=0)
    segment = text[max(boundary, start - 80) : start]
    values: list[str] = []
    for match in _CLAUSE.finditer(segment):
        value = normalize_legal_ordinal("clause", match.group("ordinal"))
        if value is not None:
            values.append(value)
    return tuple(dict.fromkeys(values))


def parse_legal_citations(text: str) -> tuple[ParsedLegalCitation, ...]:
    """Parse all supported citations without guessing unresolved document identity."""

    canonical = unicodedata.normalize("NFC", text)
    mentions = _document_mentions(canonical)
    citations: list[ParsedLegalCitation] = []
    occupied_other: set[tuple[int, int]] = set()
    for match in _FULL_COORDINATE.finditer(canonical):
        article = normalize_legal_ordinal("article", match.group("article"))
        clause = (
            normalize_legal_ordinal("clause", match.group("clause"))
            if match.group("clause")
            else None
        )
        point = (
            normalize_legal_ordinal("point", match.group("point")) if match.group("point") else None
        )
        if article is None or (point is not None and clause is None):
            continue
        document = _nearest_document(match.start(), match.end(), mentions)
        other = _order_other_coordinates(
            _nearby_other_coordinates(canonical, match.start(), occupied_other)
        )
        clauses: list[str | None] = list(_preceding_clauses(canonical, match.start()))
        if clause is not None:
            clauses.append(clause)
        clauses = list(dict.fromkeys(clauses)) or [None]
        for normalized_clause in clauses:
            citations.append(
                ParsedLegalCitation(
                    document_number=document.document_number if document else None,
                    law_identity=document.law_identity if document else None,
                    article=article,
                    clause=normalized_clause,
                    point=point if normalized_clause == clause else None,
                    other_coordinates=other,
                    canonical_start=match.start(),
                    canonical_end=match.end(),
                )
            )

    remaining = [
        match
        for match in _OTHER_COORDINATE.finditer(canonical)
        if (match.start(), match.end()) not in occupied_other
        and not any(
            full.start() <= match.start() < full.end()
            for full in _FULL_COORDINATE.finditer(canonical)
        )
    ]
    grouped: list[list[re.Match[str]]] = []
    for match in remaining:
        if grouped and match.start() - grouped[-1][-1].end() <= 40:
            grouped[-1].append(match)
        else:
            grouped.append([match])
    for group in grouped:
        coordinates = _order_other_coordinates(
            tuple(
                item for item in (_other_coordinate(match) for match in group) if item is not None
            )
        )
        if not coordinates:
            continue
        start, end = group[0].start(), group[-1].end()
        document = _nearest_document(start, end, mentions)
        citations.append(
            ParsedLegalCitation(
                document_number=document.document_number if document else None,
                law_identity=document.law_identity if document else None,
                article=None,
                clause=None,
                point=None,
                other_coordinates=coordinates,
                canonical_start=start,
                canonical_end=end,
            )
        )

    if not citations:
        for mention in mentions:
            citations.append(
                ParsedLegalCitation(
                    document_number=mention.document_number,
                    law_identity=mention.law_identity,
                    article=None,
                    clause=None,
                    point=None,
                    other_coordinates=(),
                    canonical_start=mention.start,
                    canonical_end=mention.end,
                )
            )
    unique = dict.fromkeys(citations)
    return tuple(
        sorted(
            unique,
            key=lambda item: (
                item.canonical_start,
                item.canonical_end,
                item.document_number or "",
                item.law_identity or "",
                item.article or "",
                item.clause or "",
                item.point or "",
            ),
        )
    )


def mask_legal_reference_numbers(text: str) -> str:
    """Mask citation/reference spans while retaining semantic numeric expressions."""

    canonical = unicodedata.normalize("NFC", text)
    spans = [(match.start(), match.end()) for match in _DOCUMENT_NUMBER.finditer(canonical)]
    spans.extend((match.start(), match.end()) for match in _LAW_IDENTITY.finditer(canonical))
    spans.extend((match.start(), match.end()) for match in _FULL_COORDINATE.finditer(canonical))
    spans.extend((match.start(), match.end()) for match in _OTHER_COORDINATE.finditer(canonical))
    characters = list(canonical)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


__all__ = [
    "ParsedLegalCitation",
    "mask_legal_reference_numbers",
    "normalize_legal_ordinal",
    "parse_legal_citations",
]
