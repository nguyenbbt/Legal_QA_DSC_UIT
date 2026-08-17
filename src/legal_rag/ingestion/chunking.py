"""Deterministic chunking.v1 over canonical internal context records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes, content_json_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.hierarchy import HierarchyNode, HierarchyWarning, parse_hierarchy
from legal_rag.retrieval.tokenizer import (
    RETRIEVAL_TOKENIZER_ID,
    RETRIEVAL_TOKENIZER_REVISION,
    retrieval_tokens,
)

ChunkHierarchyKind = Literal[
    "root", "part", "chapter", "section", "subsection", "article", "clause", "point"
]


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunking_version: str = "chunking.v1"
    hierarchy_max_tokens: int = 768
    window_tokens: int = 512
    overlap_tokens: int = 64
    minimum_fragment_tokens: int = 32

    def __post_init__(self) -> None:
        if self.hierarchy_max_tokens < 1 or self.window_tokens < 1:
            raise ValueError("chunk token limits must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.window_tokens:
            raise ValueError("chunk overlap must be non-negative and smaller than its window")
        if self.minimum_fragment_tokens < 1:
            raise ValueError("minimum fragment token count must be positive")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    context_id: str
    source_url: str
    hierarchy_path: tuple[str, ...]
    hierarchy_rule_id: str
    hierarchy_kind: ChunkHierarchyKind
    hierarchy_ordinal: str | None
    canonical_start: int
    canonical_end: int
    display_text: str
    retrieval_text: str
    window_index: int
    chunk_checksum: str
    context_checksum: str


@dataclass(frozen=True, slots=True)
class _Unit:
    canonical_start: int
    canonical_end: int
    hierarchy_path: tuple[str, ...]
    hierarchy_rule_id: str
    hierarchy_kind: ChunkHierarchyKind
    hierarchy_ordinal: str | None


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    chunks: tuple[ChunkRecord, ...]
    warnings: tuple[HierarchyWarning, ...]
    context_checksum: str
    config: ChunkingConfig

    def manifest_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "chunk.manifest.v1",
                "chunking_version": self.config.chunking_version,
                "tokenizer_id": RETRIEVAL_TOKENIZER_ID,
                "tokenizer_revision": RETRIEVAL_TOKENIZER_REVISION,
                "context_checksum": self.context_checksum,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "context_id": chunk.context_id,
                        "hierarchy_path": list(chunk.hierarchy_path),
                        "hierarchy_rule_id": chunk.hierarchy_rule_id,
                        "hierarchy_kind": chunk.hierarchy_kind,
                        "hierarchy_ordinal": chunk.hierarchy_ordinal,
                        "canonical_start": chunk.canonical_start,
                        "canonical_end": chunk.canonical_end,
                        "window_index": chunk.window_index,
                        "chunk_checksum": chunk.chunk_checksum,
                    }
                    for chunk in self.chunks
                ],
                "warnings": [
                    {
                        "code": warning.code,
                        "canonical_start": warning.canonical_start,
                        "message": warning.message,
                    }
                    for warning in self.warnings
                ],
            }
        )


def _context_checksum(context: ContextRecord) -> str:
    return checksum_bytes(content_json_bytes(context.model_dump(mode="json")))


def _next_boundary(nodes: tuple[HierarchyNode, ...], index: int, passage_end: int) -> int:
    node = nodes[index]
    return next(
        (
            following.canonical_start
            for following in nodes[index + 1 :]
            if following.level <= node.level
        ),
        passage_end,
    )


def _leaf_units(passage: str, nodes: tuple[HierarchyNode, ...]) -> list[_Unit]:
    candidates = {"article", "clause", "point"}
    units: list[_Unit] = []
    for index, node in enumerate(nodes):
        if node.kind not in candidates:
            continue
        boundary = _next_boundary(nodes, index, len(passage))
        has_finer_child = any(
            following.canonical_start < boundary
            and following.level > node.level
            and following.kind in candidates
            for following in nodes[index + 1 :]
        )
        if not has_finer_child:
            units.append(
                _Unit(
                    canonical_start=node.canonical_start,
                    canonical_end=boundary,
                    hierarchy_path=node.hierarchy_path,
                    hierarchy_rule_id=node.rule_id,
                    hierarchy_kind=node.kind,
                    hierarchy_ordinal=node.ordinal,
                )
            )
    if units and passage[: units[0].canonical_start].strip():
        first = units[0]
        units[0] = _Unit(
            canonical_start=0,
            canonical_end=first.canonical_end,
            hierarchy_path=first.hierarchy_path,
            hierarchy_rule_id=first.hierarchy_rule_id,
            hierarchy_kind=first.hierarchy_kind,
            hierarchy_ordinal=first.hierarchy_ordinal,
        )
    return units


def _fallback_unit(passage: str, nodes: tuple[HierarchyNode, ...]) -> _Unit:
    deepest = nodes[-1] if nodes else None
    return _Unit(
        canonical_start=0,
        canonical_end=len(passage),
        hierarchy_path=deepest.hierarchy_path if deepest else ("Văn bản",),
        hierarchy_rule_id=deepest.rule_id if deepest else "HIERARCHY_FALLBACK",
        hierarchy_kind=deepest.kind if deepest else "root",
        hierarchy_ordinal=deepest.ordinal if deepest else None,
    )


def _compatible(left: _Unit, right: _Unit, passage: str) -> bool:
    return (
        left.hierarchy_path[:-1] == right.hierarchy_path[:-1]
        and left.canonical_end <= right.canonical_start
        and not passage[left.canonical_end : right.canonical_start].strip()
    )


def _merge_short_units(
    units: list[_Unit],
    passage: str,
    minimum_tokens: int,
) -> list[_Unit]:
    merged = list(units)
    index = 0
    while index < len(merged):
        unit = merged[index]
        token_count = len(retrieval_tokens(passage[unit.canonical_start : unit.canonical_end]))
        if token_count >= minimum_tokens:
            index += 1
            continue
        if index > 0 and _compatible(merged[index - 1], unit, passage):
            previous = merged[index - 1]
            merged[index - 1] = _Unit(
                canonical_start=previous.canonical_start,
                canonical_end=unit.canonical_end,
                hierarchy_path=previous.hierarchy_path,
                hierarchy_rule_id=previous.hierarchy_rule_id,
                hierarchy_kind=previous.hierarchy_kind,
                hierarchy_ordinal=previous.hierarchy_ordinal,
            )
            del merged[index]
            continue
        if index + 1 < len(merged) and _compatible(unit, merged[index + 1], passage):
            following = merged[index + 1]
            merged[index + 1] = _Unit(
                canonical_start=unit.canonical_start,
                canonical_end=following.canonical_end,
                hierarchy_path=following.hierarchy_path,
                hierarchy_rule_id=following.hierarchy_rule_id,
                hierarchy_kind=following.hierarchy_kind,
                hierarchy_ordinal=following.hierarchy_ordinal,
            )
            del merged[index]
            continue
        index += 1
    return merged


def _chunk_id(
    *,
    config: ChunkingConfig,
    context_id: str,
    unit: _Unit,
    canonical_start: int,
    canonical_end: int,
    window_index: int,
    display_checksum: str,
) -> str:
    payload = canonical_json_bytes(
        {
            "chunking_version": config.chunking_version,
            "context_id": context_id,
            "hierarchy_path": list(unit.hierarchy_path),
            "canonical_start": canonical_start,
            "canonical_end": canonical_end,
            "window_index": window_index,
            "display_text_checksum": display_checksum,
        }
    )
    return f"chunk_{hashlib.sha256(payload).hexdigest()[:24]}"


def _build_chunk(
    context: ContextRecord,
    context_checksum: str,
    unit: _Unit,
    config: ChunkingConfig,
    *,
    canonical_start: int,
    canonical_end: int,
    window_index: int,
) -> ChunkRecord:
    display_text = context.passage[canonical_start:canonical_end]
    retrieval_text = " ".join(token.value for token in retrieval_tokens(display_text))
    display_checksum = checksum_bytes(display_text.encode("utf-8"))
    chunk_id = _chunk_id(
        config=config,
        context_id=context.context_id,
        unit=unit,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        window_index=window_index,
        display_checksum=display_checksum,
    )
    chunk_payload = {
        "chunk_id": chunk_id,
        "context_id": context.context_id,
        "hierarchy_path": list(unit.hierarchy_path),
        "hierarchy_rule_id": unit.hierarchy_rule_id,
        "hierarchy_kind": unit.hierarchy_kind,
        "hierarchy_ordinal": unit.hierarchy_ordinal,
        "canonical_start": canonical_start,
        "canonical_end": canonical_end,
        "display_text_checksum": display_checksum,
        "retrieval_text": retrieval_text,
        "window_index": window_index,
        "context_checksum": context_checksum,
    }
    return ChunkRecord(
        chunk_id=chunk_id,
        context_id=context.context_id,
        source_url=context.source_url,
        hierarchy_path=unit.hierarchy_path,
        hierarchy_rule_id=unit.hierarchy_rule_id,
        hierarchy_kind=unit.hierarchy_kind,
        hierarchy_ordinal=unit.hierarchy_ordinal,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        display_text=display_text,
        retrieval_text=retrieval_text,
        window_index=window_index,
        chunk_checksum=checksum_bytes(content_json_bytes(chunk_payload)),
        context_checksum=context_checksum,
    )


def _window_unit(
    context: ContextRecord,
    context_checksum: str,
    unit: _Unit,
    config: ChunkingConfig,
    *,
    hierarchy_free: bool,
) -> list[ChunkRecord]:
    unit_text = context.passage[unit.canonical_start : unit.canonical_end]
    tokens = retrieval_tokens(unit_text)
    threshold = config.window_tokens if hierarchy_free else config.hierarchy_max_tokens
    if len(tokens) <= threshold:
        return [
            _build_chunk(
                context,
                context_checksum,
                unit,
                config,
                canonical_start=unit.canonical_start,
                canonical_end=unit.canonical_end,
                window_index=0,
            )
        ]
    chunks: list[ChunkRecord] = []
    step = config.window_tokens - config.overlap_tokens
    for window_index, token_start in enumerate(range(0, len(tokens), step)):
        window = tokens[token_start : token_start + config.window_tokens]
        if not window:
            break
        chunks.append(
            _build_chunk(
                context,
                context_checksum,
                unit,
                config,
                canonical_start=unit.canonical_start + window[0].canonical_start,
                canonical_end=unit.canonical_start + window[-1].canonical_end,
                window_index=window_index,
            )
        )
        if token_start + config.window_tokens >= len(tokens):
            break
    return chunks


def chunk_context(
    context: ContextRecord,
    *,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Create stable retrieval chunks without mutating or re-normalizing the context."""

    active_config = config or ChunkingConfig()
    context_checksum = _context_checksum(context)
    if not context.indexable:
        return ChunkingResult((), (), context_checksum, active_config)
    hierarchy = parse_hierarchy(context.passage)
    units = _leaf_units(context.passage, hierarchy.nodes)
    hierarchy_free = not units
    if hierarchy_free:
        units = [_fallback_unit(context.passage, hierarchy.nodes)]
    units = _merge_short_units(units, context.passage, active_config.minimum_fragment_tokens)
    chunks = [
        chunk
        for unit in units
        for chunk in _window_unit(
            context,
            context_checksum,
            unit,
            active_config,
            hierarchy_free=hierarchy_free,
        )
    ]
    chunks.sort(key=lambda chunk: (chunk.canonical_start, chunk.window_index, chunk.chunk_id))
    return ChunkingResult(tuple(chunks), hierarchy.warnings, context_checksum, active_config)
