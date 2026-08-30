"""Bounded per-run cache for immutable sparse-index chunk lookups."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult


class ExpandedSparseSource(Protocol):
    index_checksum: str

    def retrieve(self, query: str, *, candidate_limit: int = 12) -> SparseRetrievalResult: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]: ...


class CachedExpandedSparseIndex:
    """Cache only immutable chunk resolution; query rankings are always recomputed."""

    def __init__(self, source: ExpandedSparseSource) -> None:
        self._source = source
        self.index_checksum = source.index_checksum
        self._contexts: OrderedDict[str, tuple[ChunkRecord, ...]] = OrderedDict()
        self._coordinates: OrderedDict[tuple[str, str | None], tuple[ChunkRecord, ...]] = (
            OrderedDict()
        )

    def retrieve(self, query: str, *, candidate_limit: int = 12) -> SparseRetrievalResult:
        return self._source.retrieve(query, candidate_limit=candidate_limit)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        cached = self._contexts.get(context_id)
        if cached is not None:
            self._contexts.move_to_end(context_id)
            return cached
        chunks = self._source.chunks_for_context(context_id)
        self._contexts[context_id] = chunks
        if len(self._contexts) > 256:
            self._contexts.popitem(last=False)
        return chunks

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        key = (hierarchy_kind, hierarchy_ordinal)
        cached = self._coordinates.get(key)
        if cached is not None:
            self._coordinates.move_to_end(key)
            return cached
        chunks = self._source.chunks_for_coordinate(hierarchy_kind, hierarchy_ordinal)
        self._coordinates[key] = chunks
        if len(self._coordinates) > 128:
            self._coordinates.popitem(last=False)
        return chunks


__all__ = ["CachedExpandedSparseIndex", "ExpandedSparseSource"]
