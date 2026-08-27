"""Hierarchy-aware legal sparse discovery over the immutable bm25.v1 store."""

from __future__ import annotations

from typing import Protocol

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic
from legal_rag.retrieval.tokenizer import retrieval_token_values

LEGAL_SPARSE_VERSION = "legal-sparse.bm25f-lite.v2"
BODY_WEIGHT = 0.75
HIERARCHY_WEIGHT = 0.25


class ExpandedSparseIndex(Protocol):
    index_checksum: str

    def retrieve(self, query: str, *, candidate_limit: int = 12) -> SparseRetrievalResult: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]: ...


class LegalSparseRetriever:
    """Reweight body discovery with a separately normalized hierarchy field."""

    def __init__(self, index: ExpandedSparseIndex, *, discovery_limit: int = 100) -> None:
        if discovery_limit < 100 or discovery_limit > 200:
            raise ValueError("legal sparse discovery limit must be within [100, 200]")
        self._index = index
        self.discovery_limit = discovery_limit
        self.index_checksum = checksum_bytes(
            content_json_bytes(
                {
                    "schema_version": "legal-sparse.config.v1",
                    "retrieval_version": LEGAL_SPARSE_VERSION,
                    "source_index_checksum": index.index_checksum,
                    "discovery_limit": discovery_limit,
                    "body_weight": BODY_WEIGHT,
                    "hierarchy_weight": HIERARCHY_WEIGHT,
                }
            )
        )

    def retrieve(self, query: str) -> SparseRetrievalResult:
        source = self._index.retrieve(query, candidate_limit=self.discovery_limit)
        if not source.candidates:
            return SparseRetrievalResult(
                source.query,
                source.query_terms,
                (),
                source.diagnostics,
                self.index_checksum,
            )
        maximum_body_score = max(candidate.sparse_score or 0.0 for candidate in source.candidates)
        query_terms = frozenset(source.query_terms)
        scored: list[RetrievalCandidate] = []
        for candidate in source.candidates:
            hierarchy_terms = frozenset(
                retrieval_token_values(" ".join(candidate.chunk.hierarchy_path).casefold())
            )
            body_score = (
                (candidate.sparse_score or 0.0) / maximum_body_score
                if maximum_body_score > 0.0
                else 0.0
            )
            hierarchy_score = (
                float(len(query_terms.intersection(hierarchy_terms))) / float(len(query_terms))
                if query_terms
                else 0.0
            )
            combined = BODY_WEIGHT * body_score + HIERARCHY_WEIGHT * hierarchy_score
            scored.append(
                RetrievalCandidate(
                    chunk=candidate.chunk,
                    exact_reference_match=candidate.exact_reference_match,
                    sparse_score=combined,
                )
            )
        ranked = tuple(
            sorted(
                scored,
                key=lambda candidate: (
                    -int(candidate.exact_reference_match),
                    -(candidate.sparse_score or 0.0),
                    candidate.chunk.chunk_id.encode("utf-8"),
                ),
            )[: self.discovery_limit]
        )
        return SparseRetrievalResult(
            query=source.query,
            query_terms=source.query_terms,
            candidates=ranked,
            diagnostics=(
                *source.diagnostics,
                RetrievalDiagnostic(
                    code="LEGAL_SPARSE_BM25F_LITE",
                    message="fixed body/hierarchy field normalization applied",
                    candidate_count=len(ranked),
                ),
            ),
            index_checksum=self.index_checksum,
        )

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return self._index.chunks_for_context(context_id)

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return self._index.chunks_for_coordinate(hierarchy_kind, hierarchy_ordinal)


__all__ = [
    "BODY_WEIGHT",
    "HIERARCHY_WEIGHT",
    "LEGAL_SPARSE_VERSION",
    "ExpandedSparseIndex",
    "LegalSparseRetriever",
]
