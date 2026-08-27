"""Stable exact/sparse candidate union without score mixing."""

from __future__ import annotations

import math

from legal_rag.retrieval.models import RetrievalCandidate


class RetrievalFusionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _merge_candidate(
    existing: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    if existing.chunk != incoming.chunk:
        raise RetrievalFusionError(
            "RETRIEVAL_CHUNK_CONFLICT",
            "one chunk ID resolves to conflicting chunk records",
        )
    sparse_scores = {
        score for score in (existing.sparse_score, incoming.sparse_score) if score is not None
    }
    if len(sparse_scores) > 1:
        raise RetrievalFusionError(
            "RETRIEVAL_SCORE_CONFLICT",
            "one chunk has conflicting sparse component scores",
        )
    sparse_score = next(iter(sparse_scores), None)
    return RetrievalCandidate(
        chunk=existing.chunk,
        exact_reference_match=(existing.exact_reference_match or incoming.exact_reference_match),
        sparse_score=sparse_score,
        dense_score=existing.dense_score
        if existing.dense_score is not None
        else incoming.dense_score,
        reranker_score=(
            existing.reranker_score
            if existing.reranker_score is not None
            else incoming.reranker_score
        ),
    )


def union_rank_candidates(
    *,
    exact: tuple[RetrievalCandidate, ...],
    sparse: tuple[RetrievalCandidate, ...],
    candidate_limit: int = 12,
) -> tuple[RetrievalCandidate, ...]:
    if candidate_limit < 1 or candidate_limit > 200:
        raise RetrievalFusionError(
            "RETRIEVAL_CANDIDATE_LIMIT_INVALID",
            "candidate limit must be within [1, 200]",
        )
    by_chunk_id: dict[str, RetrievalCandidate] = {}
    for candidate in (*exact, *sparse):
        if candidate.sparse_score is not None and not math.isfinite(candidate.sparse_score):
            raise RetrievalFusionError(
                "RETRIEVAL_SCORE_NONFINITE",
                "candidate sparse score must be finite or null",
            )
        existing = by_chunk_id.get(candidate.chunk.chunk_id)
        by_chunk_id[candidate.chunk.chunk_id] = (
            candidate if existing is None else _merge_candidate(existing, candidate)
        )
    ranked = sorted(
        by_chunk_id.values(),
        key=lambda candidate: (
            -int(candidate.exact_reference_match),
            candidate.sparse_score is None,
            -(candidate.sparse_score or 0.0),
            candidate.chunk.chunk_id,
        ),
    )
    return tuple(ranked[:candidate_limit])
