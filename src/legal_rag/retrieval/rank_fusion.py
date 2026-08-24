"""Deterministic reciprocal-rank fusion for exact/sparse/dense candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from legal_rag.retrieval.models import RetrievalCandidate


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalCandidate]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> tuple[RetrievalCandidate, ...]:
    """Fuse rankings by identity without manufacturing or score-mixing candidates."""

    if limit < 1 or rank_constant < 1:
        raise ValueError("fusion limit and rank constant must be positive")
    by_id: dict[str, RetrievalCandidate] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, candidate in enumerate(ranking, start=1):
            chunk_id = candidate.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError("one fusion ranking contains a duplicate chunk ID")
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / float(rank_constant + rank)
            existing = by_id.get(chunk_id)
            if existing is None:
                by_id[chunk_id] = candidate
                continue
            if existing.chunk != candidate.chunk:
                raise ValueError("one fusion chunk ID resolves to conflicting records")
            by_id[chunk_id] = replace(
                existing,
                exact_reference_match=(
                    existing.exact_reference_match or candidate.exact_reference_match
                ),
                sparse_score=(
                    existing.sparse_score
                    if existing.sparse_score is not None
                    else candidate.sparse_score
                ),
                dense_score=(
                    existing.dense_score
                    if existing.dense_score is not None
                    else candidate.dense_score
                ),
                reranker_score=(
                    existing.reranker_score
                    if existing.reranker_score is not None
                    else candidate.reranker_score
                ),
            )
    return tuple(
        by_id[chunk_id]
        for chunk_id in sorted(scores, key=lambda key: (-scores[key], key.encode("utf-8")))[:limit]
    )


__all__ = ["reciprocal_rank_fusion"]
