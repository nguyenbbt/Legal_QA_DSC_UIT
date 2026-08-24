"""Bounded provider-neutral reranking that cannot expand retrieval scope."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from legal_rag.retrieval.models import RetrievalCandidate


class RerankerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RerankerBackend(Protocol):
    model_id: str
    model_revision: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    backend: RerankerBackend,
    *,
    limit: int,
    maximum_candidate_count: int = 50,
) -> tuple[RetrievalCandidate, ...]:
    """Reorder only admitted candidates, preserving identity and stable ties."""

    admitted = tuple(candidates)
    if not query.strip():
        raise RerankerError("RERANK_QUERY_EMPTY", "reranker query must be non-empty")
    if limit < 1 or limit > maximum_candidate_count:
        raise RerankerError("RERANK_LIMIT_INVALID", "reranker limit is outside its bound")
    if len(admitted) > maximum_candidate_count:
        raise RerankerError(
            "RERANK_CANDIDATE_LIMIT", "reranker candidate count exceeds its declared bound"
        )
    ids = tuple(candidate.chunk.chunk_id for candidate in admitted)
    if len(ids) != len(set(ids)):
        raise RerankerError("RERANK_CANDIDATE_DUPLICATE", "reranker candidates must be unique")
    scores = tuple(
        float(score)
        for score in backend.score(
            query, tuple(candidate.chunk.retrieval_text for candidate in admitted)
        )
    )
    if len(scores) != len(admitted):
        raise RerankerError(
            "RERANK_OUTPUT_CARDINALITY", "reranker returned the wrong number of scores"
        )
    if any(not math.isfinite(score) for score in scores):
        raise RerankerError("RERANK_SCORE_NONFINITE", "reranker scores must be finite")
    rescored = tuple(
        RetrievalCandidate(
            chunk=candidate.chunk,
            exact_reference_match=candidate.exact_reference_match,
            sparse_score=candidate.sparse_score,
            dense_score=candidate.dense_score,
            reranker_score=score,
        )
        for candidate, score in zip(admitted, scores, strict=True)
    )
    return tuple(
        sorted(
            rescored,
            key=lambda candidate: (
                -(candidate.reranker_score or 0.0),
                candidate.chunk.chunk_id.encode("utf-8"),
            ),
        )[:limit]
    )


__all__ = ["RerankerBackend", "RerankerError", "rerank_candidates"]
