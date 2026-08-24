"""Model-free contracts for learned retrieval adapters."""

from __future__ import annotations

from dataclasses import replace

import pytest

from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.dense import DenseDocument, DenseIndex, DenseRetrievalError
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.retrieval.reranker import RerankerError, rerank_candidates


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        context_id="1",
        source_url="https://example.invalid/legal",
        hierarchy_path=("Điều 1",),
        hierarchy_rule_id="ARTICLE",
        hierarchy_kind="article",
        hierarchy_ordinal="1",
        canonical_start=0,
        canonical_end=len(text),
        display_text=text,
        retrieval_text=text,
        window_index=0,
        chunk_checksum="sha256:" + "1" * 64,
        context_checksum="sha256:" + "2" * 64,
    )


def test_dense_index_normalizes_and_breaks_equal_score_ties_by_id() -> None:
    index = DenseIndex(
        (
            DenseDocument(_chunk("chunk_b", "b"), (2.0, 0.0)),
            DenseDocument(_chunk("chunk_a", "a"), (1.0, 0.0)),
            DenseDocument(_chunk("chunk_c", "c"), (0.0, 1.0)),
        ),
        dimension=2,
    )

    ranked = index.retrieve((8.0, 0.0), limit=3)

    assert tuple(item.chunk.chunk_id for item in ranked) == ("chunk_a", "chunk_b", "chunk_c")
    assert ranked[0].dense_score == pytest.approx(1.0)


@pytest.mark.parametrize("vector", ((0.0, 0.0), (float("nan"), 1.0), (1.0,)))
def test_dense_index_rejects_invalid_query_vectors(vector: tuple[float, ...]) -> None:
    index = DenseIndex((DenseDocument(_chunk("chunk_a", "a"), (1.0, 0.0)),), dimension=2)

    with pytest.raises(DenseRetrievalError):
        index.retrieve(vector, limit=1)


class _ReverseReranker:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        del query
        return tuple(float(index) for index, _ in enumerate(documents))


def test_reranker_only_reorders_admitted_candidates_and_preserves_components() -> None:
    first = RetrievalCandidate(_chunk("chunk_a", "a"), True, 3.0, 0.7)
    second = RetrievalCandidate(_chunk("chunk_b", "b"), False, 2.0, 0.8)

    ranked = rerank_candidates("query", (first, second), _ReverseReranker(), limit=2)

    assert tuple(item.chunk.chunk_id for item in ranked) == ("chunk_b", "chunk_a")
    assert {item.chunk.chunk_id for item in ranked} == {first.chunk.chunk_id, second.chunk.chunk_id}
    assert ranked[1] == replace(first, reranker_score=0.0)


def test_reranker_fails_before_scoring_an_oversized_pool() -> None:
    candidate = RetrievalCandidate(_chunk("chunk_a", "a"), False, None)

    with pytest.raises(RerankerError) as error:
        rerank_candidates("query", (candidate,) * 51, _ReverseReranker(), limit=10)

    assert error.value.code == "RERANK_CANDIDATE_LIMIT"
