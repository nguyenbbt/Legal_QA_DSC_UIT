from __future__ import annotations

from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.retrieval.rank_fusion import reciprocal_rank_fusion


def _candidate(
    chunk_id: str, *, sparse: float | None = None, dense: float | None = None
) -> RetrievalCandidate:
    text = chunk_id
    chunk = ChunkRecord(
        chunk_id,
        "1",
        "https://example.invalid",
        ("Điều 1",),
        "ARTICLE",
        "article",
        "1",
        0,
        len(text),
        text,
        text,
        0,
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
    )
    return RetrievalCandidate(chunk, False, sparse, dense)


def test_rrf_fuses_identity_and_keeps_component_scores() -> None:
    sparse_a = _candidate("a", sparse=4.0)
    sparse_b = _candidate("b", sparse=3.0)
    dense_b = _candidate("b", dense=0.9)
    dense_c = _candidate("c", dense=0.8)

    ranked = reciprocal_rank_fusion(((sparse_a, sparse_b), (dense_b, dense_c)), limit=3)

    assert tuple(item.chunk.chunk_id for item in ranked) == ("b", "a", "c")
    assert ranked[0].sparse_score == 3.0
    assert ranked[0].dense_score == 0.9
