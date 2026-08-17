from __future__ import annotations

from dataclasses import replace

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, chunk_context
from legal_rag.retrieval.fusion import RetrievalFusionError, union_rank_candidates
from legal_rag.retrieval.models import RetrievalCandidate


def chunk_for(context_id: int, text: str):
    context = ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": str(context_id),
            "original_id": str(context_id),
            "original_id_kind": "json_integer",
            "source_position": context_id,
            "source_artifact": f"fixtures/context_{context_id}.json",
            "source_checksum": checksum_bytes(text.encode()),
            "name": None,
            "source_url": f"https://example.invalid/{context_id}",
            "passage": text,
            "indexable": True,
            "quarantine_reason": None,
        }
    )
    return chunk_context(
        context,
        config=ChunkingConfig(minimum_fragment_tokens=1),
    ).chunks[0]


def candidate(
    context_id: int,
    *,
    exact: bool,
    sparse: float | None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=chunk_for(context_id, f"nội dung {context_id}"),
        exact_reference_match=exact,
        sparse_score=sparse,
    )


def test_union_ranks_exact_tier_then_sparse_score_then_chunk_id() -> None:
    exact_only = candidate(1, exact=True, sparse=None)
    sparse_high = candidate(2, exact=False, sparse=3.0)
    sparse_equal = candidate(3, exact=False, sparse=3.0)
    sparse_null = candidate(4, exact=False, sparse=None)

    ranked = union_rank_candidates(
        exact=(exact_only,),
        sparse=(sparse_null, sparse_equal, sparse_high),
    )

    assert ranked[0] == exact_only
    assert [item.sparse_score for item in ranked[1:]] == [3.0, 3.0, None]
    equal_ids = [sparse_high.chunk.chunk_id, sparse_equal.chunk.chunk_id]
    assert [item.chunk.chunk_id for item in ranked[1:3]] == sorted(equal_ids)


def test_union_deduplicates_overlap_and_preserves_component_values() -> None:
    chunk = chunk_for(1, "Điều 1. Nội dung")
    exact = RetrievalCandidate(chunk=chunk, exact_reference_match=True, sparse_score=None)
    sparse = RetrievalCandidate(chunk=chunk, exact_reference_match=False, sparse_score=2.5)

    ranked = union_rank_candidates(exact=(exact,), sparse=(sparse,))

    assert ranked == (
        RetrievalCandidate(chunk=chunk, exact_reference_match=True, sparse_score=2.5),
    )


def test_union_rejects_conflicting_chunks_with_same_id() -> None:
    original = chunk_for(1, "Điều 1. Nội dung")
    conflicting = replace(original, display_text="khác")

    with pytest.raises(RetrievalFusionError) as captured:
        union_rank_candidates(
            exact=(RetrievalCandidate(original, True, None),),
            sparse=(RetrievalCandidate(conflicting, False, 1.0),),
        )

    assert captured.value.code == "RETRIEVAL_CHUNK_CONFLICT"


def test_union_keeps_only_first_twelve_candidates() -> None:
    sparse = tuple(candidate(index, exact=False, sparse=float(index)) for index in range(1, 15))

    ranked = union_rank_candidates(exact=(), sparse=sparse)

    assert len(ranked) == 12
    assert [item.sparse_score for item in ranked] == list(
        reversed([float(index) for index in range(3, 15)])
    )
