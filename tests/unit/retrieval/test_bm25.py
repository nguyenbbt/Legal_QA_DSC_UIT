from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, ChunkRecord, chunk_context
from legal_rag.retrieval.bm25 import (
    APPROVED_BM25_RUNTIME_ID,
    SparseRetrievalError,
    build_bm25_index,
)


def chunk_for(context_id: int, text: str) -> ChunkRecord:
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


def build_index(chunks: tuple[ChunkRecord, ...]):
    return build_bm25_index(
        chunks,
        corpus_checksum=checksum_bytes(b"fixture-corpus"),
        alias_manifest_checksum=checksum_bytes(b"fixture-aliases"),
        runtime_compatibility_id=APPROVED_BM25_RUNTIME_ID,
    )


def test_bm25_matches_hand_computed_sequential_binary64_score() -> None:
    first = chunk_for(1, "luật luật mẫu")
    second = chunk_for(2, "luật khác")
    index = build_index((first, second))

    result = index.retrieve("luật")

    ratio = (2.0 - 2.0 + 0.5) / (2.0 + 0.5)
    idf = math.log1p(ratio)
    avgdl = float(5) / float(2)
    length_ratio = float(3) / avgdl
    length_norm = (1.0 - 0.75) + (0.75 * length_ratio)
    numerator = 2.0 * (1.2 + 1.0)
    denominator = 2.0 + (1.2 * length_norm)
    expected = idf * (numerator / denominator)
    by_id = {candidate.chunk.chunk_id: candidate.sparse_score for candidate in result.candidates}
    assert by_id[first.chunk_id] == expected


def test_bm25_uses_log1p_last_bit_vector() -> None:
    chunk = chunk_for(1, "luật")

    score = build_index((chunk,)).retrieve("luật").candidates[0].sparse_score

    assert score == 0.2876820724517809
    assert score != math.log(1.0 + (1.0 - 1.0 + 0.5) / (1.0 + 0.5))


def test_query_terms_are_unique_and_ordered_independently_of_discovery() -> None:
    chunks = (chunk_for(1, "điều luật"), chunk_for(2, "luật khác"))
    index = build_index(chunks)

    repeated = index.retrieve("luật điều luật")
    permuted = index.retrieve("điều luật")

    assert repeated.candidates == permuted.candidates
    assert repeated.query_terms == tuple(sorted({"luật", "điều"}, key=lambda value: value.encode()))


def test_empty_index_and_empty_query_return_typed_diagnostics() -> None:
    empty = build_index(())
    non_empty = build_index((chunk_for(1, "nội dung"),))

    assert empty.retrieve("luật").diagnostics[0].code == "SPARSE_INDEX_EMPTY"
    assert non_empty.retrieve("   ").diagnostics[0].code == "SPARSE_QUERY_EMPTY"


def test_index_rejects_zero_average_length_and_runtime_mismatch() -> None:
    zero_length = replace(chunk_for(1, "x"), retrieval_text="")

    with pytest.raises(SparseRetrievalError) as zero_error:
        build_index((zero_length,))
    assert zero_error.value.code == "SPARSE_INDEX_ZERO_AVGDL"

    with pytest.raises(SparseRetrievalError) as runtime_error:
        build_bm25_index(
            (chunk_for(1, "x"),),
            corpus_checksum=checksum_bytes(b"corpus"),
            alias_manifest_checksum=checksum_bytes(b"alias"),
            runtime_compatibility_id="unapproved",
        )
    assert runtime_error.value.code == "BM25_RUNTIME_UNAPPROVED"


def test_partial_zero_length_document_remains_in_document_count() -> None:
    zero_length = replace(chunk_for(1, "x"), retrieval_text="")
    matching = chunk_for(2, "luật")

    index = build_index((zero_length, matching))
    result = index.retrieve("luật")

    assert index.document_count == 2
    assert [candidate.chunk.chunk_id for candidate in result.candidates] == [matching.chunk_id]


def test_nonfinite_score_fails_query(monkeypatch: pytest.MonkeyPatch) -> None:
    index = build_index((chunk_for(1, "luật"),))
    monkeypatch.setattr("legal_rag.retrieval.bm25.math.log1p", lambda _ratio: math.inf)

    with pytest.raises(SparseRetrievalError) as captured:
        index.retrieve("luật")

    assert captured.value.code == "SPARSE_SCORE_NONFINITE"


def test_bm25_filters_nonpositive_scores_and_keeps_only_top_twelve() -> None:
    chunks = tuple(chunk_for(index, f"luật nội dung {index}") for index in range(1, 15))
    index = build_index(chunks)

    result = index.retrieve("luật")

    assert len(result.candidates) == 12
    assert all(
        candidate.sparse_score is not None and candidate.sparse_score > 0
        for candidate in result.candidates
    )
    assert index.retrieve("không-có").candidates == ()


def test_manifest_and_retrieval_json_are_byte_stable() -> None:
    index = build_index((chunk_for(1, "luật mẫu"), chunk_for(2, "điều luật")))

    first = index.retrieve("luật điều")
    second = index.retrieve("luật điều")

    assert index.manifest_bytes() == index.manifest_bytes()
    assert first.json_bytes() == second.json_bytes()
    assert repr(first.candidates[0].sparse_score).encode() in first.json_bytes()


def test_query_term_order_is_stable_across_three_hash_seeds() -> None:
    script = (
        "from legal_rag.retrieval.bm25 import ordered_unique_query_terms;"
        "print('|'.join(ordered_unique_query_terms('luật điều luật khoản')))"
    )
    outputs = []
    for seed in ("1", "17", "999"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONIOENCODING"] = "utf-8"
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                encoding="utf-8",
                text=True,
            )
        )

    assert outputs[0] == outputs[1] == outputs[2]
