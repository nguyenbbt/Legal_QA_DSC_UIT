from __future__ import annotations

import numpy as np
import pytest

from legal_rag.retrieval.dense import DenseRetrievalError
from legal_rag.retrieval.dense_batch import (
    compare_dense_hit_replay,
    merge_top_k_blocks,
    select_top_k_rows,
)
from legal_rag.retrieval.dense_store import DenseHit


def test_select_top_k_rows_breaks_score_ties_by_utf8_chunk_id() -> None:
    scores = np.asarray([[0.2, 0.9, 0.9, 0.1], [0.3, 0.3, 0.2, 0.4]], dtype=np.float32)
    ids = ("z", "b", "a", "x")

    selected = select_top_k_rows(scores, ids, limit=2)

    assert tuple(hit.chunk_id for hit in selected[0]) == ("a", "b")
    assert tuple(hit.chunk_id for hit in selected[1]) == ("x", "b")


def test_merge_top_k_blocks_preserves_exact_global_order() -> None:
    current = (
        (DenseHit("a", 0.8), DenseHit("b", 0.6)),
        (DenseHit("x", 0.5),),
    )
    block = (
        (DenseHit("c", 0.9), DenseHit("b", 0.7)),
        (DenseHit("a", 0.5), DenseHit("z", 0.4)),
    )

    merged = merge_top_k_blocks(current, block, limit=2)

    assert merged == (
        (DenseHit("c", 0.9), DenseHit("a", 0.8)),
        (DenseHit("a", 0.5), DenseHit("x", 0.5)),
    )


def test_dense_replay_requires_identical_ids_and_bounded_score_delta() -> None:
    expected = ((DenseHit("a", 0.8), DenseHit("b", 0.7)),)
    replay = ((DenseHit("a", 0.8000001), DenseHit("b", 0.6999999)),)

    maximum_delta = compare_dense_hit_replay(expected, replay, score_tolerance=1e-6)

    assert maximum_delta == pytest.approx(1e-7)
    with pytest.raises(DenseRetrievalError) as captured:
        compare_dense_hit_replay(
            expected,
            ((DenseHit("b", 0.8), DenseHit("a", 0.7)),),
            score_tolerance=1e-6,
        )
    assert captured.value.code == "DENSE_REPLAY_DRIFT"
