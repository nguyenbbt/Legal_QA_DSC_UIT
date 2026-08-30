"""Deterministic bounded Top-K selection for blockwise dense search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from legal_rag.retrieval.dense import DenseRetrievalError
from legal_rag.retrieval.dense_store import DenseHit


def _ranked(hits: Sequence[DenseHit], limit: int) -> tuple[DenseHit, ...]:
    by_id: dict[str, float] = {}
    for hit in hits:
        if not hit.chunk_id or not np.isfinite(hit.score):
            raise DenseRetrievalError("DENSE_SCORE_NONFINITE", "dense hit is invalid")
        by_id[hit.chunk_id] = max(hit.score, by_id.get(hit.chunk_id, -float("inf")))
    ordered = sorted(by_id.items(), key=lambda item: (-item[1], item[0].encode("utf-8")))
    return tuple(DenseHit(chunk_id, score) for chunk_id, score in ordered[:limit])


def select_top_k_rows(
    scores: np.ndarray,
    chunk_ids: Sequence[str],
    *,
    limit: int,
) -> tuple[tuple[DenseHit, ...], ...]:
    """Select each score row with an exact UTF-8 ID tie-break at the boundary."""

    values = np.asarray(scores)
    ids = tuple(chunk_ids)
    if (
        values.ndim != 2
        or values.shape[1] != len(ids)
        or not ids
        or len(ids) != len(set(ids))
        or limit < 1
        or not np.isfinite(values).all()
    ):
        raise DenseRetrievalError("DENSE_SCORE_BLOCK_INVALID", "dense score block is invalid")
    selected: list[tuple[DenseHit, ...]] = []
    take = min(limit, len(ids))
    for row in values:
        if take == len(ids):
            indices = np.arange(len(ids))
        else:
            threshold = np.partition(row, len(ids) - take)[len(ids) - take]
            above = np.flatnonzero(row > threshold).tolist()
            ties = sorted(
                np.flatnonzero(row == threshold).tolist(),
                key=lambda index: ids[index].encode("utf-8"),
            )
            indices = np.asarray(above + ties[: take - len(above)], dtype=np.int64)
        selected.append(
            _ranked(
                tuple(DenseHit(ids[int(index)], float(row[int(index)])) for index in indices),
                take,
            )
        )
    return tuple(selected)


def merge_top_k_blocks(
    current: Sequence[Sequence[DenseHit]],
    incoming: Sequence[Sequence[DenseHit]],
    *,
    limit: int,
) -> tuple[tuple[DenseHit, ...], ...]:
    """Merge corresponding block candidates without retaining the full score matrix."""

    if limit < 1 or len(current) != len(incoming):
        raise DenseRetrievalError("DENSE_SCORE_BLOCK_INVALID", "dense block rows differ")
    return tuple(
        _ranked((*left, *right), limit) for left, right in zip(current, incoming, strict=True)
    )


def compare_dense_hit_replay(
    expected: Sequence[Sequence[DenseHit]],
    replay: Sequence[Sequence[DenseHit]],
    *,
    score_tolerance: float,
) -> float:
    """Prove byte-stable identities/order and bounded floating-point score parity."""

    if score_tolerance < 0.0 or len(expected) != len(replay):
        raise DenseRetrievalError("DENSE_REPLAY_DRIFT", "dense replay rows differ")
    maximum_delta = 0.0
    for expected_row, replay_row in zip(expected, replay, strict=True):
        if tuple(hit.chunk_id for hit in expected_row) != tuple(hit.chunk_id for hit in replay_row):
            raise DenseRetrievalError("DENSE_REPLAY_DRIFT", "dense replay ranking drifted")
        for expected_hit, replay_hit in zip(expected_row, replay_row, strict=True):
            delta = abs(expected_hit.score - replay_hit.score)
            if not np.isfinite(delta) or delta > score_tolerance:
                raise DenseRetrievalError("DENSE_REPLAY_DRIFT", "dense replay score drifted")
            maximum_delta = max(maximum_delta, delta)
    return maximum_delta


__all__ = ["compare_dense_hit_replay", "merge_top_k_blocks", "select_top_k_rows"]
