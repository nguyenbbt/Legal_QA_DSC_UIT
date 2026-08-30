from __future__ import annotations

import pytest

from legal_rag.retrieval.dense_sampling import stratified_positions


def test_stratified_positions_cover_both_length_sorted_extremes_deterministically() -> None:
    assert stratified_positions(total_count=10, sample_count=4) == (0, 3, 6, 9)
    assert stratified_positions(total_count=5, sample_count=5) == (0, 1, 2, 3, 4)


def test_stratified_positions_reject_invalid_sample_sizes() -> None:
    with pytest.raises(ValueError, match="sample"):
        stratified_positions(total_count=10, sample_count=1)
    with pytest.raises(ValueError, match="sample"):
        stratified_positions(total_count=10, sample_count=11)
