from __future__ import annotations

import pytest

from legal_rag.retrieval.packing_calibration import (
    PackingCalibrationCandidate,
    PackingCalibrationError,
    PackingCalibrationGroup,
    calibrate_relative_sparse_score,
)


def _candidate(evidence_id: str, score: float) -> PackingCalibrationCandidate:
    return PackingCalibrationCandidate(evidence_id=evidence_id, sparse_score=score)


def test_calibration_maximizes_train_micro_f1_deterministically() -> None:
    groups = (
        PackingCalibrationGroup(
            question_id="q1",
            split="train",
            relevant_evidence_ids=("a",),
            candidates=(_candidate("a", 10.0), _candidate("noise", 8.0)),
        ),
        PackingCalibrationGroup(
            question_id="q2",
            split="train",
            relevant_evidence_ids=("b",),
            candidates=(_candidate("wrong", 10.0), _candidate("b", 9.0)),
        ),
    )

    first = calibrate_relative_sparse_score(groups)
    second = calibrate_relative_sparse_score(tuple(reversed(groups)))

    assert first == second
    assert first.minimum_relative_sparse_score == 0.9
    assert first.true_positive == 2
    assert first.false_positive == 1
    assert first.false_negative == 0
    assert first.micro_f1 == pytest.approx(0.8)


def test_calibration_breaks_exact_f1_tie_toward_compact_higher_threshold() -> None:
    groups = (
        PackingCalibrationGroup(
            question_id="q1",
            split="train",
            relevant_evidence_ids=("a", "b"),
            candidates=(
                _candidate("a", 10.0),
                _candidate("b", 8.0),
                _candidate("noise-1", 8.0),
            ),
        ),
        PackingCalibrationGroup(
            question_id="q2",
            split="train",
            relevant_evidence_ids=("c", "d"),
            candidates=(
                _candidate("c", 10.0),
                _candidate("d", 8.0),
                _candidate("noise-2", 8.0),
            ),
        ),
        PackingCalibrationGroup(
            question_id="q3",
            split="train",
            relevant_evidence_ids=("e",),
            candidates=(_candidate("e", 10.0), _candidate("noise-3", 8.0)),
        ),
        PackingCalibrationGroup(
            question_id="q4",
            split="train",
            relevant_evidence_ids=("f",),
            candidates=(_candidate("f", 10.0),),
        ),
    )

    result = calibrate_relative_sparse_score(groups)

    assert result.minimum_relative_sparse_score == 1.0
    assert result.micro_f1 == pytest.approx(0.8)


def test_calibration_rejects_non_train_and_invalid_rankings() -> None:
    with pytest.raises(PackingCalibrationError, match="official train") as split_error:
        calibrate_relative_sparse_score(
            (
                PackingCalibrationGroup(
                    question_id="q",
                    split="development",
                    relevant_evidence_ids=("a",),
                    candidates=(_candidate("a", 1.0),),
                ),
            )
        )
    assert split_error.value.code == "PACKING_CALIBRATION_SPLIT_INVALID"

    with pytest.raises(PackingCalibrationError) as score_error:
        calibrate_relative_sparse_score(
            (
                PackingCalibrationGroup(
                    question_id="q",
                    split="train",
                    relevant_evidence_ids=("a",),
                    candidates=(_candidate("a", 1.0), _candidate("b", 2.0)),
                ),
            )
        )
    assert score_error.value.code == "PACKING_CALIBRATION_RANKING_INVALID"
