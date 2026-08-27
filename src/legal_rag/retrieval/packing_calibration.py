"""Train-only deterministic calibration for adaptive evidence packing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NoReturn


class PackingCalibrationError(Exception):
    """Stable fail-closed calibration error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise PackingCalibrationError(code, message)


@dataclass(frozen=True, slots=True)
class PackingCalibrationCandidate:
    evidence_id: str
    sparse_score: float


@dataclass(frozen=True, slots=True)
class PackingCalibrationGroup:
    question_id: str
    split: str
    relevant_evidence_ids: tuple[str, ...]
    candidates: tuple[PackingCalibrationCandidate, ...]


@dataclass(frozen=True, slots=True)
class PackingCalibrationResult:
    schema_version: Literal["evidence-packing-calibration.v1"]
    objective: Literal["micro_evidence_set_f1"]
    tie_break: Literal["higher_threshold"]
    minimum_relative_sparse_score: float
    group_count: int
    threshold_candidate_count: int
    selected_evidence_count: int
    relevant_evidence_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    micro_precision: float
    micro_recall: float
    micro_f1: float


def _validated_groups(
    groups: tuple[PackingCalibrationGroup, ...],
) -> tuple[PackingCalibrationGroup, ...]:
    if not groups:
        _fail("PACKING_CALIBRATION_EMPTY", "packing calibration groups are required")
    ordered = tuple(sorted(groups, key=lambda group: group.question_id.encode("utf-8")))
    if len({group.question_id for group in ordered}) != len(ordered):
        _fail("PACKING_CALIBRATION_ID_INVALID", "calibration question IDs must be unique")
    for group in ordered:
        if group.split != "train":
            _fail(
                "PACKING_CALIBRATION_SPLIT_INVALID",
                "packing calibration accepts official train rows only",
            )
        if (
            not group.question_id
            or not group.relevant_evidence_ids
            or len(set(group.relevant_evidence_ids)) != len(group.relevant_evidence_ids)
            or not group.candidates
        ):
            _fail(
                "PACKING_CALIBRATION_GROUP_INVALID",
                "calibration identity, relevance, and candidates must be non-empty and unique",
            )
        candidate_ids = tuple(candidate.evidence_id for candidate in group.candidates)
        scores = tuple(candidate.sparse_score for candidate in group.candidates)
        if (
            any(not evidence_id for evidence_id in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or any(not math.isfinite(score) or score <= 0.0 for score in scores)
            or any(left < right for left, right in zip(scores, scores[1:], strict=False))
        ):
            _fail(
                "PACKING_CALIBRATION_RANKING_INVALID",
                "calibration rankings require unique IDs and finite positive descending scores",
            )
    return ordered


def _score(
    groups: tuple[PackingCalibrationGroup, ...],
    threshold: float,
    *,
    maximum_evidence_count: int,
    threshold_candidate_count: int,
) -> PackingCalibrationResult:
    true_positive = 0
    false_positive = 0
    relevant_count = 0
    selected_count = 0
    for group in groups:
        primary_score = group.candidates[0].sparse_score
        selected = tuple(
            candidate.evidence_id
            for candidate in group.candidates
            if candidate.sparse_score / primary_score >= threshold
        )[:maximum_evidence_count]
        relevant = set(group.relevant_evidence_ids)
        matched = len(relevant.intersection(selected))
        true_positive += matched
        false_positive += len(selected) - matched
        relevant_count += len(relevant)
        selected_count += len(selected)
    false_negative = relevant_count - true_positive
    precision = true_positive / selected_count if selected_count else 0.0
    recall = true_positive / relevant_count
    denominator = (2 * true_positive) + false_positive + false_negative
    f1 = (2 * true_positive) / denominator if denominator else 0.0
    return PackingCalibrationResult(
        schema_version="evidence-packing-calibration.v1",
        objective="micro_evidence_set_f1",
        tie_break="higher_threshold",
        minimum_relative_sparse_score=threshold,
        group_count=len(groups),
        threshold_candidate_count=threshold_candidate_count,
        selected_evidence_count=selected_count,
        relevant_evidence_count=relevant_count,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
    )


def calibrate_relative_sparse_score(
    groups: tuple[PackingCalibrationGroup, ...],
    *,
    maximum_evidence_count: int = 3,
) -> PackingCalibrationResult:
    """Maximize train micro evidence-set F1; exact ties choose compact evidence."""

    if maximum_evidence_count < 1 or maximum_evidence_count > 3:
        _fail(
            "PACKING_CALIBRATION_COUNT_INVALID",
            "calibration evidence count must be within [1, 3]",
        )
    ordered = _validated_groups(groups)
    thresholds = {0.0, 1.0}
    for group in ordered:
        primary_score = group.candidates[0].sparse_score
        thresholds.update(candidate.sparse_score / primary_score for candidate in group.candidates)
    candidates = tuple(sorted(thresholds, reverse=True))
    scored = tuple(
        _score(
            ordered,
            threshold,
            maximum_evidence_count=maximum_evidence_count,
            threshold_candidate_count=len(candidates),
        )
        for threshold in candidates
    )
    return max(scored, key=lambda result: (result.micro_f1, result.minimum_relative_sparse_score))


__all__ = [
    "PackingCalibrationCandidate",
    "PackingCalibrationError",
    "PackingCalibrationGroup",
    "PackingCalibrationResult",
    "calibrate_relative_sparse_score",
]
