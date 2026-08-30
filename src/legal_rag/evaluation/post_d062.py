"""Deterministic reconciliation of the immutable D-062 completion evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NoReturn, cast

from legal_rag.domain.checksums import checksum_bytes


class PostD062Error(Exception):
    """Stable failure at the post-D-062 reconciliation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise PostD062Error(code, message)


def _object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PostD062Error("D063_ARTIFACT_INVALID", f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        _fail("D063_ARTIFACT_INVALID", f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _metric_rows(data: bytes, label: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    try:
        raw_lines = data.splitlines()
        for line in raw_lines:
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != "competition.per_query.v1"
                or not isinstance(value.get("question_id"), str)
                or type(value.get("meteor")) is not float
                or type(value.get("rouge_l")) is not float
            ):
                _fail("D063_ARTIFACT_INVALID", f"{label} metric row is invalid")
            question_id = cast(str, value["question_id"])
            if question_id in rows:
                _fail("D063_ARTIFACT_INVALID", f"{label} contains duplicate IDs")
            rows[question_id] = (cast(float, value["meteor"]), cast(float, value["rouge_l"]))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PostD062Error(
            "D063_ARTIFACT_INVALID", f"{label} metrics are not valid JSONL"
        ) from error
    if not rows:
        _fail("D063_ARTIFACT_INVALID", f"{label} metrics are empty")
    return rows


def _predictions(data: bytes, label: str) -> dict[str, str]:
    value = _object(data, label)
    predictions: dict[str, str] = {}
    for question_id, raw in value.items():
        if (
            not isinstance(question_id, str)
            or not isinstance(raw, dict)
            or set(raw) != {"answer"}
            or not isinstance(raw["answer"], str)
            or not raw["answer"].strip()
        ):
            _fail("D063_ARTIFACT_INVALID", f"{label} prediction row is invalid")
        predictions[question_id] = raw["answer"]
    return predictions


def _nearest_rank(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def _length_summary(answers: Mapping[str, str]) -> dict[str, int]:
    lengths = [len(answer.split()) for answer in answers.values()]
    return {
        "minimum": min(lengths),
        "p50": _nearest_rank(lengths, 50),
        "p90": _nearest_rank(lengths, 90),
        "p95": _nearest_rank(lengths, 95),
        "maximum": max(lengths),
    }


def _require_completed_gates(
    comparison: Mapping[str, Any], grounding: Mapping[str, Any], parameters: Mapping[str, Any]
) -> None:
    requirements = (
        comparison.get("baseline_prediction_bytes_replayed") is True,
        comparison.get("candidate_prediction_bytes_replayed") is True,
        comparison.get("numeric_evaluation_gate") == "passed",
        comparison.get("resource_gate") == "passed",
        grounding.get("grounding_gate") == "passed",
        grounding.get("promotion_blockers") == [],
        parameters.get("passes_parameter_gate") is True,
        type(parameters.get("system_parameter_count")) is int,
        type(parameters.get("competition_limit_exclusive")) is int,
        cast(int, parameters.get("system_parameter_count"))
        < cast(int, parameters.get("competition_limit_exclusive")),
    )
    if not all(requirements):
        _fail("D063_REQUIREMENT_UNSATISFIED", "a required D-062 completion gate is not passed")


def reconcile_d063(
    *,
    comparison_data: bytes,
    grounding_data: bytes,
    baseline_metrics_data: bytes,
    candidate_metrics_data: bytes,
    baseline_predictions_data: bytes,
    candidate_predictions_data: bytes,
    parameter_manifest_data: bytes,
    expected_question_count: int,
) -> dict[str, Any]:
    """Prove that D-063 is satisfied by existing D-062 evidence only."""

    comparison = _object(comparison_data, "comparison")
    grounding = _object(grounding_data, "grounding comparison")
    parameters = _object(parameter_manifest_data, "parameter manifest")
    _require_completed_gates(comparison, grounding, parameters)

    if comparison.get("question_count") != expected_question_count:
        _fail("D063_COUNT_MISMATCH", "comparison question count is not the frozen count")
    baseline_metrics = _metric_rows(baseline_metrics_data, "baseline")
    candidate_metrics = _metric_rows(candidate_metrics_data, "candidate")
    baseline_predictions = _predictions(baseline_predictions_data, "baseline")
    candidate_predictions = _predictions(candidate_predictions_data, "candidate")
    expected_ids = set(baseline_metrics)
    if (
        len(expected_ids) != expected_question_count
        or set(candidate_metrics) != expected_ids
        or set(baseline_predictions) != expected_ids
        or set(candidate_predictions) != expected_ids
    ):
        _fail("D063_ID_MISMATCH", "D-062 source artifacts do not contain identical IDs")

    outcomes = {"candidate_wins": 0, "ties": 0, "candidate_losses": 0}
    rouge_outcomes = {"candidate_wins": 0, "ties": 0, "candidate_losses": 0}
    for question_id in sorted(expected_ids, key=str.encode):
        baseline_meteor, baseline_rouge = baseline_metrics[question_id]
        candidate_meteor, candidate_rouge = candidate_metrics[question_id]
        meteor_key = (
            "candidate_wins"
            if candidate_meteor > baseline_meteor
            else "candidate_losses"
            if candidate_meteor < baseline_meteor
            else "ties"
        )
        rouge_key = (
            "candidate_wins"
            if candidate_rouge > baseline_rouge
            else "candidate_losses"
            if candidate_rouge < baseline_rouge
            else "ties"
        )
        outcomes[meteor_key] += 1
        rouge_outcomes[rouge_key] += 1

    inputs = {
        "comparison": checksum_bytes(comparison_data),
        "grounding": checksum_bytes(grounding_data),
        "baseline_metrics": checksum_bytes(baseline_metrics_data),
        "candidate_metrics": checksum_bytes(candidate_metrics_data),
        "baseline_predictions": checksum_bytes(baseline_predictions_data),
        "candidate_predictions": checksum_bytes(candidate_predictions_data),
        "parameter_manifest": checksum_bytes(parameter_manifest_data),
    }
    return {
        "schema_version": "post_d062.baseline.freeze.v1",
        "d063_status": "SATISFIED_BY_D062_FINALIZATION",
        "post_d062_baseline": {
            "retrieval": "Qwen/Qwen3-Reranker-0.6B base",
            "generator": "Qwen/Qwen3-1.7B G1A512",
        },
        "question_count": expected_question_count,
        "source_checksums": inputs,
        "metrics": comparison.get("metrics"),
        "meteor_outcomes": outcomes,
        "rouge_l_outcomes": rouge_outcomes,
        "answer_token_lengths": {
            "baseline": _length_summary(baseline_predictions),
            "candidate": _length_summary(candidate_predictions),
        },
        "grounding": grounding.get("rates"),
        "system_parameter_count": parameters["system_parameter_count"],
        "parameter_limit_exclusive": parameters["competition_limit_exclusive"],
        "numeric_gate": "passed",
        "grounding_gate": "passed",
        "resource_gate": "passed",
        "replay_gate": "passed",
        "new_inference_runs": 0,
    }


__all__ = ["PostD062Error", "reconcile_d063"]
